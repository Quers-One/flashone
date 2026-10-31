"""
network.py
=========
五子棋「策略/价值共享网络」(KataGo 同款双头架构)。

输入：3×19×19 的局面张量（来自 gomoku_env.Board.get_state_tensor()）
      通道 0=黑棋位置, 1=白棋位置, 2=当前执子方(全1黑/全0白)

结构：
    3 残差块骨干 (通道 64, 3×3 卷积, 零填充保尺寸)
    └── 策略头 (Policy head)：1 卷积 + 全连接 -> 361 维落子概率 (softmax)
    └── 价值头 (Value head)  ：1 卷积 + 全连接 + tanh -> [-1,1] 标量 (当前执子方胜率)

特性：
    * 支持批量输入 (B, 3, 19, 19)
    * CPU/GPU 自动切换 (cuda 可用则用 GPU，否则 CPU)
    * policy_value(state)：输入单个局面张量，返回 (策略概率数组, 价值标量)
    * 【任务 6】可选辅助头（auxiliary heads）：ownership / threat。
      默认 `aux=""` ⇒ 一个都不建，**结构与旧权重完全兼容**。
      辅助头只在 `forward_with_aux()` 里计算；`forward()`（推理与自对弈走的
      都是它）一行都不多算，所以加辅助头**不会让 MCTS 变慢**。
"""

import numpy as np

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    单个残差块：2 层 3×3 卷积 + BatchNorm + ReLU，加跳跃连接。
    输入/输出通道数相同（64），零填充保持 19×19 尺寸不变。
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # 若通道数变化则需投影，这里 in_ch==out_ch 直接跳跃
        self.identity = nn.Identity() if in_ch == out_ch else nn.Conv2d(
            in_ch, out_ch, kernel_size=1, bias=False
        )

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = h + self.identity(x)
        return torch.relu(h)


class GomokuNet(nn.Module):
    """
    五子棋策略/价值共享网络。

    头部结构说明（相比旧版的重要修复）
    ----------------------------------
    旧版两个头都是"卷积降维到 32 通道 → **展平** → 全连接"：
        policy_fc: Linear(32*19*19, 361) = 4.16M 参数（占全网 95%）
        value_fc : Linear(32*19*19, 1)    = 11.5k 参数
    这带来两个严重问题（实测确认）：
        1) 参数量极度浪费且破坏平移等变性：一次全连接就把"每个位置一个特征"
           摊平混合，等于放弃了卷积的局部性归纳偏置。
        2) **价值头会直接背下训练集**：value_fc 的容量足以对少量样本记住标签，
           实测损失在 1 个 step 内从 1.47 掉到 0.0001，而网络输出仍是噪声
           （对面 8 个局面的输出几乎相同）。价值头学不到任何东西。
    现在改为标准做法：
        * 策略头：1×1 卷积直接输出 361 个 logits（全卷积，参数量 ~46k）
        * 价值头：1×1 卷积降维 → **全局平均池化** → 小全连接 → tanh
          全局池化强制"对全盘形势做整体判断"，无法记住单个局面。

    骨干规模可用 n_blocks / channels 调整（默认 3 块/64 通道 = 与旧版权重兼容）。
    """

    BOARD = 19
    N = BOARD * BOARD  # 361 个落子点

    # 【任务 6】辅助头的名字 -> 输出通道数。多出来的通道都落在棋盘空间上
    # （(B,C,19,19)），因为辅助任务的答案本身就是逐点给出的。
    AUX_OUT_CH = {
        "ownership": 1,   # 归属：每点 -1(对方) .. +1(己方)，tanh 输出
        "threat": 2,      # 威胁：ch0=己方落此处即连五，ch1=对方落此处即连五
    }

    def __init__(self, n_blocks: int = 3, channels: int = 64,
                 value_hidden: int = 64, aux: str = "", aux_ch: int = 32):
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.channels = int(channels)
        self.aux_ch = int(aux_ch)
        # 辅助头容器：默认空 ⇒ state_dict 里一个 aux_heads.* 键都没有，
        # 因此旧权重能**严格(strict=True)加载**，老脚本/老 ckpt 全部照旧可用。
        self.aux_heads = nn.ModuleDict()

        # ---- 骨干网络 ----
        # 初始卷积：3 通道 -> channels（零填充保 19×19）
        self.stem = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.backbone = nn.Sequential(
            *[ResidualBlock(channels, channels) for _ in range(self.n_blocks)]
        )

        # ---- 策略头 (Policy head)：全卷积 ----
        # 64 -> 32 卷积 + 1×1 卷积直接给 361 个 logits，softmax 归一化
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=3, padding=1,
                                     bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_out = nn.Conv2d(32, 1, kernel_size=1)   # -> (B,1,19,19)

        # ---- 价值头 (Value head)：全局平均池化 ----
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        # 自动选择设备：cuda 可用则 GPU，否则 CPU。
        # （提到建辅助头之前：辅助头一建好就要落到正确设备上）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 权重初始化（He/Kaiming，残差网络的标准做法）
        self._init_weights()

        # 【任务 6】辅助头放在 _init_weights 之后建：_init_weights 会给所有
        # Conv2d 做 kaiming，而辅助头的输出层需要"小初始化"，建完再单独设。
        if aux:
            self.add_aux_heads(aux)

        self.to(self.device)

    def _init_weights(self):
        """He 初始化：卷积/全连接用 kaiming_normal，BN 权重置 1、偏置置 0。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # 两个输出头用小初始化，开局输出接近"均匀策略 / 和棋价值"，
        # 避免一开始就把先验推得极端
        nn.init.normal_(self.policy_out.weight, std=0.01)
        nn.init.zeros_(self.policy_out.bias)
        nn.init.normal_(self.value_fc2.weight, std=0.01)
        nn.init.zeros_(self.value_fc2.bias)

    # ------------------------------------------------------------------
    # 【任务 6】辅助头（auxiliary heads）
    # ------------------------------------------------------------------
    @classmethod
    def _parse_aux(cls, aux) -> tuple:
        """
        把 `"threat,ownership"` 解析成去重保序的名字元组并校验合法性。
        接受逗号/分号分隔的字符串，也接受 list/tuple；空串 -> ()。
        """
        if isinstance(aux, (list, tuple)):
            items = [str(a) for a in aux]
        else:
            items = str(aux or "").replace(";", ",").split(",")
        out = []
        for raw in items:
            name = raw.strip()
            if not name:
                continue
            if name not in cls.AUX_OUT_CH:
                raise ValueError(
                    f"不支持的辅助头 {name!r}；目前只支持 {sorted(cls.AUX_OUT_CH)}")
            if name not in out:
                out.append(name)
        return tuple(out)

    def _make_aux_head(self, name: str) -> nn.Module:
        """
        建一个辅助头：1×1 卷积降维 -> ReLU -> 1×1 卷积输出到棋盘空间。

        为什么**不用 BatchNorm**（与骨干和两个主头不同）：
            `fused_copy()` 是靠"把 conv+BN 逐对折进 conv"来加速推理的，它只认识
            stem/policy_conv/value_conv/backbone 这几对。辅助头里塞 BN 会有两个
            后果：① 融合逻辑必须跟着改（而辅助头根本不在推理路径上，白改）；
            ② 多一份 running stats 需要同步。头部本身很浅，且骨干末尾已有
            BN+ReLU，省掉它不影响拟合。
        """
        head = nn.Sequential(
            nn.Conv2d(self.channels, self.aux_ch, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.aux_ch, self.AUX_OUT_CH[name], kernel_size=1),
        )
        nn.init.kaiming_normal_(head[0].weight, mode="fan_out", nonlinearity="relu")
        # 输出层小初始化：开局先输出"没有威胁 / 归属不明"，
        # 不要一上来就把 logits 推得极端（威胁正样本极稀疏，见 train.aux_targets）
        nn.init.normal_(head[-1].weight, std=0.01)
        nn.init.zeros_(head[-1].bias)
        return head.to(self.device)

    def add_aux_heads(self, aux, aux_ch: int = None) -> tuple:
        """
        就地补建辅助头（已存在的保持不变），返回本次新建的名字元组。

        【为什么需要"就地补建"】任务 6 的标准用法是**热启动**：拿一个已有的
        6×128 checkpoint（例如 `--init-weights models/V3/gen_110.pt`）继续训，
        而那个权重文件里没有辅助头。如果在构造时一次性定型，就只能从零重训 ——
        白丢全部历史数据与进度。有了这个方法，主干权重原样保留，只多挂两个小头，
        优化器随后统一接管（`net.parameters()` 此时已包含辅助头）。
        """
        if aux_ch:
            self.aux_ch = int(aux_ch)
        added = []
        for name in self._parse_aux(aux):
            if name in self.aux_heads:
                continue
            self.aux_heads[name] = self._make_aux_head(name)
            added.append(name)
        return tuple(added)

    @property
    def aux_names(self) -> tuple:
        """当前网络的辅助头名字（顺序 = ModuleDict 插入顺序；空 = 没有辅助头）。"""
        return tuple(self.aux_heads.keys())

    def fused_copy(self):
        """
        返回一个**融合了 Conv+BN 的推理副本**；原网络完全不动。

        为什么用"副本"而不是"就地融合"
        ------------------------------
        就地融合会把 BN 模块换成 `Identity()`，于是 `load_state_dict` 再也还原不了
        结构（BN 的 running stats 变成多余键、conv 多出 bias）——"融合后想接着训练"
        就会毁掉权重。做成副本后这个风险从设计上消失：
            infer_net = net.fused_copy()      # 推理用
            net                                # 训练照旧
        代价只是一次 1MB 量级的 deepcopy。

        折入公式（本网络 conv 全是 bias=False，b=0）：
            y = γ·(W·x + b − μ)/√(σ²+ε) + β
              = (γ/√(σ²+ε))·W·x + ( β − γ·μ/√(σ²+ε) )
        折叠后推理时少掉**全部 13 个 BN 核函数**。小网络上 GPU 前向是
        "核函数启动延迟受限"，所以省核函数直接换速度。
        """
        import copy as _copy
        net = _copy.deepcopy(self)
        net.eval()
        if getattr(net, "_bn_fused", False):
            return net

        def fold(conv, bn):
            if not isinstance(bn, nn.BatchNorm2d):
                return
            w = conv.weight.detach()
            scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
            new_w = w * scale.view(-1, 1, 1, 1)
            base = (torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
                    if conv.bias is None else conv.bias.detach())
            new_b = bn.bias.detach() + (base - bn.running_mean.detach()) * scale
            conv.weight.data.copy_(new_w)
            if conv.bias is None:
                conv.bias = nn.Parameter(new_b)
            else:
                conv.bias.data.copy_(new_b)

        fold(net.stem, net.stem_bn)
        fold(net.policy_conv, net.policy_bn)
        fold(net.value_conv, net.value_bn)
        for blk in net.backbone:
            fold(blk.conv1, blk.bn1)
            fold(blk.conv2, blk.bn2)

        net.stem_bn = nn.Identity()
        net.policy_bn = nn.Identity()
        net.value_bn = nn.Identity()
        for blk in net.backbone:
            blk.bn1 = nn.Identity()
            blk.bn2 = nn.Identity()
        net._bn_fused = True
        return net

    # ------------------------------------------------------------------
    # 架构读写：让 checkpoint 不再隐含"3 块 × 64 通道"
    # ------------------------------------------------------------------
    @staticmethod
    def infer_arch(state_dict: dict) -> tuple:
        """
        从 state_dict 的键与形状反推 (n_blocks, channels, value_hidden)。

        【为什么必须有】放大骨干（比如 6 块 × 128 通道）之后，旧加载路径
        `GomokuNet()` + `load_state_dict(strict=False)` 会**静默**把新骨干的
        权重全当"多余层"丢掉，得到一个随机初始化的 3×64 网络 —— 评测工具照样
        跑完、照样打印比分，结论却全是假的（只有 verbose 时打一行提示）。
        所以统一改成"先反推架构 → 再建网 → 再加载"。
        """
        w = state_dict.get("stem.weight")
        if w is None:
            raise KeyError("state_dict 里没有 stem.weight，无法反推架构")
        channels = int(w.shape[0])
        n_blocks = 0
        while f"backbone.{n_blocks}.conv1.weight" in state_dict:
            n_blocks += 1
        vh = state_dict.get("value_fc1.weight")
        value_hidden = int(vh.shape[0]) if vh is not None else 64
        return n_blocks, channels, value_hidden

    @classmethod
    def infer_aux(cls, state_dict: dict) -> dict:
        """
        【任务 6】从 state_dict 反推辅助头配置。

        返回 `{"aux": "threat,ownership", "aux_ch": 32}`；没有辅助头时返回
        `{"aux": "", "aux_ch": 32}`。

        【顺序必须逐字保留，不能"规范化"】`ModuleDict` 的注册顺序就是
        `net.parameters()` 的顺序，而优化器状态（动量）是**按位置**存的。
        如果这里按 `AUX_OUT_CH` 的规范顺序输出，那么"用 `infer_aux` 的结果重建
        网络"就会得到一个参数顺序不同的网络 —— 参数**个数**相同，于是
        `load_state_dict` 不会报错，但动量会静默灌到错误的张量上。实测过：
        建网用 `threat,ownership`、反推按规范序回 `ownership,threat`，续训第一步
        就在 `_multi_tensor_adam` 里抛
        `RuntimeError: output with shape [1, 32, 1, 1] doesn't match the
        broadcast shape [2, 32, 1, 1]`（归属头的动量撞上了威胁头的梯度）。
        所以这里按 **state_dict 里的键顺序**输出。

        【为什么单开一个方法，而不是让 infer_arch 多返回几项】
        `infer_arch` 的返回值元数有 3 处调用方在断言/解包
        （`train.py:1441`、`register_models.py:75`、`test_arch.py:57-60`
        断言 `(3,64,64)` / `(6,128,64)`）。改它的元数会连带打断这些地方，
        所以这里**只增不改**。
        """
        names = []
        aux_ch = 32
        # 按 state_dict 的键顺序（= ModuleDict 注册顺序 = parameters() 顺序）遍历，
        # 只认第一个 1×1 conv 权重 `aux_heads.<name>.0.weight` 作为"这个头存在"的标志。
        suffix = ".0.weight"
        for key in (state_dict or {}):
            if not (key.startswith("aux_heads.") and key.endswith(suffix)):
                continue
            name = key[len("aux_heads."):-len(suffix)]
            if name not in names:
                names.append(name)
                aux_ch = int(state_dict[key].shape[0])
        return {"aux": ",".join(names), "aux_ch": aux_ch}

    def arch(self) -> dict:
        """当前网络的架构描述（写进 checkpoint，便于人工核对）。"""
        return {"n_blocks": int(self.n_blocks), "channels": int(self.channels),
                "value_hidden": int(self.value_fc1.weight.shape[0]),
                # 【任务 6】辅助头：老 checkpoint 没这两个键，读的时候按默认取
                "aux": ",".join(self.aux_names),
                "aux_ch": int(self.aux_ch)}

    @classmethod
    def from_arch(cls, arch: dict):
        """按 arch 字典（save_checkpoint 写出的格式）建网（含辅助头）。"""
        arch = arch or {}
        return cls(n_blocks=int(arch.get("n_blocks", 3)),
                   channels=int(arch.get("channels", 64)),
                   value_hidden=int(arch.get("value_hidden", 64)),
                   aux=arch.get("aux", "") or "",
                   aux_ch=int(arch.get("aux_ch", 32)))

    def _trunk(self, states):
        """骨干前向：把 (B,3,19,19) 局面张量变成 (B,channels,19,19) 特征图。"""
        x = torch.relu(self.stem_bn(self.stem(states)))
        return self.backbone(x)

    def _policy_value_heads(self, x):
        """在骨干特征图上算策略头与价值头（两个主头）。"""
        # 策略头：全卷积 -> (B,1,19,19) -> (B,361)
        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_out(p).flatten(1)
        policy = torch.softmax(p, dim=1)

        # 价值头：1×1 卷积 -> 全局平均池化 -> 两层全连接 -> tanh
        v = torch.relu(self.value_bn(self.value_conv(x)))
        v = v.mean(dim=(2, 3))                     # (B,32) 全局平均池化
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)   # (B,)
        return policy, value

    def forward(self, states):
        """
        前向传播（支持批量）。

        **推理与自对弈走的都是这条路径**，它**不计算辅助头** —— 所以加辅助头
        不会让 MCTS 慢一分（辅助头只在 `forward_with_aux()` 里算）。

        参数:
            states: torch.Tensor (B, 3, 19, 19)，可 batch=1

        返回:
            policy: (B, 361) 策略概率 (softmax)
            value:  (B,)     价值标量 (tanh, [-1,1])
        """
        return self._policy_value_heads(self._trunk(states))

    def forward_with_aux(self, states):
        """
        训练用前向：一次骨干前向同时算两个主头与全部辅助头。

        参数:
            states: torch.Tensor (B, 3, 19, 19)

        返回:
            policy: (B, 361) 策略概率 (softmax)
            value:  (B,)     价值标量 (tanh, [-1,1])
            aux:    dict[name] -> Tensor
                    * "ownership": (B,1,19,19)，已 tanh 到 [-1,1]
                    * "threat"   : (B,2,19,19)，**未过 sigmoid 的 logits**
                      （配 `binary_cross_entropy_with_logits`：数值更稳、
                       且支持 pos_weight —— 威胁正样本极稀疏，必须加权）
        """
        x = self._trunk(states)
        policy, value = self._policy_value_heads(x)
        aux = {}
        for name in self.aux_names:
            h = self.aux_heads[name](x)
            aux[name] = torch.tanh(h) if name == "ownership" else h
        return policy, value, aux

    def policy_value(self, state):
        """
        输入单个局面张量，返回 (策略概率数组, 价值标量)。

        参数:
            state: 单个局面张量，numpy (3,19,19) 或 torch (3,19,19)

        返回:
            policy: numpy (361,) 落子概率 (softmax)
            value:  float  价值标量 (当前执子方胜率, [-1,1])
        """
        # 转为 torch 并加 batch 维（统一转为 float32，网络卷积需浮点）
        if isinstance(state, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(state))
        else:
            t = torch.as_tensor(state)
        t = t.float().to(self.device)
        if t.dim() == 3:
            t = t.unsqueeze(0)  # (3,19,19) -> (1,3,19,19)

        with torch.no_grad():
            policy, value = self.forward(t)

        return (
            policy[0].cpu().numpy(),
            float(value[0].cpu()),
        )

    def evaluate_batch(self, states):
        """
        批量评估。

        参数:
            states: numpy (B,3,19,19) 或 torch (B,3,19,19)

        返回:
            policy: numpy (B,361)
            value:  numpy (B,)
        """
        if isinstance(states, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(states))
        else:
            t = torch.as_tensor(states)
        t = t.float().to(self.device)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        with torch.no_grad():
            policy, value = self.forward(t)
        return policy.cpu().numpy(), value.cpu().numpy()


# ----------------------------------------------------------------------
# 自测：随机输入验证输出维度正确
# ----------------------------------------------------------------------
def self_test():
    """随机输入验证：前向输出维度、policy_value 单局输出、批量输出。"""
    print("=" * 60)
    print("network.py 自测")
    print("=" * 60)

    net = GomokuNet()
    print(f"设备: {net.device}")

    # ---- 1) 单局 policy_value ----
    state = np.random.rand(3, 19, 19).astype(np.float32)
    policy, value = net.policy_value(state)
    assert policy.shape == (361,), f"policy 形状错误: {policy.shape}"
    assert np.isclose(policy.sum(), 1.0, atol=1e-5), "policy 未归一化"
    assert np.all(policy >= 0), "policy 存在负值"
    assert -1.0 <= value <= 1.0, f"value 超出 [-1,1]: {value}"
    print(f"[PASS] policy_value: policy {policy.shape} (sum={policy.sum():.6f}), value={value:.4f}")

    # ---- 2) 前向批量 (B,3,19,19) -> (B,361) / (B,) ----
    B = 8
    batch = np.random.rand(B, 3, 19, 19).astype(np.float32)
    policy_b, value_b = net.evaluate_batch(batch)
    assert policy_b.shape == (B, 361), f"批量 policy 形状错误: {policy_b.shape}"
    assert value_b.shape == (B,), f"批量 value 形状错误: {value_b.shape}"
    assert np.all(np.isclose(policy_b.sum(axis=1), 1.0, atol=1e-4)), "批量 policy 未归一化"
    print(f"[PASS] evaluate_batch: policy {policy_b.shape}, value {value_b.shape}")

    # ---- 3) torch 输入 policy_value ----
    state_t = torch.from_numpy(np.random.rand(3, 19, 19).astype(np.float32))
    policy_t, value_t = net.policy_value(state_t)
    assert policy_t.shape == (361,), f"torch policy 形状错误: {policy_t.shape}"
    print("[PASS] torch 输入 policy_value 正常")

    # ---- 4) 参数数量统计（策略头不应再占绝大部分）----
    n_params = sum(p.numel() for p in net.parameters())
    policy_head = sum(p.numel() for p in net.policy_out.parameters()) + \
        sum(p.numel() for p in net.policy_conv.parameters())
    print(f"[PASS] 网络参数: {n_params/1e6:.3f}M "
          f"(策略头仅 {policy_head/1e3:.0f}k，旧版策略 FC 占 4.16M/95%)")
    assert n_params < 1_000_000, "参数异常偏多（旧版头部问题是否复发？）"

    # ---- 5) 价值头必须能学（旧架构的致命缺陷回归测试）----
    # 旧版 value_fc = Linear(32*19*19, 1) 能直接背下小数据集：
    # 实测 loss 在 1 个 step 内从 1.47 掉到 0.0001 而输出仍是噪声。
    # 这里构造一个**可学**的任务（黑子多 -> +1），要求验证集 MSE 明显下降。
    import numpy as _np
    rng = _np.random.default_rng(0)
    N = 400
    xs = _np.zeros((N, 3, 19, 19), dtype=_np.uint8)
    ys = _np.zeros(N, dtype=_np.float32)
    for i in range(N):
        nb = int(rng.integers(1, 8))
        nw = int(rng.integers(1, 8))
        for _ in range(nb):
            xs[i, 0, int(rng.integers(0, 19)), int(rng.integers(0, 19))] = 1
        for _ in range(nw):
            xs[i, 1, int(rng.integers(0, 19)), int(rng.integers(0, 19))] = 1
        ys[i] = 1.0 if nb > nw else (-1.0 if nw > nb else 0.0)
    ps = _np.full((N, 361), 1.0 / 361, dtype=_np.float32)

    probe = None
    probe_opt = None

    def _val_mse():
        probe.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, 80, 80):
                t = torch.from_numpy(xs[i:i + 80].astype(_np.float32)).to(probe.device)
                _p, v = probe(t)
                tot += float(_np.mean((v.cpu().numpy() - ys[i:i + 80]) ** 2))
        probe.train()
        return tot

    from train import Config as _Cfg, train_step as _step
    c = _Cfg(); c.batch_size = 64; c.on_the_fly_augment = False
    # 固定初始化种子：这是一条**结构回归**测试，要的是"能不能学"，
    # 不是"这次随机初始化运气好不好"。实测不同种子下 ratio 在 0.16~0.74 之间
    # 波动，不固定种子会偶发假失败（曾出现 0.9253->0.6880 恰好卡在阈值上）。
    torch.manual_seed(1234)
    probe = GomokuNet()
    probe_opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    e0 = _val_mse()
    for _ in range(150):
        _step(probe, (xs[80:], ps[80:], ys[80:]), c, probe_opt,
              buffer=None, rng=rng, ema=None)
    e1 = _val_mse()
    print(f"[PASS] 价值头可学习性: 验证集 MSE {e0:.4f} -> {e1:.4f} "
          f"(ratio={e1 / e0:.3f}，阈值 0.70)")
    assert e1 < e0 * 0.7, \
        f"价值头学不动（MSE {e0:.4f}->{e1:.4f}），头部结构可能有问题"

    # ---- 【任务 6】辅助头：结构 + 前向 + 热启动 + 反推 ----
    #
    # 这里只做**结构与管线**回归（不测拟合质量：辅助头有没有用，只能靠独立对战
    # 用 Elo/CI 判定，不能靠 loss）。要守住的四件事：
    #   ① 默认 aux="" 时不建任何辅助头，state_dict 与老权重逐字节兼容
    #      （因此历史 ckpt 依旧能 strict=True 加载）；
    #   ② 带辅助头时 forward_with_aux 三路形状正确，且两个**主头**输出与
    #      forward() 完全一致 —— 证明辅助头没混进推理路径、不拖慢 MCTS；
    #   ③ 热启动（老权重 + add_aux_heads）不动主干任何一个参数；
    #   ④ infer_aux 能从 state_dict 反推回辅助头配置，infer_arch 元数不变。
    base = GomokuNet()
    assert base.aux_names == (), "默认不应建任何辅助头"
    assert not any(k.startswith("aux_heads.") for k in base.state_dict())

    hot = GomokuNet()
    hot.eval()          # eval 模式下前向不会更新 BN 统计量，下面的比对才干净
    trunk_before = {k: v.clone() for k, v in hot.state_dict().items()}
    hot.add_aux_heads("threat,ownership")
    assert set(hot.aux_names) == {"threat", "ownership"}
    assert all(torch.equal(trunk_before[k], hot.state_dict()[k])
               for k in trunk_before), "add_aux_heads 动了主干权重"

    _x = torch.rand(3, 3, 19, 19, device=hot.device)
    _p1, _v1, _aux = hot.forward_with_aux(_x)
    assert _aux["threat"].shape == (3, 2, 19, 19), "威胁头形状不对"
    assert _aux["ownership"].shape == (3, 1, 19, 19), "归属头形状不对"
    assert float(_aux["ownership"].abs().max()) <= 1.0
    _p2, _v2 = hot.forward(_x)
    assert torch.allclose(_p1, _p2) and torch.allclose(_v1, _v2), \
        "forward 与 forward_with_aux 的主头输出不一致 => 辅助头污染了推理路径"

    _info = GomokuNet.infer_aux(hot.state_dict())
    # 【必须逐字保留顺序，不能"规范化"】ModuleDict 的注册顺序决定
    # `net.parameters()` 的顺序，而优化器动量是按位置存的 —— 一旦反推时换了
    # 顺序，"等参数个数"的续训就会把动量灌到错误的张量上（实测：torch 在
    # `_multi_tensor_adam` 里抛 broadcast 形状错，ownership(1,32,1,1) 的
    # exp_avg 撞上 threat(2,32,1,1) 的梯度）。所以这里断言的是**相等**，
    # 不是"集合相同"。
    assert _info["aux"] == ",".join(hot.aux_names) == "threat,ownership", \
        f"infer_aux 没有逐字保留辅助头顺序：{_info['aux']!r} vs {hot.aux_names}"
    assert _info["aux_ch"] == hot.aux_ch, "infer_aux 反推 aux_ch 失败"
    assert GomokuNet.infer_aux(base.state_dict())["aux"] == ""
    # 反向顺序也要原样反推（证明它真的按存档顺序，而不是按规范顺序）
    _rev = GomokuNet(channels=8, aux="ownership,threat", aux_ch=8)
    assert GomokuNet.infer_aux(_rev.state_dict())["aux"] == "ownership,threat", \
        "infer_aux 把顺序规范化了 => 等参数个数的续训会把动量灌错位置"
    assert _rev.aux_names == ("ownership", "threat")
    _re = GomokuNet.from_arch(hot.arch())
    _re.load_state_dict(hot.state_dict(), strict=True)
    assert len(GomokuNet.infer_arch(hot.state_dict())) == 3, "infer_arch 元数被改了"
    print(f"[PASS] 辅助头: 默认无头且兼容旧权重；forward_with_aux 形状正确；"
          f"推理路径与主头输出一致；热启动不动主干；"
          f"infer_aux 反推 {_info['aux']} (aux_ch={_info['aux_ch']})")

    print("=" * 60)
    print("全部自测通过 [PASS ALL]")
    print("=" * 60)


if __name__ == "__main__":
    self_test()
