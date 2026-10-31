// sysutil.h — 平台小工具
#pragma once

#include <string>
#include <vector>

namespace gk {

// 把当前进程降到 BelowNormal，便于与游戏等前台程序共存；返回是否成功
bool set_below_normal_priority();

// 当前可执行文件的绝对路径（多进程自对弈时用来启动同名 exe 的子命令）
std::string exe_path();

// 并行启动多个子进程并等待全部结束。
//   cmds: 每个元素是一条完整命令行（不含可执行文件名，例如 "worker --net x.bin"）
//   codes: 若非空，按顺序填退出码
// 返回 true 表示全部子进程退出码都是 0。
bool run_parallel(const std::vector<std::string>& cmds, std::vector<int>* codes);

}  // namespace gk
