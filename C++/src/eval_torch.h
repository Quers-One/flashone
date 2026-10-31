// eval_torch.h — 把 TorchNet 包成 MCTS/selfplay 需要的 Evaluator
#pragma once

#include "mcts.h"
#include "net.h"

namespace gk {

class TorchEvaluator : public Evaluator {
public:
    explicit TorchEvaluator(TorchNet* net) : net_(net) {}
    void evaluate_batch(const float* states, int batch, float* policies, float* values) override {
        net_->infer_batch(states, batch, policies, values);
    }
    TorchNet* net() const { return net_; }

private:
    TorchNet* net_;
};

}  // namespace gk
