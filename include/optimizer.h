#pragma once

#include "params.h"

namespace tllm {

class AdamW {
public:
    AdamW(detail::ParamsDict decay_params, 
            detail::ParamsDict nodecay_params,
            float lr,
            float beta1,
            float beta2,
            string device);

    void step();

private:
    detail::ParamsDict decay_params_;
    detail::ParamsDict nodecay_params_;
    float lr_;
    float beta1_;
    float beta2_;
    float weight_decay_ = 0.1;
    std::unordered_map<string, std::shared_ptr<float>> moment1_;
    std::unordered_map<string, std::shared_ptr<float>> moment2_;
    float ep = 1e-8;
    index_t t;
    string device_;
};

}