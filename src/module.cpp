#include "module.h"
#include "function.h"

using namespace tllm::nn;
using tllm::Tensor;

void Module::print() {
    for (auto iter: parameters()) {
        std::cout << iter.first << std::endl;
        std::cout << iter.second << std::endl;
    }
}

void Module::to(string device) {
    for (auto iter: parameters()) {
        iter.second.get().to(device);
    }
}

Linear::Linear(index_t in_features, index_t out_features, bool bias)
  : weight_(Tensor({in_features, out_features})),
    require_bias_(bias) {
    if (require_bias_) {
        bias_ = Tensor({out_features});
    }
    for (int i = 0; i < (in_features * out_features); ++i) {
        weight_[i] = i;
    }
    if (require_bias_) {
        for (int i = 0; i < (out_features); ++i) {
            bias_[i] = i;
        } 
    } 
}

Tensor Linear::forward(Tensor& input) {
    Tensor ret = MatMul(input, weight_);
    if (require_bias_) {
        Tensor ret2 = ret + bias_;
        return ret2;
    }
    else {
        return ret;
    }
}

ParamsDict Linear::parameters() {
    if (require_bias_) {
        return {
            {"weight", weight_},
            {"bias", bias_}
        };
    }
    else {
        return {
            {"weight", weight_}
        };
    }
}



