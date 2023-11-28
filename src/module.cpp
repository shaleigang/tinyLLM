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

Linear::Linear(index_t in_features, index_t out_features)
  : weight_(Tensor({in_features, out_features})),
    bias_(Tensor({out_features})) {
    for (int i = 0; i < (in_features * out_features); ++i) {
        weight_[i] = i;
    }
    for (int i = 0; i < (out_features); ++i) {
        bias_[i] = i;
    }  
}

Tensor Linear::forward(Tensor& input) {
    Tensor ret = MatMul(input, weight_);
    Tensor ret2 = ret + bias_;
    return ret2;
    // return MatMul(input, weight_) + bias_;
}

ParamsDict Linear::parameters() {
    return {
        {"weight", weight_},
        {"bias", bias_}
    };
}



