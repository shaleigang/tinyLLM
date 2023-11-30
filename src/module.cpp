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

void Module::apply_grad(float lr) {
    for (auto iter: parameters()) {
        Tensor& t = iter.second.get();
        if (t.require_grad()) {
            t.apply_grad(lr);
        }
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
    Tensor ret = mat_mul(input, weight_);
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

LayerNorm::LayerNorm(index_t n_dim, bool bias)
  : weight_(Tensor({n_dim})),
    require_bias_(bias) {
    if (require_bias_) {
        bias_ = Tensor({n_dim});
    }
    for (int i = 0; i < (n_dim); ++i) {
        weight_[i] = i;
    }
    if (require_bias_) {
        for (int i = 0; i < (n_dim); ++i) {
            bias_[i] = i;
        } 
    } 
}

Tensor LayerNorm::forward(Tensor& input) {
    Tensor x1 = layer_norm(input);
    Tensor x2 = x1 * weight_;
    if (require_bias_) {
        Tensor x3 = x2 + bias_;
        return x3;
    }
    else {
        return x2;
    }
}

ParamsDict LayerNorm::parameters() {
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



