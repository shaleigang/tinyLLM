#pragma once

#include "tensor.h"
#include "params.h"

using tllm::detail::ParamsDict;

namespace tllm {
namespace nn {

class Module {
public:
    Module() = default;
    Module(const Module& other) = delete;
    virtual ~Module() = default;

    virtual void print();
    virtual void to(string device);
    virtual void apply_grad(float lr);

    virtual Tensor forward(Tensor& input) = 0;
    virtual ParamsDict parameters(void) = 0;
    
    virtual Tensor operator()(Tensor& input) { return forward(input); }
};

class Linear : public Module {
public:
    Linear(index_t in_features, index_t out_features, bool bias = true);
    ~Linear() = default;

    virtual Tensor forward(Tensor& input) override;
    virtual ParamsDict parameters(void) override;

protected:
    Tensor weight_;
    Tensor bias_;
    bool require_bias_;
};

class LayerNorm : public Module {
public:
    LayerNorm(index_t n_dim, bool bias = true);
    ~LayerNorm() = default;

    virtual Tensor forward(Tensor& input) override;
    virtual ParamsDict parameters(void) override;

private:
    Tensor weight_;
    Tensor bias_;
    bool require_bias_;
};

}
}