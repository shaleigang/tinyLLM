#pragma once

#include "tensor.h"
#include "params.h"

using tllm::detail::ParamsDict;

namespace tllm {
namespace nn {

class Module {
public:
    virtual Tensor forward(Tensor& input) = 0;
    virtual ParamsDict parameters(void) = 0;
    virtual ~Module() = default;

    virtual void print();
    virtual void to(string device);
};

class Linear : public Module {
public:
    Linear(index_t in_features, index_t out_features, bool bias=true);
    Linear(const Linear& other) = delete;
    ~Linear() = default;

    Tensor forward(Tensor& input) override;
    ParamsDict parameters(void) override;

protected:
    Tensor weight_;
    Tensor bias_;
    bool require_bias_;
};

}
}