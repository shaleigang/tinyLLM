#pragma once

#include "tensor.h"
#include "exp.h"
#include "util.h"

using tllm::detail::UnaryExp;
using tllm::detail::BinaryExp;

namespace tllm {
namespace detail {

class MatMulExp : public BinaryExp {
public:
    Tensor operator()(Tensor& x1, Tensor& x2);

private:
    virtual Tensor generate_ret_tensor(Tensor& x1, Tensor& x2) override;
    virtual void prepare_forward(Tensor& x1, Tensor& x2, Tensor& x) override;
    virtual void forward_process(Tensor& x1, Tensor& x2, Tensor& x) override;

    virtual void lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) override;
    virtual void rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) override;
};

}

extern detail::MatMulExp MatMul;

}