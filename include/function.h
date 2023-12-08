#pragma once

#include "tensor.h"
#include "exp.h"
#include "util.h"

#include <random>

using tllm::detail::UnaryExp;
using tllm::detail::BinaryExp;

namespace tllm {
namespace detail {

class UnaryFunc : public UnaryExp {
public:
    virtual Tensor operator()(Tensor& x1);
};

class BinaryFunc : public BinaryExp {
public:
    virtual Tensor operator()(Tensor& x1, Tensor& x2);
};

class MatMulExp : public BinaryFunc {
private:
    virtual Tensor generate_ret_tensor(Tensor& x1, Tensor& x2) override;
    virtual void prepare_forward(Tensor& x1, Tensor& x2, Tensor& x) override;
    virtual void forward_process(Tensor& x1, Tensor& x2, Tensor& x) override;

    virtual void lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) override;
    virtual void rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) override;
};

class LayerNorm : public UnaryFunc {
public:
    LayerNorm(float ep) : ep_(ep) {};

private:
    virtual void forward_process(Tensor& x1, Tensor& x) override;
    virtual void grad_fn(TensorImplPtr x1, TensorImplPtr x) override;

private:
    float ep_;
};

class Dropout : public UnaryFunc {
public:
    Dropout(float prob) : prob_(prob), limit_(100 * prob), di(0, 100), dre(time(0)) {};

private:
    virtual void forward_process(Tensor& x1, Tensor& x) override;
    virtual void grad_fn(TensorImplPtr x1, TensorImplPtr x) override;

private:
    float prob_;
    int limit_;
    std::default_random_engine dre;
    std::uniform_int_distribution<int> di;
};

class GELU : public UnaryFunc {
public:
    GELU() = default;

private:
    virtual void forward_process(Tensor& x1, Tensor& x) override;
    virtual void grad_fn(TensorImplPtr x1, TensorImplPtr x) override;

};

class Softmax : public UnaryFunc {
public:
    Softmax() = default;

private:
    virtual void forward_process(Tensor& x1, Tensor& x) override;
    virtual void grad_fn(TensorImplPtr x1, TensorImplPtr x) override;

    void before_softmax(Tensor& x1);

};

class Log : public UnaryFunc {
public:
    Log() = default;

private:
    virtual void forward_process(Tensor& x1, Tensor& x) override;
    virtual void grad_fn(TensorImplPtr x1, TensorImplPtr x) override;

};

class NLLLoss : public BinaryFunc {
public:
    NLLLoss() = default;

private:
    virtual Tensor generate_ret_tensor(Tensor& x1, Tensor& x2) override;
    virtual void prepare_forward(Tensor& x1, Tensor& x2, Tensor& x) override;
    virtual void forward_process(Tensor& x1, Tensor& x2, Tensor& x) override;

    virtual void lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) override;
    virtual void rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) override;
};

class Emb : public BinaryFunc {
public:
    Emb() = default;

private:
    virtual Tensor generate_ret_tensor(Tensor& x1, Tensor& x2) override;
    virtual void prepare_forward(Tensor& x1, Tensor& x2, Tensor& x) override;
    virtual void forward_process(Tensor& x1, Tensor& x2, Tensor& x) override;

    virtual void lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) override;
    virtual void rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) override;
};

}

namespace F {
extern detail::MatMulExp mat_mul;
extern detail::LayerNorm layer_norm;
extern detail::Softmax softmax;
extern detail::Log log;
extern detail::NLLLoss nll_loss;
extern detail::Emb emb;

Tensor cross_entropy(Tensor& x1, Tensor& x2);
void causal_mask_fill(Tensor& att);
}


}