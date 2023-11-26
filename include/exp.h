#pragma once

#include "tensor.h"
#include "util.h"

namespace tllm {
namespace detail {

using tllm::Tensor;

class Exp { };

class UnaryExp : public Exp {
public:
    Tensor forward(Tensor& x1);

private:
    virtual Tensor generate_ret_tensor(Tensor& x1);
    virtual void prepare_forward(Tensor& x1, Tensor& x);
    virtual void forward_process(Tensor& x1, Tensor& x) = 0;

    virtual void grad_fn(TensorImplPtr x1, TensorImplPtr x) = 0;
};

class BinaryExp : public Exp {
public:
    Tensor forward(Tensor& x1, Tensor& x2);

private:
    virtual Tensor generate_ret_tensor(Tensor& x1, Tensor& x2);
    virtual void prepare_forward(Tensor& x1, Tensor& x2, Tensor& x);
    virtual void forward_process(Tensor& x1, Tensor& x2, Tensor& x) = 0;

    virtual void lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) = 0;
    virtual void rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) = 0;
};

class Minus : public UnaryExp {
public:
    static Minus& get();

private:
    virtual void forward_process(Tensor& x1, Tensor& x) override;
    virtual void grad_fn(TensorImplPtr x1, TensorImplPtr x) override;
};

class Add : public BinaryExp {
public:
    static Add& get();

private:
    virtual void forward_process(Tensor& x1, Tensor& x2, Tensor& x) override;
    virtual void lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) override;
    virtual void rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) override;
};

class Sub : public BinaryExp {
public:
    static Sub& get();

private:
    virtual void forward_process(Tensor& x1, Tensor& x2, Tensor& x) override;
    virtual void lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) override;
    virtual void rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) override;
};

class Mul : public BinaryExp {
public:
    static Mul& get();

private:
    virtual void forward_process(Tensor& x1, Tensor& x2, Tensor& x) override;
    virtual void lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) override;
    virtual void rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) override;
};

}

}