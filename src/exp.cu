#include <iostream>

#include "exp.h"

using namespace tllm::detail;
using tllm::Tensor;

using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;

__global__ void add_kernel(float* ret, float* data1, float* data2, index_t dsize);
__global__ void sub_kernel(float* ret, float* data1, float* data2, index_t dsize);
__global__ void minus_kernel(float* ret, float* data1, index_t dsize);

Tensor UnaryExp::forward(Tensor& x1) {
    Tensor x = generate_ret_tensor(x1);
    prepare_forward(x1, x);
    forward_process(x1, x);
    return std::move(x);
}

Tensor UnaryExp::generate_ret_tensor(Tensor& x1) {
    return Tensor(x1.shape(), x1.device());
}

void UnaryExp::prepare_forward(Tensor& x1, Tensor& x) {
    // 1. check

    // 2. add node to return Tensor
    GraphNodePtr node = std::make_shared<UnaryGraphNode>(x.get());
    node->setGradFn(std::bind(&UnaryExp::grad_fn, this, _1, _2));
    x.setNode(node);

    // 3.increase input Tensor ref count and contiguous
    x1.increaseRef();
    x1.contiguous();

    return;
}

Tensor BinaryExp::forward(Tensor& x1, Tensor& x2) {
    Tensor x = generate_ret_tensor(x1, x2);
    prepare_forward(x1, x2, x);
    forward_process(x1, x2, x);
    return std::move(x);
}

Tensor BinaryExp::generate_ret_tensor(Tensor& x1, Tensor& x2) {
    return Tensor(x1.shape(), x1.device());
}

void BinaryExp::prepare_forward(Tensor& x1, Tensor& x2, Tensor& x) {
    // 1. check
    if (x1.device() != x2.device()) {
        std::cout << "TensorImplPtr must in same device." << std::endl;
        exit(0);
    }

    if (x1.shape() != x2.shape()) {
        std::cout << "TensorImplPtr not in same shape." << std::endl;
        exit(0);
    }

    // 2. add node to return Tensor
    GraphNodePtr node = std::make_shared<BinaryGraphNode>(x1.get(), x2.get());
    node->setGradFnL(std::bind(&BinaryExp::lhs_grad_fn, this, _1, _2, _3));
    node->setGradFnR(std::bind(&BinaryExp::rhs_grad_fn, this, _1, _2, _3));
    x.setNode(node);
    // 3. increase input Tensor ref count and contiguous
    x1.increaseRef();
    x2.increaseRef();
    x1.contiguous();
    x2.contiguous();

    return;
}

void Minus::forward_process(Tensor& x1, Tensor& x) {
    if (x.device() == "cpu") {
        for (int i = 0; i < x.dsize(); ++i) {
            x1[i] = -x[i];
        }
        return;
    }
    else {
        const int block_size = 256;
        const int grid_size = (x.dsize() + 255) / 256;
        minus_kernel<<<grid_size, block_size>>>(x1.data(), x.data(), x.dsize());
        return;
    }

}

__global__ void minus_kernel(float* ret, float* data, index_t dsize) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dsize) {
        ret[idx] = -data[idx];
    }
}

Minus& Minus::get() {
    static Minus minus_;
    return minus_;
}

void Minus::grad_fn(TensorImplPtr x1, TensorImplPtr x) {
    // 1. cal grad
    if (x1->device() == "cpu") {
        for (int i = 0; i < x1->dsize(); ++i) {
            x1->grad_[i] -= x->grad_[i];
        }
    }
    else {
        const int block_size = 256;
        const int grid_size = (x1->dsize() + 255) / 256;
        sub_kernel<<<grid_size, block_size>>>(x1->grad_, x1->grad_, x->grad_, x1->dsize());
    }

    // 2. decrease ref count
    x1->decreaseRef();
}

void Add::forward_process(Tensor& x1, Tensor& x2, Tensor& x) {
    if (x1.device() == "cpu") {
        for (int i = 0; i < x1.dsize(); ++i) {
            x[i] = x1[i] + x2[i];
        }
        return;
    }
    else {
        const int block_size = 256;
        const int grid_size = (x1.dsize() + 255) / 256;
        add_kernel<<<grid_size, block_size>>>(x.data(), x1.data(), x2.data(), x1.dsize());
        return;
    }
};

Add& Add::get() {
    static Add add_;
    return add_;
}

__global__ void add_kernel(float* ret, float* data1, float* data2, index_t dsize) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dsize) {
        ret[idx] = data1[idx] + data2[idx];
    }
}

void Add::lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
    // 1. cal grad
    if (x1->device() == "cpu") {
        for (int i = 0; i < x1->dsize(); ++i) {
            x1->grad_[i] += x->grad_[i];
        }
    }
    else {
        const int block_size = 256;
        const int grid_size = (x1->dsize() + 255) / 256;
        add_kernel<<<grid_size, block_size>>>(x1->grad_, x1->grad_, x->grad_, x1->dsize());
    }

    // 2. decrease ref count
    x1->decreaseRef();
}

void Add::rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
    if (x1->device() == "cpu") {
        for (int i = 0; i < x2->dsize(); ++i) {
            x2->grad_[i] += x->grad_[i];
        }
    }
    else {
        const int block_size = 256;
        const int grid_size = (x2->dsize() + 255) / 256;
        add_kernel<<<grid_size, block_size>>>(x2->grad_, x2->grad_, x->grad_, x2->dsize());
    }
    x2->decreaseRef();
}


void Sub::forward_process(Tensor& x1, Tensor& x2, Tensor& x) {
    if (x1.device() == "cpu") {
        for (int i = 0; i < x1.dsize(); ++i) {
            x[i] = x1[i] - x2[i];
        }
        return;
    }
    else {
        const int block_size = 256;
        const int grid_size = (x1.dsize() + 255) / 256;
        sub_kernel<<<grid_size, block_size>>>(x.data(), x1.data(), x2.data(), x1.dsize());
        return;
    }
};

Sub& Sub::get() {
    static Sub sub_;
    return sub_;
}

__global__ void sub_kernel(float* ret, float* data1, float* data2, index_t dsize) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dsize) {
        ret[idx] = data1[idx] - data2[idx];
    }
}

void Sub::lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
    // 1. cal grad
    if (x1->device() == "cpu") {
        for (int i = 0; i < x1->dsize(); ++i) {
            x1->grad_[i] += x->grad_[i];
        }
    }
    else {
        const int block_size = 256;
        const int grid_size = (x1->dsize() + 255) / 256;
        add_kernel<<<grid_size, block_size>>>(x1->grad_, x1->grad_, x->grad_, x1->dsize());
    }

    // 2. decrease ref count
    x1->decreaseRef();
}

void Sub::rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
    if (x1->device() == "cpu") {
        for (int i = 0; i < x2->dsize(); ++i) {
            x2->grad_[i] -= x->grad_[i];
        }
    }
    else {
        const int block_size = 256;
        const int grid_size = (x2->dsize() + 255) / 256;
        sub_kernel<<<grid_size, block_size>>>(x2->grad_, x2->grad_, x->grad_, x2->dsize());
    }
    x2->decreaseRef();
}

// void Mul::forward_process(Tensor& x1, Tensor& x2, Tensor& x) {

//     if (x1.device() == "cpu") {
//         //TODO
//         return;
//     }
//     else {
//         const int block_size = 256;
//         const int grid_size = (x1.dsize() + 255) / 256;
//         add_kernel<<<grid_size, block_size>>>(ret.data(), x1.data(), x2.data(), x1.dsize());
//         ret.setNode(node);
//         return ret;
//     }
// }