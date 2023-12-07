#include <cassert>
#include <iostream>

#include "exp.h"

using namespace tllm::detail;
using tllm::Tensor;

using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;

__global__ void add_kernel(float* ret, float* data1, float* data2,
                           index_t dsize1, index_t dsize2);
__global__ void self_add_kernel(float* data1, float* data2, index_t dsize1,
                                index_t dsize2);
__global__ void sub_kernel(float* ret, float* data1, float* data2,
                           index_t dsize1, index_t dsize2);
__global__ void self_sub_kernel(float* data1, float* data2, index_t dsize1,
                                index_t dsize2);
__global__ void minus_kernel(float* ret, float* data1, index_t dsize);
__global__ void mul_kernel(float* ret, float* data1, float* data2,
                           index_t dsize1, index_t dsize2);
__global__ void mul_backward_kernel(float* ret, float* data1, float* data2,
                                    index_t n, index_t dsize, index_t dsize1,
                                    index_t dsize2);
__global__ void scalar_add_kernel(float* ret, float* data1, float val,
                                  index_t dsize1);
__global__ void scalar_mul_backward_kernel(float* ret, float* data1, float val,
                                           index_t dsize1);

Tensor UnaryExp::forward(Tensor& x1) {
  Tensor x = generate_ret_tensor(x1);
  prepare_forward(x1, x);
  increase_ref_contiguous(x1);
  forward_process(x1, x);
  return x;
}

Tensor UnaryExp::generate_ret_tensor(Tensor& x1) {
  return Tensor(x1.shape(), x1.device());
}

void UnaryExp::prepare_forward(Tensor& x1, Tensor& x) {
  // 1. check

  // 2. add node to return Tensor
  GraphNodePtr node = std::make_shared<UnaryGraphNode>(x1.get());
  node->setGradFn(std::bind(&UnaryExp::grad_fn, this, _1, _2));
  x.setNode(node);

  return;
}

void UnaryExp::increase_ref_contiguous(Tensor& x1) {
  x1.increaseRef();
  x1.contiguous();
}

Tensor BinaryExp::forward(Tensor& x1, Tensor& x2) {
  Tensor x = generate_ret_tensor(x1, x2);
  prepare_forward(x1, x2, x);
  increase_ref_contiguous(x1, x2);
  forward_process(x1, x2, x);
  return x;
}

Tensor BinaryExp::generate_ret_tensor(Tensor& x1, Tensor& x2) {
  return Tensor(x1.shape(), x1.device());
}

void BinaryExp::prepare_forward(Tensor& x1, Tensor& x2, Tensor& x) {
  // 1. check
  if (x1.device() != x2.device()) {
    std::cout << "TensorImplPtr must in same device." << std::endl;
    assert(false);
  }

  if (x1.shape() != x2.shape()) {
    if (x1.ndim() < x2.ndim()) {
      std::cout << "LHS ndim smaller than RHS ndim. TensorImplPtr not in same "
                   "shape and can not broadcast."
                << std::endl;
      assert(false);
    }
    index_t l = x2.ndim();
    for (int i = 1; i <= l; ++i) {
      if (x1.shape()[x1.ndim() - i] != x2.shape()[x2.ndim() - i]) {
        std::cout << x1.shape()[x1.ndim() - i] << " "
                  << x2.shape()[x2.ndim() - i] << std::endl;
        std::cout << "TensorImplPtr not in same shape and can not broadcast."
                  << std::endl;
        assert(false);
      }
    }
  }

  // 2. add node to return Tensor
  GraphNodePtr node = std::make_shared<BinaryGraphNode>(x1.get(), x2.get());
  node->setGradFnL(std::bind(&BinaryExp::lhs_grad_fn, this, _1, _2, _3));
  node->setGradFnR(std::bind(&BinaryExp::rhs_grad_fn, this, _1, _2, _3));
  x.setNode(node);

  return;
}

void BinaryExp::increase_ref_contiguous(Tensor& x1, Tensor& x2) {
  x1.increaseRef();
  x2.increaseRef();
  x1.contiguous();
  x2.contiguous();
}

void Minus::forward_process(Tensor& x1, Tensor& x) {
  if (x.device() == "cpu") {
    for (int i = 0; i < x.dsize(); ++i) {
      x1[i] = -x[i];
    }
    return;
  } else {
    const int block_size = 256;
    const int grid_size = (x.dsize() + 255) / 256;
    minus_kernel<<<grid_size, block_size>>>(x1.data(), x.data(), x.dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
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
  } else {
    const int block_size = 256;
    const int grid_size = (x1->dsize() + 255) / 256;
    sub_kernel<<<grid_size, block_size>>>(x1->grad_, x1->grad_, x->grad_,
                                          x1->dsize(), x->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }

  // 2. decrease ref count
  // x1->decreaseRef();
}

void Add::forward_process(Tensor& x1, Tensor& x2, Tensor& x) {
  if (x1.device() == "cpu") {
    for (int i = 0; i < x1.dsize(); ++i) {
      x[i] = x1[i] + x2[i % x2.dsize()];
    }
    return;
  } else {
    const int block_size = 256;
    const int grid_size = (x1.dsize() + 255) / 256;
    add_kernel<<<grid_size, block_size>>>(x.data(), x1.data(), x2.data(),
                                          x1.dsize(), x2.dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
    return;
  }
};

Add& Add::get() {
  static Add add_;
  return add_;
}

__global__ void add_kernel(float* ret, float* data1, float* data2,
                           index_t dsize1, index_t dsize2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize1) {
    ret[idx] = data1[idx] + data2[idx % dsize2];
  }
}

__global__ void self_add_kernel(float* data1, float* data2, index_t dsize1,
                                index_t dsize2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize2) {
    atomicAdd(data1 + idx % dsize1, *(data2 + idx));
  }
}

void Add::lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
  // 1. cal grad
  if (x1->device() == "cpu") {
    for (int i = 0; i < x1->dsize(); ++i) {
      x1->grad_[i] += x->grad_[i];
    }
  } else {
    const int block_size = 256;
    const int grid_size = (x->dsize() + 255) / 256;
    self_add_kernel<<<grid_size, block_size>>>(x1->grad_, x->grad_, x1->dsize(),
                                               x->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }

  // 2. decrease ref count
  // x1->decreaseRef();
}

void Add::rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
  if (x1->device() == "cpu") {
    for (int i = 0; i < x->dsize(); ++i) {
      x2->grad_[i % x2->dsize()] += x->grad_[i];
    }
  } else {
    const int block_size = 256;
    const int grid_size = (x->dsize() + 255) / 256;
    self_add_kernel<<<grid_size, block_size>>>(x2->grad_, x->grad_, x2->dsize(),
                                               x->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
  // x2->decreaseRef();
}

void Sub::forward_process(Tensor& x1, Tensor& x2, Tensor& x) {
  if (x1.device() == "cpu") {
    for (int i = 0; i < x1.dsize(); ++i) {
      x[i] = x1[i] - x2[i % x2.dsize()];
    }
    return;
  } else {
    const int block_size = 256;
    const int grid_size = (x1.dsize() + 255) / 256;
    sub_kernel<<<grid_size, block_size>>>(x.data(), x1.data(), x2.data(),
                                          x1.dsize(), x2.dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
    return;
  }
};

Sub& Sub::get() {
  static Sub sub_;
  return sub_;
}

__global__ void sub_kernel(float* ret, float* data1, float* data2,
                           index_t dsize1, index_t dsize2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize1) {
    ret[idx] = data1[idx] - data2[idx % dsize2];
  }
}

__global__ void self_sub_kernel(float* data1, float* data2, index_t dsize1,
                                index_t dsize2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize2) {
    atomicAdd(data1 + idx % dsize1, -(*(data2 + idx)));
  }
}

void Sub::lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
  // 1. cal grad
  if (x1->device() == "cpu") {
    for (int i = 0; i < x1->dsize(); ++i) {
      x1->grad_[i] += x->grad_[i];
    }
  } else {
    const int block_size = 256;
    const int grid_size = (x->dsize() + 255) / 256;
    self_add_kernel<<<grid_size, block_size>>>(x1->grad_, x->grad_, x1->dsize(),
                                               x->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }

  // 2. decrease ref count
  // x1->decreaseRef();
}

void Sub::rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
  if (x1->device() == "cpu") {
    for (int i = 0; i < x->dsize(); ++i) {
      x2->grad_[i % x2->dsize()] -= x->grad_[i];
    }
  } else {
    const int block_size = 256;
    const int grid_size = (x->dsize() + 255) / 256;
    self_sub_kernel<<<grid_size, block_size>>>(x2->grad_, x->grad_, x2->dsize(),
                                               x->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
  // x2->decreaseRef();
}

void Mul::forward_process(Tensor& x1, Tensor& x2, Tensor& x) {
  if (x1.device() == "cpu") {
    for (int i = 0; i < x1.dsize(); ++i) {
      x[i] = x1[i] * x2[i % x2.dsize()];
    }
    return;
  } else {
    const int block_size = 256;
    const int grid_size = (x1.dsize() + 255) / 256;
    mul_kernel<<<grid_size, block_size>>>(x.data(), x1.data(), x2.data(),
                                          x1.dsize(), x2.dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
    return;
  }
}

Mul& Mul::get() {
  static Mul mul_;
  return mul_;
}

__global__ void mul_kernel(float* ret, float* data1, float* data2,
                           index_t dsize1, index_t dsize2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize1) {
    ret[idx] = data1[idx] * data2[idx % dsize2];
  }
}

void Mul::lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
  // 1. cal grad
  if (x1->device() == "cpu") {
    for (int i = 0; i < x1->dsize(); ++i) {
      x1->grad_[i] += (x->grad_[i] * (*x2)[i % x2->dsize()]);
    }
  } else {
    const int block_size = 256;
    const int grid_size = (x->dsize() + 255) / 256;
    mul_backward_kernel<<<grid_size, block_size>>>(
        x1->grad_, x->grad_, x2->data_, x->dsize(), x1->dsize(), x->dsize(),
        x2->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }

  // 2. decrease ref count
  // x1->decreaseRef();
}

void Mul::rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
  // 1. cal grad
  if (x2->device() == "cpu") {
    for (int i = 0; i < x->dsize(); ++i) {
      x2->grad_[i % x2->dsize()] += (x->grad_[i] * (*x1)[i]);
    }
  } else {
    const int block_size = 256;
    const int grid_size = (x->dsize() + 255) / 256;
    mul_backward_kernel<<<grid_size, block_size>>>(
        x2->grad_, x->grad_, x1->data_, x->dsize(), x2->dsize(), x->dsize(),
        x1->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }

  // 2. decrease ref count
  // x2->decreaseRef();
}

__global__ void mul_backward_kernel(float* ret, float* data1, float* data2,
                                    index_t n, index_t dsize, index_t dsize1,
                                    index_t dsize2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    atomicAdd(ret + idx % dsize, data1[idx % dsize1] * data2[idx % dsize2]);
  }
}

// template<int c>
// ScalarAddImp<c>& ScalarAdd::get() {
//     return ScalarAddImp<count++>.get();
// }

ScalarAdd::ScalarAdd(float val) : val_(val) {}

void ScalarAdd::prepare_forward(Tensor& x1, Tensor& x) {
  // 1. check

  // 2. add node to return Tensor
  GraphNodePtr node = std::make_shared<UnaryGraphNode>(x1.get());
  node->setGradFn(std::bind(&ScalarAdd::grad_fn, shared_from_this(), _1, _2));
  x.setNode(node);

  return;
}

__global__ void scalar_add_kernel(float* ret, float* data1, float val,
                                  index_t dsize1) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize1) {
    ret[idx] = data1[idx] + val;
  }
}

void ScalarAdd::forward_process(Tensor& x1, Tensor& x) {
  if (x1.device() == "cpu") {
    for (int i = 0; i < x1.dsize(); ++i) {
      x[i] = x1[i] + val_;
    }
  } else {
    const int block_size = 256;
    const int grid_size = (x1.dsize() + 255) / 256;
    scalar_add_kernel<<<grid_size, block_size>>>(x.data(), x1.data(), val_,
                                                 x1.dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
  return;
}

void ScalarAdd::grad_fn(TensorImplPtr x1, TensorImplPtr x) {
  if (x1->device() == "cpu") {
    for (int i = 0; i < x1->dsize(); ++i) {
      x1->grad_[i] += x->grad_[i];
    }
  } else {
    const int block_size = 256;
    const int grid_size = (x1->dsize() + 255) / 256;
    self_add_kernel<<<grid_size, block_size>>>(x1->grad_, x->grad_, x1->dsize(),
                                               x->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
  // x1->decreaseRef();
  return;
}

ScalarMul::ScalarMul(float val) : val_(val) {}

void ScalarMul::prepare_forward(Tensor& x1, Tensor& x) {
  // 1. check

  // 2. add node to return Tensor
  GraphNodePtr node = std::make_shared<UnaryGraphNode>(x1.get());
  node->setGradFn(std::bind(&ScalarMul::grad_fn, shared_from_this(), _1, _2));
  x.setNode(node);

  return;
}

__global__ void scalar_mul_kernel(float* ret, float* data1, float val,
                                  index_t dsize1) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize1) {
    ret[idx] = data1[idx] * val;
  }
}

void ScalarMul::forward_process(Tensor& x1, Tensor& x) {
  if (x1.device() == "cpu") {
    for (int i = 0; i < x1.dsize(); ++i) {
      x[i] = x1[i] * val_;
    }
  } else {
    const int block_size = 256;
    const int grid_size = (x1.dsize() + 255) / 256;
    scalar_mul_kernel<<<grid_size, block_size>>>(x.data(), x1.data(), val_,
                                                 x1.dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
  return;
}

void ScalarMul::grad_fn(TensorImplPtr x1, TensorImplPtr x) {
  if (x1->device() == "cpu") {
    for (int i = 0; i < x1->dsize(); ++i) {
      x1->grad_[i] += (x->grad_[i] * val_);
    }
  } else {
    const int block_size = 256;
    const int grid_size = (x1->dsize() + 255) / 256;
    scalar_mul_backward_kernel<<<grid_size, block_size>>>(x1->grad_, x->grad_,
                                                          val_, x->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
  // x1->decreaseRef();
  return;
}

__global__ void scalar_mul_backward_kernel(float* ret, float* data1, float val,
                                           index_t dsize1) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize1) {
    atomicAdd(ret + idx, (*(data1 + idx)) * val);
  }
}
