#include <curand_kernel.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <numeric>

#include "function.h"

namespace tllm {
namespace F {
detail::MatMulExp mat_mul;
detail::LayerNorm layer_norm(0.00001);
detail::Softmax softmax;
detail::Log log;
detail::NLLLoss nll_loss;
detail::Emb emb;

Tensor cross_entropy(Tensor& x1, Tensor& x2) {
  Tensor x1_softmax = softmax(x1);
  Tensor x1_softmax_log = log(x1_softmax);
  return nll_loss(x1_softmax_log, x2);
}

__global__ void causal_mask_fill_kernel(float* att, index_t T, index_t dsize);
void causal_mask_fill(Tensor& att) {
  // att (B, nh, T, T)
  auto shape = att.shape();
  assert(shape.size() == 4);
  assert(shape[2] == shape[3]);

  index_t B = shape[0];
  index_t nh = shape[1];
  index_t T = shape[2];

  if (att.device() == "cpu") {
    for (index_t b = 0; b < B; ++b) {
      for (index_t h = 0; h < nh; ++h) {
        for (index_t i = 0; i < T; ++i) {
          for (index_t j = i + 1; j < T; ++j) {
            att[{b, h, i, j}] = FLT_MIN;
          }
        }
      }
    }
  } else {
    dim3 grid_size((T + 25) / 26, (T + 25) / 26, B * nh);
    dim3 block_size(26, 26);
    causal_mask_fill_kernel<<<grid_size, block_size>>>(att.data(), T, att.dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}

__global__ void causal_mask_fill_kernel(float* att, index_t T, index_t dsize) {
  // printf("%d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
  const int head_idx = blockIdx.z * T * T;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int x = blockIdx.x * blockDim.y + threadIdx.x;
  if (y < T && x < T && x > y) {
      att[head_idx + y * T + x] = FLT_MIN;
  }
}

}  // namespace F
}  // namespace tllm

using namespace tllm::detail;

using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;

__global__ void matmul_kernel(float* x, float* x1, float* x2, index_t z_,
                              index_t y_, index_t x_, index_t d_,
                              index_t dsize2);
__global__ void matmul_backward_kernel(float* x, float* x1, float* x2,
                                       index_t z_, index_t y_, index_t x_,
                                       index_t d_, index_t dsize,
                                       index_t dsize1, index_t dsize2);
__global__ void layer_norm_kernel(float* x1, float* x, int N, index_t dim,
                                  index_t dsize, float ep);
__global__ void layer_norm_backward_kernel(float* x1_grad, float* x1_data,
                                           float* x_grad, float* x_data, int N,
                                           index_t dim, index_t dsize,
                                           float ep);
__global__ void drouput_kernel(float* x1, float* x, float prob, index_t dsize,
                               unsigned long long seed);
__global__ void dropout_backward_kernel(float* x1_grad, float* x_grad,
                                        float* x_data, float prob,
                                        index_t dsize);
__global__ void gelu_kernel(float* x1, float* x, index_t dsize);
__global__ void gelu_backward_kernel(float* x1_grad, float* x1_data,
                                     float* x_grad, index_t dsize);
__global__ void softmax_kernel(float* x1, float* x, int N, index_t dim,
                               index_t dsize);
__global__ void softmax_kernel(float* x1, float* x, int N, index_t dim,
                               index_t dsize);
__global__ void softmax_backward_kernel(float* x1_grad, float* x1_data,
                                        float* x_grad, float* x_data, int N,
                                        index_t dim, index_t dsize);
__global__ void log_kernel(float* x1, float* x, index_t dsize);
__global__ void log_backward_kernel(float* x1_grad, float* x1_data,
                                    float* x_grad, index_t dsize);
__global__ void nllloss_kernel(float* x1, float* x2, float* x, index_t len,
                               index_t dsize);
__global__ void nllloss_backward_kernel(float* x1_grad, float* x1_data,
                                        float* x2_data, float* x_grad,
                                        index_t len, index_t dsize);
__global__ void emb_backward_kernel(float* idx, float* emb_grad, float* x_grad,
                                    int N, index_t dim, index_t dsize);

Tensor UnaryFunc::operator()(Tensor& x1) { return forward(x1); }

Tensor BinaryFunc::operator()(Tensor& x1, Tensor& x2) {
  return forward(x1, x2);
}

Tensor MatMulExp::generate_ret_tensor(Tensor& x1, Tensor& x2) {
  auto shape1 = x1.shape();
  auto shape2 = x2.shape();
  if (shape1[shape1.size() - 1] != shape2[shape2.size() - 2]) {
    std::cout << "MatMul shape not match." << std::endl;
    assert(false);
  }
  shape1[shape1.size() - 1] = shape2[shape2.size() - 1];
  return Tensor(shape1, x1.device());
}

void MatMulExp::prepare_forward(Tensor& x1, Tensor& x2, Tensor& x) {
  // 1. check
  if (x1.device() != x2.device()) {
    std::cout << "TensorImplPtr must in same device." << std::endl;
    assert(false);
  }

  if (x1.ndim() < x2.ndim()) {
    std::cout << "LHS ndim smaller than RHS ndim. TensorImplPtr not in same "
                 "shape and can not broadcast."
              << std::endl;
    assert(false);
  }
  index_t l = x2.ndim();
  for (int i = 3; i <= l; ++i) {
    if (x1.shape()[x1.ndim() - i] != x2.shape()[x2.ndim() - i]) {
      std::cout << x1.shape()[x1.ndim() - i] << " " << x2.shape()[x2.ndim() - i]
                << std::endl;
      std::cout << "TensorImplPtr not in same shape and can not broadcast."
                << std::endl;
      assert(false);
    }
  }

  // 2. add node to return Tensor
  GraphNodePtr node = std::make_shared<BinaryGraphNode>(x1.get(), x2.get());
  node->setGradFnL(std::bind(&MatMulExp::lhs_grad_fn, this, _1, _2, _3));
  node->setGradFnR(std::bind(&MatMulExp::rhs_grad_fn, this, _1, _2, _3));
  x.setNode(node);

  x2.transpose(x2.ndim() - 2, x2.ndim() - 1);

  return;
}

void MatMulExp::forward_process(Tensor& x1, Tensor& x2, Tensor& x) {
  auto shape1 = x1.shape();
  auto shape2 = x2.shape();
  auto shape = x.shape();
  index_t dim = 1;
  for (int i = 0; i < shape.size() - 2; ++i) {
    dim *= shape[i];
  }
  index_t dim2 = 1;
  for (int i = 0; i < shape2.size() - 2; ++i) {
    dim2 *= shape2[i];
  }
  x1.view({dim, shape1[shape1.size() - 2], shape1[shape1.size() - 1]});
  x2.view({dim2, shape2[shape2.size() - 2], shape2[shape2.size() - 1]});
  x.view({dim, shape[shape.size() - 2], shape[shape.size() - 1]});
  if (x1.device() == "cpu") {
    for (index_t m = 0; m < dim; ++m) {
      for (index_t i = 0; i < shape[shape.size() - 2]; ++i) {
        for (index_t j = 0; j < shape[shape.size() - 1]; ++j) {
          x[{m, i, j}] = 0;
          for (index_t p = 0; p < shape1[shape1.size() - 1]; ++p) {
            x[{m, i, j}] += (x1[{m, i, p}] * x2[{m % dim2, j, p}]);
          }
        }
      }
    }
  } else {
    const dim3 grid_size((shape[shape.size() - 1] + 15) / 16,
                         (shape[shape.size() - 2] + 15) / 16, dim);
    const dim3 block_size(16, 16);
    // std::cout << dim << " " << shape[shape.size() - 2] << " " <<
    // shape[shape.size() - 1] << " " << shape1[shape1.size() - 1] << std::endl;
    matmul_kernel<<<grid_size, block_size>>>(
        x.data(), x1.data(), x2.data(), dim, shape[shape.size() - 2],
        shape[shape.size() - 1], shape1[shape1.size() - 1], x2.dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
  x1.view(shape1);
  x2.view(shape2);
  x.view(shape);
  x2.transpose(x2.ndim() - 2, x2.ndim() - 1);
  return;
}

__global__ void matmul_kernel(float* x, float* x1, float* x2, index_t z_,
                              index_t y_, index_t x_, index_t d_,
                              index_t dsize2) {
  const int id_z = blockIdx.z;
  const int id_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int id_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int x_offset = id_z * y_ * x_ + id_y * x_ + id_x;
  const int x1_offset = id_z * y_ * d_ + id_y * d_;
  const int x2_offset = id_z * x_ * d_ + id_x * d_;
  if (id_z < z_ && id_y < y_ && id_x < x_) {
    x[x_offset] = 0;
    // printf("%d %d %d %d %d\n", x_offset, id_z, id_y, id_x, gridDim.y);
    for (int i = 0; i < d_; ++i) {
      x[x_offset] += x1[x1_offset + i] * x2[(x2_offset + i) % dsize2];
    }
  }
}

void MatMulExp::lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2,
                            TensorImplPtr x) {
  x1->contiguous();
  x2->contiguous();
  x->contiguous();
  auto shape1 = x1->shape();
  auto shape2 = x2->shape();
  auto shape = x->shape();

  index_t dim = 1;
  for (int i = 0; i < shape.size() - 2; ++i) {
    dim *= shape[i];
  }
  index_t dim2 = 1;
  for (int i = 0; i < shape2.size() - 2; ++i) {
    dim2 *= shape2[i];
  }
  x1->view({dim, shape1[shape1.size() - 2], shape1[shape1.size() - 1]});
  x2->view({dim2, shape2[shape2.size() - 2], shape2[shape2.size() - 1]});
  x->view({dim, shape[shape.size() - 2], shape[shape.size() - 1]});

  if (x1->device() == "cpu") {
    for (index_t m = 0; m < dim; ++m) {
      for (index_t i = 0; i < shape[shape.size() - 2]; ++i) {
        for (index_t p = 0; p < shape1[shape1.size() - 1]; ++p) {
          for (index_t j = 0; j < shape[shape.size() - 1]; ++j) {
            x1->grad_[x1->get_offset({m, i, p})] +=
                (x->grad_[x->get_offset({m, i, j})] *
                 x2->data_[x2->get_offset({m % dim2, p, j})]);
          }
        }
      }
    }
  } else {
    const dim3 grid_size((shape1[shape1.size() - 1] + 15) / 16,
                         (shape1[shape1.size() - 2] + 15) / 16, dim);
    const dim3 block_size(16, 16);
    matmul_backward_kernel<<<grid_size, block_size>>>(
        x1->grad_, x->grad_, x2->data_, dim, shape1[shape1.size() - 2],
        shape1[shape1.size() - 1], shape2[shape2.size() - 1], x1->dsize(),
        x->dsize(), x2->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
  x1->view(shape1);
  x2->view(shape2);
  x->view(shape);

  return;
}

__global__ void matmul_backward_kernel(float* x, float* x1, float* x2,
                                       index_t z_, index_t y_, index_t x_,
                                       index_t d_, index_t dsize,
                                       index_t dsize1, index_t dsize2) {
  const int id_z = blockIdx.z;
  const int id_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int id_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int x_offset = id_z * y_ * x_ + id_y * x_ + id_x;
  const int x1_offset = id_z * y_ * d_ + id_y * d_;
  const int x2_offset = id_z * x_ * d_ + id_x * d_;
  if (id_z < z_ && id_y < y_ && id_x < x_) {
    for (int i = 0; i < d_; ++i) {
      atomicAdd(x + x_offset % dsize,
                x1[(x1_offset + i) % dsize1] * x2[(x2_offset + i) % dsize2]);
      // printf("%.2f %.2f %.2f %d\n", x[x_offset % dsize], x1[(x1_offset + i) %
      // dsize1], x2[(x2_offset + i) % dsize2], x_offset);
    }
  }
}

void MatMulExp::rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2,
                            TensorImplPtr x) {
  auto shape1 = x1->shape();
  auto shape2 = x2->shape();
  auto shape = x->shape();
  x1->transpose(x1->ndim() - 2, x1->ndim() - 1);
  x->transpose(x->ndim() - 2, x->ndim() - 1);
  x1->contiguous();
  x2->contiguous();
  x->contiguous();
  index_t dim = 1;
  for (int i = 0; i < shape.size() - 2; ++i) {
    dim *= shape[i];
  }
  index_t dim2 = 1;
  for (int i = 0; i < shape2.size() - 2; ++i) {
    dim2 *= shape2[i];
  }

  x1->view({dim, shape1[shape1.size() - 1], shape1[shape1.size() - 2]});
  x2->view({dim2, shape2[shape2.size() - 2], shape2[shape2.size() - 1]});
  x->view({dim, shape[shape.size() - 1], shape[shape.size() - 2]});
  if (x1->device() == "cpu") {
    for (index_t m = 0; m < dim; ++m) {
      for (index_t i = 0; i < shape[shape.size() - 2]; ++i) {
        for (index_t j = 0; j < shape[shape.size() - 1]; ++j) {
          for (index_t p = 0; p < shape1[shape1.size() - 1]; ++p) {
            x2->grad_[x2->get_offset({m % dim2, p, j})] +=
                (x1->data_[x1->get_offset({m, p, i})] *
                 x->grad_[x->get_offset({m, j, i})]);
          }
        }
      }
    }
  } else {
    const dim3 grid_size((shape2[shape2.size() - 1] + 15) / 16,
                         (shape2[shape2.size() - 2] + 15) / 16, dim);
    const dim3 block_size(16, 16);
    matmul_backward_kernel<<<grid_size, block_size>>>(
        x2->grad_, x1->data_, x->grad_, dim, shape2[shape2.size() - 2],
        shape2[shape2.size() - 1], shape1[shape1.size() - 2], x2->dsize(),
        x1->dsize(), x->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }

  x1->transpose(x1->ndim() - 2, x1->ndim() - 1);
  x->transpose(x->ndim() - 2, x->ndim() - 1);

  x1->contiguous();
  x->contiguous();

  x1->view(shape1);
  x2->view(shape2);
  x->view(shape);

  // x2->decreaseRef();
  return;
}

void LayerNorm::forward_process(Tensor& x1, Tensor& x) {
  if (x1.device() == "cpu") {
    std::memcpy(x.data(), x1.data(), sizeof(float) * x1.dsize());
    index_t dim = x1.shape()[x1.shape().size() - 1];
    for (int b = 0; b < x1.dsize() / dim; ++b) {
      index_t offset = b * dim;
      float mean =
          std::accumulate(x1.data() + offset, x1.data() + offset + dim, 0.0) /
          dim;
      float accum = 0.0;
      std::for_each(x1.data() + offset, x1.data() + offset + dim,
                    [&](const float d) { accum += (d - mean) * (d - mean); });

      float stdev = sqrt(accum / dim + ep_);
      std::for_each(x.data() + offset, x.data() + offset + dim, [&](float& d) {
        d = d - mean;
        d = d / stdev;
      });
    }
  } else {
    auto shape = x.shape();
    // cudaMemcpy(x.data(), x1.data(), sizeof(float) * x1.dsize(),
    // cudaMemcpyDeviceToDevice);
    const dim3 grid_size(
        1, shape[shape.size() - 2],
        x.dsize() / (shape[shape.size() - 2] * shape[shape.size() - 1]));
    const int block_size = min(
        512, (shape[shape.size() - 1] % 2 == 0 ? shape[shape.size() - 1]
                                               : shape[shape.size() - 1] + 1));
    layer_norm_kernel<<<grid_size, block_size,
                        sizeof(float) * shape[shape.size() - 1]>>>(
        x1.data(), x.data(),
        (shape[shape.size() - 1] + block_size - 1) / block_size,
        shape[shape.size() - 1], x.dsize(), ep_);
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}

__global__ void layer_norm_kernel(float* x1, float* x, int N, index_t dim,
                                  index_t dsize, float ep) {
  const index_t z = blockIdx.z;
  const index_t y = blockIdx.y;
  const index_t thread_idx = threadIdx.x;

  const index_t head_offset = z * gridDim.y * dim + y * dim;

  extern __shared__ float data[];
  for (int i = 0; i < N; ++i) {
    index_t offset = thread_idx * N + i;
    if (offset < dim) {
      if ((head_offset + offset) < dsize) {
        data[offset] = x1[head_offset + offset];
      } else {
        data[offset] = 0;
      }
    } else {
      data[offset] = 0;
    }
  }
  __syncthreads();
  for (int gap = (blockDim.x * N) >> 1; gap > 0; gap >>= 1) {
    for (int i = 0; i < N; ++i) {
      int offset = thread_idx * N + i;
      if (offset < gap) {
        data[offset] += data[offset + gap];
      }
    }
    __syncthreads();
  }
  __syncthreads();
  float mean = data[0] / dim;

  for (int i = 0; i < N; ++i) {
    int offset = thread_idx * N + i;
    if (offset < dim && head_offset + offset < dsize) {
      data[offset] =
          (x1[head_offset + offset] - mean) * (x1[head_offset + offset] - mean);
    } else {
      data[offset] = 0;
    }
  }
  __syncthreads();
  for (int gap = blockDim.x * N >> 1; gap > 0; gap >>= 1) {
    for (int i = 0; i < N; ++i) {
      int offset = thread_idx * N + i;
      if (offset < gap) {
        data[offset] += data[offset + gap];
      }
    }
    __syncthreads();
  }
  __syncthreads();
  float accum = data[0];
  float stdev = sqrt(accum / dim + ep);

  for (int i = 0; i < N; ++i) {
    int offset = thread_idx * N + i;
    if (offset < dim && head_offset + offset < dsize) {
      x[head_offset + offset] = (x1[head_offset + offset] - mean) / stdev;
    }
  }
}

void LayerNorm::grad_fn(TensorImplPtr x1, TensorImplPtr x) {
  if (x1->device() == "cpu") {
    index_t dim = x1->shape()[x1->shape().size() - 1];
    for (int b = 0; b < x1->dsize() / dim; ++b) {
      index_t offset = b * dim;
      float mean =
          std::accumulate(x1->data_ + offset, x1->data_ + offset + dim, 0.0) /
          dim;
      float accum = 0.0;
      std::for_each(x1->data_ + offset, x1->data_ + offset + dim,
                    [&](const float d) { accum += (d - mean) * (d - mean); });

      float stdev = sqrt(accum / dim + ep_);

      float a1 = -(1 / (dim * stdev));
      // float a2 = stdev * (accum / dim + ep_);

      float grad_accum =
          std::accumulate(x->grad_ + offset, x->grad_ + offset + dim, 0.0) /
          (dim * stdev);
      grad_accum *= a1;

      float grad_data_accum = 0.0;
      for (int i = 0; i < dim; ++i) {
        grad_data_accum += x->grad_[offset + i] * x->data_[offset + i];
      }
      grad_data_accum *= a1;

      for (int i = 0; i < dim; ++i) {
        x1->grad_[offset + i] += (x->grad_[offset + i] / stdev + grad_accum +
                                  x->data_[offset + i] * grad_data_accum);
      }
    }
  } else {
    auto shape = x->shape();
    const dim3 grid_size(
        1, shape[shape.size() - 2],
        x->dsize() / (shape[shape.size() - 2] * shape[shape.size() - 1]));
    const int block_size = min(
        512, (shape[shape.size() - 1] % 2 == 0 ? shape[shape.size() - 1]
                                               : shape[shape.size() - 1] + 1));
    layer_norm_backward_kernel<<<grid_size, block_size,
                                 sizeof(float) * shape[shape.size() - 1]>>>(
        x1->grad_, x1->data_, x->grad_, x->data_,
        (shape[shape.size() - 1] + block_size - 1) / block_size,
        shape[shape.size() - 1], x->dsize(), ep_);
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
  // x->to("cpu");
  // x1->to("cpu");
  // for (int i = 0; i < x->dsize(); ++i) {
  //   if (std::isnan(x->grad_[i])) {
  //     std::cout << "before ln grad has nan." << std::endl;
  //     exit(0);
  //   }
  // }
  // for (int i = 0; i < x1->dsize(); ++i) {
  //   if (std::isnan(x1->grad_[i])) {
  //     std::cout << "after ln grad has nan." << std::endl;
  //     exit(0);
  //   }
  // }
  // x->to("cuda");
  // x1->to("cuda");
}

__global__ void layer_norm_backward_kernel(float* x1_grad, float* x1_data,
                                           float* x_grad, float* x_data, int N,
                                           index_t dim, index_t dsize,
                                           float ep) {
  const int z = blockIdx.z;
  const int y = blockIdx.y;
  const int thread_idx = threadIdx.x;

  const int head_offset = z * gridDim.y * dim + y * dim;

  extern __shared__ float data[];
  for (int i = 0; i < N; ++i) {
    int offset = thread_idx * N + i;
    if (offset < dim && head_offset + offset < dsize) {
      data[offset] = x1_data[head_offset + offset];
    } else {
      data[offset] = 0;
    }
  }
  __syncthreads();
  for (int gap = (blockDim.x * N) >> 1; gap > 0; gap >>= 1) {
    for (int i = 0; i < N; ++i) {
      int offset = thread_idx * N + i;
      if (offset < gap) {
        data[offset] += data[offset + gap];
      }
    }
    __syncthreads();
  }
  __syncthreads();
  float mean = data[0] / dim;

  for (int i = 0; i < N; ++i) {
    int offset = thread_idx * N + i;
    if (offset < dim && head_offset + offset < dsize) {
      data[offset] = (x1_data[head_offset + offset] - mean) *
                     (x1_data[head_offset + offset] - mean);
    } else {
      data[offset] = 0;
    }
  }
  __syncthreads();
  for (int gap = blockDim.x * N >> 1; gap > 0; gap >>= 1) {
    for (int i = 0; i < N; ++i) {
      int offset = thread_idx * N + i;
      if (offset < gap) {
        data[offset] += data[offset + gap];
      }
    }
    __syncthreads();
  }
  __syncthreads();
  float accum = data[0];
  float stdev = sqrt(accum / dim + ep);

  float a1 = -(1 / (dim * stdev));

  for (int i = 0; i < N; ++i) {
    int offset = thread_idx * N + i;
    if (offset < dim && head_offset + offset < dsize) {
      data[offset] = x_grad[head_offset + offset];
    } else {
      data[offset] = 0;
    }
  }
  __syncthreads();
  for (int gap = blockDim.x * N >> 1; gap > 0; gap >>= 1) {
    for (int i = 0; i < N; ++i) {
      int offset = thread_idx * N + i;
      if (offset < gap) {
        data[offset] += data[offset + gap];
      }
    }
    __syncthreads();
  }
  __syncthreads();
  float grad_accum = data[0] * a1;

  for (int i = 0; i < N; ++i) {
    int offset = thread_idx * N + i;
    if (offset < dim && head_offset + offset < dsize) {
      data[offset] =
          x_grad[head_offset + offset] * x_data[head_offset + offset];
    } else {
      data[offset] = 0;
    }
  }
  __syncthreads();
  for (int gap = blockDim.x * N >> 1; gap > 0; gap >>= 1) {
    for (int i = 0; i < N; ++i) {
      int offset = thread_idx * N + i;
      if (offset < gap) {
        data[offset] += data[offset + gap];
      }
    }
    __syncthreads();
  }
  __syncthreads();
  float grad_data_accum = data[0] * a1;

  // for (int i = 0; i < N; ++i) {
  //     int offset = thread_idx * N + i;
  //     if (offset < dim && head_offset + offset < dsize) {
  //         data[offset] = x1_data[head_offset + offset];
  //     }
  //     else {
  //         data[offset] = 0;
  //     }
  // }
  // __syncthreads();

  for (int i = 0; i < N; ++i) {
    int offset = thread_idx * N + i;
    if (offset < dim && head_offset + offset < dsize) {
      x1_grad[head_offset + offset] += x_grad[head_offset + offset] / stdev;
      x1_grad[head_offset + offset] +=
          grad_accum + grad_data_accum * x_data[head_offset + offset];
      // if (std::isnan(x1_grad[head_offset + offset])) {
      //   printf("%.2f %.2f %.2f %.2f %.2f\n", stdev, p1, p2,
      //   x1_data[head_offset + offset], mean);
      // }
    }
  }
}

void Dropout::forward_process(Tensor& x1, Tensor& x) {
#ifdef EVAL
  x = x1 * 1;
  return;
#endif
  if (x1.device() == "cpu") {
    for (int i = 0; i < x1.dsize(); ++i) {
      if (di(dre) < limit_) {
        x[i] = 0;
      } else {
        x[i] = x1[i] / (1 - prob_);
      }
    }
  } else {
    const int block_size = 512;
    const int grid_size = (x1.dsize() + 511) / 512;
    drouput_kernel<<<grid_size, block_size>>>(x1.data(), x.data(), prob_,
                                              x1.dsize(), time(nullptr));
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}

__global__ void drouput_kernel(float* x1, float* x, float prob, index_t dsize,
                               unsigned long long seed) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    if (curand_uniform(&state) < prob) {
      x[idx] = 0;
    } else {
      x[idx] = x1[idx] / (1 - prob);
    }
  }
}

void Dropout::grad_fn(TensorImplPtr x1, TensorImplPtr x) {
  if (x1->device() == "cpu") {
#ifdef EVAL
    memcpy(x1->grad_, x->grad_, sizeof(float) * x->dsize());
    return;
#endif
    for (int i = 0; i < x1->dsize(); ++i) {
      if (fabs((*x)[i]) < 1e-6) {
        continue;
      } else {
        x1->grad_[i] += (x->grad_[i] / (1 - prob_));
      }
    }
  } else {
#ifdef EVAL
    cudaMemcpy(x1->grad_, x->grad_, sizeof(float) * x->dsize(),
               cudaMemcpyDeviceToDevice);
    return;
#endif
    const int block_size = 512;
    const int grid_size = (x1->dsize() + 511) / 512;
    dropout_backward_kernel<<<grid_size, block_size>>>(
        x1->grad_, x->grad_, x->data_, prob_, x1->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}

__global__ void dropout_backward_kernel(float* x1_grad, float* x_grad,
                                        float* x_data, float prob,
                                        index_t dsize) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize) {
    if (fabs(x_data[idx]) < 1e-6) {
      return;
    } else {
      x1_grad[idx] += x_grad[idx] / (1 - prob);
    }
  }
}

void GELU::forward_process(Tensor& x1, Tensor& x) {
  if (x1.device() == "cpu") {
    const float n = sqrt(2 / M_PI);
    for (int i = 0; i < x1.dsize(); ++i) {
      x[i] = 0.5 * x1[i] * (1 + tanh(n * (x1[i] + 0.044715 * pow(x1[i], 3))));
    }
  } else {
    const int block_size = 512;
    const int grid_size = (x1.dsize() + 511) / 512;
    gelu_kernel<<<grid_size, block_size>>>(x1.data(), x.data(), x1.dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}

__global__ void gelu_kernel(float* x1, float* x, index_t dsize) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize) {
    const float n = sqrt(2 / M_PI);
    x[idx] =
        0.5 * x1[idx] * (1 + tanh(n * (x1[idx] + 0.044715 * pow(x1[idx], 3))));
  }
}

void GELU::grad_fn(TensorImplPtr x1, TensorImplPtr x) {
  if (x1->device() == "cpu") {
    const float n = sqrt(2 / M_PI);
    for (int i = 0; i < x1->dsize(); ++i) {
      const float th = tanh(n * ((*x1)[i] + 0.044715 * pow((*x1)[i], 3)));
      x1->grad_[i] +=
          x->grad_[i] *
          (0.5 * (1 + th) + 0.5 * (*x1)[i] * (1 - th * th) * n *
                                (1 + 0.044715 * 3 * (*x1)[i] * (*x1)[i]));
    }
  } else {
    const int block_size = 512;
    const int grid_size = (x1->dsize() + 511) / 512;
    gelu_backward_kernel<<<grid_size, block_size>>>(x1->grad_, x1->data_,
                                                    x->grad_, x1->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}

__global__ void gelu_backward_kernel(float* x1_grad, float* x1_data,
                                     float* x_grad, index_t dsize) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize) {
    const float n = sqrt(2 / M_PI);
    const float th = tanh(n * (x1_data[idx] + 0.044715 * pow(x1_data[idx], 3)));
    x1_grad[idx] +=
        x_grad[idx] *
        (0.5 * (1 + th) + 0.5 * x1_data[idx] * (1 - th * th) * n *
                              (1 + 0.044715 * 3 * x1_data[idx] * x1_data[idx]));
  }
}

__global__ void before_softmax_kernel(float* x1, index_t dim,
                                      index_t sum_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < sum_size) {
    index_t offset = idx * dim;
    float max_num = x1[offset];
    for (int i = 0; i < dim; ++i) {
      max_num = max(max_num, x1[offset + i]);
    }
    for (int i = 0; i < dim; ++i) {
      x1[offset + i] -= max_num;
    }
  }
}

void Softmax::before_softmax(Tensor& x1) {
  if (x1.device() == "cpu") {
    index_t dim = x1.shape()[x1.shape().size() - 1];
    for (int b = 0; b < x1.dsize() / dim; ++b) {
      index_t offset = b * dim;
      float max_num = x1[offset];
      for (int i = 0; i < dim; ++i) {
        max_num = max(max_num, x1[offset + i]);
      }
      for (int i = 0; i < dim; ++i) {
        x1[offset + i] -= max_num;
      }
    }
  } else {
    index_t dim = x1.shape()[x1.shape().size() - 1];
    const int block_size = 512;
    const int grid_size = (x1.dsize() / dim + 511) / 512;
    before_softmax_kernel<<<grid_size, block_size>>>(x1.data(), dim,
                                                     x1.dsize() / dim);
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}

void Softmax::forward_process(Tensor& x1, Tensor& x) {
  // before_softmax(x1);
  if (x1.device() == "cpu") {
    index_t dim = x1.shape()[x1.shape().size() - 1];
    for (int b = 0; b < x1.dsize() / dim; ++b) {
      index_t offset = b * dim;
      float exp_sum = 0;
      for (int i = 0; i < dim; ++i) {
        exp_sum += exp(x1[offset + i]);
      }
      for (int i = 0; i < dim; ++i) {
        x[offset + i] = exp(x1[offset + i]) / exp_sum;
      }
    }
  } else {
    auto shape = x.shape();
    // cudaMemcpy(x.data(), x1.data(), sizeof(float) * x1.dsize(),
    // cudaMemcpyDeviceToDevice);
    const dim3 grid_size(
        1, shape[shape.size() - 2],
        x.dsize() / (shape[shape.size() - 2] * shape[shape.size() - 1]));
    const int block_size = min(
        512, (shape[shape.size() - 1] % 2 == 0 ? shape[shape.size() - 1]
                                               : shape[shape.size() - 1] + 1));
    softmax_kernel<<<grid_size, block_size,
                     sizeof(float) *
                         ((shape[shape.size() - 1] + block_size - 1) /
                          block_size * block_size)>>>(
        x1.data(), x.data(),
        (shape[shape.size() - 1] + block_size - 1) / block_size,
        shape[shape.size() - 1], x.dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}

__global__ void softmax_kernel(float* x1, float* x, int N, index_t dim,
                               index_t dsize) {
  const index_t z = blockIdx.z;
  const index_t y = blockIdx.y;
  const index_t thread_idx = threadIdx.x;

  const index_t head_offset = z * gridDim.y * dim + y * dim;

  extern __shared__ float data[];
  for (int i = 0; i < N; ++i) {
    index_t offset = thread_idx * N + i;
    if (offset < dim && (head_offset + offset) < dsize) {
        data[offset] = exp(x1[head_offset + offset]);
    } else {
      data[offset] = 0;
    }
  }
  __syncthreads();
  for (int gap = (blockDim.x * N) >> 1; gap > 0; gap >>= 1) {
    for (int i = 0; i < N; ++i) {
      int offset = thread_idx * N + i;
      if (offset < gap) {
        data[offset] += data[offset + gap];
      }
    }
    __syncthreads();
  }
  __syncthreads();
  float exp_sum = data[0];

  for (int i = 0; i < N; ++i) {
    int offset = thread_idx * N + i;
    if (offset < dim && head_offset + offset < dsize) {
      x[head_offset + offset] = exp(x1[head_offset + offset]) / exp_sum;
      assert(x[head_offset + offset] < 1);
    }
  }
}

void Softmax::grad_fn(TensorImplPtr x1, TensorImplPtr x) {
  if (x1->device() == "cpu") {
    index_t dim = x1->shape()[x1->shape().size() - 1];
    for (int b = 0; b < x1->dsize() / dim; ++b) {
      index_t offset = b * dim;
      float sum = 0;
      for (int i = 0; i < dim; ++i) {
        sum += x->data_[offset + i] * x->grad_[offset + i];
      }
      for (int i = 0; i < dim; ++i) {
        x1->grad_[offset + i] += (x->data_[offset + i] * x->grad_[offset + i] -
                                  x->data_[offset + i] * sum);
      }
    }
  } else {
    auto shape = x->shape();
    const dim3 grid_size(
        1, shape[shape.size() - 2],
        x->dsize() / (shape[shape.size() - 2] * shape[shape.size() - 1]));
    const int block_size = min(
        512, (shape[shape.size() - 1] % 2 == 0 ? shape[shape.size() - 1]
                                               : shape[shape.size() - 1] + 1));
    softmax_backward_kernel<<<grid_size, block_size,
                              3 * sizeof(float) *
                                  ((shape[shape.size() - 1] + block_size - 1) /
                                   block_size * block_size)>>>(
        x1->grad_, x1->data_, x->grad_, x->data_,
        (shape[shape.size() - 1] + block_size - 1) / block_size,
        shape[shape.size() - 1], x->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}

__global__ void softmax_backward_kernel(float* x1_grad, float* x1_data,
                                        float* x_grad, float* x_data, const int N,
                                        index_t dim, index_t dsize) {
  const index_t z = blockIdx.z;
  const index_t y = blockIdx.y;
  const index_t thread_idx = threadIdx.x;

  const index_t head_offset = z * gridDim.y * dim + y * dim;

  extern __shared__ float shared_data[];
  float* x_data_s = shared_data;
  float* x_grad_s = (float*)&x_data_s[N * blockDim.x];
  float* data = (float*)&x_grad_s[N * blockDim.x];

  for (int i = 0; i < N; ++i) {
    index_t offset = thread_idx * N + i;
    if (offset < dim && (head_offset + offset) < dsize) {
        x_data_s[offset] = x_data[head_offset + offset];
        x_grad_s[offset] = x_grad[head_offset + offset];
    } else {
      x_data_s[offset] = 0;
      x_grad_s[offset] = 0;
    }
  }
  __syncthreads();

  for (int i = 0; i < N; ++i) {
    index_t offset = thread_idx * N + i;
    if (offset < dim && (head_offset + offset) < dsize) {
        data[offset] =
            x_data_s[offset] * x_grad_s[offset];
    } else {
      data[offset] = 0;
    }
  }
  __syncthreads();
  for (int gap = (blockDim.x * N) >> 1; gap > 0; gap >>= 1) {
    for (int i = 0; i < N; ++i) {
      int offset = thread_idx * N + i;
      if (offset < gap) {
        data[offset] += data[offset + gap];
      }
    }
    __syncthreads();
  }
  __syncthreads();
  float sum = data[0];

  for (int i = 0; i < N; ++i) {
    int offset = thread_idx * N + i;
    if (offset < dim && head_offset + offset < dsize) {
      x1_grad[head_offset + offset] +=
          (x_data_s[offset] * x_grad_s[offset] -
           x_data_s[offset] * sum);
    }
  }
}

void Log::forward_process(Tensor& x1, Tensor& x) {
  if (x1.device() == "cpu") {
    for (int i = 0; i < x1.dsize(); ++i) {
      x[i] = log(x1[i]);
    }
  } else {
    const int block_size = 512;
    const int grid_size = (x1.dsize() + 511) / 512;
    log_kernel<<<grid_size, block_size>>>(x1.data(), x.data(), x1.dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}

__global__ void log_kernel(float* x1, float* x, index_t dsize) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize) {
    x[idx] = log(x1[idx]);
  }
}

void Log::grad_fn(TensorImplPtr x1, TensorImplPtr x) {
  if (x1->device() == "cpu") {
    for (int i = 0; i < x1->dsize(); ++i) {
      x1->grad_[i] += (x->grad_[i] / (x1->data_[i] + 1e-8));
    }
  } else {
    const int block_size = 512;
    const int grid_size = (x1->dsize() + 511) / 512;
    log_backward_kernel<<<grid_size, block_size>>>(x1->grad_, x1->data_,
                                                   x->grad_, x1->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}

__global__ void log_backward_kernel(float* x1_grad, float* x1_data,
                                    float* x_grad, index_t dsize) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize) {
    x1_grad[idx] += (x_grad[idx] / (x1_data[idx] + 1e-8));
  }
}

Tensor NLLLoss::generate_ret_tensor(Tensor& x1, Tensor& x2) {
  Tensor ret({1});
  ret.data()[0] = 0;
  ret.grad()[0] = 1;
  ret.to(x1.device());
  return ret;
}

void NLLLoss::prepare_forward(Tensor& x1, Tensor& x2, Tensor& x) {
  // 1. check
  if (x1.device() != x2.device()) {
    std::cout << "TensorImplPtr must in same device." << std::endl;
    assert(false);
  }

  assert(x1.ndim() == 2);
  assert(x2.ndim() == 1);
  assert(x1.shape()[0] == x2.dsize());

  // 2. add node to return Tensor
  GraphNodePtr node = std::make_shared<BinaryGraphNode>(x1.get(), x2.get());
  node->setGradFnL(std::bind(&NLLLoss::lhs_grad_fn, this, _1, _2, _3));
  node->setGradFnR(std::bind(&NLLLoss::rhs_grad_fn, this, _1, _2, _3));
  x.setNode(node);

  return;
}

void NLLLoss::forward_process(Tensor& x1, Tensor& x2, Tensor& x) {
  if (x1.device() == "cpu") {
    for (int i = 0; i < x2.dsize(); ++i) {
      index_t offset = x1.shape()[1] * i;
      x[0] += (-x1[offset + (int)x2[i]]);
    }
    x[0] /= x2.dsize();
  } else {
    const int block_size = 512;
    const int grid_size = (x2.dsize() + 511) / 512;
    nllloss_kernel<<<grid_size, block_size, sizeof(float) * 512>>>(
        x1.data(), x2.data(), x.data(), x1.shape()[1], x2.dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
    x.cpu();
    cudaDeviceSynchronize();
    x[0] /= x2.dsize();
    x.cuda();
  }
}

__global__ void nllloss_kernel(float* x1, float* x2, float* x, index_t len,
                               index_t dsize) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ float data[];
  if (idx < dsize) {
    data[threadIdx.x] = -x1[idx * len + (int)x2[idx]];
  } else {
    data[threadIdx.x] = 0;
  }
  __syncthreads();

  for (int gap = blockDim.x >> 1; gap > 0; gap >>= 1) {
    if (threadIdx.x < gap) {
      data[threadIdx.x] += data[threadIdx.x + gap];
    }
    __syncthreads();
  }
  __syncthreads();
  float loss = data[0];

  if (threadIdx.x == 0) {
    atomicAdd(x, loss);
  }
  __syncthreads();
  // if (idx == 0) {
  //     x[0] /= dsize;
  // }
  // __syncthreads();
}

void NLLLoss::lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
  if (x1->device() == "cpu") {
    for (int i = 0; i < x2->dsize(); ++i) {
      index_t offset = x1->shape()[1] * i;
      x1->grad_[offset + (int)(x2->data_[i])] += -(
          x->grad_[0] / x2->dsize() * x1->data_[offset + (int)(x2->data_[i])]);
    }
  } else {
    const int block_size = 512;
    const int grid_size = (x2->dsize() + 511) / 512;
    nllloss_backward_kernel<<<grid_size, block_size>>>(
        x1->grad_, x1->data_, x2->data_, x->grad_, x1->shape()[1], x2->dsize());
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}

void NLLLoss::rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
  return;
}

__global__ void nllloss_backward_kernel(float* x1_grad, float* x1_data,
                                        float* x2_data, float* x_grad,
                                        index_t len, index_t dsize) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize) {
    x1_grad[idx * len + (int)(x2_data[idx])] +=
        -(x_grad[0] / dsize * x1_data[idx * len + (int)(x2_data[idx])]);
  }
}

Tensor Emb::generate_ret_tensor(Tensor& idx, Tensor& emb) {
  if (idx.shape().size() == 1) {
    Tensor x({idx.shape()[0], emb.shape()[1]}, emb.device());
    return x;
  } else {
    Tensor x({idx.shape()[0], idx.shape()[1], emb.shape()[1]}, emb.device());
    return x;
  }
}

void Emb::prepare_forward(Tensor& idx, Tensor& emb, Tensor& x) {
  // 1. check
  if (idx.device() != emb.device()) {
    std::cout << "TensorImplPtr must in same device." << std::endl;
    assert(false);
  }

  assert(emb.shape().size() == 2);

  // 2. add node to return Tensor
  GraphNodePtr node = std::make_shared<BinaryGraphNode>(idx.get(), emb.get());
  node->setGradFnL(std::bind(&Emb::lhs_grad_fn, this, _1, _2, _3));
  node->setGradFnR(std::bind(&Emb::rhs_grad_fn, this, _1, _2, _3));
  x.setNode(node);
}

void Emb::forward_process(Tensor& idx, Tensor& emb, Tensor& x) {
  index_t vocab_size = emb.shape()[0];
  index_t hidden_dim = emb.shape()[1];
  if (idx.device() == "cpu") {
    for (int i = 0; i < idx.dsize(); ++i) {
      index_t offset = (index_t)idx[i];
      assert(offset < vocab_size);
      memcpy(x.data() + hidden_dim * i, emb.data() + offset * hidden_dim,
             sizeof(float) * hidden_dim);
    }
  } else {
    idx.cpu();
    for (int i = 0; i < idx.dsize(); ++i) {
      index_t offset = (index_t)idx[i];
      assert(offset < vocab_size);
      cudaMemcpy(x.data() + hidden_dim * i, emb.data() + offset * hidden_dim,
                 sizeof(float) * hidden_dim, cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
    idx.cuda();
  }
}

void Emb::lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
  return;
}

__global__ void emb_backward_kernel(float* idx, float* emb_grad, float* x_grad,
                                    int N, index_t dim, index_t dsize) {
  const index_t id = blockIdx.x;
  const index_t thread_idx = threadIdx.x;

  if (id < dsize) {
    index_t vocab_id = (index_t)idx[id];
    index_t head_offset = thread_idx * N;
    for (int i = 0; i < N; ++i) {
      if (head_offset + i < dim) {
        atomicAdd(emb_grad + vocab_id * dim + head_offset + i,
                  x_grad[id * dim + head_offset + i]);
      }
    }
  }
  __syncthreads();
}

void Emb::rhs_grad_fn(TensorImplPtr idx, TensorImplPtr emb, TensorImplPtr x) {
  index_t hidden_dim = emb->shape()[1];
  if (idx->device() == "cpu") {
    for (int i = 0; i < idx->dsize(); ++i) {
      index_t offset = (index_t)idx->data_[i];
      for (int d = 0; d < hidden_dim; ++d) {
        emb->grad_[offset * hidden_dim + d] += x->grad_[i * hidden_dim + d];
      }
    }
  } else {
    int grid_size = idx->dsize();
    int block_size = min(512, (emb->shape()[1] % 2 == 0 ? emb->shape()[1]
                                                        : emb->shape()[1] + 1));
    emb_backward_kernel<<<grid_size, block_size>>>(
        idx->data_, emb->grad_, x->grad_,
        (hidden_dim + block_size - 1) / block_size, hidden_dim, idx->dsize());

    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}