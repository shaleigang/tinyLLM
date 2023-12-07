#include <cassert>

#include "optimizer.h"

using namespace tllm;

bool del_cpu(float* p) { delete[] p; }

bool del_gpu(float* p) {
  cudaFree(p);
  auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
}

AdamW::AdamW(detail::ParamsDict decay_params, detail::ParamsDict nodecay_params,
             float lr, float beta1, float beta2, string device)
    : decay_params_(decay_params),
      nodecay_params_(nodecay_params),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      t(0),
      device_(device) {
  for (auto& iter : decay_params_) {
    string name = iter.first;
    Tensor& p = iter.second.get();
    if (device_ == "cpu") {
      std::shared_ptr<float> t1(new float[p.dsize()], del_cpu);
      memset(t1.get(), 0, p.dsize() * sizeof(float));

      std::shared_ptr<float> t2(new float[p.dsize()], del_cpu);
      memset(t2.get(), 0, p.dsize() * sizeof(float));

      moment1_.insert({name, t1});
      moment2_.insert({name, t2});
    } else {
      float* m1;
      cudaMalloc((void**)&m1, p.dsize() * sizeof(float));
      cudaMemset(m1, 0, p.dsize() * sizeof(float));
      auto error = cudaGetLastError();
      if (cudaSuccess != error) {
        printf("%s\n", cudaGetErrorString(error));
        assert(false);
      }
      std::shared_ptr<float> t1(m1, del_gpu);

      float* m2;
      cudaMalloc((void**)&m2, p.dsize() * sizeof(float));
      cudaMemset(m2, 0, p.dsize() * sizeof(float));
      error = cudaGetLastError();
      if (cudaSuccess != error) {
        printf("%s\n", cudaGetErrorString(error));
        assert(false);
      }
      std::shared_ptr<float> t2(m2, del_gpu);

      moment1_.insert({name, t1});
      moment2_.insert({name, t2});
    }
  }
  for (auto& iter : nodecay_params_) {
    string name = iter.first;
    Tensor& p = iter.second.get();
    if (device_ == "cpu") {
      std::shared_ptr<float> t1(new float[p.dsize()], del_cpu);
      memset(t1.get(), 0, p.dsize() * sizeof(float));

      std::shared_ptr<float> t2(new float[p.dsize()], del_cpu);
      memset(t2.get(), 0, p.dsize() * sizeof(float));

      moment1_.insert({name, t1});
      moment2_.insert({name, t2});
    } else {
      float* m1;
      cudaMalloc((void**)&m1, p.dsize() * sizeof(float));
      cudaMemset(m1, 0, p.dsize() * sizeof(float));
      auto error = cudaGetLastError();
      if (cudaSuccess != error) {
        printf("%s\n", cudaGetErrorString(error));
        assert(false);
      }
      std::shared_ptr<float> t1(m1, del_gpu);

      float* m2;
      cudaMalloc((void**)&m2, p.dsize() * sizeof(float));
      cudaMemset(m2, 0, p.dsize() * sizeof(float));
      error = cudaGetLastError();
      if (cudaSuccess != error) {
        printf("%s\n", cudaGetErrorString(error));
        assert(false);
      }
      std::shared_ptr<float> t2(m2, del_gpu);

      moment1_.insert({name, t1});
      moment2_.insert({name, t2});
    }
  }
}

__global__ void adamw_kernel(float* grad, float* data, float* m1, float* m2, index_t t,
                             float lr, float beta1, float beta2, float ep,
                             float weight_decay, index_t dsize) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dsize) {
    m1[i] = beta1 * m1[i] + (1 - beta1) * grad[i];
    m2[i] = beta2 * m2[i] + (1 - beta2) * grad[i] * grad[i];
    float lr_n = lr * sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t));
    grad[i] = -lr_n * (m1[i] / (sqrt(m2[i]) + ep) + weight_decay * data[i]);
  }
}

void AdamW::step() {
  ++t;
  for (auto& iter : decay_params_) {
    string name = iter.first;
    Tensor& p = iter.second.get();
    std::shared_ptr<float> m1 = moment1_[name];
    std::shared_ptr<float> m2 = moment2_[name];

    if (device_ == "cpu") {
      for (int i = 0; i < p.dsize(); ++i) {
        m1.get()[i] = beta1_ * m1.get()[i] + (1 - beta1_) * p.grad()[i];
        m2.get()[i] =
            beta2_ * m2.get()[i] + (1 - beta2_) * p.grad()[i] * p.grad()[i];
        float lr = lr_ * sqrt(1 - pow(beta2_, t)) / (1 - pow(beta1_, t));
        p.grad()[i] = -lr * (m1.get()[i] / (sqrt(m2.get()[i]) + ep) +
                             weight_decay_ * p.data()[i]);
      }
    } else {
      const int block_size = 256;
      const int grid_size = (p.dsize() + 255) / 256;
      adamw_kernel<<<grid_size, block_size>>>(p.grad(), p.data(), m1.get(), m2.get(), t,
                                              lr_, beta1_, beta2_, ep,
                                              weight_decay_, p.dsize());
      cudaDeviceSynchronize();
      auto error = cudaGetLastError();
      if (cudaSuccess != error) {
        printf("%s\n", cudaGetErrorString(error));
        assert(false);
      }
    }
    p.apply_grad();
  }

  for (auto& iter : nodecay_params_) {
    string name = iter.first;
    Tensor& p = iter.second.get();
    std::shared_ptr<float> m1 = moment1_[name];
    std::shared_ptr<float> m2 = moment2_[name];

    if (device_ == "cpu") {
      for (int i = 0; i < p.dsize(); ++i) {
        m1.get()[i] = beta1_ * m1.get()[i] + (1 - beta1_) * p.grad()[i];
        m2.get()[i] =
            beta2_ * m2.get()[i] + (1 - beta2_) * p.grad()[i] * p.grad()[i];
        float lr = lr_ * sqrt(1 - pow(beta2_, t)) / (1 - pow(beta1_, t));
        p.grad()[i] = -lr * (m1.get()[i] / (sqrt(m2.get()[i]) + ep));
      }
    } else {
      const int block_size = 256;
      const int grid_size = (p.dsize() + 255) / 256;
      adamw_kernel<<<grid_size, block_size>>>(p.grad(), p.data(), m1.get(), m2.get(), t,
                                              lr_, beta1_, beta2_, ep, 0,
                                              p.dsize());
      cudaDeviceSynchronize();
      auto error = cudaGetLastError();
      if (cudaSuccess != error) {
        printf("%s\n", cudaGetErrorString(error));
        assert(false);
      }
    }
    p.apply_grad();
  }
}