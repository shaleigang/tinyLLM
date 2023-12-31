#include <cassert>
#include <fstream>
#include <sys/stat.h>

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

__global__ void adamw_kernel(float* grad, float* data, float* m1, float* m2,
                             index_t t, float lr, float beta1, float beta2,
                             float ep, float weight_decay, index_t dsize) {
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
    p.get()->recovery(0);
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
      const int block_size = 512;
      const int grid_size = (p.dsize() + 511) / 512;
      adamw_kernel<<<grid_size, block_size>>>(p.grad(), p.data(), m1.get(),
                                              m2.get(), t, lr_, beta1_, beta2_,
                                              ep, weight_decay_, p.dsize());
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
    p.get()->recovery(0);
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
      const int block_size = 512;
      const int grid_size = (p.dsize() + 511) / 512;
      adamw_kernel<<<grid_size, block_size>>>(p.grad(), p.data(), m1.get(),
                                              m2.get(), t, lr_, beta1_, beta2_,
                                              ep, 0, p.dsize());
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

void AdamW::save(string path) {
  struct stat st;

    if (stat(path.c_str(), &st) == -1) {
        if (mkdir(path.c_str(), 0775) == -1) {
            perror("mkdir error");
            return;
        }
    }

    string bin_path = path + "optimizer.bin";
    string offset_path = path + "optimizer.index";

    std::ofstream data_out(bin_path, std::ios::binary | std::ios::trunc);
    std::ofstream offset_out(offset_path, std::ios::trunc);

    data_out.write((char*)&t, sizeof(index_t));

    for (auto iter : decay_params_) {
        std::shared_ptr<float> data_raw1 = moment1_[iter.first];
        std::shared_ptr<float> data_raw2 = moment2_[iter.first];
        index_t len = sizeof(float) * iter.second.get().dsize();
        offset_out << iter.first << " " << len << std::endl;
        if (device_ == "cpu") {
          data_out.write((char*)data_raw1.get(), len);
          data_out.write((char*)data_raw2.get(), len);
        }
        else {
          float* data = (float*)malloc(len);
          cudaMemcpy(data, data_raw1.get(), len, cudaMemcpyDeviceToHost);
          data_out.write((char*)data, len);
          cudaMemcpy(data, data_raw2.get(), len, cudaMemcpyDeviceToHost);
          data_out.write((char*)data, len);
          free(data);
        }
    }
    for (auto iter : nodecay_params_) {
        std::shared_ptr<float> data_raw1 = moment1_[iter.first];
        std::shared_ptr<float> data_raw2 = moment2_[iter.first];
        index_t len = sizeof(float) * iter.second.get().dsize();
        offset_out << iter.first << " " << len << std::endl;
        if (device_ == "cpu") {
          data_out.write((char*)data_raw1.get(), len);
          data_out.write((char*)data_raw2.get(), len);
        }
        else {
          float* data = (float*)malloc(len);
          cudaMemcpy(data, data_raw1.get(), len, cudaMemcpyDeviceToHost);
          data_out.write((char*)data, len);
          cudaMemcpy(data, data_raw2.get(), len, cudaMemcpyDeviceToHost);
          data_out.write((char*)data, len);
          free(data);
        }
    }
    data_out.close();
    offset_out.close();
}

void AdamW::load(string path) {
  if (path == "restart") {
    return;
  }

  string bin_path = path + "optimizer.bin";
    string offset_path = path + "optimizer.index";

    std::ifstream data_in(bin_path, std::ios::binary);
    std::ifstream offset_in(offset_path);

    if (!data_in.good()) {
        std::cout << "Can not find bin file in \"" << path << "\"" << std::endl;
        exit(0);
    }
    if (!offset_in.good()) {
        std::cout << "Can not find index file in \"" << path << "\"" << std::endl;
        exit(0);
    }

    data_in.read((char*)&t, sizeof(index_t));

    string name;
    index_t len;
    while (offset_in >> name >> len) {
        if (device_ == "cpu") {
          data_in.read((char*)moment1_[name].get(), len);
          data_in.read((char*)moment2_[name].get(), len);
        }
        else {
          float* data = (float*)malloc(len);
          data_in.read((char*)data, len);
          cudaMemcpy(moment1_[name].get(), data, len, cudaMemcpyHostToDevice);
          data_in.read((char*)data, len);
          cudaMemcpy(moment2_[name].get(), data, len, cudaMemcpyHostToDevice);
          free(data);
        }
    }
}