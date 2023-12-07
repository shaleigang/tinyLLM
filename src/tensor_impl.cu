#include <algorithm>
#include <cassert>
#include <cstring>

#include "tensor_impl.h"

using namespace tllm::detail;

__global__ void contiguous_kernel(float* src, float* dst, index_t dsize,
                                  index_t ndim, index_t* stride,
                                  index_t* order_stride);

__global__ void operator_index_kernel(float* data, float* target, index_t idx);


TensorImpl::TensorImpl(std::initializer_list<index_t> dims, string device,
               bool require_grad)
    : dims_(dims), require_grad_(require_grad), device_(device), ref_count_(0) {
  // init stride_
  stride_ = std::vector<index_t>(dims_.size(), 0);
  int stride = 1;
  for (int i = dims_.size() - 1; i >= 0; --i) {
    stride_[i] = stride;
    stride *= dims_[i];
  }

  dsize_ = stride;
  // malloc for data_ and grad_
  if (device_ == "cpu") {
    data_ = (float*)malloc(sizeof(float) * stride);
    grad_ = (float*)malloc(sizeof(float) * stride);
    memset(grad_, 0, sizeof(float) * stride);
  } else if (device_ == "cuda") {
    cudaMalloc((void**)&data_, sizeof(float) * stride);
    cudaMalloc((void**)&grad_, sizeof(float) * stride);
    cudaMemset(grad_, 0, sizeof(float) * stride);
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  } else {
    std::cout << "Not support device." << std::endl;
    assert(false);
  }
}

TensorImpl::TensorImpl(std::vector<index_t> dims, string device,
               bool require_grad)
    : dims_(dims), require_grad_(require_grad), device_(device), ref_count_(0) {
  // init stride_
  stride_ = std::vector<index_t>(dims_.size(), 0);
  int stride = 1;
  for (int i = dims_.size() - 1; i >= 0; --i) {
    stride_[i] = stride;
    stride *= dims_[i];
  }

  dsize_ = stride;
  // malloc for data_ and grad_
  if (device_ == "cpu") {
    data_ = (float*)malloc(sizeof(float) * stride);
    grad_ = (float*)malloc(sizeof(float) * stride);
    memset(grad_, 0, sizeof(float) * stride);

  } else if (device_ == "cuda") {
    cudaMalloc((void**)&data_, sizeof(float) * stride);
    cudaMalloc((void**)&grad_, sizeof(float) * stride);
    cudaMemset(grad_, 0, sizeof(float) * stride);
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  } else {
    std::cout << "Not support device." << std::endl;
    assert(false);
  }
}

TensorImpl::TensorImpl(const TensorImpl& t)
    : require_grad_(t.require_grad_),
      dims_(t.dims_),
      stride_(t.stride_),
      device_(t.device_),
      dsize_(t.dsize_),
      ref_count_(0) {
  if (device_ == "cpu") {
    data_ = (float*)malloc(sizeof(float) * t.dsize());
    std::memcpy(data_, t.data_, sizeof(float) * t.dsize());
    grad_ = (float*)malloc(sizeof(float) * t.dsize());
    std::memcpy(grad_, t.grad_, sizeof(float) * t.dsize());
  } else {
    cudaMalloc((void**)&data_, sizeof(float) * t.dsize());
    cudaMemcpy(data_, t.data_, sizeof(float) * t.dsize(), cudaMemcpyDeviceToDevice);
    cudaMalloc((void**)&grad_, sizeof(float) * t.dsize());
    cudaMemcpy(grad_, t.grad_, sizeof(float) * t.dsize(), cudaMemcpyDeviceToDevice);
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}

TensorImpl::TensorImpl(TensorImpl&& t) {
    device_ = t.device_;
    require_grad_ = t.require_grad_;
    dims_ = t.dims_;
    stride_ = t.stride_;
    dsize_ = t.dsize_;
    next_node_ptr_ = t.next_node_ptr_;
    ref_count_ = t.ref_count_;
    data_ = t.data_;
    grad_ = t.grad_;
    t.data_ = nullptr;
    t.grad_ = nullptr;
}

TensorImpl::~TensorImpl() {
  if (device_ == "cpu") {
    free(data_);
    free(grad_);
  } else {
    cudaFree(data_);
    cudaFree(grad_);
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
}

TensorImpl& TensorImpl::operator=(const TensorImpl& t) {
  device_ = t.device_;
  require_grad_ = t.require_grad_;
  dims_ = t.dims_;
  stride_ = t.stride_;
  dsize_ = t.dsize_;
  ref_count_ = 0;
  if (device_ == "cpu") {
    free(data_);
    data_ = (float*)malloc(sizeof(float) * t.dsize());
    std::memcpy(data_, t.data_, sizeof(float) * t.dsize());
    free(grad_);
    grad_ = (float*)malloc(sizeof(float) * t.dsize());
    std::memcpy(grad_, t.grad_, sizeof(float) * t.dsize());
  } else {
    cudaFree(data_);
    cudaMalloc((void**)&data_, sizeof(float) * t.dsize());
    cudaMemcpy(data_, t.data_, sizeof(float) * t.dsize(), cudaMemcpyDeviceToDevice);
    cudaFree(grad_);
    cudaMalloc((void**)&grad_, sizeof(float) * t.dsize());
    cudaMemcpy(grad_, t.grad_, sizeof(float) * t.dsize(), cudaMemcpyDeviceToDevice);
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
  return *this;
}

float& TensorImpl::operator[](std::initializer_list<index_t> ids) {
  assert(ids.size() == ndim());
  index_t offset = 0, i = 0;
  for (auto index : ids) {
    assert(index < dims_[i]);
    offset += index * stride_[i++];
  }
  if (device_ == "cpu")
    return data_[offset];
  else {
    std::cout << "only support non const index on cpu TensorImpl." << std::endl;
    assert(false);
  }
}

float TensorImpl::operator[](std::initializer_list<index_t> ids) const {
  assert(ids.size() == ndim());
  index_t offset = 0, i = 0;
  for (auto index : ids) {
    assert(index < dims_[i]);
    offset += index * stride_[i++];
  }
  if (device_ == "cpu")
    return data_[offset];
  else {
    float* target = (float*)malloc(sizeof(float));
    cudaMemcpy(target, data_ + offset, sizeof(float), cudaMemcpyDeviceToHost);
    float ret = *target;
    free(target);
    return ret;
  }
}

float& TensorImpl::operator[](std::vector<index_t> ids) {
  assert(ids.size() == ndim());
  index_t offset = 0, i = 0;
  for (auto index : ids) {
    assert(index < dims_[i]);
    offset += index * stride_[i++];
  }
  if (device_ == "cpu")
    return data_[offset];
  else {
    std::cout << "only support non const index on cpu TensorImpl." << std::endl;
    assert(false);
  }
}

float TensorImpl::operator[](std::vector<index_t> ids) const {
  assert(ids.size() == ndim());
  index_t offset = 0, i = 0;
  for (auto index : ids) {
    assert(index < dims_[i]);
    offset += index * stride_[i++];
  }
  if (device_ == "cpu")
    return data_[offset];
  else {
    float* target = (float*)malloc(sizeof(float));
    cudaMemcpy(target, data_ + offset, sizeof(float), cudaMemcpyDeviceToHost);
    float ret = *target;
    free(target);
    return ret;
  }
}

float& TensorImpl::operator[](index_t offset) {
  if (device_ == "cpu")
    return data_[offset];
  else {
    std::cout << "only support index on cpu TensorImpl." << std::endl;
    assert(false);
  }
}

float TensorImpl::operator[](index_t offset) const {
  if (device_ == "cpu")
    return data_[offset];
  else {
    float target = 0;
    float* target_device = nullptr;
    cudaMalloc((void**)&target_device, sizeof(float));
    operator_index_kernel<<<1, 1>>>(data_, target_device, offset);
    cudaMemcpy(&target, target_device, sizeof(float), cudaMemcpyDeviceToHost);
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
    return target;
  }
}

index_t TensorImpl::get_offset(std::initializer_list<index_t> ids) const {
  assert(ids.size() == ndim());
  index_t offset = 0, i = 0;
  for (auto index : ids) {
    assert(index < dims_[i]);
    offset += index * stride_[i++];
  }
  return offset;
}


index_t TensorImpl::get_offset(std::vector<index_t> ids) const {
  assert(ids.size() == ndim());
  index_t offset = 0, i = 0;
  for (auto index : ids) {
    assert(index < dims_[i]);
    offset += index * stride_[i++];
  }
  return offset;
}

void TensorImpl::transpose(index_t dim1, index_t dim2) {
  std::swap(dims_[dim1], dims_[dim2]);
  std::swap(stride_[dim1], stride_[dim2]);

  shape_recovery_queue_.push_back(
    std::bind(&TensorImpl::transpose_recovery, this,
            std::vector<index_t>{dim1, dim2})
  );

  return;
}

void TensorImpl::transpose_recovery(std::vector<index_t> dims) {
  std::swap(dims_[dims[0]], dims_[dims[1]]);
  std::swap(stride_[dims[0]], stride_[dims[1]]);
  return;
}

void TensorImpl::view_recovery(std::vector<index_t> dims) {
  assert(is_contiguous());
  std::vector<index_t> newDims;
  index_t newDsize = 1;
  for (auto d : dims) {
    newDims.push_back(d);
    newDsize *= d;
  }
  assert(newDsize == dsize());
  dims_ = newDims;
  index_t stride = 1;
  stride_ = std::vector<index_t>(newDims.size(), 0);
  for (int i = newDims.size() - 1; i >= 0; --i) {
    stride_[i] = stride;
    stride *= newDims[i];
  }
}

void TensorImpl::view(std::initializer_list<index_t> dims) {
  assert(is_contiguous());
  std::vector<index_t> old_shape = shape();
  std::vector<index_t> newDims;
  index_t newDsize = 1;
  for (auto d : dims) {
    newDims.push_back(d);
    newDsize *= d;
  }
  assert(newDsize == dsize());
  dims_ = newDims;
  index_t stride = 1;
  stride_ = std::vector<index_t>(newDims.size(), 0);
  for (int i = newDims.size() - 1; i >= 0; --i) {
    stride_[i] = stride;
    stride *= newDims[i];
  }

  shape_recovery_queue_.push_back(
    std::bind(&TensorImpl::view_recovery, this, old_shape)
  );
}

void TensorImpl::view(std::vector<index_t> dims) {
  assert(is_contiguous());
  std::vector<index_t> old_shape = shape();
  std::vector<index_t> newDims;
  index_t newDsize = 1;
  for (auto d : dims) {
    newDims.push_back(d);
    newDsize *= d;
  }
  assert(newDsize == dsize());
  dims_ = newDims;
  index_t stride = 1;
  stride_ = std::vector<index_t>(newDims.size(), 0);
  for (int i = newDims.size() - 1; i >= 0; --i) {
    stride_[i] = stride;
    stride *= newDims[i];
  }

  shape_recovery_queue_.push_back(
    std::bind(&TensorImpl::view_recovery, this, old_shape)
  );
}

void TensorImpl::to(string dev) {
  assert(dev == "cpu" || dev == "cuda");
  if (dev == device_) return;
  if (dev == "cpu") {
    float* newData = (float*)malloc(sizeof(float) * dsize_);
    cudaMemcpy(newData, data_, sizeof(float) * dsize_, cudaMemcpyDeviceToHost);
    cudaFree(data_);
    data_ = newData;
    float* newGrad = (float*)malloc(sizeof(float) * dsize_);
    cudaMemcpy(newGrad, grad_, sizeof(float) * dsize_,
                 cudaMemcpyDeviceToHost);
    cudaFree(grad_);
    grad_ = newGrad;
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  } else {
    float* newData = nullptr;
    cudaMalloc((void**)&newData, sizeof(float) * dsize_);
    cudaMemcpy(newData, data_, sizeof(float) * dsize_, cudaMemcpyHostToDevice);
    free(data_);
    data_ = newData;
    float* newGrad = nullptr;
    cudaMalloc((void**)&newGrad, sizeof(float) * dsize_);
    cudaMemcpy(newGrad, grad_, sizeof(float) * dsize_,
                 cudaMemcpyHostToDevice);
    free(grad_);
    grad_ = newGrad;
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }
  device_ = dev;
}

bool TensorImpl::is_contiguous() const {
  index_t stride = 1;
  for (int i = stride_.size() - 1; i >= 0; --i) {
    if (stride_[i] != stride) return false;
    stride *= dims_[i];
  }
  return true;
}

void TensorImpl::contiguous() {
  if (is_contiguous()) return;

  std::vector<index_t> order_stride(dims_.size(), 0);
  int stride = 1;
  for (int i = dims_.size() - 1; i >= 0; --i) {
    order_stride[i] = stride;
    stride *= dims_[i];
  }
  if (device_ == "cpu") {
    float* newData = (float*)malloc(sizeof(float) * dsize());
    int count = 0;
    std::vector<index_t> pos(dims_.size(), 0);
    while (count < dsize_) {
      index_t offset = 0, i = 0;
      for (auto index : pos) {
        assert(index < dims_[i]);
        offset += index * stride_[i++];
      }
      newData[count] = data_[offset];

      for (int i = dims_.size() - 1; i >= 0; --i) {
        if (pos[i] == dims_[i] - 1) {
          pos[i] = 0;
        } else {
          ++pos[i];
          break;
        }
      }
      ++count;
    }
    free(data_);
    data_ = newData;
    {
      float* newGrad = (float*)malloc(sizeof(float) * dsize());
      int count = 0;
      std::vector<index_t> pos(dims_.size(), 0);
      while (count < dsize_) {
        index_t offset = 0, i = 0;
        for (auto index : pos) {
          assert(index < dims_[i]);
          offset += index * stride_[i++];
        }
        newGrad[count] = grad_[offset];

        for (int i = dims_.size() - 1; i >= 0; --i) {
          if (pos[i] == dims_[i] - 1) {
            pos[i] = 0;
          } else {
            ++pos[i];
            break;
          }
        }
        ++count;
      }
      free(grad_);
      grad_ = newGrad;
    }
  } else {
    float* newData = nullptr;
    index_t* stride_device = nullptr;
    index_t* order_stride_device = nullptr;
    cudaMalloc((void**)&newData, dsize_ * sizeof(float));
    cudaMalloc((void**)&stride_device, ndim() * sizeof(index_t));
    cudaMalloc((void**)&order_stride_device, ndim() * sizeof(index_t));
    cudaMemcpy(stride_device, stride_.data(), ndim() * sizeof(index_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(order_stride_device, order_stride.data(),
               ndim() * sizeof(index_t), cudaMemcpyHostToDevice);
    auto error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
    const int block_size = 256;
    const int grid_size = (dsize_ + 255) / 256;

    contiguous_kernel<<<grid_size, block_size>>>(data_, newData, dsize_,
                                                 dims_.size(), stride_device,
                                                 order_stride_device);

    cudaFree(data_);
    data_ = newData;
    {
      float* newGrad = nullptr;
      cudaMalloc((void**)&newGrad, dsize_ * sizeof(float));
      auto error = cudaGetLastError();
      if (cudaSuccess != error) {
        printf("%s\n", cudaGetErrorString(error));
        assert(false);
      }

      const int block_size = 256;
      const int grid_size = (dsize_ + 255) / 256;

      contiguous_kernel<<<grid_size, block_size>>>(grad_, newGrad, dsize_,
                                                   dims_.size(), stride_device,
                                                   order_stride_device);

      cudaFree(grad_);
      grad_ = newGrad;
    }
    cudaFree(stride_device);
    cudaFree(order_stride_device);
    error = cudaGetLastError();
    if (cudaSuccess != error) {
      printf("%s\n", cudaGetErrorString(error));
      assert(false);
    }
  }

  stride_ = order_stride;
}

__global__ void contiguous_kernel(float* src, float* dst, index_t dsize,
                                  index_t ndim, index_t* stride,
                                  index_t* order_stride) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dsize) {
    index_t temp = idx;
    index_t offset = 0;
    for (index_t i = 0; i < ndim; ++i) {
      index_t pos = temp / order_stride[i];
      temp = temp - pos * order_stride[i];
      offset += pos * stride[i];
    }
    dst[idx] = src[offset];
  }
}

__global__ void operator_index_kernel(float* data, float* target, index_t idx) {
  *target = data[idx];
}

void TensorImpl::recovery() {
  while( !shape_recovery_queue_.empty()) {
    contiguous();
    shape_recovery_queue_.back()();
    shape_recovery_queue_.pop_back();
  }
}

namespace tllm {
namespace detail {

std::ostream& operator<<(std::ostream& out, TensorImpl& t) {
    string old_device = t.device();
    t.to("cpu");
    index_t col = t.shape()[t.ndim() - 1];
    std::vector<index_t> order_stride(t.dims_.size(), 0);
    int stride = 1;
    for (int i = t.dims_.size() - 1; i >= 0; --i) {
        order_stride[i] = stride;
        stride *= t.dims_[i];
    }
    out << "data: " << std::endl;
    std::vector<index_t> pos(t.dims_.size(), 0);
    for(int i = 0; i < t.dsize(); ++i) {
        int idx = i;
        for (int p = 0; p < t.dims_.size(); ++p) {
            pos[p] = idx / order_stride[p];
            idx %= order_stride[p];
        }
        out << t[pos] << ", ";
        if ((i + 1) % col == 0) {
            out << std::endl;
        }
    }

        out << "grad: " << std::endl;
        for(int i = 0; i < t.dsize(); ++i) {
            out << t.grad_[i] << ", ";
            if ((i + 1) % col == 0) {
                out << std::endl;
        }
    }
    t.to(old_device);
    
    return out;
}

}
}

void TensorImpl::backward() {
    if (next_node_ptr_.get() == nullptr) return;
    if (ref_count_ == 0) {
        next_node_ptr_->backward(shared_from_this());
        next_node_ptr_.reset();
    }
}

void TensorImpl::zero_grad() {
    if (device_ == "cpu") {
        memset(grad_, 0, sizeof(float) * dsize_);
    }
    else {
        cudaMemset(grad_, 0, sizeof(float) * dsize_);
    }
}

__global__ void apply_grad_kernel(float* data, float* grad, index_t dsize) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dsize) {
        data[idx] -= grad[idx];
    }
}

void TensorImpl::apply_grad() {
    if (device_ == "cpu") {
        for(int i = 0; i < dsize_; ++i) {
            data_[i] -= grad_[i];
        }
    }
    else {
        const int block_size = 256;
        const int grid_size = (dsize_ + 255) / 256;
        apply_grad_kernel<<<grid_size, block_size>>>(data_, grad_, dsize_);
    }
    zero_grad();
}

UnaryGraphNode::UnaryGraphNode(TensorImplPtr t)
  : t_(t) { }

void UnaryGraphNode::backward(TensorImplPtr t) {
  t->recovery();
  grad_fn_(t_, t);
  t_->decreaseRef();
  t_->backward();
}

BinaryGraphNode::BinaryGraphNode(TensorImplPtr tl, TensorImplPtr tr)
  : tl_(tl), tr_(tr) { }

void BinaryGraphNode::backward(TensorImplPtr t) {
  t->recovery();
  grad_fn_l_(tl_, tr_, t);
  grad_fn_r_(tl_, tr_, t);
  tl_->decreaseRef();
  tr_->decreaseRef();
  tl_->backward();
  tr_->backward();
}
