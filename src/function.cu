#include "function.h"

using namespace tllm::detail;

namespace tllm {

detail::MatMulExp MatMul;

}

__global__ void matmul_kernel(float* x, float* x1, float* x2, index_t z_, index_t y_, index_t x_, index_t d_);
__global__ void matmul_backward_kernel(float* x, float* x1, float* x2, index_t z_, index_t y_, index_t x_, index_t d_);


Tensor MatMulExp::operator()(Tensor& x1, Tensor& x2) {
    return forward(x1, x2);
}

Tensor MatMulExp::generate_ret_tensor(Tensor& x1, Tensor& x2) {
    auto shape1 = x1.shape();
    auto shape2 = x2.shape();
    shape1[shape1.size() - 1] = shape2[shape2.size() - 1];
    if (shape1[shape1.size() - 1] != shape2[shape2.size() - 2]) {
        std::cout << "MatMul shape not match. " << std::endl;
        exit(0);
    }
    x2.transpose(x2.ndim() - 2, x2.ndim() - 1);
    return Tensor(shape1, x1.device());
}

void MatMulExp::forward_process(Tensor& x1, Tensor& x2, Tensor& x) {
    auto shape1 = x1.shape();
    auto shape2 = x2.shape();
    auto shape = x.shape();
    index_t dim = 1;
    for (int i = 0; i < shape.size() - 2; ++i) {
        dim *= shape[i];
    }
    x1.view({dim, shape1[shape1.size() - 2], shape1[shape1.size() - 1]});
    x2.view({dim, shape2[shape2.size() - 2], shape2[shape2.size() - 1]});
    x.view({dim, shape[shape.size() - 2], shape[shape.size() - 1]});

    if (x1.device() == "cpu") {
        for (index_t m = 0; m < dim; ++m) {
            for (index_t i = 0; i < shape[shape.size() - 2]; ++i) {
                for (index_t j = 0; j < shape[shape.size() - 1]; ++j) {
                    x[{m, i, j}] = 0;
                    for (index_t p = 0; p < shape1[shape1.size() - 1]; ++p) {
                        x[{m, i, j}] += (x1[{m, i, p}] * x2[{m, j, p}]);
                    }
                }
            }
        }
    }
    else {
        const dim3 grid_size((shape[shape.size() - 2] + 15)/16,
                            (shape[shape.size() - 1] + 15)/16,
                            dim);
        const dim3 block_size(16, 16);
        matmul_kernel<<<grid_size, block_size>>>(x.data(), x1.data(), x2.data(), dim, shape[shape.size() - 2], shape[shape.size() - 1], shape1[shape1.size() - 1]);
    }

    x1.view(shape1);
    x2.view(shape2);
    x.view(shape);
    x2.transpose(x2.ndim() - 2, x2.ndim() - 1);

    return;
}

__global__ void matmul_kernel(float* x, float* x1, float* x2, index_t z_, index_t y_, index_t x_, index_t d_) {
    const int id_z = blockIdx.z;
    const int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int x_offset = id_z * y_ * x_ + id_y * x_ + id_x;
    const int x1_offset = id_z * y_ * d_ + id_y * d_;
    const int x2_offset = id_z * x_ * d_ + id_x * d_;
    if (id_z < z_ && id_y < y_ && id_x < x_) {
        x[x_offset] = 0;
        for (int i = 0; i < d_; ++i) {
            x[x_offset] += x1[x1_offset + i] * x2[x2_offset + i];
        }
    }
}

void MatMulExp::lhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
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
    x1->view({dim, shape1[shape1.size() - 2], shape1[shape1.size() - 1]});
    x2->view({dim, shape2[shape2.size() - 2], shape2[shape2.size() - 1]});
    x->view({dim, shape[shape.size() - 2], shape[shape.size() - 1]});

    if (x1->device() == "cpu") {
        for (index_t m = 0; m < dim; ++m) {
            for (index_t i = 0; i < shape[shape.size() - 2]; ++i) {
                for (index_t j = 0; j < shape[shape.size() - 1]; ++j) {
                    for (index_t p = 0; p < shape1[shape1.size() - 1]; ++p) {
                        x1->grad_[x1->get_offset({m, i, j})] += (x->grad_[x->get_offset({m, i, p})] * x2->data_[x2->get_offset({m, j, p})]);
                    }
                }
            }
        }
    }
    else {
        const dim3 grid_size((shape[shape.size() - 2] + 15)/16,
                            (shape[shape.size() - 1] + 15)/16,
                            dim);
        const dim3 block_size(16, 16);
        matmul_kernel<<<grid_size, block_size>>>(x1->grad_, x->grad_, x2->data_, dim, shape[shape.size() - 2], shape[shape.size() - 1], shape1[shape1.size() - 1]);
    }

    x1->view(shape1);
    x2->view(shape2);
    x->view(shape);

    x1->decreaseRef();

    return;
}

__global__ void matmul_backward_kernel(float* x, float* x1, float* x2, index_t z_, index_t y_, index_t x_, index_t d_) {
    const int id_z = blockIdx.z;
    const int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int x_offset = id_z * y_ * x_ + id_y * x_ + id_x;
    const int x1_offset = id_z * y_ * d_ + id_y * d_;
    const int x2_offset = id_z * x_ * d_ + id_x * d_;
    if (id_z < z_ && id_y < y_ && id_x < x_) {
        for (int i = 0; i < d_; ++i) {
            x[x_offset] += x1[x1_offset + i] * x2[x2_offset + i];
        }
    }
}

void MatMulExp::rhs_grad_fn(TensorImplPtr x1, TensorImplPtr x2, TensorImplPtr x) {
    x1->transpose(x1->ndim() - 2, x1->ndim() - 1);
    x->transpose(x->ndim() - 2, x->ndim() - 1);
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
    x1->view({dim, shape1[shape1.size() - 2], shape1[shape1.size() - 1]});
    x2->view({dim, shape2[shape2.size() - 2], shape2[shape2.size() - 1]});
    x->view({dim, shape[shape.size() - 2], shape[shape.size() - 1]});

    if (x1->device() == "cpu") {
        for (index_t m = 0; m < dim; ++m) {
            for (index_t i = 0; i < shape[shape.size() - 2]; ++i) {
                for (index_t j = 0; j < shape[shape.size() - 1]; ++j) {
                    for (index_t p = 0; p < shape1[shape1.size() - 1]; ++p) {
                        x2->grad_[x2->get_offset({m, i, j})] += (x1->data_[x1->get_offset({m, i, p})] * x->grad_[x->get_offset({m, j, p})]);
                    }
                }
            }
        }
    }
    else {
        const dim3 grid_size((shape[shape.size() - 2] + 15)/16,
                            (shape[shape.size() - 1] + 15)/16,
                            dim);
        const dim3 block_size(16, 16);
        matmul_backward_kernel<<<grid_size, block_size>>>(x2->grad_, x1->data_, x->grad_, dim, shape[shape.size() - 2], shape[shape.size() - 1], shape1[shape1.size() - 1]);
    }

    x1->view(shape1);
    x2->view(shape2);
    x->view(shape);

    x1->transpose(x1->ndim() - 2, x1->ndim() - 1);
    x->transpose(x->ndim() - 2, x->ndim() - 1);

    x2->decreaseRef();

    return;
}