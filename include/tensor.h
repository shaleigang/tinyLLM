#pragma once

#include "tensor_impl.h"
#include "util.h"

namespace tllm {

using detail::TensorImplPtr;
using detail::GraphNodePtr;

class Tensor {
public:
    Tensor();
    Tensor(TensorImplPtr t);
    Tensor(std::initializer_list<index_t> dims, string device="cpu", bool require_grad=true);
    Tensor(std::vector<index_t> dims, string device="cpu", bool require_grad=true);
    Tensor(const Tensor& t);
    Tensor(Tensor&& t);
    ~Tensor() = default;
    Tensor& operator=(const Tensor& t);

    index_t dsize () const;
    index_t ndim() const;
    std::vector<index_t> shape() const;
    std::vector<index_t> stride() const;
    bool is_contiguous() const;
    void contiguous();

    float& operator[](std::initializer_list<index_t> ids);
    float operator[](std::initializer_list<index_t> ids) const;
    float& operator[](std::vector<index_t> ids);
    float operator[](std::vector<index_t> ids) const;
    float& operator[](index_t offset);
    float operator[](index_t offset) const;
    index_t get_offset(std::initializer_list<index_t> ids) const;
    index_t get_offset(std::vector<index_t> ids) const;

    void transpose(index_t dim1, index_t dim2);
    void view(std::initializer_list<index_t> dims);
    void view(std::vector<index_t> dims);

    string device() const;
    void to(string dev);
    void cuda();
    void cpu();
    bool require_grad();
    void enable_grad();
    void disable_grad();
    TensorImplPtr get();
    float* data();
    float* grad();
    
    void setNode(GraphNodePtr n);
    GraphNodePtr getNode();
    void increaseRef();
    void decreaseRef();
    index_t getRef();
    void backward();
    void zero_grad();
    void apply_grad();

    friend std::ostream& operator<<(std::ostream& out, Tensor& t);
    Tensor operator+(Tensor& t);
    Tensor operator-(Tensor& t);
    Tensor operator-();
    Tensor operator*(Tensor& t);
    Tensor operator+(float val);
    Tensor operator*(float val);

private:
    TensorImplPtr tensor_;
};


}

