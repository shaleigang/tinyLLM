#pragma once

#include <initializer_list>
#include <vector>
#include <memory>
#include <iostream>
#include <functional>

#include "util.h"

namespace tllm {
namespace detail {

class GraphNode;
using GraphNodePtr=std::shared_ptr<detail::GraphNode>;

class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
public:
    TensorImpl() : dsize_(0), data_(nullptr), grad_(nullptr) {}
    TensorImpl(std::initializer_list<index_t> dims, string device="cpu", bool require_grad=true);
    TensorImpl(std::vector<index_t> dims, string device="cpu", bool require_grad=true);
    TensorImpl(const TensorImpl& t);
    TensorImpl(TensorImpl&& t);
    ~TensorImpl();
    TensorImpl& operator=(const TensorImpl& t);

    index_t dsize () const { return dsize_; };
    index_t ndim() const { return dims_.size(); }
    std::vector<index_t> shape() const { return dims_; }
    std::vector<index_t> stride() const { return stride_; }
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

    string device() const { return device_; }
    void to(string dev);
    bool require_grad() { return require_grad_; }
    void enable_grad() { require_grad_ = true; }
    void disable_grad() { require_grad_ = false; }
    
    void setNode(GraphNodePtr n) { next_node_ptr_ = n; }
    GraphNodePtr getNode() { return next_node_ptr_; }
    void increaseRef() { ++ref_count_; }
    void decreaseRef() { --ref_count_; }
    index_t getRef() { return ref_count_; }
    void backward();
    void zero_grad();
    void apply_grad(float lr);

    friend std::ostream& operator<<(std::ostream& out, TensorImpl& t);

private:
    string device_;
    bool require_grad_;

    std::vector<index_t> dims_;
    std::vector<index_t> stride_;
    index_t dsize_;

    GraphNodePtr next_node_ptr_;
    index_t ref_count_;

public:
    float* data_;
    float* grad_;
};

typedef std::shared_ptr<TensorImpl> TensorImplPtr;
typedef std::function<void(TensorImplPtr, TensorImplPtr)> UnaryGradFn;
typedef std::function<void(TensorImplPtr, TensorImplPtr, TensorImplPtr)> BinaryGradFn;

class GraphNode {
public:
    virtual void backward(TensorImplPtr t) = 0;

    virtual void setGradFn(UnaryGradFn f) = 0;
    virtual void setGradFnL(BinaryGradFn f) = 0;
    virtual void setGradFnR(BinaryGradFn f) = 0;
};

class UnaryGraphNode : public GraphNode {
public:
    UnaryGraphNode(TensorImplPtr t);
    void setGradFn(UnaryGradFn f) override { grad_fn_ = f; }
    void setGradFnL(BinaryGradFn f) override { return; };
    void setGradFnR(BinaryGradFn f) override { return; };

    void backward(TensorImplPtr t) override;

private:
    TensorImplPtr t_;
    UnaryGradFn grad_fn_;
};

class BinaryGraphNode : public GraphNode {
public:
    BinaryGraphNode(TensorImplPtr tl, TensorImplPtr tr);
    void setGradFn(UnaryGradFn f) override { return; }
    void setGradFnL(BinaryGradFn f) override { grad_fn_l_ = f; }
    void setGradFnR(BinaryGradFn f) override { grad_fn_r_ = f; }

    void backward(TensorImplPtr t) override;

private:
    TensorImplPtr tl_;
    TensorImplPtr tr_;
    BinaryGradFn grad_fn_l_;
    BinaryGradFn grad_fn_r_;
};
}

}

