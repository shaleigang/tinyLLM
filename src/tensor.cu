#include "tensor.h"
#include "exp.h"

using namespace tllm;
using tllm::detail::TensorImpl;

Tensor::Tensor()
  : tensor_(std::make_shared<TensorImpl>()) {}

Tensor::Tensor(TensorImplPtr t)
  : tensor_(t) {}

Tensor::Tensor(std::initializer_list<index_t> dims, string device, bool require_grad)
  : tensor_(std::make_shared<TensorImpl>(std::move(dims), device, require_grad)) {}

Tensor::Tensor(std::vector<index_t> dims, string device, bool require_grad)
  : tensor_(std::make_shared<TensorImpl>(std::move(dims), device, require_grad)) {}

Tensor::Tensor(const Tensor& t)
  : tensor_(std::make_shared<TensorImpl>(*(t.tensor_))) {}

Tensor::Tensor(Tensor&& t)
  : tensor_(t.tensor_) {
    t.tensor_.reset();
}

Tensor& Tensor::operator=(const Tensor& t) {
    tensor_.reset();
    tensor_ = std::make_shared<TensorImpl>(*(t.tensor_));
}

index_t Tensor::dsize() const {
    return tensor_->dsize();
}

index_t Tensor::ndim() const {
    return tensor_->ndim();
}

std::vector<index_t> Tensor::shape() const {
    return tensor_->shape();
}

std::vector<index_t> Tensor::stride() const {
    return  tensor_->stride();
}

bool Tensor::is_contiguous() const {
    return tensor_->is_contiguous();
}

void Tensor::contiguous() {
    return tensor_->contiguous();
}

float& Tensor::operator[](std::initializer_list<index_t> ids) {
    return std::ref(tensor_->operator[](std::move(ids)));
}

float Tensor::operator[](std::initializer_list<index_t> ids) const {
    return tensor_->operator[](std::move(ids));
}

float& Tensor::operator[](std::vector<index_t> ids) {
    return std::ref(tensor_->operator[](std::move(ids)));
}

float Tensor::operator[](std::vector<index_t> ids) const {
    return tensor_->operator[](std::move(ids));
}

float& Tensor::operator[](index_t offset) {
    return std::ref(tensor_->operator[](offset));
}

float Tensor::operator[](index_t offset) const {
    return tensor_->operator[](offset);
}

index_t Tensor::get_offset(std::initializer_list<index_t> ids) const {
    tensor_->get_offset(std::move(ids));
}

index_t Tensor::get_offset(std::vector<index_t> ids) const {
    tensor_->get_offset(std::move(ids));
}

void Tensor::transpose(index_t dim1, index_t dim2) {
    tensor_->transpose(dim1, dim2);
}

void Tensor::view(std::initializer_list<index_t> dims) {
    tensor_->view(std::move(dims));
}

void Tensor::view(std::vector<index_t> dims) {
    tensor_->view(std::move(dims));
}

string Tensor::device() const {
    return tensor_->device();
}

void Tensor::to(string dev) {
    tensor_->to(std::move(dev));
}

bool Tensor::require_grad() {
    return tensor_->require_grad();
}

TensorImplPtr Tensor::get() {
    return tensor_;
}

float* Tensor::data() {
    return tensor_->data_;
}

float* Tensor::grad() {
    return tensor_->grad_;
}

void Tensor::setNode(GraphNodePtr n) {
    tensor_->setNode(n);
}

GraphNodePtr Tensor::getNode() {
    return tensor_->getNode();
}

void Tensor::increaseRef() {
    tensor_->increaseRef();
}

void Tensor::decreaseRef() {
    tensor_->decreaseRef();
}

index_t Tensor::getRef() {
    return tensor_->getRef();
}

void Tensor::backward() {
    tensor_->backward();
}

void Tensor::zero_grad() {
    tensor_->zero_grad();
}

void Tensor::apply_grad(float lr) {
    tensor_->apply_grad(lr);
}

namespace tllm {

std::ostream& operator<<(std::ostream& out, Tensor& t) {
    out << *(t.tensor_);
    return out;
}

}

Tensor Tensor::operator+(Tensor& t) {
    return detail::Add::get().forward(*this, t);
}

Tensor Tensor::operator-(Tensor& t) {
    return detail::Sub::get().forward(*this, t);
}

Tensor Tensor::operator-() {
    return detail::Minus::get().forward(*this);
}

Tensor Tensor::operator*(Tensor& t) {
    return detail::Mul::get().forward(*this, t);
}

Tensor Tensor::operator+(float val) {
    return std::make_shared<detail::ScalarAdd>(val)->forward(*this);
}

Tensor Tensor::operator*(float val) {
    return std::make_shared<detail::ScalarMul>(val)->forward(*this);
}




















