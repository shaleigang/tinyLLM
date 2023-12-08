#pragma once

#include "function.h"
#include "params.h"
#include "tensor.h"

using tllm::detail::ParamsDict;

namespace tllm {
namespace nn {

class Module {
 public:
  Module() : device_("cpu") {}
  Module(const Module& other) = delete;
  virtual ~Module() = default;

  void print();
  void to(string device);
  void cuda();
  void cpu();
  int get_num_params();
  string device() { return device_; }
  void save(string path);
  void load(string path);

  virtual Tensor forward(Tensor& input) = 0;
  virtual ParamsDict parameters(void) = 0;

  virtual Tensor operator()(Tensor& input) { return forward(input); }

private:
    string device_;
};

class Linear : public Module {
 public:
  Linear(index_t in_features, index_t out_features, bool bias = false);
  ~Linear() = default;

  virtual Tensor forward(Tensor& input) override;
  virtual ParamsDict parameters(void) override;

 protected:
  Tensor weight_;
  Tensor bias_;
  bool require_bias_;
};

class LayerNorm : public Module {
 public:
  LayerNorm(index_t n_dim, bool bias = true);
  ~LayerNorm() = default;

  virtual Tensor forward(Tensor& input) override;
  virtual ParamsDict parameters(void) override;

 private:
  Tensor weight_;
  Tensor bias_;
  bool require_bias_;
};

using detail::Dropout;
using detail::GELU;

class MLP : public Module {
 public:
  MLP(index_t in_dim, index_t hidden_dim, index_t out_dim, float dropout = 0.2,
      bool bias = false);
  ~MLP() = default;

  virtual Tensor forward(Tensor& input) override;
  virtual ParamsDict parameters(void) override;

 private:
  Linear c_fc;
  GELU gelu;
  Linear c_proj;
  // Dropout dropout;
};

class CausalSelfAttention : public Module {
 public:
  CausalSelfAttention(index_t n_emdb, index_t n_head, float dropout);
  ~CausalSelfAttention() = default;

  virtual Tensor forward(Tensor& input) override;
  virtual ParamsDict parameters(void) override;

 private:
  Linear c_attn_q;
  Linear c_attn_k;
  Linear c_attn_v;

  Linear c_proj;

  // Dropout attn_dropout;
  // Dropout resid_dropout;

 private:
  index_t n_head_;
  index_t n_emdb_;
};

class TransformerBlock : public Module {
 public:
  TransformerBlock(index_t n_embd, index_t n_head, float dropout, bool bias);
  ~TransformerBlock() = default;

  virtual Tensor forward(Tensor& input) override;
  virtual ParamsDict parameters(void) override;

 private:
  LayerNorm ln_1;
  CausalSelfAttention attn;
  LayerNorm ln_2;
  MLP mlp;
};

class Embedding : public Module {
public:
    Embedding(index_t vocab_size, index_t n_embd);
    Embedding(index_t vocab_size, index_t n_embd, Tensor& weight);
    ~Embedding() = default;

    virtual Tensor forward(Tensor& idx) override;
    virtual ParamsDict parameters(void) override;
private:
    Tensor& embs;
    Tensor embs_real;
    // Tensor embs;

    index_t vocab_size_;
    index_t n_embd_;
};

}  // namespace nn
}  // namespace tllm