#include "module.h"
#include "function.h"

#include <cassert>

using namespace tllm::nn;
using tllm::Tensor;

void Module::print() {
    for (auto iter: parameters()) {
        std::cout << iter.first << std::endl;
        std::cout << iter.second << std::endl;
    }
}

void Module::to(string device) {
    for (auto iter: parameters()) {
        iter.second.get().to(device);
    }
    device_ = device;
}

void Module::cuda() {
    to("cuda");
}

void Module::cpu() {
    to("cpu");
}

// void Module::apply_grad() {
//     for (auto iter: parameters()) {
//         Tensor& t = iter.second.get();
//         if (t.require_grad()) {
//             t.apply_grad();
//         }
//     }
// }

int Module::get_num_params() {
    int count = 0;
    for (auto iter : parameters()) {
        count += iter.second.get().dsize();
    }
    return count;
}

Linear::Linear(index_t in_features, index_t out_features, bool bias)
  : weight_(Tensor({in_features, out_features})),
    require_bias_(bias) {
    if (require_bias_) {
        bias_ = Tensor({out_features});
    }
    for (int i = 0; i < (in_features * out_features); ++i) {
        weight_[i] = i;
    }
    if (require_bias_) {
        for (int i = 0; i < (out_features); ++i) {
            bias_[i] = i;
        } 
    } 
}

Tensor Linear::forward(Tensor& input) {
    Tensor ret = F::mat_mul(input, weight_);
    if (require_bias_) {
        Tensor ret2 = ret + bias_;
        return ret2;
    }
    else {
        return ret;
    }
}

ParamsDict Linear::parameters() {
    if (require_bias_) {
        return {
            {"weight", weight_},
            {"bias", bias_}
        };
    }
    else {
        return {
            {"weight", weight_}
        };
    }
}

LayerNorm::LayerNorm(index_t n_dim, bool bias)
  : weight_(Tensor({n_dim})),
    require_bias_(bias) {
    if (require_bias_) {
        bias_ = Tensor({n_dim});
    }
    for (int i = 0; i < (n_dim); ++i) {
        weight_[i] = i;
    }
    if (require_bias_) {
        for (int i = 0; i < (n_dim); ++i) {
            bias_[i] = i;
        } 
    } 
}

Tensor LayerNorm::forward(Tensor& input) {
    Tensor x1 = F::layer_norm(input);
    Tensor x2 = x1 * weight_;
    if (require_bias_) {
        Tensor x3 = x2 + bias_;
        return x3;
    }
    else {
        return x2;
    }
}

ParamsDict LayerNorm::parameters() {
    if (require_bias_) {
        return {
            {"weight", weight_},
            {"bias", bias_}
        };
    }
    else {
        return {
            {"weight", weight_}
        };
    }
}

MLP::MLP(index_t in_dim, index_t hidden_dim, index_t out_dim, float dropout, bool bias)
  : c_fc(in_dim, hidden_dim, bias),
    gelu(),
    c_proj(hidden_dim, out_dim, bias),
    dropout(dropout) {}

Tensor MLP::forward(Tensor& input) {
    Tensor x1 = c_fc(input);
    Tensor x2 = gelu(x1);
    Tensor x3 = c_proj(x2);
    Tensor x4 = dropout(x3);
    return x4;
}

ParamsDict MLP::parameters() {
    return {
        {"c_fc", c_fc.parameters()},
        {"c_proj", c_proj.parameters()}
    };
}

CausalSelfAttention::CausalSelfAttention(index_t n_emdb, index_t n_head, float dropout)
  : c_attn_q(n_emdb, n_emdb),
    c_attn_k(n_emdb, n_emdb),
    c_attn_v(n_emdb, n_emdb),
    c_proj(n_emdb, n_emdb),
    attn_dropout(dropout),
    resid_dropout(dropout),
    n_head_(n_head),
    n_emdb_(n_emdb) {
    assert (n_emdb % n_head == 0);
}

Tensor CausalSelfAttention::forward(Tensor& input) {
    auto input_shape = input.shape();
    assert(input_shape.size() == 3);
    index_t B = input_shape[0];
    index_t T = input_shape[1];
    index_t C = input_shape[2];

    Tensor q = c_attn_q(input); q.view({B, T, n_head_, C / n_head_}); q.transpose(1, 2); // B, nh, T, hs
    Tensor k = c_attn_k(input); k.view({B, T, n_head_, C / n_head_}); k.transpose(1, 2); // B, nh, T, hs
    Tensor v = c_attn_v(input); v.view({B, T, n_head_, C / n_head_}); v.transpose(1, 2); // B, nh, T, hs
    
    k.transpose(2, 3);
    Tensor att = F::mat_mul(q, k);
    Tensor att2 = att * (1.0 / sqrt(C / n_head_));
    F::causal_mask_fill(att2);
    Tensor att_softmax = F::softmax(att2);
    Tensor att_dropout = attn_dropout(att_softmax);
    Tensor y = F::mat_mul(att_dropout, v); // (B, nh, T, hs)

    y.transpose(1, 2);
    y.contiguous();
    y.view({B, T, C});

    Tensor y_proj = c_proj(y);
    Tensor y_dropout = resid_dropout(y_proj);
    
    return y_dropout;
}

ParamsDict CausalSelfAttention::parameters() {
    return {
        {"c_attn_q", c_attn_q.parameters()},
        {"c_attn_k", c_attn_k.parameters()},
        {"c_attn_v", c_attn_v.parameters()},
        {"c_proj", c_proj.parameters()}
    };
}

TransformerBlock::TransformerBlock(index_t n_embd, index_t n_head, float dropout, bool bias)
  : ln_1(n_embd, bias),
    attn(n_embd, n_head, dropout),
    ln_2(n_embd, bias),
    mlp(n_embd, 4 * n_embd, n_embd, dropout, bias) {}

Tensor TransformerBlock::forward(Tensor& input) {
    Tensor x_ln_1 = ln_1(input);
    Tensor x_attn = attn(x_ln_1);
    Tensor x1 = input + x_attn;
    Tensor x_ln_2 = ln_2(x1);
    Tensor x_mlp = mlp(x_ln_2);
    Tensor x2 = x1 + x_mlp;
    return x2;
}

ParamsDict TransformerBlock::parameters() {
    return {
        {"ln_1", ln_1.parameters()},
        {"attn", attn.parameters()},
        {"ln_2", ln_2.parameters()},
        {"mlp", mlp.parameters()}
    };
}

Embedding::Embedding(index_t vocab_size, index_t n_embd)
  : embs_real({n_embd, vocab_size}),
    vocab_size_(vocab_size),
    n_embd_(n_embd),
    embs(embs_real) {}

Embedding::Embedding(index_t vocab_size, index_t n_embd, Tensor& weight)
  : vocab_size_(vocab_size),
    n_embd_(n_embd),
    embs(weight) {}



Tensor Embedding::forward(Tensor& idx) {
    embs.transpose(0, 1);
    Tensor ret = F::mat_mul(idx, embs);
    embs.transpose(1, 0);
    return ret;
}

ParamsDict Embedding::parameters() {
    return {
        {"embs", embs}
    };
}