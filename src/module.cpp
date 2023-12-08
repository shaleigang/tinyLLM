#include "module.h"
#include "function.h"

#include <cassert>
#include <fstream>
#include <sys/stat.h>

using namespace tllm::nn;
using tllm::Tensor;

std::ostream& tllm::operator<<(std::ostream&output, std::vector<index_t> shape) {
    output << "(";
    for (int i = 0; i < shape.size() - 1; ++i) {
        output << shape[i] << ", ";
    }
    output << shape[shape.size() - 1] << ")";
    return output;
}

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

int Module::get_num_params() {
    int count = 0;
    for (auto iter : parameters()) {
        count += iter.second.get().dsize();
    }
    return count;
}

void Module::save(string path) {
    string old_device = device();
    cpu();

    struct stat st;

    if (stat(path.c_str(), &st) == -1) {
        if (mkdir(path.c_str(), 0775) == -1) {
            perror("mkdir error");
            return;
        }
    }

    string bin_path = path + "model.bin";
    string offset_path = path + "model.index";

    std::ofstream data_out(bin_path, std::ios::binary | std::ios::trunc);
    std::ofstream offset_out(offset_path);

    for (auto iter : parameters()) {
        Tensor& t = iter.second.get();
        offset_out << iter.first << " " << sizeof(float) * t.dsize() << std::endl;
        data_out.write((char*)t.data(), sizeof(float) * t.dsize());
    }
    to(old_device);
}

void Module::load(string path) {
    string old_device = device();
    cpu();

    string bin_path = path + "model.bin";
    string offset_path = path + "model.index";

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

    auto params = parameters();
    string name;
    index_t len;

    while (offset_in >> name >> len) {
        data_in.read((char*)params[name].data(), len);
    }

    to(old_device);
}

Linear::Linear(index_t in_features, index_t out_features, bool bias)
  : weight_(Tensor({in_features, out_features})),
    require_bias_(bias) {
    if (require_bias_) {
        bias_ = Tensor({out_features});
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
    // for (int i = 0; i < (n_dim); ++i) {
    //     weight_[i] = i;
    // }
    // if (require_bias_) {
    //     for (int i = 0; i < (n_dim); ++i) {
    //         bias_[i] = i;
    //     } 
    // } 
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
    c_proj(hidden_dim, out_dim, bias) {}
    // dropout(dropout) {}

Tensor MLP::forward(Tensor& input) {
    Tensor x1 = c_fc(input);
    Tensor x2 = gelu(x1);
    Tensor x3 = c_proj(x2);
    // Tensor x4 = dropout(x3);
    return x3;
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
    // attn_dropout(dropout),
    // resid_dropout(dropout),
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
    // Tensor att_dropout = attn_dropout(att_softmax);
    // Tensor y = F::mat_mul(att_dropout, v); // (B, nh, T, hs)
    Tensor y = F::mat_mul(att_softmax, v); // (B, nh, T, hs)

    // q.transpose(1, 2); q.contiguous(); q.view({B, T, C});
    // k.transpose(2, 3); k.transpose(1, 2); k.contiguous(); k.view({B, T, C});
    // v.transpose(1, 2); v.contiguous(); v.view({B, T, C});

    y.transpose(1, 2);
    y.contiguous();
    y.view({B, T, C});

    Tensor y_proj = c_proj(y);
    // Tensor y_dropout = resid_dropout(y_proj);
    
    // y.view({B, n_head_, T, C / n_head_}); y.transpose(1, 2); y.contiguous();


    // return y_dropout;
    return y_proj;
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
  : embs_real({vocab_size, n_embd}),
    vocab_size_(vocab_size),
    n_embd_(n_embd),
    embs(embs_real) {}

Embedding::Embedding(index_t vocab_size, index_t n_embd, Tensor& weight)
  : vocab_size_(vocab_size),
    n_embd_(n_embd),
    embs(weight) {
        assert(embs.shape()[0] == n_embd);
        assert(embs.shape()[1] == vocab_size);
    }

// Embedding::Embedding(index_t vocab_size, index_t n_embd)
//   : vocab_size_(vocab_size),
//     n_embd_(n_embd),
//     embs({vocab_size, n_embd}) {}



Tensor Embedding::forward(Tensor& idx) {
    if (embs.shape()[0] != vocab_size_) {
        embs.transpose(0, 1);
        Tensor ret = F::emb(idx, embs);
        embs.transpose(1, 0);
        return ret;
    }
    Tensor ret = F::emb(idx, embs);
    
    return ret;
}

ParamsDict Embedding::parameters() {
    return {
        {"embs", embs}
    };
}