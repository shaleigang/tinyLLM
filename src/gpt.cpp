#include "gpt.h"

#include <cassert>
#include <thread>
#include <cstring>

using namespace tllm;

GPT::GPT(index_t n_layer, index_t n_embd, index_t n_head, index_t vocab_size, index_t block_size, float dropout, bool bias)
  : wpe(block_size, n_embd),
    // drop(dropout),
    ln_f(n_embd, bias),
    lm_head(n_embd, vocab_size, false),
    wte(vocab_size, n_embd, lm_head.parameters()["weight"]),
    block_size_(block_size),
    vocab_size_(vocab_size) {
    for (int i = 0; i < n_layer; ++i) {
        blocks.push_back(std::move(std::make_unique<nn::TransformerBlock>(n_embd, n_head, dropout, bias)));
    }
    init_weight();
}

Tensor GPT::forward(Tensor& idx) {
    auto shape = idx.shape();
    index_t batch = shape[0];
    index_t T = shape[1];
    assert(T < block_size_);
    idx.disable_grad();
    Tensor pos_ids = get_pos_ids(T);
    pos_ids.to(device());
    Tensor pos_emb = wpe(pos_ids);
    Tensor tok_emb = wte(idx);
    // for (auto i : tok_emb.shape()) {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;
    Tensor x = tok_emb + pos_emb;
    
    // Tensor x = drop(tok_emb);
    for (int i = 0; i < blocks.size(); ++i) {
        x = (*blocks[i])(x);
    }
    x = ln_f(x);
    Tensor logits = lm_head(x);
    return logits;
}

Tensor GPT::forward(Tensor& idx, Tensor& targets) {
    auto shape = idx.shape();
    index_t batch = shape[0];
    index_t T = shape[1];
    assert(T < block_size_);
    idx.disable_grad();

    Tensor pos_ids = get_pos_ids(T);
    pos_ids.to(device());
    Tensor pos_emb = wpe(pos_ids);
    Tensor tok_emb = wte(idx);
    Tensor x = tok_emb + pos_emb;
    // Tensor x = drop(tok_emb);
    std::vector<Tensor> vec;
    vec.push_back(std::move(x));
    for (int i = 0; i < blocks.size(); ++i) {
        Tensor temp = (*blocks[i])(vec[vec.size() - 1]);
        vec.push_back(std::move(temp));
    }
    Tensor x_l = ln_f(vec[vec.size() - 1]);
    Tensor logits = lm_head(x_l); // (B, T, vocab_size)
    auto shape_l = logits.shape();
    logits.view({logits.dsize() / vocab_size_, vocab_size_}); 
    Tensor loss = F::cross_entropy(logits, targets);

    logits.view(shape_l);
    return loss;
}

ParamsDict GPT::parameters() {
    ParamsDict blocks_parm{};
    for (int i = 0; i < blocks.size(); ++i) {
        blocks_parm.insert_parmdict("block" + std::to_string(i), blocks[i]->parameters());
    }
    return {
        {"wpe", wpe.parameters()},
        {"ln_f", ln_f.parameters()},
        {"lm_head", lm_head.parameters()},
        {"blocks", blocks_parm}
    };
}

Tensor GPT::get_pos_ids(index_t T) {
    Tensor pos({T}, "cpu", false);
    for (int i = 0; i < T; ++i) {
        pos[i] = i;
    }
    return pos;
}

void random_gen(float* t, index_t dsize) {
    std::thread::id tid = std::this_thread::get_id();
    std::default_random_engine e(time(0) + *(unsigned int*)&tid);
    std::normal_distribution<float> n(0, 0.02);
    for (int i = 0; i < dsize; ++i) {
        t[i] = n(e);
    }
}

void GPT::init_weight() {
    std::vector<std::thread> threads;
    for (auto iter : parameters()) {
        Tensor& weight = iter.second.get();
        threads.push_back(std::thread(random_gen, weight.data(), weight.dsize()));
    }
    for (std::thread& t : threads) {
        t.join();
    }
}