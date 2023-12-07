#pragma once

#include "module.h"

using namespace tllm;

class GPT : public nn::Module {
public:
    GPT(index_t n_layer,
        index_t n_embd,
        index_t n_head,
        index_t vocab_size,
        index_t block_size,
        float dropout,
        bool bias);

    ~GPT() = default;

    virtual Tensor forward(Tensor& idx) override;
    Tensor forward(Tensor& idx, Tensor& targets); // target: (vocab_size) = B * T
    virtual ParamsDict parameters(void) override;

    void init_weight();
    Tensor get_pos_ids(index_t T);

private:
    nn::Embedding wpe;
    // nn::Dropout drop;
    nn::LayerNorm ln_f;
    std::vector<std::unique_ptr<nn::TransformerBlock>> blocks;
    nn::Linear lm_head;
    nn::Embedding wte;

private:
    index_t block_size_;
    index_t vocab_size_;
};