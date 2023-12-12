#pragma once
#include <unordered_map>

#include "tensor.h"

namespace tllm {
// namespace detail {
// struct TokenIndex
// {
//     string str;
//     int id;
// };
// }

// using detail::TokenIndex;

class Tokenizer {
public:
    Tokenizer(string path, int vocab_size);
    ~Tokenizer() = default;

    Tensor encode(string text, int8_t bos, int8_t eos);
    string decode(int prev_token, int token);
    void saft_print(string piece);

private:
    int str_lookup(string str);

private:
    std::vector<string> vocab_;
    std::vector<float> vocab_scores_;
    std::unordered_map<string, int> vocab_map_;
    unsigned char byte_pieces_[512];

    int vocab_size_;
    index_t max_token_length_;
};


}