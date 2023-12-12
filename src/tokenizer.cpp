#include "tokenizer.h"

#include <fstream>
#include <cassert>

using namespace tllm;

Tokenizer::Tokenizer(string path, int vocab_size)
  : vocab_size_(vocab_size) {

    vocab_.reserve(vocab_size_);
    vocab_scores_.reserve(vocab_size_);

    for (int i = 0; i < 256; ++i) {
        byte_pieces_[i * 2] = (unsigned char)i;
        byte_pieces_[i * 2 + 1] = '\0';
    }

    std::ifstream data_in(path, std::ios::binary);
    if (!data_in.good()) {
        std::cout << "Can not find tokenizer file in \"" << path << "\"" << std::endl;
        exit(0);
    }

    data_in.read((char*)&max_token_length_, sizeof(int));

    int len;
    float score;
    char* buffer = new char[4096];
    for (int i = 0; i < vocab_size_; ++i) {
      data_in.read((char*)&score, sizeof(float));
      vocab_scores_.push_back(score);

      data_in.read((char*)&len, sizeof(int));
      assert(len < 4096);
      data_in.read(buffer, len);
      buffer[len] = '\0';
      vocab_.emplace_back(buffer);
    }

    data_in.close();
    delete [] buffer;

    for (int i = 0; i < vocab_.size(); ++i) {
      vocab_map_[vocab_[i]] = i;
    }
}

string Tokenizer::decode(int prev_token, int token) {
  string piece = vocab_[token];
  if (prev_token == 1 && piece[0] == ' ') { 
    piece.erase(0, 1);
  }

  unsigned char byte_val;
  if (sscanf(piece.c_str(), "<0x%02hhX>", &byte_val) == 1) {
      piece = string(reinterpret_cast<char*>(byte_pieces_ + byte_val * 2));
  }

  return piece;
}

void Tokenizer::saft_print(string piece) {
  if (piece.empty()) { return; }
  if (piece[0] == '\0') { return; }
  if (piece[1] == '\0') {
      unsigned char byte_val = piece[0];
      if (!(isprint(byte_val) || isspace(byte_val))) {
          return; // bad byte, don't print it
      }
  }
  std::cout << piece;
}

int Tokenizer::str_lookup(string str) {
    auto iter = vocab_map_.find(str);
    if (iter != vocab_map_.end()) {
      return iter->second;
    }
    else {
      return -1;
    }
}

Tensor Tokenizer::encode(string text, int8_t bos, int8_t eos) {
  std::vector<int> tokens;
  char* str_buffer = new char(max_token_length_ * 2 + 1 + 2);
  size_t str_len = 0;

  if (bos) tokens.push_back(1);

  if (text[0] != '\0') {
    int dummy_prefix = str_lookup(" ");
    tokens.push_back(dummy_prefix);
  }

  const char *s = text.c_str();
  for (int c = 0; c < text.size(); c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((s[c] & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = s[c]; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((s[c + 1] & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens.push_back(id);
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens.push_back((unsigned char)str_buffer[i] + 3);
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < tokens.size(); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            int id = str_lookup(vocab_[tokens[i]] + vocab_[tokens[i+1]]);
            if (id != -1 && vocab_scores_[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = vocab_scores_[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < tokens.size(); i++) {
            tokens[i] = tokens[i+1];
        }
        tokens.pop_back();
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens.push_back(2);

    free(str_buffer);
    index_t len = tokens.size();
    Tensor ret({1, len});
    for (int i = 0; i < len; ++i) {
      ret[i] = tokens[i];
    }
    return ret;
}