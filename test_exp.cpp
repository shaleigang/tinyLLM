#include "tensor.h"
#include "exp.h"
#include "function.h"
#include "module.h"
#include "gpt.h"
#include "optimizer.h"
#include "dataloader.h"
#include "tokenizer.h"

#include <unistd.h>
#include <cassert>

using namespace tllm;

int main() {

    Tokenizer tokenizer("/home/slg/work/tinyLLM/data/tok4096.bin", 4096);
    Tensor tokens = tokenizer.encode("Once up on a time, there was a man lived in the forest.", 1, 0);

    std::cout << tokens.dsize() << std::endl;
    int prev_token = -1;
    for (int i = 0; i < tokens.dsize(); ++i) {
        string piece = tokenizer.decode(prev_token, tokens[i]);
        prev_token = tokens[i];
        tokenizer.saft_print(piece);
        std::cout << " ";
    }
    std::cout << std::endl;

    return 0;
}