#include "tensor.h"
#include "optimizer.h"
#include "gpt.h"
#include "dataloader.h"
#include "function.h"

#include <cstring>
#include <cmath>
#include <cassert>

using namespace tllm;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: ./chat [model_path]" << std::endl;
        exit(0);
    }

    GPT gpt(6, 64, 4, 4096, 256, 0.2, false);
    gpt.load(argv[1]);
    gpt.cuda();

    Tokenizer tokenizer("/home/slg/work/tinyLLM/data/tok4096.bin", 4096);

    while (true) {
        std::cout << "User>> ";
        string text;
        getline(std::cin, text);
        if (text == "q") {
            break;
        }
        std::cout << "tllm>> ";
        gpt.generate(text, tokenizer);
        std::cout << std::endl;
    }
    


    return 0;
}