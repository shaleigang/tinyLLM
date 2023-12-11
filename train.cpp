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
    if (argc != 4) {
        std::cout << "Usage: ./train [model_path] [start_epoch] [start_iter]" << std::endl;
        exit(0);
    }

    GPT gpt(6, 64, 4, 4096, 256, 0.2, false);
    gpt.load(argv[1]);
    int start_iter = atoi(argv[3]);
    int start_epoch = atoi(argv[2]);
    std::cout << "model loaded." << std::endl;
    gpt.cuda();


    std::cout << "GPT model " << gpt.get_num_params() / 1e6 << "M" <<std::endl;

    

    ParamsDict decay_params;
    ParamsDict nodecay_params;
    for (auto iter : gpt.parameters()) {
        if (iter.second.get().ndim() >= 2) {
            decay_params.insert(iter);
        }
        else {
            nodecay_params.insert(iter);
        }
    }

    AdamW adamw(decay_params, nodecay_params, 0.001, 0.9, 0.95, "cuda");

    for (int e = start_epoch; e < 3; ++e) {
        TinyStoriesLoader loader("../data/tok4096/", 128, 256);
        float loss_g = 0;
        for (int i = 0; i < loader.get_iter_len(); ++i) {
            if (start_iter > 0) {
                --start_iter;
                continue;
            }

            auto ret = loader.next();
            Tensor& data = ret.first;
            Tensor& label = ret.second;
            data.to("cuda");
            label.to("cuda");
            label.view({label.dsize()});

            Tensor loss = gpt.forward(data, label);
            loss.backward();

            if (i % 1 == 0) {
                loss.cpu();
                std::cout << "[" << i << "/" << loader.get_iter_len() << "] " << "loss: " << loss[0] <<std::endl;
                loss_g = loss[0];
            }

            if (i != 0 && i % 500 == 0) {
                gpt.save("/home/slg/work/tinyLLM/ckpt/epoch" + std::to_string(e) + "_" + std::to_string(i) + "_" + std::to_string(loss[0]) + "/");
                std::cout << "model saved" << std::endl;
            }

            adamw.step();
        }
        std::cout << "saving model" << std::endl;
        gpt.save("/home/slg/work/tinyLLM/ckpt/0_epoch" + std::to_string(e) + "_" + std::to_string(loss_g) + "/");
        std::cout << "model saved" << std::endl;
    }
    return 0;
}