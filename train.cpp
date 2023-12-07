#include "tensor.h"
#include "optimizer.h"
#include "gpt.h"
#include "dataloader.h"
#include "function.h"

#include <cstring>
#include <cmath>

using namespace tllm;

int main() {
    GPT gpt(6, 64, 8, 4096, 256, 0.2, false);
    // gpt.load("/home/slg/work/tinyLLM/ckpt/");
    // std::cout << "model loaded." << std::endl;
    // GPT gpt(6, 8, 2, 10, 5, 0.2, false);
    gpt.cuda();
    // for (auto iter : gpt.parameters()) {
    //         std::cout << iter.first << std::endl;
            // std::cout << iter.second.get() << std::endl;
            // std::cout << iter.first << std::endl;
        // }

    std::cout << "GPT model " << gpt.get_num_params() / 1e6 << "M" <<std::endl;

    TinyStoriesLoader loader("../data/tok4096/", 128, 256, 4096);

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

    AdamW adamw(decay_params, nodecay_params, 5e-4, 0.9, 0.95, "cuda");

    for (int i = 0; i < loader.get_iter_len(); ++i) {
    // for (int i = 0; i < 5000; ++i) {
        auto ret = loader.next();
        Tensor& data = ret.first;
        Tensor& label = ret.second;
        data.to("cuda");
        label.to("cuda");
        // Tensor data = Tensor({2, 4, 10});
        // memset(data.data(), 0, sizeof(float) * 80);
        // Tensor label = Tensor({2, 4});
        // memset(label.data(), 0, sizeof(float) * 8);
        // for (int p = 0; p < 8; ++p) {
        //     data[p * 10 + p] = 1;
        //     label[p] = p;
        // }
        // std::cout << "data: " << std::endl;
        // std::cout << data << std::endl;
        // std::cout << "label: " << std::endl;
        // std::cout << label << std::endl;
        
        label.view({label.dsize()});
        Tensor loss = gpt.forward(data, label);
        loss.backward();
        loss.cpu();
        std::cout << "[" << i << "/" << loader.get_iter_len() << "] " << "loss: " << loss[0] <<std::endl;
    
        adamw.step();

        if (i != 0 && i % 1000 == 0) {
            std::cout << "saving model" << std::endl;
            gpt.save("/home/slg/work/tinyLLM/ckpt/" + std::to_string(i) + "_" + std::to_string(loss[0]) + "/");
        }
    }
    return 0;
}