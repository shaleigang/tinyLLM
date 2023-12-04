#include "tensor.h"
#include "exp.h"
#include "function.h"
#include "module.h"
#include "gpt.h"
#include "optimizer.h"
#include "dataloader.h"

using namespace tllm;


// Tensor get_pos_ids(index_t T) {
//         Tensor pos({T, 10}, "cpu");
//         for (int i = 0; i < T; ++i) {
//             index_t offset = 10 * i + i;
//             pos[offset] = 1;
//         }
//         return pos;
    // }

int main() {
    // Tensor t1({2,2,5,5}, "cpu", true);
    // for (int i = 0; i < 100; ++i) {
    //     t1[i] = i;
    // }
    // t1.to("cuda");

    // t1 = t1 + 1;
    // t1 = t1 + 2;

    TinyStoriesLoader loader("../data/tok4096/", 1, 5, 4096);
    nn::Embedding emb(4096, 4096);
    emb.cuda();
    AdamW adamw(emb.parameters(), {}, 0.001, 0.9, 0.999, "cuda");
    for (int i = 0; i < loader.get_iter_len(); ++i) {
        auto ret = loader.next();
        Tensor data = ret.first;
        data.to("cuda");
        auto embs = emb(data);
        std::cout << embs <<std::endl;
        std::cout << ret.second <<std::endl;
        break;
    }
    adamw.step();


    // std::cout << t1 <<std::endl;
    // std::cout << t2 <<std::endl;
    // std::cout << t3 <<std::endl;
    // linear.print();
    // std::cout << t4 <<std::endl;
    // std::cout << t5 <<std::endl;
    // std::cout << t6 <<std::endl;

    return 0;
}