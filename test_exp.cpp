#include "tensor.h"
#include "exp.h"
#include "function.h"
#include "module.h"
#include "gpt.h"

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

    // F::causal_mask_fill(t1);
    
    GPT model(1, 1024, 8, 4000, 1024, 0.2, false);
    auto parms = model.parameters();
    for (auto iter : parms) {
        std::cout << iter.first << std::endl;
        // std::cout << iter.second.get() << std::endl;
    }
    std::cout << model.get_num_params() / 1e6  << "M" << std::endl;

    // Tensor t1 = get_pos_ids(5);


    // std::cout << t1 <<std::endl;
    // std::cout << t2 <<std::endl;
    // std::cout << t3 <<std::endl;
    // linear.print();
    // std::cout << t4 <<std::endl;
    // std::cout << t5 <<std::endl;
    // std::cout << t6 <<std::endl;

    return 0;
}