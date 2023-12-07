#include "tensor.h"
#include "exp.h"
#include "function.h"
#include "module.h"
#include "gpt.h"
#include "optimizer.h"
#include "dataloader.h"

#include <unistd.h>

using namespace tllm;

int main() {
    nn::MLP mlp(8,32,8);
    std::cout << "ctor 2" << std::endl;
    for (int i = 0; i < 1; ++i) {
        Tensor t1({2,4,8}, "cpu", true);
        for (int i = 0; i < 64; ++i) {
            t1[i] = i;
        }
        std::cout << "ctor 1" << std::endl;

        Tensor t2 = mlp(t1);

        std::cout << "ctor 3" << std::endl;
        for (int i = 0; i < t2.dsize(); ++i) {
            t2.grad()[i] = 1;
        }
        t2.backward();
        std::cout << "dtor 2" << std::endl;
    }
    std::cout << "dtor 2" << std::endl;
    
    


    return 0;
}