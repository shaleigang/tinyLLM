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
    Tensor idx1({2, 8});
    for (int i = 0; i < idx1.dsize(); ++i) {
        idx1[i] = i % 10;
    }
    idx1.cuda();
    Tensor t1 = F::softmax(idx1);
    t1.cpu();
    for (int i = 0; i < t1.dsize(); ++i) {
        t1.grad()[i] = i;
    }
    t1.cuda();
    t1.backward();

    Tensor idx2({2, 8});
    for (int i = 0; i < idx2.dsize(); ++i) {
        idx2[i] = i % 10;
    }
    Tensor t2 = F::softmax(idx2);
    for (int i = 0; i < t2.dsize(); ++i) {
        t2.grad()[i] = i;
    }
    t2.backward();
    
    std::cout << idx1 << std::endl;
    std::cout << t1 << std::endl;
    std::cout << idx2 << std::endl;
    std::cout << t2 << std::endl;


    return 0;
}