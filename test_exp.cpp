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
    Tensor idx({2, 5});
    for (int i = 0; i < idx.dsize(); ++i) {
        idx[i] = i;
    }
    // idx.cuda();

    Tensor liner({10, 20});
    for (int i = 0; i < liner.dsize(); ++i) {
        liner[i] = i;
    }
    // liner.cuda();

    nn::Embedding emb(20, 10, liner);
    // emb.cuda();

    Tensor t = emb(idx);
    Tensor t2 = F::mat_mul(t, liner);

    t2.cpu();
    for (int i = 0; i < t2.dsize(); ++i) {
        t2.grad()[i] = 1;
    }
    // t2.cuda();
    t2.backward();
    
    std::cout << idx << std::endl;
    std::cout << liner << std::endl;
    std::cout << emb.parameters()["embs"] << std::endl;
    std::cout << t << std::endl;
    std::cout << t2 << std::endl;


    return 0;
}