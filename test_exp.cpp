#include "tensor.h"
#include "exp.h"
#include "function.h"
#include "module.h"
#include "gpt.h"
#include "optimizer.h"
#include "dataloader.h"

#include <unistd.h>
#include <cassert>

using namespace tllm;

int main() {

    GPT gpt(6, 64, 4, 4096, 256, 0.2, false);
    gpt.save("/home/slg/work/tinyLLM/ckpt/test/");
    gpt.cuda();

    GPT gpt2(6, 64, 4, 4096, 256, 0.2, false);
    gpt2.load("/home/slg/work/tinyLLM/ckpt/test/");
    auto param2 = gpt2.parameters();

    gpt.cpu();
    for (auto iter : gpt.parameters()) {
        string name = iter.first;
        Tensor& t = iter.second.get();
        Tensor& t2 = param2[name];
        for (int i = 0; i < t.dsize(); ++i) {
            assert(t[i] == t2[i]);
        }
    }

    return 0;
}