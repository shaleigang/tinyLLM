#include "tensor.h"
#include "exp.h"
#include "function.h"
#include "module.h"

using namespace tllm;

int main() {
    Tensor t1({2,3,3}, "cpu", true);
    for (int i = 0; i < 18; ++i) {
        t1[i] = i;
    }

    Tensor t2({2,3}, "cpu", true);
    for (int i = 0; i < 6; ++i) {
        t2[i] = i % 3;
    }
    t1.to("cuda");
    t2.to("cuda");

    // nn::LayerNorm norm(3);
    // norm.to("cuda");
    // nn::Dropout dropout(0.5);
    t1.view({6,3});
    t2.view({6});
    Tensor t3 = F::cross_entropy(t1, t2);

    t3.to("cpu");
    t3.grad()[0] = 1;
    t3.to("cuda");
    t3.backward();

    // t3.to("cpu");
    // for (int i = 0; i < t3.dsize(); ++i) {
    //     t3.grad()[i] = 1;
    //     if ((i - 1) % 3 == 0) {
    //         t2.grad()[i] = 10;
    //     }
    // }

    // t2.to("cuda");
    // t2.backward();


    // t1.apply_grad(0.1);
    // t2.apply_grad(0.1);
    // t3.apply_grad(0.1);
    // t4.apply_grad(0.1);
    // t5.apply_grad(0.1);
    // t6.apply_grad(0.1);

    std::cout << t1 <<std::endl;
    std::cout << t2 <<std::endl;
    std::cout << t3 <<std::endl;
    // linear.print();
    // std::cout << t4 <<std::endl;
    // std::cout << t5 <<std::endl;
    // std::cout << t6 <<std::endl;

    return 0;
}