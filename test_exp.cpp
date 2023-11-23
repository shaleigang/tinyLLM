#include "tensor.h"
#include "exp.h"

using namespace tllm;

int main() {
    Tensor t1({2,3,4}, "cpu", true);
    for (int i = 0; i < 24; ++i) {
        t1[i] = i;
    }
    t1.to("cuda");

    Tensor t2(t1);
    Tensor t3(t1);

    Tensor t4 = t1 + t2 + t3;

    Tensor t5(t4);
    Tensor t6 = t5 - t4 + t1;

    t6.to("cpu");
    for (int i = 0; i < t6.dsize(); ++i) {
        t6.grad()[i] = 1;
    }

    t6.to("cuda");
    t6.backward();


    t1.apply_grad(0.1);
    t2.apply_grad(0.1);
    t3.apply_grad(0.1);
    t4.apply_grad(0.1);
    t5.apply_grad(0.1);
    t6.apply_grad(0.1);

    std::cout << t1 <<std::endl;
    std::cout << t2 <<std::endl;
    std::cout << t3 <<std::endl;
    std::cout << t4 <<std::endl;
    std::cout << t5 <<std::endl;
    std::cout << t6 <<std::endl;

    return 0;
}