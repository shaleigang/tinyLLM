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

    // t1.to("cuda");

    // Tensor t2({3,3}, "cpu", true);
    // for (int i = 0; i < 9; ++i) {
    //     t2[i] = 1;
    // }
    Tensor t2 = t1 + 6;

    Tensor t3 = t2 * 2;

    Tensor t4 = t3 + 2;
    // t1.to("cuda");
    // t2.to("cuda");

    // Tensor t3 = t1 * t2;

    nn::Linear linear(3, 6, false);
    // linear.to("cuda");
    Tensor t5 = linear.forward(t4);

    // Tensor t2(t1);
    // Tensor t3 = MatMul(t1, t2);
    // Tensor t3(t1);

    // Tensor t4 = t1 + t2 + t3;

    // Tensor t5(t4);
    // Tensor t6 = t4 * t1;

    // t3.to("cpu");
    for (int i = 0; i < t5.dsize(); ++i) {
        t5.grad()[i] = 1;
    }

    // t3.to("cuda");
    t5.backward();


    // t1.apply_grad(0.1);
    // t2.apply_grad(0.1);
    // t3.apply_grad(0.1);
    // t4.apply_grad(0.1);
    // t5.apply_grad(0.1);
    // t6.apply_grad(0.1);

    std::cout << t1 <<std::endl;
    std::cout << t2 <<std::endl;
    std::cout << t3 <<std::endl;
    linear.print();
    std::cout << t4 <<std::endl;
    std::cout << t5 <<std::endl;
    // std::cout << t6 <<std::endl;

    return 0;
}