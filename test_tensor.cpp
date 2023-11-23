#include "tensor.h"

using namespace tllm;

int main() {
    // std::cout << "Hello!" << std::endl;
    Tensor t({2,3,4}, "cpu", true);
    for (int i = 0; i < 24; ++i) {
        t[i] = i;
    }
    std::cout << t;
    std::cout << "contiguous: " << t.is_contiguous() << std::endl;
    std::cout << "dsize: " << t.dsize() << std::endl;
    std::cout << "ndim: " << t.ndim() << std::endl;
    auto shape = t.shape();
    std::cout << "shape: ";
    for(int i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;
    auto stride = t.stride();
    std::cout << "stride: ";
    for(int i = 0; i < stride.size(); ++i) {
        std::cout << stride[i] << " ";
    }
    std::cout << std::endl;


    t.view({2, 12});
    std::cout << "t.view({2, 12});" << std::endl;
    std::cout << t;
    std::cout << "contiguous: " << t.is_contiguous() << std::endl;
    std::cout << "t[1][10] = " << t[{1, 10}] << std::endl;
    shape = t.shape();
    std::cout << "shape: ";
    for(int i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;
    stride = t.stride();
    std::cout << "stride: ";
    for(int i = 0; i < stride.size(); ++i) {
        std::cout << stride[i] << " ";
    }
    std::cout << std::endl;

    t.view({2, 3, 4});
    std::cout << "t.view({2, 3, 4});" << std::endl;
    std::cout << t;
    std::cout << "contiguous: " << t.is_contiguous() << std::endl;
    shape = t.shape();
    std::cout << "shape: ";
    for(int i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;
    stride = t.stride();
    std::cout << "stride: ";
    for(int i = 0; i < stride.size(); ++i) {
        std::cout << stride[i] << " ";
    }
    std::cout << std::endl;

    
    t.transpose(1, 2);
    std::cout << "t.transpose(1, 2);" << std::endl;
    std::cout << t;
    std::cout << "contiguous: " << t.is_contiguous() << std::endl;
    shape = t.shape();
    std::cout << "shape: ";
    for(int i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;
    stride = t.stride();
    std::cout << "stride: ";
    for(int i = 0; i < stride.size(); ++i) {
        std::cout << stride[i] << " ";
    }
    std::cout << std::endl;

    t.to("cuda");
    std::cout << t.device() << std::endl << t;

    std::cout << "contiguous: " << t.is_contiguous() << std::endl;
    t.contiguous();
    std::cout << "contiguous: " << t.is_contiguous() << std::endl;
    std::cout << t;

    t.to("cpu");
    Tensor t2 = t;
    t2[{0,0,1}] = 56;
    std::cout << t2 << std::endl;
    std::cout << "= t2. {0,0,1}: " << t;

    Tensor t3(t);
    t3[{0,0,1}] = 100;
    std::cout << t3 << std::endl;
    std::cout << "() t3. {0,0,1}: " << t;


    return 0;
}