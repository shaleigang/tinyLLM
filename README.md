# tinyLLM
Train a tiny language model using pure C++ from scratch without any third-party libraries. CUDA is supported.

For learning purposes only.
# Introduction
This project implements a simple deep learning framework and trains a tiny language model based on it.
The core parts include:
* Tensor implemention
* Automatic differentiation
* Dynamic construction of computational graphs
* Gradient backpropagation

<!-- This project implements some commonly used functions and modules in language model, and supports both CPU and GPU.
|Modules|Functions|
|:-:|:-:|
|Linear|+|
|MLP|-|
|GELU|*|
|Dropout|mat_mul|
|LayerNorm|nlloss|
|CausalSelfAttention|log|
|TransformerBlock|softmax|
|Embedding|cross_entropy| -->

# GPT-style LM
This project refers to [nanoGPT](https://github.com/karpathy/nanoGPT) and [Llama2.c](https://github.com/karpathy/llama2.c) to build a GPT-style language model, trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. All modules involved in the model, such as **Transformer Block**, **AdamW Optimizer**, **Loss Function**, **Tokenizer**, and **DataLoader**, are all built from scratch using C++.

You can use [train.cpp](./train.cpp) to train the model. The inference code is also very simple:
```C++
#include "gpt.h"
#include "tokenizer.h"

using namespace tllm;

int main() {

    GPT gpt(6, 64, 4, 4096, 256, 0.2, false);   # Declare a GPT model
    gpt.load("Path to saved model.");           # Load model from ckeckpoint
    gpt.cuda();                                 # Move model to GPU

    Tokenizer tokenizer("Path to tokenizer file.", 4096);   # Load tokenizer
    
    gpt.generate("Your Prompt", tokenizer);     # Get model response

    return 0;
}

```
It is worth noting that this project did not build the code for training a tokenizer. It uses the code of Llama2.c to train a tokenizer with a vocab_size of 4096. Moreover, the data used to train the model is also pretokenizered with the tokenizer.

Please refer to the **custom tokenizers** part of [Llama2.c](https://github.com/karpathy/llama2.c) to obtain the corresponding training data filea and convert the tokenizer to binary format. The tokenizer of this project uses the same structure as Llama2.c, and the converted tokenizer bin file can be used directly.

Unfortunately, there are still some problems with the training of the language model in this project. When the loss drops from 8 to about 3, it will stop declining, and the performance of the model is not very good. I am still working on this...