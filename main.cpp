#include <iostream>
#include "Tensor.h"
#include "GGUFLoader.h"

int main()
{
    Tensor a;
    a.rows = 2;
    a.cols = 2;
    a.data = {1.0f, 2.0f, 3.0f, 4.0f};

    Tensor b;
    b.rows = 2;
    b.cols = 2;
    b.data = {5.0f, 6.0f, 7.0f, 8.0f};

    Tensor c = matmul(a, b);
    for (int i = 0; i < c.rows; i++)
    {
        for (int j = 0; j < c.cols; j++)
        {
            std::cout << c.data[i * c.cols + j] << " ";
        }
        std::cout << "\n";
    }

    // RMSNorm test
    Tensor x;
    x.rows = 1;
    x.cols = 2;
    x.data = {3.0f, 4.0f};

    Tensor weight;
    weight.rows = 1;
    weight.cols = 2;
    weight.data = {1.0f, 1.0f};

    Tensor y = rms_norm(x, weight, 1e-5f);

    std::cout << "rms_norm(x, weight) = ";
    for (int i = 0; i < y.data.size(); i++)
    {
        std::cout << y.data[i] << " ";
    }
    std::cout << "\n";

    // GGUF parser test
    load_gguf("tiny-random-LlamaForCausalLM.gguf");

    return 0;
}