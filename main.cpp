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

    // GGUF parser test — update this path to match your file
    load_gguf("tiny-random-LlamaForCausalLM.gguf");

    return 0;
}
