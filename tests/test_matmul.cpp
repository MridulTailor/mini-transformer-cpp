#include "Tensor.h"

#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

namespace
{
bool tensors_match(const Tensor &expected, const Tensor &actual)
{
    if (expected.rows != actual.rows || expected.cols != actual.cols)
    {
        return false;
    }

    for (std::size_t i = 0; i < expected.data.size(); i++)
    {
        const float difference = std::fabs(expected.data[i] - actual.data[i]);
        const float tolerance = 1e-4f + 1e-5f * std::fabs(expected.data[i]);
        if (difference > tolerance)
        {
            std::cerr << "Mismatch at index " << i << ": expected "
                      << expected.data[i] << ", actual " << actual.data[i]
                      << ", difference " << difference << '\n';
            return false;
        }
    }
    return true;
}

Tensor random_tensor(int rows, int cols, std::mt19937 &generator)
{
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    Tensor tensor{std::vector<float>(rows * cols), rows, cols};
    for (float &value : tensor.data)
    {
        value = distribution(generator);
    }
    return tensor;
}
} // namespace

int main()
{
    std::mt19937 generator(12345);
    const std::vector<std::tuple<int, int, int>> shapes = {
        {1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {7, 4, 9}, {11, 13, 6}};

    for (const auto &[m, k, n] : shapes)
    {
        const Tensor a = random_tensor(m, k, generator);
        const Tensor b = random_tensor(k, n, generator);
        const Tensor naive = matmul(a, b);
        const Tensor optimized = matmul_optimized(a, b);
        if (!tensors_match(naive, optimized))
        {
            std::cerr << "MatMul mismatch for " << m << "x" << k << "x" << n << '\n';
            return 1;
        }
    }

    std::cout << "All MatMul correctness tests passed.\n";
    return 0;
}
