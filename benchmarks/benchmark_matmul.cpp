#include "Tensor.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace
{
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

template <typename MatMul>
double benchmark(MatMul matmul_function, const Tensor &a, const Tensor &b, int repetitions)
{
    volatile float checksum = 0.0f;
    const auto start = std::chrono::steady_clock::now();
    for (int repetition = 0; repetition < repetitions; repetition++)
    {
        const Tensor result = matmul_function(a, b);
        checksum += result.data[0];
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    if (checksum == 0.123456f)
    {
        std::cerr << "Unexpected checksum.\n";
    }
    return std::chrono::duration<double, std::milli>(elapsed).count() / repetitions;
}
} // namespace

int main()
{
    constexpr int repetitions = 10;
    std::mt19937 generator(12345);
    const std::vector<int> sizes = {64, 256, 512};

    std::cout << "repetitions: " << repetitions << '\n';
    std::cout << std::left << std::setw(12) << "M x K x N"
              << std::right << std::setw(18) << "naive (ms)"
              << std::setw(22) << "optimized (ms)"
              << std::setw(14) << "speedup" << '\n';

    for (const int size : sizes)
    {
        const Tensor a = random_tensor(size, size, generator);
        const Tensor b = random_tensor(size, size, generator);
        const double naive_ms = benchmark(matmul, a, b, repetitions);
        const double optimized_ms = benchmark(matmul_optimized, a, b, repetitions);
        std::cout << std::left << std::setw(12)
                  << (std::to_string(size) + " x " + std::to_string(size) + " x " +
                      std::to_string(size))
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(18) << naive_ms
                  << std::setw(22) << optimized_ms
                  << std::setw(14) << (naive_ms / optimized_ms) << "x\n";
    }
}
