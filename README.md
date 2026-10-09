# Mini Transformer C++

A small C++17 transformer implementation with basic tensor operations, attention, and GGUF model loading.

## Requirements

- CMake 3.16 or newer
- A C++17 compiler
- The GGUF model file `tiny-random-LlamaForCausalLM.gguf` in the project root

## Build

```bash
cmake -S . -B build
cmake --build build
```

For the optimized kernel benchmark and correctness test, use a Release build:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target test_matmul benchmark_matmul
ctest --test-dir build-release --output-on-failure
./build-release/benchmark_matmul
```

## Run

```bash
./build/mini_transformer
```

The model file is copied to `build/` during configuration so the executable can load it from its working directory.

## VS Code

Open the project folder in VS Code, select a C++ compiler, and use the CMake Tools extension to configure and build the project. Launch `mini_transformer` with the C/C++ debugger, using `build/` as the working directory.

## Project Layout

- `Tensor.*` - tensor operations and transformer functions
- `GGUFLoader.*` - GGUF metadata and tensor loading
- `tests/test_matmul.cpp` - naive versus optimized MatMul correctness test
- `benchmarks/benchmark_matmul.cpp` - repeated MatMul latency benchmark
- `main.cpp` - executable entry point and runtime test
- `CMakeLists.txt` - build configuration

## MatMul optimization

`matmul` is retained as the naive reference implementation. `matmul_optimized`
uses `i-k-j` loop ordering and initializes the output once, making the
innermost access to matrix `B` contiguous.

The correctness test uses seeded randomized FP32 matrices with rectangular
shapes and compares both implementations with an absolute tolerance of
`1e-4` plus a small relative tolerance for larger values.

The following results were measured with 10 repetitions per implementation on
an Apple M1 Pro (arm64), using Apple clang 17.0.0, C++17, and CMake Release
flags (`-O3 -DNDEBUG`):

| Matrix size (M x K x N) | Naive latency (ms) | Optimized latency (ms) | Speedup |
| --- | ---: | ---: | ---: |
| 64 x 64 x 64 | 0.154 | 0.024 | 6.54x |
| 256 x 256 x 256 | 15.958 | 1.297 | 12.31x |
| 512 x 512 x 512 | 146.633 | 10.271 | 14.28x |

### Resume bullets

- Implemented transformer components from scratch in C++17, including
  multi-head attention, scaled dot-product attention, normalization, and
  feedforward layers.
- Developed a lightweight GGUF loader for parsing model metadata and tensor
  weights.
- Optimized FP32 matrix multiplication using cache-friendly loop reordering,
  achieving 6.54x-14.28x speedup over the naive implementation across
  benchmarked workloads.
- Validated optimized kernels against reference implementations and benchmarked
  inference primitives across multiple matrix dimensions.