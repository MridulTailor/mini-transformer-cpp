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
- `main.cpp` - executable entry point and runtime test
- `CMakeLists.txt` - build configuration