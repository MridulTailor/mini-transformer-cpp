cmake_minimum_required(VERSION 3.16)

project(mini_transformer_cpp LANGUAGES CXX)

add_executable(mini_transformer
    main.cpp
    Tensor.cpp
    GGUFLoader.cpp
)

target_compile_features(mini_transformer PRIVATE cxx_std_17)

target_compile_options(mini_transformer PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
)

configure_file(
    tiny-random-LlamaForCausalLM.gguf
    tiny-random-LlamaForCausalLM.gguf
    COPYONLY
)