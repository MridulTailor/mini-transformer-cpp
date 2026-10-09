cmake_minimum_required(VERSION 3.16)

project(mini_transformer_cpp LANGUAGES CXX)

include(CTest)

add_executable(mini_transformer
    main.cpp
    Tensor.cpp
    GGUFLoader.cpp
)

target_compile_features(mini_transformer PRIVATE cxx_std_17)

target_compile_options(mini_transformer PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
)

add_executable(benchmark_matmul
    benchmarks/benchmark_matmul.cpp
    Tensor.cpp
)
target_compile_features(benchmark_matmul PRIVATE cxx_std_17)
target_include_directories(benchmark_matmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(benchmark_matmul PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
)

if(BUILD_TESTING)
    add_executable(test_matmul
        tests/test_matmul.cpp
        Tensor.cpp
    )
    target_compile_features(test_matmul PRIVATE cxx_std_17)
    target_include_directories(test_matmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(test_matmul PRIVATE
        $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
    )
    add_test(NAME matmul_correctness COMMAND test_matmul)
endif()

configure_file(
    tiny-random-LlamaForCausalLM.gguf
    tiny-random-LlamaForCausalLM.gguf
    COPYONLY
)