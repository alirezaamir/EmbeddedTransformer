cmake_minimum_required(VERSION 3.20)
project(FvllMontiTransformer)

set(CMAKE_CXX_STANDARD 17)

set(GCC_COVERAGE_COMPILE_FLAGS "-DDEVELOP -DLOAD_SKIP -DSA -DSA_SIZE=16")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

file(GLOB transformer_SRC
        "transformer_layers/*.h"
        "transformer_layers/*.cc"
        "SYLT-FFT/*.h"
        )

add_executable(FvllMontiTransformer epilepsyTransformer.cpp ${transformer_SRC})
