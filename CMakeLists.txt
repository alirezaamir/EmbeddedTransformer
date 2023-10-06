cmake_minimum_required(VERSION 3.20)
project(FvllMontiTransformer)

set(CMAKE_C_STANDARD 11)

set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

file(GLOB transformer_SRC
        "transformer_C/*.h"
        "SYLT-FFT/*.h"
        )

add_executable(FvllMontiTransformer epilepsyTransformer.c ${transformer_SRC})

# Link the math library
target_link_libraries(FvllMontiTransformer m)