#pragma once

// #include <cstddef>
// #include <vector>
// #include <unordered_map>
// #include <string>
#include "util.h"
#include "../accelerator/smm_gem.h"
#include "../transformer.h"

class Dense {
public:
    Dense(std::size_t input_dim, std::size_t output_dim, quant_bit_width *weight, uint32_t *flag);

    ~Dense();

    void compute(std::size_t seq_len, quant_bit_width *input, quant_bit_width *output);

private:
    void multiplyweight(std::size_t seq_len, quant_bit_width *input, quant_bit_width *output);

    void addbias(std::size_t seq_len, quant_bit_width *output);

    std::size_t input_size_;
    std::size_t output_size_;
    quant_bit_width *weight; // shape [input_size_, output_size_]
    uint32_t *flag; // shape [input_size_/KERNEL_DIM, output_size_/KERNEL_DIM/32]
    quant_bit_width *bias;   // shape [output_size_]

};