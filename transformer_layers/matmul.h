//
// Created by alireza on 9/15/23.
//

#ifndef FVLLMONTITRANSFORMER_MATMUL_H
#define FVLLMONTITRANSFORMER_MATMUL_H

#include "util.h"
#include "../transformer.h"

class MatMul {
public:
    static void multiply(std::size_t seq_len, quant_bit_width *input,
                         quant_bit_width *weight, quant_bit_width *output,
                         std::size_t input_size, std::size_t output_size);

    static void scale(quant_bit_width* input, int shift_scale, std::size_t mat_size);
};


#endif //FVLLMONTITRANSFORMER_MATMUL_H
