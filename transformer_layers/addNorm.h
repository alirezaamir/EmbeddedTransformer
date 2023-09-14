//
// Created by alireza on 3/2/22.
//
#include "util.h"
#include "iostream"
#include "../transformer.h"
#ifndef FVLLMONTITRANSFORMER_ADDNORM_H
#define FVLLMONTITRANSFORMER_ADDNORM_H

class AddNormalize{
public:
    AddNormalize(std::size_t, std::size_t, std::size_t, std::size_t);
    AddNormalize(std::size_t, std::size_t, quant_bit_width*weight, quant_bit_width *bias);
    void compute(uint32_t *input, uint32_t *output);
    void compute(quant_bit_width *input, quant_bit_width *output);
    void normalize(quant_bit_width *input);
    void computeRearranged(uint32_t *input, uint32_t *output);
private:
    std::size_t seq_len_;
    std::size_t input_dim_;
    std::size_t kernel_dim_;
    std::size_t max_col_;

    quant_bit_width *weight_; // shape [input_size_]
    quant_bit_width *bias_;   // shape [input_size_]

};

#endif //FVLLMONTITRANSFORMER_ADDNORM_H
