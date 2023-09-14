//
// Created by alireza on 3/3/22.
//

#ifndef FVLLMONTITRANSFORMER_SMM_GEM_H
#define FVLLMONTITRANSFORMER_SMM_GEM_H

#include <cstddef>
#include <cstdint>
#include "../transformer.h"

void conventionalCompute(std::size_t seq_len, const uint32_t * input, uint32_t * output, uint32_t *weight,
                         std::size_t input_size_, std::size_t output_size_);
int conventionalCompute(std::size_t seq_len, const  quant_bit_width * input, quant_bit_width * output,
                         quant_bit_width *weight,
                         std::size_t input_size_, std::size_t output_size_);
void tiledCompute(std::size_t seq_len, const uint32_t * input, uint32_t * output, uint32_t *weight,
                         std::size_t input_size_, std::size_t output_size_);

void tiledL1Compute(std::size_t seq_len, const uint32_t * input, uint32_t * output, uint32_t *weight,
                  std::size_t input_size_, std::size_t output_size_);

void simdCompute(std::size_t seq_len, const uint32_t * input, uint32_t * output, uint32_t *weight,
                    std::size_t input_size_, std::size_t output_size_);


void smmCompute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                uint32_t *flag, std::size_t input_size_, std::size_t output_size_, bool sparse);

void smmComputeRearranged(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                          uint32_t *flag, std::size_t input_size_, std::size_t output_size_, bool sparse);

void simdCompute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                          uint32_t *flag, std::size_t input_size_, std::size_t output_size_, bool sparse);

void simdComputeRearranged(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                 uint32_t *flag, std::size_t input_size_, std::size_t output_size_, bool sparse);

void smmComputeEigen(std::size_t seq_len, const int8_t *input, int8_t *output, int8_t *weights,
                     std::size_t input_size_, std::size_t output_size_);


void print_arr(uint32_t* array, int n, int p);

#endif //FVLLMONTITRANSFORMER_SMM_GEM_H
