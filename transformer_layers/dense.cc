#include "dense.h"
#include <exception>
//#include <mkl.h>
#include <memory.h>
#include <iostream>

Dense::Dense(std::size_t input_size, std::size_t output_size,
             quant_bit_width *weightDense, uint32_t *flagDense) {
    input_size_ = input_size;
    output_size_ = output_size;
    std::cout << "Input Size : " << input_size_ << std::endl;
    std::cout << "Output Size : " << output_size_ << std::endl;
    weight = weightDense;
    flag = flagDense;
    bias = nullptr;
}

Dense::Dense(std::size_t input_size, std::size_t output_size,
             quant_bit_width *weightDense, quant_bit_width *biasDense) {
    input_size_ = input_size;
    output_size_ = output_size;
    weight = weightDense;
    bias = biasDense;
}

Dense::~Dense() {
//    delete weight;
//    delete[] bias;
}

void Dense::multiplyweight(std::size_t seq_len, quant_bit_width *input, quant_bit_width *output) {
    for (int length = 0; length < seq_len; length++) {
        for (int out_idx = 0; out_idx < output_size_; out_idx++) {
            auto *weight_ptr = weight + out_idx;
            auto *output_ptr = output + (length * output_size_) + out_idx;
            auto *input_ptr = input + (length * input_size_);
            int32_t sum = 0;
            for (int i = 0; i < input_size_; i++) {
//                if (length == 0){
//                    std::cout << *weight_ptr << " x " << *input_ptr << " = " << MUL(*weight_ptr, * input_ptr) << std::endl;
//                }
                sum += MUL_HQ(*weight_ptr, * input_ptr);
                input_ptr++;
                weight_ptr += output_size_;
            }
            *(output_ptr) = (quant_bit_width) (sum >> NUM_FRACTION_BITS);
        }
    }
}

void Dense::addbias(std::size_t seq_len, quant_bit_width *output) {

    for (std::size_t idx = 0; idx < seq_len; idx++) {
        for (std::size_t feature_idx = 0; feature_idx < output_size_; feature_idx++) {
            output[idx * output_size_ + feature_idx] += bias[feature_idx];
        }
    }
}

void Dense::compute(std::size_t seq_len, quant_bit_width *input, quant_bit_width *output) {
    // input shape [batch_size, input_size_]
    // output shape [batch_size, output_size_]

    multiplyweight(seq_len, input, output);
    // add bias vector here
    if (bias != nullptr) {
        addbias(seq_len, output);
    }
}
