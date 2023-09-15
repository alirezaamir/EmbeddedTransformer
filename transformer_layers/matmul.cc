//
// Created by alireza on 9/15/23.
//

#include "matmul.h"

void MatMul::multiply(std::size_t seq_len, quant_bit_width *input,
                      quant_bit_width *weight, quant_bit_width *output,
                      std::size_t input_size, std::size_t output_size) {
    for (int length = 0; length < seq_len; length++) {
        for (int out_idx = 0; out_idx < output_size; out_idx++) {
            auto *weight_ptr = weight + out_idx;
            auto *output_ptr = output + (length * output_size) + out_idx;
            auto *input_ptr = input + (length * input_size);
            int32_t sum = 0;
            for (int i = 0; i < input_size; i++) {
                //                if (length == 0){
                //                    std::cout << *weight_ptr << " x " << *input_ptr << " = " << MUL(*weight_ptr, * input_ptr) << std::endl;
                //                }
                sum += MUL_HQ(*weight_ptr, * input_ptr);
                input_ptr++;
                weight_ptr += output_size;
            }
            *(output_ptr) = (quant_bit_width) (sum >> NUM_FRACTION_BITS);
        }
    }
}

void MatMul::scale(quant_bit_width* input, int shift_scale, std::size_t mat_size){
    for (int i = 0; i < mat_size; i++){
        *input = (*input) >> shift_scale;
        input++;
    }

}
