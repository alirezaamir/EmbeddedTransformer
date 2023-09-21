//
// Created by alireza on 3/2/22.
//

#include "addNorm.h"
#include <cmath>

AddNormalize::AddNormalize(std::size_t seq_len, std::size_t input_dim,
                           std::size_t kernelDim, std::size_t maxCol) {
    input_dim_ = input_dim;
    seq_len_ = seq_len;
    kernel_dim_ = kernelDim;
    max_col_ = maxCol;
}

void AddNormalize::compute(uint32_t *input, uint32_t *output) {
    for (int i =0; i< seq_len_; i++){
        auto* input_ptr = (int8_t*) (input + i * (input_dim_ >> 2));
        auto* output_ptr = (int8_t*) (output + i * (input_dim_ >> 2));
        int32_t sum = 0;
        for (int j=0; j< input_dim_; j++){
            *output_ptr = (int8_t) (*output_ptr + *input_ptr);
            sum += *output_ptr;
            output_ptr ++;
            input_ptr ++;
        }

        output_ptr = (int8_t*) (output + i * (input_dim_ >> 2));
        auto mean = (int32_t) (sum / input_dim_);
        int32_t variance = 0;
        for (int j=0; j< input_dim_; j++){
            variance+= (*output_ptr++ - mean) ^ 2; // Assuming that the values are fixed-point with 2 digit of fraction.
        }
        variance = variance / (int) input_dim_;
        double sd = sqrt((double) variance);
        auto sd_inv = (int32_t) ((1<<2)/(sd + 1)); // prevent zero divide! // Assuming that the values are fixed-point with 2 digit of fraction.

        output_ptr = (int8_t*) (output + i * (input_dim_ >> 2));
        for (int j=0; j< input_dim_; j++){
            *output_ptr = (int8_t) ((*output_ptr - mean) * (sd_inv) >> 2);
            output_ptr ++;
        }

    }
}

void AddNormalize::compute(quant_bit_width *input, quant_bit_width *output) {
    std::size_t width= seq_len_;
    std::size_t height = seq_len_;
    for (int i =0; i< seq_len_; i++){
        auto* input_ptr = input + i * (input_dim_);
        auto* output_ptr = output + i * (input_dim_);
        quant_bit_width sum = 0;
        for (int j=0; j< input_dim_; j++){
            *output_ptr = (*output_ptr + *input_ptr);
            sum += *output_ptr;
            output_ptr ++;
            input_ptr ++;
        }

        output_ptr = output + i * (input_dim_);
        auto mean = (quant_bit_width)( (float) sum / (float) input_dim_);
        // TODO: integer to float function
        quant_bit_width variance = 0;
        for (int j=0; j< input_dim_; j++){
            variance+= (*output_ptr - mean) * (*output_ptr - mean);
            output_ptr++;
        }
        variance = (float) variance / (float) input_dim_;
        double sd = sqrt((double) variance);
        // TODO: integer to float function
        auto sd_inv = (float) (1/(sd + 0.0001)); // prevent zero divide!
        output_ptr = output + i * (input_dim_);
        for (int j=0; j< input_dim_; j++){
            *output_ptr = ((*output_ptr - mean) * (sd_inv));
            output_ptr ++;
        }

    }
}

AddNormalize::AddNormalize(std::size_t seq_len, std::size_t input_dim,
                           quant_bit_width * weight, quant_bit_width * bias) {
    input_dim_ = input_dim;
    seq_len_ = seq_len;
    weight_ = weight;
    bias_ = bias;
}

void AddNormalize::normalize(quant_bit_width *input, quant_bit_width *input_normalized) {
    for (int i =0; i< seq_len_; i++){

        auto* input_ptr = input + i * (input_dim_);
        auto* input_normalized_ptr = input_normalized + i * (input_dim_);

        int sum = 0;
        for (int j=0; j< input_dim_; j++){
            sum += *input_ptr;
            input_ptr ++;
        }

        input_ptr = input + i * (input_dim_);
        auto mean = (quant_bit_width)( (float) sum / (float) input_dim_);

        int64_t variance = 0;
        for (int j=0; j< input_dim_; j++){
            variance += MUL_HQ((*input_ptr - mean), (*input_ptr - mean));
            input_ptr++;
        }
        variance = SHIFT(variance);
        float variance_float = (float) variance / (float) (input_dim_);
        variance_float = variance_float / (float) (1 << NUM_FRACTION_BITS);
        float sd = sqrtf(variance_float);
        auto sd_inv = (float) (1/(sd + 0.00001)); // prevent zero divide!
        auto sd_inv_int = (quant_bit_width) (sd_inv * (1 << NUM_FRACTION_BITS));
        input_ptr = input + i * (input_dim_);
        input_normalized_ptr = input_normalized + i * (input_dim_);

        for (int j=0; j< input_dim_; j++){
            *input_normalized_ptr = (quant_bit_width) MUL((*input_ptr - mean), sd_inv_int);
            *input_normalized_ptr = (quant_bit_width) (MUL((*input_normalized_ptr), weight_[j]) + bias_[j]);
            input_ptr ++;
            input_normalized_ptr ++;
        }

    }
}

void AddNormalize::add(quant_bit_width* input, quant_bit_width* to_be_added){
    int32_t sum;
    for (int i =0; i< seq_len_ * input_dim_; i++){
        sum =  input[i] + to_be_added[i];
        if ((quant_bit_width) sum != sum)     // In case of overflow in 16 bits
            input[i] = (sum>0) ? INT16_MAX : INT16_MIN;
        else
            input[i] = (quant_bit_width) sum;

    }
}


void AddNormalize::computeRearranged(uint32_t *input, uint32_t *output) {
    auto* input_ptr = (int8_t*) (input );
    auto* output_ptr = (int8_t*) (output);
    for (int i =0; i< seq_len_* input_dim_; i++){
        *output_ptr = (int8_t) (*output_ptr + *input_ptr);
        output_ptr ++;
        input_ptr ++;
    }

    for (int i=0; i< seq_len_; i++){
        output_ptr = ((int8_t*) output) + i*kernel_dim_;
        int sum = 0;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            for (int k=0; k< kernel_dim_; k++) {
                sum += *(output_ptr+k);
            }
            output_ptr += seq_len_* kernel_dim_;
        }

        auto mean = (int32_t) (sum / input_dim_);
        int32_t variance = 0;
        output_ptr = (int8_t*) output + i*kernel_dim_;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            for (int k=0; k< kernel_dim_; k++) {
                variance+= (*(output_ptr+k) - mean) ^ 2; // Assuming that the values are fixed-point with 2 digit of fraction.
            }
            output_ptr += seq_len_* kernel_dim_;
        }

        variance = variance / (int) input_dim_;
        double sd = sqrt((double) variance);
        auto sd_inv = (int32_t) ((1<<2)/(sd + 1)); // prevent zero divide! // Assuming that the values are fixed-point with 2 digit of fraction.

        output_ptr = (int8_t*) output + i*kernel_dim_;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            for (int k=0; k< kernel_dim_; k++) {
                *(output_ptr+k) = (int8_t) ((*(output_ptr+k) - mean) * (sd_inv) >> 2);
            }
            output_ptr += seq_len_* kernel_dim_;
        }
    }
}