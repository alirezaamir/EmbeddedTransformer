//
// Created by alireza on 10/6/23.
//

#ifndef FVLLMONTITRANSFORMER_SOFTMAXC_H
#define FVLLMONTITRANSFORMER_SOFTMAXC_H


#include <stddef.h>
#include <stdint.h>
#include <math.h>

typedef struct {
} Softmax;

void compute(Softmax* softmax, int16_t* input, size_t seq_len);

// softmax.c



void compute(Softmax* softmax, int16_t* input, size_t seq_len) {
    size_t width = seq_len;
    float input_float = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        int16_t max_val = input[i * seq_len];
        for (int j = 1; j < width; j++) {
            if (input[i * seq_len + j] > max_val) {
                max_val = input[i * seq_len + j];
            }
        }
        for (int j = 0; j < width; j++) {
            input[i * seq_len + j] = (int16_t) fmax(input[i * seq_len + j] - max_val, -32767);
        }
        int32_t sum = 0;
        for (int j = 0; j < width; j++) {
            input_float = (float) input[i * seq_len + j] / (float) (1 << NUM_FRACTION_BITS);
            input_float = expf(input_float);
            input[i * seq_len + j] = (int16_t) (input_float * (1 << NUM_FRACTION_BITS));
            sum += input[i * seq_len + j];
        }
        float sum_float = (float) sum / (float) (1 << NUM_FRACTION_BITS);
        float sum_inv = (float) (1 / (sum_float + 0.00001)); // prevent zero divide!
        int16_t sum_inv_int = (int16_t) (sum_inv * (1 << NUM_FRACTION_BITS));
        for (int j = 0; j < width; j++) {
            input[i * seq_len + j] = (int16_t) MUL(input[i * seq_len + j], sum_inv_int);
        }
    }
}


#endif //FVLLMONTITRANSFORMER_SOFTMAXC_H
