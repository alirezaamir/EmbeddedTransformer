//
// Created by alireza on 10/6/23.
//

#ifndef FVLLMONTITRANSFORMER_SELFATTENTIONC_H
#define FVLLMONTITRANSFORMER_SELFATTENTIONC_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "dense_layerC.h"
#include "softmaxC.h"
#include "transposeC.h"
#include "matMulC.h"

typedef struct {
    Dense* query_layer;
    Dense* key_layer;
    Dense* value_layer;
    Softmax* softmax;
    int16_t* query_layer_out;
    int16_t* key_layer_out;
    int16_t* key_transposed_layer_out;
    int16_t* value_layer_out;
    int16_t* attention_scores;
    size_t pre_seq_len;
    size_t head_hidden_size;
} SingleHeadSelfAttn;

SingleHeadSelfAttn* create_SingleHeadSelfAttn(size_t pre_seq_len, size_t input_dim, size_t head_hidden_size, int16_t** weightVector);
void destroy_SingleHeadSelfAttn(SingleHeadSelfAttn* self_attn);
void compute_SingleHeadSelfAttn(SingleHeadSelfAttn* self_attn, int16_t* input, int16_t* output, int16_t* qkv, int16_t* intermediate);




SingleHeadSelfAttn* create_SingleHeadSelfAttn(size_t pre_seq_len, size_t input_dim, size_t head_hidden_size, int16_t** weightVector) {
    SingleHeadSelfAttn* self_attn = (SingleHeadSelfAttn*) malloc(sizeof(SingleHeadSelfAttn));
    self_attn->pre_seq_len = pre_seq_len;
    self_attn->head_hidden_size = head_hidden_size;

    self_attn->query_layer = createDense(input_dim, head_hidden_size, weightVector[0], NULL);
    self_attn->key_layer = createDense(input_dim, head_hidden_size, weightVector[1], NULL);
    self_attn->value_layer = createDense(input_dim, head_hidden_size, weightVector[2], NULL);
    self_attn->softmax = createSoftmax();

    self_attn->query_layer_out = (int16_t*) malloc(pre_seq_len * head_hidden_size * sizeof(int16_t));
    self_attn->key_layer_out = (int16_t*) malloc(pre_seq_len * head_hidden_size * sizeof(int16_t));
    self_attn->key_transposed_layer_out = (int16_t*) malloc(pre_seq_len * head_hidden_size * sizeof(int16_t));
    self_attn->value_layer_out = (int16_t*) malloc(pre_seq_len * head_hidden_size * sizeof(int16_t));
    self_attn->attention_scores = (int16_t*) malloc(pre_seq_len * pre_seq_len * sizeof(int16_t));

    return self_attn;
}

void destroy_SingleHeadSelfAttn(SingleHeadSelfAttn* self_attn) {
    free(self_attn->query_layer_out);
    free(self_attn->key_layer_out);
    free(self_attn->key_transposed_layer_out);
    free(self_attn->value_layer_out);
    free(self_attn->attention_scores);

    destroyDense(self_attn->query_layer);
    destroyDense(self_attn->key_layer);
    destroyDense(self_attn->value_layer);

    free(self_attn);
}

void compute_SingleHeadSelfAttn(SingleHeadSelfAttn* self_attn, int16_t* input, int16_t* output, int16_t* qkv, int16_t* intermediate) {
    self_attn->query_layer_out = qkv;
    self_attn->key_layer_out = qkv + self_attn->pre_seq_len * self_attn->head_hidden_size;
    self_attn->value_layer_out = qkv + 2 * self_attn->pre_seq_len * self_attn->head_hidden_size;
    self_attn->key_transposed_layer_out = qkv + 3 * self_attn->pre_seq_len * self_attn->head_hidden_size;

    computeDense(self_attn->query_layer, self_attn->pre_seq_len, input, self_attn->query_layer_out);
    computeDense(self_attn->key_layer, self_attn->pre_seq_len, input, self_attn->key_layer_out);
    computeDense(self_attn->value_layer, self_attn->pre_seq_len, input, self_attn->value_layer_out);

    transpose_quant(self_attn->key_layer_out, self_attn->key_transposed_layer_out, self_attn->pre_seq_len, self_attn->head_hidden_size);
    MatMul_scale(self_attn->key_transposed_layer_out, 1, self_attn->pre_seq_len * self_attn->head_hidden_size);

    MatMul_multiply(self_attn->pre_seq_len, self_attn->query_layer_out, self_attn->key_transposed_layer_out, intermediate, self_attn->head_hidden_size, self_attn->pre_seq_len);

    computeSoftmax(self_attn->softmax, intermediate, self_attn->pre_seq_len);
    MatMul_multiply(self_attn->pre_seq_len, intermediate, self_attn->value_layer_out, output, self_attn->pre_seq_len, self_attn->head_hidden_size);
}

#endif //FVLLMONTITRANSFORMER_SELFATTENTIONC_H
