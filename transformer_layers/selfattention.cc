#include "selfattention.h"
#include "memory.h"
#include <cmath>
#include <iostream>
//#include <cstdint>

SingleHeadSelfAttn::SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                       quant_bit_width **weightVector, uint32_t **flagVector, std::size_t kernel_dim,
                                       std::size_t max_col) {

    pre_seq_len_ = pre_seq_len;
    head_hidden_size_ = head_hidden_size;
    kernel_size_ = kernel_dim;
    max_col_ = max_col;

    query_layer = new Dense(input_dim, head_hidden_size, weightVector[0], flagVector[0]);
    key_layer = new Dense(input_dim, head_hidden_size, weightVector[1], flagVector[1]);
    value_layer = new Dense(input_dim, head_hidden_size, weightVector[2], flagVector[2]);
    softmax = new Softmax();

    query_layer_out = new quant_bit_width[pre_seq_len * head_hidden_size]();
    key_layer_out = new quant_bit_width[pre_seq_len * head_hidden_size]();
    key_transposed_layer_out = new quant_bit_width[pre_seq_len * head_hidden_size]();
    value_layer_out = new quant_bit_width[pre_seq_len * head_hidden_size]();
    attention_scores = new quant_bit_width[pre_seq_len * pre_seq_len]();
}

SingleHeadSelfAttn::~SingleHeadSelfAttn() {

    delete[] query_layer_out;
    delete[] key_layer_out;
    delete[] key_transposed_layer_out;
    delete[] value_layer_out;
    delete[] attention_scores;

    delete query_layer;
    delete key_layer;
    delete value_layer;
    delete softmax;
}


SingleHeadSelfAttn::SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                       quant_bit_width **weightVector) {

    pre_seq_len_ = pre_seq_len;
    head_hidden_size_ = head_hidden_size;

    query_layer = new Dense(input_dim, head_hidden_size_, weightVector[0], (quant_bit_width*) nullptr);
    key_layer = new Dense(input_dim, head_hidden_size_, weightVector[1], (quant_bit_width*) nullptr);
    value_layer = new Dense(input_dim, head_hidden_size_, weightVector[2], (quant_bit_width*) nullptr);
    softmax = new Softmax();

    query_layer_out = new quant_bit_width[pre_seq_len * head_hidden_size]();
    key_layer_out = new quant_bit_width[pre_seq_len * head_hidden_size]();
    key_transposed_layer_out = new quant_bit_width[pre_seq_len * head_hidden_size]();
    value_layer_out = new quant_bit_width[pre_seq_len * head_hidden_size]();
    attention_scores = new quant_bit_width[pre_seq_len * pre_seq_len]();
}


void SingleHeadSelfAttn::compute(quant_bit_width *input, quant_bit_width *output, quant_bit_width *qkv,
                                 quant_bit_width* intermediate) {
    query_layer_out = qkv;
    key_layer_out = qkv + pre_seq_len_* head_hidden_size_;
    value_layer_out = qkv + 2 * pre_seq_len_* head_hidden_size_;
    key_transposed_layer_out = qkv + 3 * pre_seq_len_* head_hidden_size_;
    query_layer->compute(pre_seq_len_, input, query_layer_out);
    key_layer->compute(pre_seq_len_, input, key_layer_out);
    value_layer->compute(pre_seq_len_, input, value_layer_out);

    Transpose::transpose(key_layer_out, key_transposed_layer_out, pre_seq_len_, head_hidden_size_);
    MatMul::scale(key_transposed_layer_out, 1, pre_seq_len_*head_hidden_size_);

    MatMul::multiply(pre_seq_len_, query_layer_out, key_transposed_layer_out, intermediate,
                     head_hidden_size_, pre_seq_len_);

    softmax->compute(intermediate, pre_seq_len_);
    MatMul::multiply(pre_seq_len_, intermediate, value_layer_out, output, pre_seq_len_, head_hidden_size_);

}