//
// Created by alireza on 9/14/23.
//

#include "tokenPosEmebedding.h"



TokenPosEmbedding::TokenPosEmbedding(quant_bit_width* pos_matrix, quant_bit_width* cls_token_vector,
                                     std::size_t seq_len, std::size_t input_dim, std::size_t pos_dim) {

    seq_len_ = seq_len;
    input_dim_ = input_dim;
    pos_matrix_ = pos_matrix;
    cls_token_vector_ = cls_token_vector;
    pos_matrix_dim_ = pos_dim;
}

TokenPosEmbedding::~TokenPosEmbedding()= default;


void TokenPosEmbedding::clsConcatenate(quant_bit_width *input, quant_bit_width *concatenated_input) {

    // Copy cls_token_ into the concatenated array column-wise at the beginning
    for (std::size_t i = 0; i < seq_len_; ++i) {
        concatenated_input[i] = cls_token_vector_[i];
    }

    // Copy the input array into the concatenated array
    for (std::size_t i = 0; i < seq_len_ * input_dim_; ++i) {
        concatenated_input[i + input_dim_] = input[i];
    }

}

void TokenPosEmbedding::posEmbedding(quant_bit_width *input) {
    for (std::size_t i = 0; i < seq_len_; ++i) {
        for (std::size_t j = 0; j < input_dim_; ++j) {
            input[i * input_dim_ + j] += pos_matrix_[i* input_dim_ + j];
        }
    }
}
