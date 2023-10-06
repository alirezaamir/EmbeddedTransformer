//
// Created by alireza on 10/6/23.
//

#ifndef FVLLMONTITRANSFORMER_TOKENPOSEMBEDDINGC_H
#define FVLLMONTITRANSFORMER_TOKENPOSEMBEDDINGC_H

typedef struct {
    quant_bit_width* cls_token_vector;
    quant_bit_width* pos_matrix;
    size_t seq_len;
    size_t input_dim;
    size_t pos_matrix_dim;
} TokenPosEmbedding;

void clsConcatenate(TokenPosEmbedding* tpe, quant_bit_width* input, quant_bit_width* concatenated_input) {
    // Copy cls_token_ into the concatenated array column-wise at the beginning
    for (size_t i = 0; i < tpe->input_dim; ++i) {
        concatenated_input[i] = tpe->cls_token_vector[i];
    }
    // Copy the input array into the concatenated array
    for (size_t i = 0; i < tpe->seq_len * tpe->input_dim; ++i) {
        concatenated_input[i + tpe->input_dim] = input[i];
    }
}

void posEmbedding(TokenPosEmbedding* tpe, quant_bit_width* input) {
    for (size_t i = 0; i < (tpe->seq_len + 1); ++i) {
        for (size_t j = 0; j < tpe->input_dim; ++j) {
            input[i * tpe->input_dim + j] += tpe->pos_matrix[i * tpe->input_dim + j];
        }
    }
}

#endif //FVLLMONTITRANSFORMER_TOKENPOSEMBEDDINGC_H
