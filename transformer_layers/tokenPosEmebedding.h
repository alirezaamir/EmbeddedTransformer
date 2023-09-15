//
// Created by alireza on 9/14/23.
//

#ifndef FVLLMONTITRANSFORMER_TOKENPOSEMEBEDDING_H
#define FVLLMONTITRANSFORMER_TOKENPOSEMEBEDDING_H

#include "util.h"
#include "../transformer.h"
#include "iostream"

class TokenPosEmbedding{
public:
    TokenPosEmbedding(quant_bit_width* pos_matrix, quant_bit_width* cls_token_vector,
                      std::size_t seq_len, std::size_t input_dim, std::size_t pos_matrix_dim);
    ~TokenPosEmbedding();
    void clsConcatenate(quant_bit_width* input, quant_bit_width* concatenated_input);
    void posEmbedding(quant_bit_width* input);

private:
    quant_bit_width* cls_token_vector_;
    quant_bit_width * pos_matrix_;
    std::size_t seq_len_;
    std::size_t input_dim_;
    std::size_t pos_matrix_dim_;
};

#endif //FVLLMONTITRANSFORMER_TOKENPOSEMEBEDDING_H
