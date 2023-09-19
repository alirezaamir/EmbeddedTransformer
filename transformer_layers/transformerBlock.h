//
// Created by alireza on 3/2/22.
//
#include "selfattention.h"
#include "addNorm.h"
#include "dense.h"
#include "tokenPosEmebedding.h"
#include "../transformer.h"

#ifndef FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H
#define FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H

class TransformerBlock{
public:
    TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size, std::size_t num_heads,
                     std::size_t ff_size, quant_bit_width ** weightVector, uint32_t ** flagVector,
                     std::size_t kernelDim, std::size_t maxCol);

    TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size, std::size_t num_heads,
                     std::size_t ff_size, quant_bit_width ** weightVector,
                     quant_bit_width ** biasVector, quant_bit_width*, quant_bit_width* );

    virtual ~TransformerBlock();

    void compute(std::size_t seq_len, quant_bit_width *input, quant_bit_width *output);
    void computeFixedPoint(std::size_t seq_len, quant_bit_width *input,
                           quant_bit_width *input_normalized, quant_bit_width *output,
                           quant_bit_width* intermediate, quant_bit_width* qkv);

private:
    std::size_t num_heads_;
    std::size_t head_hidden_size_;
    std::size_t input_dim_;
    std::size_t ff_size_;
    SingleHeadSelfAttn* selfatten[16];
    quant_bit_width* multihead_out;
    quant_bit_width* condense_out;
    quant_bit_width* intermediateFF;
    quant_bit_width* intermediateFFBlockWise;
    AddNormalize* addNorm;
    AddNormalize* addNorm2;
    AddNormalize* transformer_layer_0_0_addNorm;
    AddNormalize* transformer_layer_0_1_addNorm;
    TokenPosEmbedding* token;
    Dense* condense;
    Dense* feedForward0;
    Dense* feedForward1;
    Dense* patchEmbedding;

#ifndef REARRANGE
    quant_bit_width* multihead_out_reshape;
#endif

};

#endif //FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H
