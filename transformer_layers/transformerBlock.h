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

    ~TransformerBlock();

    void compute(std::size_t seq_len, quant_bit_width *input, quant_bit_width *output);
    void computeFixedPoint(std::size_t seq_len, quant_bit_width *input,
                           quant_bit_width *input_normalized, quant_bit_width *output,
                           quant_bit_width* intermediate, quant_bit_width* qkv);

private:
    std::size_t num_heads_;
    std::size_t head_hidden_size_;
    std::size_t input_dim_;
    std::size_t ff_size_;
    SingleHeadSelfAttn* selfatten[NUM_LAYERS*NUM_HEAD];
    quant_bit_width* multihead_out;
    quant_bit_width* condense_out;
    quant_bit_width* intermediateFF;
    quant_bit_width* intermediateFFBlockWise;
    AddNormalize* addNorm;
    AddNormalize* addNorm2;
    AddNormalize* transformer_layer_0_addNorm[NUM_LAYERS];
    AddNormalize* transformer_layer_1_addNorm[NUM_LAYERS];
    AddNormalize* mlp_head_norm;
    TokenPosEmbedding* token;
    Dense* condense[NUM_LAYERS];
    Dense* feedForward0[NUM_LAYERS];
    Dense* feedForward1[NUM_LAYERS];
    Dense* patchEmbedding;
    Dense* mlp_head_linear;

#ifndef REARRANGE
    quant_bit_width* multihead_out_reshape;
#endif

};

#endif //FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H
