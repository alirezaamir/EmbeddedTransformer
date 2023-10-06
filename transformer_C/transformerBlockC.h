//
// Created by alireza on 10/6/23.
//

#ifndef FVLLMONTITRANSFORMER_TRANSFORMERBLOCK_H
#define FVLLMONTITRANSFORMER_TRANSFORMERBLOCK_H

#include <stddef.h>
#include <stdint.h>
#include "selfattention.h"
#include "addNorm.h"
#include "dense.h"
#include "tokenPosEmebedding.h"
#include "../transformer.h"

typedef struct {
    size_t num_heads_;
    size_t head_hidden_size_;
    size_t input_dim_;
    size_t ff_size_;
    SingleHeadSelfAttn* selfatten[NUM_LAYERS*NUM_HEAD];
    int16_t* multihead_out;
    int16_t* condense_out;
    int16_t* intermediateFF;
    int16_t* intermediateFFBlockWise;
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
    int16_t* multihead_out_reshape;
    #endif
} TransformerBlock;

TransformerBlock* createTransformerBlock(size_t pre_seq_len, size_t input_dim, size_t head_hidden_size, size_t num_heads, size_t ff_size, int16_t** weightVector, int16_t** biasVector, int16_t* clsTokenVector, int16_t* posMatrix);
void destroyTransformerBlock(TransformerBlock* transformerBlock);
void computeFixedPoint(TransformerBlock* transformerBlock, size_t seq_len, int16_t* input, int16_t* input_normalized, int16_t* output, int16_t* intermediate, int16_t* qkv);

TransformerBlock* createTransformerBlock(size_t pre_seq_len, size_t input_dim, size_t head_hidden_size, size_t num_heads, size_t ff_size, int16_t** weightVector, int16_t** biasVector, int16_t* clsTokenVector, int16_t* posMatrix) {
    TransformerBlock* transformerBlock = (TransformerBlock*) malloc(sizeof(TransformerBlock));
    transformerBlock->num_heads_ = num_heads;
    transformerBlock->head_hidden_size_ = head_hidden_size;
    transformerBlock->input_dim_ = input_dim;
    transformerBlock->ff_size_ = ff_size;

    transformerBlock->addNorm = create_AddNormalize(pre_seq_len, D_EMBEDDING, weightVector[0], biasVector[0]);
    transformerBlock->patchEmbedding = create_Dense(D_EMBEDDING, D_MODEL, weightVector[1], biasVector[1]);
    transformerBlock->addNorm2 = create_AddNormalize(pre_seq_len, D_MODEL, weightVector[2], biasVector[2]);
    transformerBlock->token = create_TokenPosEmbedding(posMatrix, clsTokenVector, pre_seq_len, input_dim, D_SEQ + 1);

    for (int l = 0; l < 4; l++) {
        transformerBlock->transformer_layer_0_addNorm[l] = create_AddNormalize((pre_seq_len + 1), D_MODEL, weightVector[l * 17 + 3], biasVector[l * 17 + 3]);

        for (int n = 0; n < num_heads; n++) {
            transformerBlock->selfatten[l * num_heads + n] = create_SingleHeadSelfAttn((pre_seq_len + 1), input_dim, head_hidden_size, weightVector + l * 17 + 4 + n * 3);
        }

        transformerBlock->condense[l] = create_Dense(num_heads * head_hidden_size, input_dim, weightVector[l * 17 + num_heads * 3 + 4], biasVector[l * 17 + num_heads * 3 + 4]);

        transformerBlock->transformer_layer_1_addNorm[l] = create_AddNormalize((pre_seq_len + 1), input_dim, weightVector[l * 17 + num_heads * 3 + 5], biasVector[l * 17 + num_heads * 3 + 5]);
        transformerBlock->feedForward0[l] = create_Dense(input_dim, ff_size, weightVector[l * 17 + num_heads * 3 + 6], biasVector[l * 17 + num_heads * 3 + 6]);
        transformerBlock->feedForward1[l] = create_Dense(ff_size, input_dim, weightVector[l * 17 + num_heads * 3 + 7], biasVector[l * 17 + num_heads * 3 + 7]);
    }

    transformerBlock->mlp_head_norm = create_AddNormalize(1, D_MODEL, weightVector[(NUM_LAYERS - 1) * 17 + NUM_HEAD * 3 + 8], biasVector[(NUM_LAYERS - 1) * 17 + NUM_HEAD * 3 + 8]);
    transformerBlock->mlp_head_linear = create_Dense(D_MODEL, D_MODEL, weightVector[(NUM_LAYERS - 1) * 17 + NUM_HEAD * 3 + 9], biasVector[(NUM_LAYERS - 1) * 17 + NUM_HEAD * 3 + 9]);

    return transformerBlock;
}


void destroyTransformerBlock(TransformerBlock* transformerBlock) {
    // Free dynamically allocated memory

    free(transformerBlock);
}

void computeFixedPoint(TransformerBlock* transformerBlock, size_t seq_len, int16_t* input, int16_t* input_normalized, int16_t* output, int16_t* intermediate, int16_t* qkv) {
    normalize(transformerBlock->addNorm, input, input);
    compute_Dense(transformerBlock->patchEmbedding, seq_len, input, output);
    normalize(transformerBlock->addNorm2, output, output);

    clsConcatenate(transformerBlock->token, output, input);
    seq_len++;
    posEmbedding(transformerBlock->token, input);

    for (int l = 0; l < 4; l++) {
        normalize(transformerBlock->transformer_layer_0_addNorm[l], input, input_normalized);

        for (int n = 0; n < NUM_HEAD; n++) {
            compute_SingleHeadSelfAttn(transformerBlock->selfatten[l * NUM_HEAD + n], input_normalized, output + n * (seq_len * transformerBlock->head_hidden_size_), qkv, intermediate);
        }
        multihead_transpose(output, intermediate, seq_len, transformerBlock->head_hidden_size_, transformerBlock->num_heads_);

        compute_Dense(transformerBlock->condense[l], seq_len, intermediate, output);

        add(transformerBlock->transformer_layer_0_addNorm[l], input, output);

        normalize(transformerBlock->transformer_layer_1_addNorm[l], input, input_normalized);
        compute_Dense(transformerBlock->feedForward0[l], seq_len, input_normalized, intermediate);
        activation(transformerBlock->feedForward0[l], seq_len * transformerBlock->ff_size_, intermediate, intermediate);

        compute_Dense(transformerBlock->feedForward1[l], seq_len, intermediate, output);
        add(transformerBlock->transformer_layer_1_addNorm[l], input, output);
    }

    normalize(transformerBlock->mlp_head_norm, input, input_normalized);
    compute_Dense(transformerBlock->mlp_head_linear, 1, input_normalized, output);
}



#endif //FVLLMONTITRANSFORMER_TRANSFORMERBLOCK_H
