//
// Created by alireza on 9/13/23.
//

#include <iostream>
#include "transformer.h"
#include "data_cpp/data.cpp"
#include "data_cpp/signal.cpp"
#include "transformer_layers/transformerBlock.h"

quant_bit_width out[(D_SEQ+1) * D_MODEL];
quant_bit_width intermediate[(D_SEQ+1) * (D_SEQ+1)];
quant_bit_width qkv[(D_SEQ+1) * D_MODEL];
quant_bit_width input_normalized[(D_SEQ+1) * D_MODEL];
float error_check(const quant_bit_width* groundTruth, const quant_bit_width* output, std::size_t length){
    long error = 0;
    for (int i=0; i<length; i++){
        if (i<10)
            std::cout << groundTruth[i] << " , " << output[i] << std::endl;
        error += MUL_HQ(groundTruth[i] - output[i], groundTruth[i] - output[i]);
//        if ((groundTruth[i] - output[i] > 20) || (groundTruth[i] - output[i] < -20) )
//            std::cout<< i << ": " << groundTruth[i] << " , " << output[i] << std::endl;
    }
    error = (error >> NUM_FRACTION_BITS);

    return (float) error/ (float) length;
}


void inference(){
    quant_bit_width * weightVec[3*NUM_HEAD+8];
    quant_bit_width * biasVec[3*NUM_HEAD+8];
    quant_bit_width * clsTokenVector;
    quant_bit_width * posMatrix;

    weightVec[0] = to_patch_embedding_layer_norm1_weight;
    biasVec[0] = to_patch_embedding_layer_norm1_bias;
    weightVec[1] = to_patch_embedding_linear_weight;
    biasVec[1] = to_patch_embedding_linear_bias;
    weightVec[2] = to_patch_embedding_layer_norm2_weight;
    biasVec[2] = to_patch_embedding_layer_norm2_bias;
    clsTokenVector = cls_token;
    posMatrix = pos_embedding;
    weightVec[3] = transformer_layers_0_0_norm_weight;
    biasVec[3] = transformer_layers_0_0_norm_bias;

    int weightVectorIndex = 4;
    int biasVectorIndex = 4;
    weightVec[weightVectorIndex++] = transformer_layers_0_0_fn_to_qkv_weight_Q_H0;
    weightVec[weightVectorIndex++] = transformer_layers_0_0_fn_to_qkv_weight_K_H0;
    weightVec[weightVectorIndex++] = transformer_layers_0_0_fn_to_qkv_weight_V_H0;
    weightVec[weightVectorIndex++] = transformer_layers_0_0_fn_to_qkv_weight_Q_H1;
    weightVec[weightVectorIndex++] = transformer_layers_0_0_fn_to_qkv_weight_K_H1;
    weightVec[weightVectorIndex++] = transformer_layers_0_0_fn_to_qkv_weight_V_H1;
    weightVec[weightVectorIndex++] = transformer_layers_0_0_fn_to_qkv_weight_Q_H2;
    weightVec[weightVectorIndex++] = transformer_layers_0_0_fn_to_qkv_weight_K_H2;
    weightVec[weightVectorIndex++] = transformer_layers_0_0_fn_to_qkv_weight_V_H2;
    weightVec[weightVectorIndex++] = transformer_layers_0_0_fn_to_qkv_weight_Q_H3;
    weightVec[weightVectorIndex++] = transformer_layers_0_0_fn_to_qkv_weight_K_H3;
    weightVec[weightVectorIndex++] = transformer_layers_0_0_fn_to_qkv_weight_V_H3;
    for (int i=0; i<3*4; i++)
        biasVec[biasVectorIndex++] = (quant_bit_width *) nullptr;

    weightVec[weightVectorIndex++] = transformer_layers_0_0_fn_projection_weight;
    biasVec[biasVectorIndex++] = transformer_layers_0_0_fn_projection_bias;

    weightVec[weightVectorIndex++] = transformer_layers_0_1_norm_weight;
    biasVec[biasVectorIndex++] = transformer_layers_0_1_norm_bias;

    weightVec[weightVectorIndex++] = transformer_layers_0_1_fn_ff1_weight;
    weightVec[weightVectorIndex++] = transformer_layers_0_1_fn_ff2_weight;
    biasVec[biasVectorIndex++] = transformer_layers_0_1_fn_ff1_bias;
    biasVec[biasVectorIndex++] = transformer_layers_0_1_fn_ff2_bias;

    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec, biasVec,
                               clsTokenVector, posMatrix);
    selfatten.computeFixedPoint(D_SEQ, input_signal, input_normalized, out, intermediate, qkv);

    std::cout<<"Error value : " << error_check(transformer_layers_0_1_fn_add,
                                               input_signal, (D_SEQ + 1) * D_MODEL);
    std::cout << std::endl;
}

int main() {
    inference();
    return 0;
}