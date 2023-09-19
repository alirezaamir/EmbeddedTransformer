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
float error_check(const quant_bit_width* groundTruth, const quant_bit_width* output, std::size_t length){
    long error = 0;
    for (int i=0; i<length; i++){
        if (i<10)
            std::cout << groundTruth[i] << " , " << output[i] << std::endl;
        error += MUL_HQ(groundTruth[i] - output[i], groundTruth[i] - output[i]);
//        if ((groundTruth[i] - output[i] > 100) || (groundTruth[i] - output[i] < -100) )
//            std::cout<< i << ": " << groundTruth[i] << " , " << output[i] << std::endl;
    }
    error = (error >> NUM_FRACTION_BITS);

    return (float) error/ (float) length;
}


void inference(){
    quant_bit_width * weightVec[3*NUM_HEAD+3];
    quant_bit_width * biasVec[3*NUM_HEAD+3];
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

    int startIndex = 4;
    weightVec[startIndex + 0] = transformer_layers_0_0_fn_to_qkv_weight_Q_H0;
    weightVec[startIndex + 1] = transformer_layers_0_0_fn_to_qkv_weight_K_H0;
    weightVec[startIndex + 2] = transformer_layers_0_0_fn_to_qkv_weight_V_H0;
    weightVec[startIndex + 3] = transformer_layers_0_0_fn_to_qkv_weight_Q_H1;
    weightVec[startIndex + 4] = transformer_layers_0_0_fn_to_qkv_weight_K_H1;
    weightVec[startIndex + 5] = transformer_layers_0_0_fn_to_qkv_weight_V_H1;
    weightVec[startIndex + 6] = transformer_layers_0_0_fn_to_qkv_weight_Q_H2;
    weightVec[startIndex + 7] = transformer_layers_0_0_fn_to_qkv_weight_K_H2;
    weightVec[startIndex + 8] = transformer_layers_0_0_fn_to_qkv_weight_V_H2;
    weightVec[startIndex + 9] = transformer_layers_0_0_fn_to_qkv_weight_Q_H3;
    weightVec[startIndex + 10] = transformer_layers_0_0_fn_to_qkv_weight_K_H3;
    weightVec[startIndex + 11] = transformer_layers_0_0_fn_to_qkv_weight_V_H3;
    for (int i=0; i<3*4; i++)
        biasVec[i+startIndex] = (quant_bit_width *) nullptr;

    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec, biasVec,
                               clsTokenVector, posMatrix);
    selfatten.computeFixedPoint(D_SEQ, input_signal, out, intermediate);

    std::cout<<"Error value : " << error_check(transformer_layers_0_0_fn_attn, intermediate, (D_SEQ + 1) * (D_SEQ + 1));
    std::cout << std::endl;
}

int main() {
    inference();
    return 0;
}