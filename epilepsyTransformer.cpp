//
// Created by alireza on 9/13/23.
//

#include <iostream>
#include "transformer.h"
#include "data_cpp/data.cpp"
#include "data_cpp/signal.cpp"
#include "transformer_layers/transformerBlock.h"

quant_bit_width out[D_SEQ * D_MODEL];
float error_check(const quant_bit_width* groundTruth, const quant_bit_width* output, std::size_t length){
    long error = 0;
    for (int i=0; i<length; i++){
        if (i<10)
            std::cout << groundTruth[i] << " , " << output[i] << std::endl;
        error += MUL_HQ(groundTruth[i] - output[i], groundTruth[i] - output[i]);
    }
    error = (error >> NUM_FRACTION_BITS);

    return (float) error/ (float) length;
}


void inference(){
    quant_bit_width * weightVec[3*NUM_HEAD+3];
    quant_bit_width * biasVec[3*NUM_HEAD+3];

    weightVec[0] = to_patch_embedding_layer_norm1_weight;
    biasVec[0] = to_patch_embedding_layer_norm1_bias;
    weightVec[1] = to_patch_embedding_linear_weight;
    biasVec[1] = to_patch_embedding_linear_bias;


    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec, biasVec);
    selfatten.computeFixedPoint(D_SEQ, input_signal, out);

    std::cout<<"Error value : " << error_check(to_patch_embedding_linear, out, D_SEQ * D_MODEL) <<std::endl;
}

int main() {
    inference();
    return 0;
}