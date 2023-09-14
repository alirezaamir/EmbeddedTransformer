//
// Created by alireza on 9/13/23.
//

#include <iostream>
#include "transformer.h"
#include "data_cpp/data.cpp"
#include "data_cpp/signal.cpp"
#include "transformer_layers/transformerBlock.h"


float error_check(const quant_bit_width* groundTruth, const quant_bit_width* output, std::size_t length){
    long error = 0;
    for (int i=0; i<length; i++){
        error += MUL(groundTruth[i] - output[i], groundTruth[i] - output[i]);
    }
    return (float) error/ (float) length;
}


void inference(){
    quant_bit_width * weightVec[3*NUM_HEAD+3];
    quant_bit_width * biasVec[3*NUM_HEAD+3];

    weightVec[0] = to_patch_embedding_layer_norm1_weight;
    biasVec[0] = to_patch_embedding_layer_norm1_bias;


    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec, biasVec);
    selfatten.computeFixedPoint(D_SEQ, input_signal, nullptr);

    std::cout<<"Error value : " << error_check(to_patch_embedding_layer_norm1, input_signal, D_SEQ * D_EMBEDDING) <<std::endl;
}

int main() {
    inference();
    return 0;
}