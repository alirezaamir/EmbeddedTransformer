//
// Created by alireza on 9/13/23.
//

#include <iostream>
#include "transformer.h"
#include "data_cpp/signal.cpp"
#include "transformer_layers/transformerBlock.h"
#include "transformer_layers/weightsAndBiases.h"

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
    quant_bit_width * weightVec[NUM_LAYERS*(3*NUM_HEAD+5)+3];
    quant_bit_width * biasVec[NUM_LAYERS*(3*NUM_HEAD+5)+3];
    WeightsAndBiases::getWeights(weightVec);
    WeightsAndBiases::getBiases(biasVec);
    quant_bit_width * clsTokenVector = WeightsAndBiases::getClassToken();
    quant_bit_width * posMatrix = WeightsAndBiases::getPosEmbedding();

    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec, biasVec,
                               clsTokenVector, posMatrix);
    selfatten.computeFixedPoint(D_SEQ, input_signal, input_normalized, out, intermediate, qkv);

    std::cout<<"Error value : " << error_check(transformer_layers_3_1_fn_add,
                                               input_signal, (D_SEQ + 1) * D_MODEL);
    std::cout << std::endl;
}

int main() {
    inference();
    return 0;
}