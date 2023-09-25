//
// Created by alireza on 9/20/23.
//

#ifndef FVLLMONTITRANSFORMER_WEIGHTSANDBIASES_H
#define FVLLMONTITRANSFORMER_WEIGHTSANDBIASES_H

#include "util.h"
#include "../transformer.h"

class WeightsAndBiases{
private:
public:
    static void getWeights(quant_bit_width * weightVec[]);
    static void getBiases(quant_bit_width * biasVec[]);
    static quant_bit_width * getPosEmbedding();
    static quant_bit_width * getClassToken();
};


#endif //FVLLMONTITRANSFORMER_WEIGHTSANDBIASES_H
