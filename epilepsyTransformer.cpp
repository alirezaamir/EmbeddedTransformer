//
// Created by alireza on 9/13/23.
//

#include <iostream>
#include <cmath>
#include "transformer.h"
#include "data_cpp/signal.cpp"
#include "data_cpp/signal_fft.cpp"
#include "transformer_layers/transformerBlock.h"
#include "transformer_layers/weightsAndBiases.h"
#include "SYLT-FFT/fft.h"

void prototype_distances(quant_bit_width* prototypeVec,
                          const quant_bit_width* modelOutput,
                          int32_t* distVec,
                          std::size_t prototypeLength,
                          int prototypeNums){
    for (int p=0; p< prototypeNums; p++){
        long dist = 0;
        quant_bit_width * prototypePtr = prototypeVec + (p * prototypeLength);
        for (int i=0; i<prototypeLength; i++){
            dist += MUL_HQ(prototypePtr[i] - modelOutput[i], prototypePtr[i] - modelOutput[i]);
        }
        dist = (dist >> NUM_FRACTION_BITS);
        distVec[p] = (int32_t) dist;
    }
}


void transformerInference(quant_bit_width * transformerInput,
                          quant_bit_width * transformerOutput,
                          quant_bit_width* input_normalized,
                          quant_bit_width* qkv,
                          quant_bit_width* intermediate){
    quant_bit_width * weightVec[NUM_LAYERS*(3*NUM_HEAD+5)+5];
    quant_bit_width * biasVec[NUM_LAYERS*(3*NUM_HEAD+5)+5];
    WeightsAndBiases::getWeights(weightVec);
    WeightsAndBiases::getBiases(biasVec);
    quant_bit_width * clsTokenVector = WeightsAndBiases::getClassToken();
    quant_bit_width * posMatrix = WeightsAndBiases::getPosEmbedding();

    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec, biasVec,
                               clsTokenVector, posMatrix);
    selfatten.computeFixedPoint(D_SEQ, transformerInput, input_normalized,
                                transformerOutput, intermediate, qkv);
}

quant_bit_width compute_log_amp(int32_t real, int32_t imag){
    real = MUL_HQ(real, 25) >> (NUM_FRACTION_BITS - 9);
    imag = MUL_HQ(imag, 25) >> (NUM_FRACTION_BITS - 9);
    auto real2 = MUL_LONG(real, real) >> NUM_FRACTION_BITS;
    auto imag2 = MUL_LONG(imag, imag) >> NUM_FRACTION_BITS;
    float pow2 = (float)(real2 + imag2) / (float) (1<< NUM_FRACTION_BITS);
    float amp = sqrtf(pow2);
    float stft = logf(amp+ 1e-10f);
    auto stft_int = (quant_bit_width) (stft * (1<<NUM_FRACTION_BITS));

    return stft_int;
}

void initialize_stft(fft_complex_t* data, const quant_bit_width * raw_input_signal){
    // Initialize each element of the data array
    for (int i = 0; i < 256; i++) {
        data[i].r = (MUL_HQ(raw_input_signal[i], hanning[i])) ; // Set the real part
        data[i].i = 0; // Set the imaginary part to 0
    }
    for (int i = 256; i < 512; i++) {  // Padding for nfft=2
        data[i].r = 0; // Set the real part to 0
        data[i].i = 0; // Set the imaginary part to 0
    }
}

void rearrange_error(const quant_bit_width* stftVec,
                     const quant_bit_width* rearrangedVec,
                     int length){
    for (int i=0; i<length; i++){
        int error = stftVec[i] - rearrangedVec[i];
        bool sign_error = (bool) (((long)stftVec[i] * (long)rearrangedVec[i]) <= 0);
        if (error > 100 && sign_error){
            std::cout << "Error in "<<  i << " : " << error <<std::endl;

            std::cout << "Calc: " << stftVec[i] << std::endl;
            std::cout << "Ground truth: "<< rearrangedVec[i] << std::endl;
            std::cout << std::endl;
        }
    }
}

void stft_rearrange(quant_bit_width* rawInputSignal, quant_bit_width* stftVec,
                    std::size_t patchHeight, std::size_t patchWidth){
    fft_complex_t data[512];
    int overlap = 64;
    for (int ch=0; ch<20; ch++){
        for (int time_step=0; time_step<15; time_step++){
            quant_bit_width* rawSignalPtr = rawInputSignal + ch * 3072 + (256 - overlap) * time_step;
            initialize_stft(data, rawSignalPtr);
            fft_fft(data, 9);
            quant_bit_width * stftVecPtr = stftVec
                    + ch * 15 * 160
                    + (time_step / patchWidth) * patchWidth * patchHeight
                    + (time_step % patchWidth);
            for (int index =0 ; index < patchHeight; index++){
                quant_bit_width stft_int = compute_log_amp(data[index].r, data[index].i);
                *stftVecPtr = stft_int;
                stftVecPtr += patchWidth;
//                error_check(stft_int, index, data[index].r, data[index].i);
            }

            stftVecPtr += patchHeight * patchWidth * 2;
            for (int index = patchHeight ; index < 2*patchHeight; index++){
                quant_bit_width stft_int = compute_log_amp(data[index].r, data[index].i);
                *stftVecPtr = stft_int;
                stftVecPtr += patchWidth;
//                error_check(stft_int, index, data[index].r, data[index].i);
            }
        }
    }
}

int main() {
    // Make memory map
    quant_bit_width* stftVec = raw_signal;
    quant_bit_width* rawInputSignal = raw_signal + 160*15;

    quant_bit_width* out = raw_signal + 160*15*20;//[(D_SEQ+1) * D_MODEL];
    quant_bit_width* intermediate = raw_signal + 16*1024;  // +32KB //[(D_SEQ+1) * (D_SEQ+1)];
    quant_bit_width* qkv = out + 2048; // [(D_SEQ+1) * D_MODEL];
    quant_bit_width* input_normalized = out + 4096;//[(D_SEQ+1) * D_MODEL];
    int32_t distances[2];

    stft_rearrange(rawInputSignal, stftVec, 80, 5);
    transformerInference(stftVec, out, input_normalized, qkv, intermediate);
    prototype_distances(prototypes, out, distances, D_MODEL, 2);
    std::cout<<"Distances : " << std::endl;
    for (int i = 0; i< 2; i++)
        std::cout<<"From the prototype of class " << i << " = " << distances[i] <<  std::endl;
    return 0;
}