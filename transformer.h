//
// Created by alireza on 10/25/21.
//

#ifndef FVLLMONTITRANSFORMER_TRANSFORMER_H
#define FVLLMONTITRANSFORMER_TRANSFORMER_H

#define NUM_FRACTION_BITS 12
#define MUL(x, y) (int32_t) (((int32_t)(x) * (int32_t)(y)) >> NUM_FRACTION_BITS)

#define D_Q 4
#define D_SEQ 120
#define D_MODEL 16
#define NUM_HEAD 4
#define D_FF 4
#define D_EMBEDDING 400

typedef int16_t quant_bit_width;

#endif //FVLLMONTITRANSFORMER_TRANSFORMER_H
