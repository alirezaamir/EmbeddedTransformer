//
// Created by alireza on 3/2/22.
//

#include "transformerBlock.h"
#include "debuggerFunctions.h"

TransformerBlock::TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                   std::size_t num_heads, std::size_t ff_size, quant_bit_width ** weightVector,
                                   uint32_t ** flagVector,
                                   std::size_t kernelDim, std::size_t maxCol) {

    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;
    input_dim_ = input_dim;

    for (int n =0; n< num_heads; n++){
        selfatten[n] = new SingleHeadSelfAttn(pre_seq_len, input_dim, head_hidden_size, weightVector+n*3,
                                              flagVector+n*3,
                                              kernelDim, maxCol);
    }

    condense[0] = new Dense(num_heads* head_hidden_size, input_dim, weightVector[num_heads * 3],
                         flagVector[num_heads * 3]);

    multihead_out = new quant_bit_width[pre_seq_len * num_heads * head_hidden_size]();
    condense_out = new quant_bit_width[pre_seq_len * input_dim]();
    intermediateFF = new quant_bit_width[pre_seq_len * ff_size]();

#ifndef REARRANGE
    multihead_out_reshape = new quant_bit_width[pre_seq_len * num_heads * head_hidden_size]();
#endif

    addNorm = new AddNormalize(pre_seq_len, input_dim, kernelDim, maxCol);
    feedForward0[0] = new Dense(input_dim, ff_size, weightVector[num_heads * 3+ 1],
                             flagVector[num_heads * 3 + 1]);
    feedForward1[0] = new Dense(ff_size, input_dim, weightVector[num_heads * 3 + 2],
                             flagVector[num_heads * 3 + 2]);
}

TransformerBlock::~TransformerBlock() = default;


void TransformerBlock::compute(std::size_t seq_len, quant_bit_width *input, quant_bit_width *output) {
    system("m5 resetstats");
    for (int n=0; n<num_heads_; n++){
        std::cout << "Head : " << n << std::endl;
        selfatten[n]->compute(seq_len, input, multihead_out + n * (seq_len * head_hidden_size_));
    }

#ifndef REARRANGE
    Transpose::multihead_transpose(multihead_out, multihead_out_reshape,
                                   seq_len, head_hidden_size_, num_heads_);
    multihead_out = multihead_out_reshape;
#endif

    std::cout << "Condense"  << std::endl;
    condense[0]->compute(seq_len, multihead_out, condense_out);

    std::cout << "Add Norm"  << std::endl;
#ifdef REARRANGE
    addNorm->computeRearranged(input, condense_out);
#else
    addNorm->compute(input, condense_out);
#endif

    system("m5 dumpresetstats");

    std::cout << "Feed Forward 0"  << std::endl;
    feedForward0[0]->compute(seq_len, condense_out, intermediateFF);

    std::cout << "Feed Forward 1"  << std::endl;
    feedForward1[0]->compute(seq_len, intermediateFF, output);

    std::cout << "Add Norm"  << std::endl;
#ifdef REARRANGE
    std::cout << "Add norm rearranged"  << std::endl;
    addNorm->computeRearranged(condense_out, output);
#else
    std::cout << "Add norm TiCSAT"  << std::endl;
    addNorm->compute(condense_out, output);
#endif
    system("m5 dumpresetstats");

//    std::string filename = "kernel.bin";
//#ifdef REARRANGE
//    auto keyBlockWise = new uint32_t [seq_len* input_dim_ >> 2]();
//    blockWise2RowWise(output, keyBlockWise, seq_len, input_dim_ >> 2);
//    write_weight_to_file(filename, keyBlockWise, seq_len, input_dim_ /4);
//#endif
//    uint32_t rearrange[seq_len*input_dim_ >>2];
//    read_weight_from_file(filename, rearrange, seq_len, input_dim_ >>2);
//    unsigned int error_total = 0;
//    for (int elem =0; elem < seq_len*input_dim_ >>2 ; elem++){
//        unsigned int error = (output[elem] - rearrange[elem]);
//        error_total += (error>0 ? error : -error);
//        if (error){
//#ifndef REARRANGE
//            std::cout<< "ERROR: "<< elem << std::endl;
//            printf("%x, %x\n", output[elem] , rearrange[elem]);
//#endif
//        }
//    }
//    std::cout<< "Total Error Output: " << error_total<<std::endl;

}


TransformerBlock::TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                   std::size_t num_heads, std::size_t ff_size, quant_bit_width ** weightVector,
                                   quant_bit_width ** biasVector, quant_bit_width* clsTokenVector,
                                   quant_bit_width* posMatrix) {

    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;
    input_dim_ = input_dim;
    ff_size_ = ff_size;

    addNorm = new AddNormalize(pre_seq_len, D_EMBEDDING, weightVector[0], biasVector[0]);
    patchEmbedding = new Dense(D_EMBEDDING, D_MODEL, weightVector[1], biasVector[1]);
    addNorm2 = new AddNormalize(pre_seq_len, D_MODEL, weightVector[2], biasVector[2]);
    token = new TokenPosEmbedding(posMatrix, clsTokenVector, pre_seq_len, input_dim, D_SEQ+1);


    for (int l=0; l<4; l++){
        transformer_layer_0_addNorm[l] = new AddNormalize((pre_seq_len + 1), D_MODEL, weightVector[l* 17 +3],
                                                          biasVector[l*17 + 3]);

        for (int n =0; n< num_heads; n++){
            selfatten[l*num_heads + n] = new SingleHeadSelfAttn((pre_seq_len+1), input_dim, head_hidden_size, weightVector+l*17+4+n*3);
        }

        condense[l] = new Dense(num_heads* head_hidden_size, input_dim, weightVector[l*17 + num_heads * 3 + 4],
                                biasVector[l*17 + num_heads * 3 + 4]);

        transformer_layer_1_addNorm[l] = new AddNormalize((pre_seq_len + 1), input_dim, weightVector[l*17 + num_heads * 3 + 5],
                                                          biasVector[l*17 + num_heads * 3 + 5]);
        feedForward0[l] = new Dense(input_dim, ff_size, weightVector[l*17 + num_heads * 3+ 6],
                                    biasVector[l*17 + num_heads * 3 + 6]);
        feedForward1[l] = new Dense(ff_size, input_dim, weightVector[l*17 + num_heads * 3 + 7],
                                    biasVector[l*17 + num_heads * 3 + 7]);
    }

    mlp_head_norm = new AddNormalize(1, D_MODEL, weightVector[(NUM_LAYERS-1)*17 + NUM_HEAD *3 + 8 ],
                                     biasVector[(NUM_LAYERS-1)*17 + NUM_HEAD *3 + 8 ]);
    mlp_head_linear = new Dense(D_MODEL, D_MODEL, weightVector[(NUM_LAYERS-1)*17 + NUM_HEAD *3 + 9 ],
                                biasVector[(NUM_LAYERS-1)*17 + NUM_HEAD *3 + 9 ]);
}

void TransformerBlock::computeFixedPoint(std::size_t seq_len, quant_bit_width *input,
                                         quant_bit_width *input_normalized,
                                         quant_bit_width *output, quant_bit_width *intermediate ,
                                         quant_bit_width *qkv) {
    addNorm->normalize(input, input);
    patchEmbedding->compute(seq_len, input, output);
    addNorm2->normalize(output, output);

    token->clsConcatenate(output, input);
    seq_len++;
    token->posEmbedding(input);


    for (int l=0; l< 4; l++){
        transformer_layer_0_addNorm[l]->normalize(input, input_normalized);

        for (int n=0; n<NUM_HEAD; n++){
            std::cout << "Head : " << n << std::endl;
            selfatten[l*NUM_HEAD + n]->compute(input_normalized, output + n * (seq_len * head_hidden_size_), qkv, intermediate);
        }
        Transpose::multihead_transpose(output, intermediate,
                                       seq_len, head_hidden_size_, num_heads_);

        condense[l]->compute(seq_len, intermediate, output);

        transformer_layer_0_addNorm[l]->add(input, output);

        transformer_layer_1_addNorm[l]->normalize(input, input_normalized);
        feedForward0[l]->compute(seq_len, input_normalized, intermediate);
        feedForward0[l]->activation(seq_len*ff_size_, intermediate, intermediate);

        feedForward1[l]->compute(seq_len, intermediate, output);
        transformer_layer_1_addNorm[l]->add(input, output);
    }

    mlp_head_norm->normalize(input, input_normalized);
    mlp_head_linear->compute(1, input_normalized, output);

}
