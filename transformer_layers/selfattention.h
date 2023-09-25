#include "util.h"

#include "dense.h"
#include "softmax.h"
#include "transpose.h"
#include "matmul.h"
#include "../accelerator/smm_gem.h"
#include "../transformer.h"

class SingleHeadSelfAttn{
    public:
        SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim_, std::size_t head_hidden_size,
                           quant_bit_width** weightVector, uint32_t** flagVector, std::size_t , std::size_t);
        SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim_, std::size_t head_hidden_size,
                           quant_bit_width** weightVector);
        ~SingleHeadSelfAttn();
        void compute(quant_bit_width *input, quant_bit_width *output, quant_bit_width *qkv, quant_bit_width *intermediate);

    private:
        Dense* query_layer;
        Dense* key_layer;
        Dense* value_layer;
        Softmax* softmax;

        quant_bit_width* query_layer_out;
        quant_bit_width* key_layer_out;
        quant_bit_width* key_transposed_layer_out;
        quant_bit_width* value_layer_out;
        quant_bit_width* attention_scores;

        std::size_t pre_seq_len_;
        std::size_t head_hidden_size_;
        std::size_t kernel_size_;
        std::size_t max_col_;
};
