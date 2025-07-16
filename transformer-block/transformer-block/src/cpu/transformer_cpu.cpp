#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include "operations_cpu.h"
#include "transformer_common.h"

void transformer_block_cpu(const std::vector<float>& input_embeddings,
                            const std::vector<float>& weights_q,
                            const std::vector<float>& weights_k,
                            const std::vector<float>& weights_v,
                            const std::vector<float>& ff_weights1,
                            const std::vector<float>& ff_weights2,
                            std::vector<float>& output,
                            int batch_size, int seq_length, int d_model) {
    // Step 1: Compute Q, K, V
    std::vector<float> Q(batch_size * d_model);
    std::vector<float> K(batch_size * d_model);
    std::vector<float> V(batch_size * d_model);
    
    matmul(input_embeddings, weights_q, Q, batch_size, seq_length, d_model);
    matmul(input_embeddings, weights_k, K, batch_size, seq_length, d_model);
    matmul(input_embeddings, weights_v, V, batch_size, seq_length, d_model);

    // Step 2: Compute attention scores
    std::vector<float> scores(batch_size * seq_length * seq_length);
    matmul_transpose(Q, K, scores, batch_size, seq_length, d_model);

    // Step 3: Apply softmax to scores
    std::vector<float> attention_scores(batch_size * seq_length * seq_length);
    softmax(scores, attention_scores, batch_size, seq_length);

    // Step 4: Compute output O
    std::vector<float> O(batch_size * d_model);
    matmul(attention_scores, V, O, batch_size, seq_length, d_model);

    // Step 5: Residual connection
    for (int i = 0; i < O.size(); ++i) {
        O[i] += input_embeddings[i];
    }

    // Step 6: Layer normalization
    std::vector<float> normalized_output(batch_size * d_model);
    layer_norm(O, normalized_output, batch_size, d_model);

    // Step 7: Feed Forward Network
    std::vector<float> ff_output(batch_size * d_model);
    matmul(normalized_output, ff_weights1, ff_output, batch_size, d_model, d_model);
    relu(ff_output, ff_output, batch_size * d_model);
    matmul(ff_output, ff_weights2, output, batch_size, d_model, d_model);
}

int main() {
    // Load data from binary files
    std::vector<float> input_embeddings = load_binary<float>("data/input_embeddings.bin");
    std::vector<float> weights_q = load_binary<float>("data/weights_q.bin");
    std::vector<float> weights_k = load_binary<float>("data/weights_k.bin");
    std::vector<float> weights_v = load_binary<float>("data/weights_v.bin");
    std::vector<float> ff_weights1 = load_binary<float>("data/ff_weights1.bin");
    std::vector<float> ff_weights2 = load_binary<float>("data/ff_weights2.bin");

    int batch_size = 128;
    int seq_length = 256;
    int d_model = 768;

    std::vector<float> output(batch_size * d_model);

    // Execute the transformer block
    transformer_block_cpu(input_embeddings, weights_q, weights_k, weights_v, ff_weights1, ff_weights2, output, batch_size, seq_length, d_model);

    // Save output to binary file
    save_binary(output, "out/output.bin");

    return 0;
}