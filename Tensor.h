#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

struct Tensor
{
    std::vector<float> data;
    int rows;
    int cols;
};

Tensor add(const Tensor &a, const Tensor &b);
Tensor matmul(const Tensor &a, const Tensor &b);
Tensor matmul_optimized(const Tensor &a, const Tensor &b);
void softmax(std::vector<float> &x);
Tensor transpose(const Tensor &a);
Tensor slice_columns(const Tensor &a, int col_start, int col_end);
Tensor concat_columns(const Tensor &a, const Tensor &b);
void relu(std::vector<float> &x);
Tensor feedforward(const Tensor &x, const Tensor &W1, const Tensor &W2);
Tensor layer_norm(const Tensor &x);
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps);
Tensor attention(const Tensor &Q, const Tensor &K, const Tensor &V);
Tensor multi_head_attention(const Tensor &Q, const Tensor &K, const Tensor &V, int num_heads);
Tensor transformer_layer(const Tensor& x, int num_heads,
                             const Tensor& Wq, const Tensor& Wk, const Tensor& Wv,
                             const Tensor& W1, const Tensor& W2);
Tensor stacked_transformer(const Tensor &x, int num_heads, int num_layers,
                           const std::vector<Tensor> &W1_list,
                           const std::vector<Tensor> &W2_list, const std::vector<Tensor> &Wq_list,
                           const std::vector<Tensor> &Wk_list, const std::vector<Tensor> &Wv_list);

#endif
