#include "Tensor.h"
#include <cmath>

Tensor add(const Tensor &a, const Tensor &b)
{
    Tensor result;
    result.rows = a.rows;
    result.cols = a.cols;
    result.data.resize(result.rows * result.cols);

    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            result.data[i * result.cols + j] = a.data[i * a.cols + j] + b.data[i * b.cols + j];
        }
    }
    return result;
}

Tensor matmul(const Tensor &a, const Tensor &b)
{
    Tensor result;
    result.rows = a.rows;
    result.cols = b.cols;
    result.data.resize(result.rows * result.cols);

    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < b.cols; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < a.cols; k++)
            {
                sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }
            result.data[i * result.cols + j] = sum;
        }
    }
    return result;
}

void softmax(std::vector<float> &x)
{
    for (int i = 0; i < x.size(); i++)
    {
        x[i] = exp(x[i]);
    }
    float summ = 0.0f;
    for (float n : x)
    {
        summ += n;
    }
    for (int i = 0; i < x.size(); i++)
    {
        x[i] = x[i] / summ;
    }
}

Tensor transpose(const Tensor &a)
{
    Tensor result;
    result.rows = a.cols;
    result.cols = a.rows;
    result.data.resize(result.rows * result.cols);

    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            result.data[i * result.cols + j] = a.data[j * a.cols + i];
        }
    }
    return result;
}

Tensor slice_columns(const Tensor &a, int col_start, int col_end)
{
    int new_cols = col_end - col_start;
    Tensor result;
    result.rows = a.rows;
    result.cols = new_cols;
    result.data.resize(result.rows * result.cols);

    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            result.data[i * result.cols + j] = a.data[i * a.cols + (col_start + j)];
        }
    }
    return result;
}

Tensor concat_columns(const Tensor &a, const Tensor &b)
{
    int new_cols = a.cols + b.cols;
    Tensor result;
    result.rows = a.rows;
    result.cols = new_cols;
    result.data.resize(result.rows * result.cols);

    for (int i = 0; i < result.rows; i++)
    {
        std::vector<float> row(result.cols);
        for (int j = 0; j < a.cols; j++)
        {
            row[j] = a.data[i * a.cols + j];
        }
        for (int j = 0; j < b.cols; j++)
        {
            row[a.cols + j] = b.data[i * b.cols + j];
        }
        for (int k = 0; k < result.cols; k++)
        {
            result.data[i * result.cols + k] = row[k];
        }
    }
    return result;
}

void relu(std::vector<float> &x)
{
    for (float &i : x)
    {
        if (i < 0)
        {
            i = 0;
        }
    }
}

Tensor feedforward(const Tensor &x, const Tensor &W1, const Tensor &W2)
{
    Tensor result = matmul(x, W1);
    relu(result.data);
    result = matmul(result, W2);
    return result;
}

Tensor layer_norm(const Tensor &x)
{
    Tensor result = x;
    float epsilon = 1e-5f;

    for (int i = 0; i < result.rows; i++)
    {
        float mean = 0.0f;
        float variance = 0.0f;
        for (int j = 0; j < result.cols; j++)
        {
            mean += result.data[i * result.cols + j];
        }
        mean = mean / result.cols;

        for (int j = 0; j < result.cols; j++)
        {
            variance += std::pow((result.data[i * result.cols + j] - mean), 2) / result.cols;
        }
        for (int j = 0; j < result.cols; j++)
        {
            result.data[i * result.cols + j] = (result.data[i * result.cols + j] - mean) / std::sqrt(variance + epsilon);
        }
    }
    return result;
}

Tensor attention(const Tensor &Q, const Tensor &K, const Tensor &V)
{
    Tensor K_t = transpose(K);
    int d_k = Q.cols;
    Tensor result = matmul(Q, K_t);
    for (int i = 0; i < result.rows * result.cols; i++)
    {
        result.data[i] = result.data[i] / sqrt(d_k);
    }

    for (int i = 0; i < result.rows; i++)
    {
        std::vector<float> row(result.cols);
        for (int j = 0; j < result.cols; j++)
        {
            row[j] = result.data[i * result.cols + j];
        }
        softmax(row);
        for (int j = 0; j < result.cols; j++)
        {
            result.data[i * result.cols + j] = row[j];
        }
    }

    result = matmul(result, V);
    return result;
}

Tensor multi_head_attention(const Tensor &Q, const Tensor &K, const Tensor &V, int num_heads)
{
    int d_model = Q.cols;
    int d_k = d_model / num_heads;

    std::vector<Tensor> outputs(num_heads);

    for (int i = 0; i < num_heads; i++)
    {
        int col_start = i * d_k;
        int col_end = (i + 1) * d_k;
        Tensor Q_sliced = slice_columns(Q, col_start, col_end);
        Tensor K_sliced = slice_columns(K, col_start, col_end);
        Tensor V_sliced = slice_columns(V, col_start, col_end);

        Tensor head_output = attention(Q_sliced, K_sliced, V_sliced);
        outputs[i] = head_output;
    }

    Tensor result = outputs[0];
    for (int i = 1; i < num_heads; i++)
    {
        result = concat_columns(result, outputs[i]);
    }
    return result;
}

Tensor transformer_layer(const Tensor &x, int num_heads, const Tensor &W1, const Tensor &W2)
{
    Tensor attn_out = multi_head_attention(x, x, x, num_heads);
    Tensor x2 = layer_norm(add(x, attn_out));

    Tensor ff_out = feedforward(x2, W1, W2);
    Tensor x3 = layer_norm(add(x2, ff_out));

    return x3;
}

Tensor stacked_transformer(const Tensor &x, int num_heads, int num_layers,
                           const std::vector<Tensor> &W1_list,
                           const std::vector<Tensor> &W2_list)
{
    Tensor current = x;
    for (int i = 0; i < num_layers; i++)
    {
        current = transformer_layer(current, num_heads, W1_list[i], W2_list[i]);
    }
    return current;
}
