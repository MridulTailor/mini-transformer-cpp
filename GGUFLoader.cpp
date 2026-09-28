#include "GGUFLoader.h"
#include "Tensor.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <string>
#include <cstring>

const GGUFTensorInfo *find_tensor(const std::vector<GGUFTensorInfo> &infos, const std::string &name)
{
    for (const auto &info : infos)
    {
        if (info.name == name)
            return &info;
    }
    return nullptr;
}

std::string read_gguf_string(std::ifstream &file)
{
    uint64_t len;
    file.read(reinterpret_cast<char *>(&len), sizeof(len));

    std::string s(len, '\0');
    file.read(&s[0], len);
    return s;
}

float f16_to_f32(uint16_t h)
{
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h & 0x7C00) >> 10;
    uint32_t mantissa = h & 0x03FF;

    uint32_t f32_bits;

    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            f32_bits = sign; // zero
        }
        else
        {
            // subnormal number
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x0400) == 0)
            {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03FF;
            f32_bits = sign | (exponent << 23) | (mantissa << 13);
        }
    }
    else if (exponent == 0x1F)
    {
        f32_bits = sign | 0x7F800000 | (mantissa << 13); // inf/nan
    }
    else
    {
        exponent = exponent - 15 + 127;
        f32_bits = sign | (exponent << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &f32_bits, sizeof(result));
    return result;
}

Tensor load_tensor_data(std::ifstream &file, const GGUFTensorInfo &info, uint64_t data_section_start)
{
    Tensor t;
    if (info.shape.size() >= 2) {
        t.rows = info.shape[1];   // true memory layout: (out x in)
        t.cols = info.shape[0];
    } else {
        t.rows = 1;
        t.cols = info.shape[0];
    }

    uint64_t num_elements = 1;
    for (uint64_t d : info.shape)
        num_elements *= d;

    file.seekg(data_section_start + info.offset);

    t.data.resize(num_elements);

    if (info.dtype == 1)
    { // F16
        for (uint64_t i = 0; i < num_elements; i++)
        {
            uint16_t raw;
            file.read(reinterpret_cast<char *>(&raw), sizeof(raw));
            t.data[i] = f16_to_f32(raw);
        }
    }
    else if (info.dtype == 0)
    { // F32
        file.read(reinterpret_cast<char *>(t.data.data()), num_elements * sizeof(float));
    }
    else
    {
        std::cout << "Unsupported dtype: " << info.dtype << "\n";
    }

    if (info.shape.size() >= 2) {
        return transpose(t);      // now (in x out), ready for x @ W
    }
    return t;
}

template <typename T, typename A>
int arg_max(std::vector<T, A> const& vec) {
  return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}


void load_gguf(const std::string &path)
{
    std::vector<std::string> vocab;
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::cout << "Failed to open file!\n";
        return;
    }

    char magic[4];
    file.read(magic, 4);
    std::cout << "Magic bytes: " << magic[0] << magic[1] << magic[2] << magic[3] << "\n";

    uint32_t version;
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    std::cout << "Version: " << version << "\n";

    uint64_t tensor_count;
    file.read(reinterpret_cast<char *>(&tensor_count), sizeof(tensor_count));
    std::cout << "Tensor count: " << tensor_count << "\n";

    uint64_t metadata_kv_count;
    file.read(reinterpret_cast<char *>(&metadata_kv_count), sizeof(metadata_kv_count));
    std::cout << "Metadata KV count: " << metadata_kv_count << "\n";

    for (uint64_t i = 0; i < metadata_kv_count; i++)
    {
        std::string key = read_gguf_string(file);

        uint32_t type;
        file.read(reinterpret_cast<char *>(&type), sizeof(type));

        std::cout << "Key: " << key << " | Type: " << type << " | Value: ";

        if (type == 8)
        { // STRING
            std::string value = read_gguf_string(file);
            std::cout << value;
        }
        else if (type == 4)
        { // UINT32
            uint32_t value;
            file.read(reinterpret_cast<char *>(&value), sizeof(value));
            std::cout << value;
        }
        else if (type == 9)
        { // ARRAY
            uint32_t element_type;
            file.read(reinterpret_cast<char *>(&element_type), sizeof(element_type));

            uint64_t count;
            file.read(reinterpret_cast<char *>(&count), sizeof(count));

            std::cout << "(array of " << count << " elements, type " << element_type << ")";

            for (uint64_t j = 0; j < count; j++)
            {
                if (element_type == 8)
                {
                    std::string s = read_gguf_string(file);
                    if (key == "tokenizer.ggml.tokens")
                        vocab.push_back(s);
                }
                else
                {
                    uint32_t discard;
                    file.read(reinterpret_cast<char *>(&discard), sizeof(discard));
                }
            }
        }
        else if (type == 6)
        { // FLOAT32
            float value;
            file.read(reinterpret_cast<char *>(&value), sizeof(value));
            std::cout << value;
        }
        else if (type == 5)
        { // INT32
            int32_t value;
            file.read(reinterpret_cast<char *>(&value), sizeof(value));
            std::cout << value;
        }
        else
        {
            std::cout << "(type " << type << " not handled)";
        }

        std::cout << "\n";
    }

    std::vector<GGUFTensorInfo> tensor_infos;

    for (uint64_t i = 0; i < tensor_count; i++)
    {
        GGUFTensorInfo info;

        info.name = read_gguf_string(file);

        uint32_t n_dims;
        file.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));

        for (uint32_t d = 0; d < n_dims; d++)
        {
            uint64_t dim;
            file.read(reinterpret_cast<char *>(&dim), sizeof(dim));
            info.shape.push_back(dim);
        }

        file.read(reinterpret_cast<char *>(&info.dtype), sizeof(info.dtype));
        file.read(reinterpret_cast<char *>(&info.offset), sizeof(info.offset));

        tensor_infos.push_back(info);

        std::cout << "Tensor: " << info.name << " | shape: [";
        for (size_t s = 0; s < info.shape.size(); s++)
        {
            std::cout << info.shape[s];
            if (s + 1 < info.shape.size())
                std::cout << ", ";
        }
        std::cout << "] | dtype: " << info.dtype << " | offset: " << info.offset << "\n";
    }

    uint64_t data_section_start = file.tellg();
    uint64_t alignment = 32; // GGUF default; check "general.alignment" metadata key if present
    data_section_start = ((data_section_start + alignment - 1) / alignment) * alignment;

    Tensor q_weight = load_tensor_data(file, tensor_infos[1], data_section_start); // tensor_infos[1] = blk.0.attn_q.weight
    std::cout << "First few values of blk.0.attn_q.weight: ";
    for (int i = 0; i < 5; i++)
        std::cout << q_weight.data[i] << " ";
    std::cout << "\n";

    std::vector<Tensor> W1_list, W2_list;
    std::vector<Tensor> Wq_list, Wk_list, Wv_list;  // NEW

    for (int layer = 0; layer < 2; layer++) {
        std::string up_name = "blk." + std::to_string(layer) + ".ffn_up.weight";
        std::string down_name = "blk." + std::to_string(layer) + ".ffn_down.weight";
        std::string q_name = "blk." + std::to_string(layer) + ".attn_q.weight";     
        std::string k_name = "blk." + std::to_string(layer) + ".attn_k.weight";     
        std::string v_name = "blk." + std::to_string(layer) + ".attn_v.weight";     

        const GGUFTensorInfo* up_info = find_tensor(tensor_infos, up_name);
        const GGUFTensorInfo* down_info = find_tensor(tensor_infos, down_name);
        const GGUFTensorInfo* q_info = find_tensor(tensor_infos, q_name);           
        const GGUFTensorInfo* k_info = find_tensor(tensor_infos, k_name);          
        const GGUFTensorInfo* v_info = find_tensor(tensor_infos, v_name);         

        Tensor W1 = load_tensor_data(file, *up_info, data_section_start);
        Tensor W2 = load_tensor_data(file, *down_info, data_section_start);
        Tensor Wq = load_tensor_data(file, *q_info, data_section_start);        
        Tensor Wk = load_tensor_data(file, *k_info, data_section_start);          
        Tensor Wv = load_tensor_data(file, *v_info, data_section_start);         

        W1_list.push_back(W1);
        W2_list.push_back(W2);
        Wq_list.push_back(Wq); 
        Wk_list.push_back(Wk); 
        Wv_list.push_back(Wv); 
    }

    const GGUFTensorInfo *embd_info = find_tensor(tensor_infos, "token_embd.weight");
    Tensor full_embd = load_tensor_data(file, *embd_info, data_section_start);

    int token_id = 1;
    Tensor x;
    x.rows = 1;
    x.cols = 16;
    x.data.resize(16);
    for (int i = 0; i < 16; i++)
    {
        x.data[i] = full_embd.data[i * 32000 + token_id];
    }

    Tensor result = stacked_transformer(x, 4, 2, W1_list, W2_list, Wq_list, Wk_list, Wv_list);
    const GGUFTensorInfo* out_info = find_tensor(tensor_infos, "output.weight");
    Tensor W_out = load_tensor_data(file, *out_info, data_section_start);  // 16 x 32000
    Tensor result_normed = layer_norm(result);
    Tensor logits = matmul(result_normed, W_out);
    softmax(logits.data);
    int best = arg_max(logits.data);
    float total = 0.0f;
    for (float p : logits.data) total += p;
    std::cout << "Predicted token: \"" << vocab[best] << "\"\n";
    std::cout << "Predicted token id: " << best << "\n";
    std::cout << "Probability: " << logits.data[best] << "\n";
    std::cout << "Sum of probabilities: " << total << "\n";
    for (float v : result.data)
        std::cout << v << " ";
    std::cout << "\n";
}
