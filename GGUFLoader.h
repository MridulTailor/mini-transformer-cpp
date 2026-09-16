#ifndef GGUFLOADER_H
#define GGUFLOADER_H
#include <iostream>
#include <string>

struct GGUFTensorInfo
{
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t dtype;
    uint64_t offset;
};

void load_gguf(const std::string &path);
const GGUFTensorInfo *find_tensor(const std::vector<GGUFTensorInfo> &infos, const std::string &name);

#endif
