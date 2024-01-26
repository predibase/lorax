#ifndef _safetensors_h
#define _safetensors_h

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cstdio>

class STFile
{
public:
    STFile(const char* filename);
    ~STFile();

    void load
    (
        torch::Tensor target,
        size_t offset,
        size_t length,
        bool gpu
    );

    int file_descriptor;
    size_t filesize;
    size_t padded_size;
    size_t block_size;
    void* aligned_buffer;
};

void safetensors_pinned_buffer();
void safetensors_free_pinned_buffer();

uintptr_t safetensors_open(const char* filename);
void safetensors_close(uintptr_t handle);

void safetensors_load
(
    uintptr_t handle,
    torch::Tensor target,
    size_t offset,
    size_t length
);

#endif