#include "safetensors.h"

#include <c10/cuda/CUDAGuard.h>
#include "util.h"

#ifdef __linux__
#include <aio.h>
#include <atomic>
#include <thread>
#include <limits>
#include <cerrno>
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#endif

#define MAX_BLOCK_SIZE (128*1024)
#define MAX_PAGES 4
#define PAGESIZE (16*1024*1024)
#define Q_DEPTH 1
#define PINNED_MEMORY (MAX_PAGES * PAGESIZE)

//#define ST_DEBUG

void* pinned_buffer = nullptr;
void* aligned_buffer = nullptr;

struct STPage
{
public:
    int file_descriptor;
    #ifdef __linux__
    size_t file_a;
    size_t file_b;
    long access;
    std::atomic<int> locks;
    char* ptr;
    #endif
};

STPage pages[MAX_PAGES];
long serial = 1;

void safetensors_pinned_buffer()
{
    #ifdef __linux__

    if (pinned_buffer) return;
    cudaMallocHost((void**) &pinned_buffer, PINNED_MEMORY + MAX_BLOCK_SIZE);
    TORCH_CHECK(pinned_buffer, "Unable to allocate pinned memory");
    aligned_buffer = (void*) ((char *)(((uintptr_t)pinned_buffer + MAX_BLOCK_SIZE - 1) & ~(MAX_BLOCK_SIZE - 1)));

    for (int i = 0; i < MAX_PAGES; i++)
    {
        pages[i].file_descriptor = -1;
        pages[i].file_a = 0;
        pages[i].file_b = 0;
        pages[i].access = -1;
        pages[i].locks.store(0);
        pages[i].ptr = ((char*) aligned_buffer) + i * PAGESIZE;
    }

    #endif
}

void safetensors_free_pinned_buffer()
{
    #ifdef __linux__

    if (!pinned_buffer) return;
    cudaFreeHost((void*) pinned_buffer);
    pinned_buffer = nullptr;
    aligned_buffer = nullptr;

    #endif
}

STPage* get_cache_page(size_t file_descriptor, size_t block_size, size_t filesize, size_t file_a, size_t file_b)
{
    #ifdef __linux__

    #ifdef ST_DEBUG
    printf("-- get cache page\n");
    DBGX3(file_descriptor, file_a, file_b);
    DBGX2(block_size, filesize);
    #endif

    // Find existing page in cache

    for (int i = 0; i < MAX_PAGES; i++)
    {
        if (pages[i].file_descriptor == file_descriptor &&
            pages[i].file_a == file_a &&
            pages[i].file_b == file_b)
        {
            pages[i].access = serial++;
            return &pages[i];
        }
    }

    // Find page to evict

    int oldest_i = -1;
    long oldest = std::numeric_limits<long>::max();

    while (oldest_i == -1)
    {
        for (int i = 0; i < MAX_PAGES; i++)
        {
            if (pages[i].locks.load() > 0) continue;
            if (pages[i].access < oldest)
            {
                oldest_i = i;
                oldest = pages[i].access;
            }
        }
    }

    #ifdef ST_DEBUG
    printf("-- evict page\n");
    DBGX(oldest_i);
    #endif

    // Load page

    #ifdef ST_DEBUG
    printf("-- load page\n");
    DBGX3(file_descriptor, file_a, file_b);
    #endif

    int p = oldest_i;
    pages[p].access = serial++;
    pages[p].file_a = file_a;
    pages[p].file_b = file_b;
    pages[p].file_descriptor = file_descriptor;

    aiocb aiocb_list[Q_DEPTH];
    size_t read_lens[Q_DEPTH];
    int num_chunks = 0;

    size_t q_chunk = PAGESIZE / Q_DEPTH;
    size_t q_a = file_a;
    char* page_ptr = pages[p].ptr;

    for (int i = 0; i < Q_DEPTH; ++i)
    {
        size_t q_b = q_a + q_chunk;
        if (q_b > filesize) q_b = filesize;

        size_t read_len = q_b - q_a;
        read_lens[i] = read_len;
        //read_len = DIVIDE(read_len, 2 * block_size) * 2 * block_size;
        read_len = q_chunk;

        memset(&aiocb_list[i], 0, sizeof(aiocb));
        aiocb_list[i].aio_fildes = file_descriptor;
        aiocb_list[i].aio_buf = page_ptr;
        aiocb_list[i].aio_nbytes = read_len;
        aiocb_list[i].aio_offset = q_a;

        #ifdef ST_DEBUG
        DBGX3(q_a, q_b, read_len);
        DBGX2(filesize, read_lens[i]);
        #endif

        aio_read(&aiocb_list[i]);
        num_chunks++;

        if (q_b >= filesize) break;

        page_ptr += q_chunk;
        q_a += q_chunk;
    }

    q_a = file_a;

    for (int i = 0; i < num_chunks; ++i)
    {
        struct aiocb *aiocb_active[1];
        aiocb_active[0] = &aiocb_list[i];
        aio_suspend(aiocb_active, 1, NULL);

        int err = aio_error(&aiocb_list[i]);

        #ifdef ST_DEBUG
        DBGX(err);
        #endif

        TORCH_CHECK(err == 0, "Async read error (1)");

        ssize_t bytes_read = aio_return(&aiocb_list[i]);

        #ifdef ST_DEBUG
        DBGX2(bytes_read, read_lens[i]);
        #endif

        TORCH_CHECK(bytes_read == read_lens[i], "Async read error (2)");
    }

    return &pages[p];

    #else
    TORCH_CHECK(false, "fasttensors only supported on Linux");
    return NULL;
    #endif
}

uintptr_t safetensors_open(const char* filename)
{
    #ifdef __linux__

    STFile* f = new STFile(filename);
    return reinterpret_cast<uintptr_t> (f);


    #else
    TORCH_CHECK(false, "fasttensors only supported on Linux");
    return 0;
    #endif
}

void safetensors_close(uintptr_t handle)
{
    #ifdef __linux__

    STFile* f = reinterpret_cast<STFile*> (handle);
    delete f;

    #else
    TORCH_CHECK(false, "fasttensors only supported on Linux");
    #endif
}

STFile::STFile(const char* filename)
{
    #ifdef __linux__

    file_descriptor = open(filename, O_RDONLY | O_DIRECT);
    TORCH_CHECK(file_descriptor != -1, "Safetensors file I/O error");

    struct stat sb;
    auto res = fstat(file_descriptor, &sb);
    TORCH_CHECK(res != -1, "Safetensors fstat failed");
    filesize = sb.st_size;
    block_size = sb.st_blksize;
    padded_size = DIVIDE(filesize, block_size) * block_size;
    TORCH_CHECK(block_size <= MAX_BLOCK_SIZE, "Block size too large")

    #else
    TORCH_CHECK(false, "fasttensors only supported on Linux");
    #endif
}

STFile::~STFile()
{
    #ifdef __linux__

    close(file_descriptor);

    #else
    TORCH_CHECK(false, "fasttensors only supported on Linux");
    #endif
}

void dec_lock(cudaStream_t stream, cudaError_t status, void *user_data)
{
    #ifdef __linux__
    STPage* p = (STPage*) user_data;
    p->locks--;
    #endif
}

void STFile::load
(
    torch::Tensor target,
    size_t offset,
    size_t length,
    bool gpu
)
{
    #ifdef __linux__

    safetensors_pinned_buffer();

    #ifdef ST_DEBUG
    printf("-- load tensor\n");
    DBGX2(offset, length);
    DBGI(length);
    #endif

    // Get cache pages

    size_t file_b = offset / PAGESIZE * PAGESIZE;
    size_t file_c = DIVIDE(offset + length, PAGESIZE) * PAGESIZE;

    // Loop over pages

    size_t file_a = file_b;
    size_t tensor_offset = 0;

    while (tensor_offset < length)
    {
        file_a = file_b;
        file_b += PAGESIZE;

        STPage* page = get_cache_page(file_descriptor, block_size, filesize, file_a, file_b);
        ssize_t left = offset - file_a;
        if (left < 0) left = 0;
        ssize_t right = offset + length - file_a;
        if (right > PAGESIZE) right = PAGESIZE;
        ssize_t copy_len = right - left;

        #ifdef ST_DEBUG
        printf("-- copy chunk\n");
        DBGX3(left, right, copy_len);
        DBGX(tensor_offset);
        DBGI(copy_len);
        #endif

        char* src = page->ptr + left;
        char* dst = ((char*) target.data_ptr()) + tensor_offset;

        if (gpu)
        {
            page->locks++;
            cudaMemcpyAsync(dst, src, copy_len, cudaMemcpyHostToDevice);
            cudaStreamAddCallback(NULL, dec_lock, (void*) page, 0);
        }
        else
        {
            memcpy(dst, src, copy_len);
        }

        //cudaDeviceSynchronize();

        tensor_offset += copy_len;
    }

    #else
    TORCH_CHECK(false, "fasttensors only supported on Linux");
    #endif
}

void safetensors_load
(
    uintptr_t handle,
    torch::Tensor target,
    size_t offset,
    size_t length
)
{
    #ifdef __linux__

    STFile* f = reinterpret_cast<STFile*> (handle);
    c10::optional<torch::Device> device = torch::device_of(target);

    if (device.has_value() && device->type() == torch::kCPU)
    {
        f->load(target, offset, length, false);
    }
    else
    {
        const at::cuda::OptionalCUDAGuard device_guard(device);
        f->load(target, offset, length, true);
    }

    #else
    TORCH_CHECK(false, "fasttensors only supported on Linux");
    #endif
}

