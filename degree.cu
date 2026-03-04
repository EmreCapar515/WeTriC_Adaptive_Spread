#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

/* ================= DEVICE HELPERS ================= */

__device__ __forceinline__
unsigned int get_spread_for_degree(unsigned int degree) {
    if (degree < 32)   return 0;
    if (degree < 128)  return 1;
    if (degree < 1024) return 2;
    return 3;
}

/* ================= KERNELS ================= */

__global__ void degree_kernel(
    const int* src,
    const int* dst,
    unsigned int* degree,
    int num_edges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        atomicAdd(&degree[src[tid]], 1);
        atomicAdd(&degree[dst[tid]], 1);
    }
}

__global__ void degree_bucket_kernel(
    const unsigned int* degree,
    unsigned int* buckets,
    int num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        unsigned int d = degree[tid];
        unsigned int bucket = get_spread_for_degree(d);
        atomicAdd(&buckets[bucket], 1);
    }
}

/* ================= MTX PARSER ================= */

void read_mtx(
    const std::string& filename,
    int& num_vertices,
    std::vector<int>& src,
    std::vector<int>& dst
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file\n";
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    int rows, cols, nnz;
    sscanf(line.c_str(), "%d %d %d", &rows, &cols, &nnz);
    num_vertices = rows;

    int r, c;
    double val;
    while (file >> r >> c >> val) {
        r--; c--;
        if (r != c) {
            src.push_back(r);
            dst.push_back(c);
        }
    }
}

/* ================= MAIN ================= */

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " graph.mtx\n";
        return 1;
    }

    std::vector<int> h_src, h_dst;
    int num_vertices;

    read_mtx(argv[1], num_vertices, h_src, h_dst);
    int num_edges = h_src.size();

    int *d_src, *d_dst;
    unsigned int *d_degree, *d_buckets;

    CUDA_CHECK(cudaMalloc(&d_src, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dst, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_degree, num_vertices * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_buckets, 4 * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst, h_dst.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_degree, 0, num_vertices * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_buckets, 0, 4 * sizeof(unsigned int)));

    int block = 256;

    degree_kernel<<<(num_edges + block - 1) / block, block>>>(
        d_src, d_dst, d_degree, num_edges
    );

    degree_bucket_kernel<<<(num_vertices + block - 1) / block, block>>>(
        d_degree, d_buckets, num_vertices
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned int h_buckets[4];
    CUDA_CHECK(cudaMemcpy(h_buckets, d_buckets,
                          4 * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    /* ============ OUTPUT ============ */

    std::cout << "Degree < 32   : " << h_buckets[0] << "\n";
    std::cout << "Degree < 128  : " << h_buckets[1] << "\n";
    std::cout << "Degree < 1024 : " << h_buckets[2] << "\n";
    std::cout << "Degree >=1024 : " << h_buckets[3] << "\n";

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_degree);
    cudaFree(d_buckets);

    return 0;
}
