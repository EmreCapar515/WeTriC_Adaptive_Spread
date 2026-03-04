/* Wedge-Parallel Triangle Counting with Adaptive Spread
 * Original authors: Jeffrey Spaan, Kuan-Hsun Chen, David Bader, Ana-Lucia Varbanescu
 * Adaptive spread additions and refactoring: Emre + Masud
 *
 * Key features:
 *  - Wedge-parallel kernel with cooperation.
 *  - Adjacency bitmap for dense tail vertices.
 *  - Adaptive spread autotuning to select the fastest wedges/thread.
 *  - Uses native CUDA atomicAdd (no overload issues).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>
#include <math.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_sort.cuh>

#define CHECK_BOUNDS 1
#define RESET_DEVICE 0
#define BINSEARCH_CONSTANT 1

#define BINSEARCH_CONSTANT_LEVELS 12
#define BINSEARCH_CONSTANT_CACHE_SIZE ((1 << BINSEARCH_CONSTANT_LEVELS) - 1)

#define UINT_t   uint32_t
#define INT_t    int32_t
#define ULONG_t  uint64_t

// For adjacency bitmap (32 bits per word)
#define UINT_WIDTH 32

#define max2(a,b) ((a)>(b)?(a):(b))
#define min2(a,b) ((a)<(b)?(a):(b))

static struct timeval tp;
static struct timezone tzp;

#define get_seconds() (gettimeofday(&tp, &tzp), (double)tp.tv_sec + (double)tp.tv_usec / 1000000.0)

#define checkCudaErrors(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

enum preprocess_t { PREPROCESS_CPU = 0, PREPROCESS_GPU, PREPROCESS_GPU_CONSTRAINED };

typedef struct {
  UINT_t   numVertices;
  UINT_t   numEdges;
  UINT_t*  rowPtr;
  UINT_t*  colInd;
} GRAPH_TYPE;

typedef struct {
  UINT_t src;
  UINT_t dst;
} edge_t;

typedef struct {
  UINT_t id;
  UINT_t *edges;
  UINT_t num_edges;
} preprocess_vertex_t;

typedef struct {
  double copy;
  double exec;
} GPU_time;

/*********
 *  GPU  *
 *********/

#if BINSEARCH_CONSTANT
__constant__ ULONG_t c_binary_search_cache[BINSEARCH_CONSTANT_CACHE_SIZE];
#endif

__device__ INT_t linear_search_GPU(const UINT_t* list, const UINT_t start, const UINT_t end, const UINT_t target) {
  for (UINT_t i=start; i<end; i++) {
    if (list[i] == target)      return i;
    else if (list[i] > target)  break;
  }
  return -1;
}

__device__ INT_t binary_search_GPU(const UINT_t* list, const UINT_t start, const UINT_t end, const UINT_t target) {
  UINT_t s=start, e=end, mid;
  while (s < e) {
    mid = (s + e) >> 1;
    if (list[mid] == target) return mid;
    if (list[mid] < target)  s = mid + 1;
    else                     e = mid;
  }
  return -1;
}

__device__ UINT_t binary_search_closest_ULONG_GPU(const ULONG_t* list, const UINT_t start, const UINT_t end, const ULONG_t target) {
  UINT_t s=start, e=end, mid;
  while (s < e) {
    mid = (s + e) >> 1;
    if (list[mid] < target + 1) s = mid + 1;
    else                        e = mid;
  }
  return max2(start, (s > 0) ? s - 1 : 0);
}

#if BINSEARCH_CONSTANT
__device__ UINT_t binary_search_closest_ULONG_constant_GPU(const ULONG_t *list, const UINT_t start, const UINT_t end, const ULONG_t target) {
  ULONG_t mid;
  UINT_t g_s = start, g_e = end, g_mid;
  UINT_t c_index = 0;

#pragma unroll
  for (UINT_t iter=0; iter<BINSEARCH_CONSTANT_LEVELS; iter++) {
    mid = c_binary_search_cache[c_index];
    g_mid = (g_s + g_e) >> 1;

    c_index = c_index * 2 + 1;
    if (mid < target + 1) {
      c_index += 1;
      g_s = g_mid + 1;
    } else {
      g_e = g_mid;
    }
  }
  g_s = max2(start, (g_s > 0) ? g_s - 1 : 0);
  return binary_search_closest_ULONG_GPU(list, g_s, g_e, target);
}
#endif

__global__ void tc_GPU_kernel(const UINT_t *g_Ap, const UINT_t *g_Ai,
                              const ULONG_t *g_wedgeSum, const ULONG_t wedgeSum_total,
                              const UINT_t num_vertices,
                              ULONG_t *g_total_count,
                              const UINT_t spread,
                              const UINT_t *g_adjacency_matrix,
                              const UINT_t adjacency_matrix_len,
                              const ULONG_t adjacency_matrix_size) {
  const ULONG_t i_start = ((ULONG_t) blockIdx.x * blockDim.x + threadIdx.x) * spread;

  extern __shared__ UINT_t sdata[];
  UINT_t *shared_src   = sdata;                              // [blockDim.x * spread]
  UINT_t *shared_dst   = &sdata[blockDim.x * spread];        // [blockDim.x * spread]
  UINT_t *shared_count = &sdata[2 * blockDim.x * spread];    // [1]

  UINT_t v=0, w=0, u=0, vb=0, ve=0, d_v=0, w_i=0, u_i=0, wedges=0, i_v=0;
  UINT_t s_i = threadIdx.x * spread;

  // Materialize this thread's wedge batch into shared memory
  for (ULONG_t i = i_start; i < min2(i_start + spread, wedgeSum_total); i++, i_v++, s_i++) {
    if (i == i_start) {
#if BINSEARCH_CONSTANT
      v = binary_search_closest_ULONG_constant_GPU(g_wedgeSum, 0, num_vertices, i_start);
#else
      v = binary_search_closest_ULONG_GPU(g_wedgeSum, 0, num_vertices, i_start);
#endif
      vb = g_Ap[v];
      ve = g_Ap[v+1];
      d_v = ve - vb;
      wedges = (d_v * (d_v - 1)) >> 1;
      i_v = i_start - g_wedgeSum[v];

      // Map linear wedge index i_v to (w_i, u_i) in upper triangular coordinates
      // Numerically stable tweak inside sqrt to avoid tiny negative due to fp rounding.
      w_i = d_v - 2 - (UINT_t)(sqrt((double)(wedges - i_v) * 2.0 - 1.75) - 0.5);
      u_i = i_v + w_i + 1 - wedges + (((d_v - w_i) * ((d_v - w_i) - 1)) >> 1);

      w = g_Ai[vb + w_i];
    } else if (i_v >= wedges) {
      // Advance to next vertex with degree >= 2
      do {
        v++;
        vb = ve;
        ve = g_Ap[v+1];
        d_v = ve - vb;
      } while (d_v < 2 && v + 1 < num_vertices);

      wedges = (d_v * (d_v - 1)) >> 1;
      i_v = 0;
      w_i = 0;
      u_i = 1;
      w = g_Ai[vb];
    } else {
      // Advance inside current vertex's wedge grid
      u_i++;
      if (u_i >= d_v) {
        w_i++;
        u_i = w_i + 1;
        w = g_Ai[vb + w_i];
      }
    }

    shared_src[s_i] = w;
    shared_dst[s_i] = g_Ai[vb + u_i];
  }

  if (threadIdx.x == 0) *shared_count = 0;
  __syncthreads();

  // Cooperation: transposed iteration to share N(w) accesses among adjacent threads
  for (s_i = threadIdx.x; s_i < (blockDim.x * spread); s_i += blockDim.x) {
#if CHECK_BOUNDS
    if (s_i >= (wedgeSum_total - (((ULONG_t) blockIdx.x * blockDim.x) * spread))) break;
#endif
    w = shared_src[s_i];
    u = shared_dst[s_i];

    // If w in adjacency bitmap band, probe bitmap; otherwise CSR search
    if (w >= (max2(num_vertices, adjacency_matrix_len) - adjacency_matrix_len)) {
      ULONG_t adjacency_i = (adjacency_matrix_size
                           - ((((ULONG_t)(num_vertices - w)) * (ULONG_t)((num_vertices - w) - 1)) >> 1))
                           + (ULONG_t)(u - w - 1);
      bool found = (g_adjacency_matrix[adjacency_i >> 5] & (1u << (adjacency_i & 31))) > 0;
      if (found) atomicAdd(shared_count, 1);
    } else {
      UINT_t wb = g_Ap[w];
      UINT_t we = g_Ap[w+1];
      if (we - wb < 2) {
        if (linear_search_GPU(g_Ai, wb, we, u) >= 0) atomicAdd(shared_count, 1);
      } else {
        if (binary_search_GPU(g_Ai, wb, we, u) >= 0) atomicAdd(shared_count, 1);
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0)
    atomicAdd((unsigned long long int *) g_total_count, (unsigned long long int)(*shared_count));
}

/*********
 *  CPU  *
 *********/

static void assert_malloc(const void *ptr) {
  if (ptr == NULL) {
    fprintf(stderr,"ERROR: failed to allocate host memory.\n");
    exit(EXIT_FAILURE);
  }
}

void build_binary_search_cache(ULONG_t *src, ULONG_t *cache, UINT_t level, UINT_t max_level, UINT_t i, UINT_t s, UINT_t e) {
  if (level < max_level) {
    UINT_t mid = (s + e) / 2;
    cache[i] = src[mid];
    build_binary_search_cache(src, cache, level+1, max_level, i*2 + 1, s, mid);
    build_binary_search_cache(src, cache, level+1, max_level, i*2 + 2, mid+1, e);
  }
}

static inline size_t smem_usage_bytes(UINT_t num_threads, UINT_t spread) {
  // two vectors of size (num_threads * spread) plus one scalar counter
  return (2ULL * num_threads * spread + 1ULL) * sizeof(UINT_t);
}

ULONG_t tc_GPU(const GRAPH_TYPE *graph, UINT_t spread, UINT_t adjacency_matrix_len, GPU_time *t) {
  UINT_t *d_Ap = nullptr;
  UINT_t *d_Ai = nullptr;
  ULONG_t *d_wedgeSum = nullptr;
  UINT_t *d_adjacency_matrix = nullptr;
  ULONG_t *d_total_count = nullptr;

  cudaEvent_t GPU_copy_start, GPU_copy_stop, GPU_exec_start, GPU_exec_stop;
  float GPU_copy_elapsed=0.f, GPU_exec_elapsed=0.f;
  checkCudaErrors(cudaEventCreate(&GPU_copy_start));
  checkCudaErrors(cudaEventCreate(&GPU_copy_stop));
  checkCudaErrors(cudaEventCreate(&GPU_exec_start));
  checkCudaErrors(cudaEventCreate(&GPU_exec_stop));

  // Build adjacency bitmap for last 'adjacency_matrix_len' vertices (upper triangular)
  ULONG_t adjacency_matrix_size = (((ULONG_t) adjacency_matrix_len) * ((ULONG_t) adjacency_matrix_len - 1)) / 2;
  UINT_t *h_adjacency_matrix = (UINT_t *) calloc(adjacency_matrix_size / UINT_WIDTH + 1, sizeof(UINT_t));
  assert_malloc(h_adjacency_matrix);

  for (UINT_t v = graph->numVertices - min2(graph->numVertices, adjacency_matrix_len); v < graph->numVertices; v++) {
    for (UINT_t i = graph->rowPtr[v]; i < graph->rowPtr[v+1]; i++) {
      UINT_t w = graph->colInd[i];
      if (w <= v) continue; // only upper triangular edges
      ULONG_t adjacency_i = (adjacency_matrix_size
                           - ((((ULONG_t)(graph->numVertices - v)) * (ULONG_t)((graph->numVertices - v) - 1)) / 2))
                           + (ULONG_t)(w - v - 1);
      h_adjacency_matrix[adjacency_i >> 5] |= (1u << (adjacency_i & 31));
    }
  }

  // Wedge prefix sums
  ULONG_t *h_wedgeSum = (ULONG_t *) malloc((graph->numVertices + 1) * sizeof(ULONG_t));
  assert_malloc(h_wedgeSum);
  h_wedgeSum[0] = 0;
  for (UINT_t v = 0; v < graph->numVertices; v++) {
    UINT_t d_v = graph->rowPtr[v+1] - graph->rowPtr[v];
    h_wedgeSum[v+1] = h_wedgeSum[v] + (d_v < 2 ? 0 : ((ULONG_t)d_v * (d_v - 1)) / 2);
  }
  ULONG_t wedgeSum_total = h_wedgeSum[graph->numVertices];

#if BINSEARCH_CONSTANT
  ULONG_t *h_wedgeSum_cache = (ULONG_t *) malloc(BINSEARCH_CONSTANT_CACHE_SIZE * sizeof(ULONG_t));
  assert_malloc(h_wedgeSum_cache);
  build_binary_search_cache(h_wedgeSum, h_wedgeSum_cache, 0, BINSEARCH_CONSTANT_LEVELS, 0, 0, graph->numVertices);
#endif

  // Device allocation + H2D copies
  checkCudaErrors(cudaEventRecord(GPU_copy_start));

  checkCudaErrors(cudaMalloc((void **)&d_Ap, (graph->numVertices + 1) * sizeof(UINT_t)));
  checkCudaErrors(cudaMalloc((void **)&d_Ai, graph->numEdges * sizeof(UINT_t)));
  checkCudaErrors(cudaMalloc((void **)&d_wedgeSum, (graph->numVertices + 1) * sizeof(ULONG_t)));
  checkCudaErrors(cudaMalloc((void **)&d_adjacency_matrix, (adjacency_matrix_size / 32 + 1) * sizeof(UINT_t)));
  checkCudaErrors(cudaMalloc((void **)&d_total_count, 1 * sizeof(ULONG_t)));

  checkCudaErrors(cudaMemcpy(d_Ap, graph->rowPtr, (graph->numVertices + 1) * sizeof(UINT_t), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Ai, graph->colInd, graph->numEdges * sizeof(UINT_t), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_wedgeSum, h_wedgeSum, (graph->numVertices + 1) * sizeof(ULONG_t), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_adjacency_matrix, h_adjacency_matrix, (adjacency_matrix_size / 32 + 1) * sizeof(UINT_t), cudaMemcpyHostToDevice));

#if BINSEARCH_CONSTANT
  checkCudaErrors(cudaMemcpyToSymbol(c_binary_search_cache, h_wedgeSum_cache, BINSEARCH_CONSTANT_CACHE_SIZE * sizeof(ULONG_t)));
#endif

  checkCudaErrors(cudaMemset(d_total_count, 0, 1 * sizeof(ULONG_t)));

  checkCudaErrors(cudaEventRecord(GPU_copy_stop));
  checkCudaErrors(cudaEventSynchronize(GPU_copy_stop));
  checkCudaErrors(cudaEventElapsedTime(&GPU_copy_elapsed, GPU_copy_start, GPU_copy_stop));
  t->copy += GPU_copy_elapsed;

  // Kernel launch configuration
  const UINT_t num_threads = 128;
  ULONG_t tiles = (wedgeSum_total + (ULONG_t)(spread * num_threads) - 1ULL) / (ULONG_t)(spread * num_threads);
  ULONG_t num_blocks = max2((ULONG_t)1, tiles);

  if (num_blocks > (((ULONG_t)1 << 31) - 1)) {
    fprintf(stderr, "ERROR: maximum grid size reached.\n");
    exit(EXIT_FAILURE);
  }

  dim3 grid(num_blocks, 1, 1);
  dim3 threads(num_threads, 1, 1);

  size_t smem_size = smem_usage_bytes(num_threads, spread);
  cudaDeviceProp prop{};
  checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
  if (smem_size > prop.sharedMemPerBlock) {
    fprintf(stderr, "ERROR: requested shared memory (%zu) exceeds device limit (%zu). Lower spread.\n",
            smem_size, (size_t)prop.sharedMemPerBlock);
    exit(EXIT_FAILURE);
  }

  // Execute
  checkCudaErrors(cudaEventRecord(GPU_exec_start));
  tc_GPU_kernel<<<grid, threads, (int)smem_size>>>(d_Ap, d_Ai, d_wedgeSum, wedgeSum_total,
                                                   graph->numVertices, d_total_count,
                                                   spread, d_adjacency_matrix,
                                                   adjacency_matrix_len, adjacency_matrix_size);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaEventRecord(GPU_exec_stop));
  checkCudaErrors(cudaEventSynchronize(GPU_exec_stop));
  checkCudaErrors(cudaEventElapsedTime(&GPU_exec_elapsed, GPU_exec_start, GPU_exec_stop));
  t->exec += GPU_exec_elapsed;

  ULONG_t h_total_count = 0;
  checkCudaErrors(cudaMemcpy(&h_total_count, d_total_count, 1 * sizeof(ULONG_t), cudaMemcpyDeviceToHost));

  // Cleanup
  checkCudaErrors(cudaFree(d_Ap));
  checkCudaErrors(cudaFree(d_Ai));
  checkCudaErrors(cudaFree(d_wedgeSum));
  checkCudaErrors(cudaFree(d_adjacency_matrix));
  checkCudaErrors(cudaFree(d_total_count));

  checkCudaErrors(cudaEventDestroy(GPU_copy_start));
  checkCudaErrors(cudaEventDestroy(GPU_copy_stop));
  checkCudaErrors(cudaEventDestroy(GPU_exec_start));
  checkCudaErrors(cudaEventDestroy(GPU_exec_stop));

  free(h_wedgeSum);
  free(h_adjacency_matrix);
#if BINSEARCH_CONSTANT
  free(h_wedgeSum_cache);
#endif

#if RESET_DEVICE
  checkCudaErrors(cudaDeviceReset());
#endif

  return h_total_count;
}

/*********
 * Sorting helpers
 *********/

static int compareInt_t(const void *a, const void *b) {
  UINT_t arg1 = *(const UINT_t *)a;
  UINT_t arg2 = *(const UINT_t *)b;
  if (arg1 < arg2) return -1;
  if (arg1 > arg2) return 1;
  return 0;
}

static int compareEdge_t(const void *a, const void *b) {
  edge_t arg1 = *(const edge_t *) a;
  edge_t arg2 = *(const edge_t *) b;
  if (arg1.src < arg2.src) return -1;
  if (arg1.src > arg2.src) return 1;
  if ((arg1.src == arg2.src) && (arg1.dst < arg2.dst)) return -1;
  if ((arg1.src == arg2.src) && (arg1.dst > arg2.dst)) return 1;
  return 0;
}

static int compare_vertex_degree_ascending(const void *a, const void *b) {
  preprocess_vertex_t arg1 = *(const preprocess_vertex_t *) a;
  preprocess_vertex_t arg2 = *(const preprocess_vertex_t *) b;
  if (arg1.num_edges < arg2.num_edges) return -1;
  if (arg1.num_edges > arg2.num_edges) return 1;
  return 0;
}

struct edge_decomposer_t {
  __host__ __device__ ::cuda::std::tuple<unsigned int&, unsigned int&> operator()(edge_t& key) const {
    return {key.src, key.dst};
  }
};

struct preprocess_vertex_decomposer_t {
  __host__ __device__ ::cuda::std::tuple<unsigned int&> operator()(preprocess_vertex_t& key) const {
    return {key.num_edges};
  }
};

edge_t *sort_edges_GPU(edge_t *d_in, edge_t *d_out, const UINT_t num_edges, bool use_double_buffer) {
  std::uint8_t* d_temp_storage{};
  std::size_t temp_storage_bytes{};

  if (use_double_buffer) {
    cub::DoubleBuffer<edge_t> d_keys(d_in, d_out);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_edges, edge_decomposer_t{});
    checkCudaErrors(cudaMalloc((void **) &d_temp_storage, temp_storage_bytes));
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_edges, edge_decomposer_t{});
    checkCudaErrors(cudaFree(d_temp_storage));
    return d_keys.Current();
  } else {
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_edges, edge_decomposer_t{});
    checkCudaErrors(cudaMalloc((void **) &d_temp_storage, temp_storage_bytes));
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_edges, edge_decomposer_t{});
    checkCudaErrors(cudaFree(d_temp_storage));
    return d_out;
  }
}

preprocess_vertex_t *sort_vertices_GPU(preprocess_vertex_t *d_in, preprocess_vertex_t *d_out, const UINT_t num_vertices, bool use_double_buffer) {
  std::uint8_t* d_temp_storage{};
  std::size_t temp_storage_bytes{};

  if (use_double_buffer) {
    cub::DoubleBuffer<preprocess_vertex_t> d_keys(d_in, d_out);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_vertices, preprocess_vertex_decomposer_t{});
    checkCudaErrors(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes));
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_vertices, preprocess_vertex_decomposer_t{});
    checkCudaErrors(cudaFree(d_temp_storage));
    return d_keys.Current();
  } else {
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_vertices, preprocess_vertex_decomposer_t{});
    checkCudaErrors(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes));
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_vertices, preprocess_vertex_decomposer_t{});
    checkCudaErrors(cudaFree(d_temp_storage));
    return d_out;
  }
}

UINT_t *sort_colInd_GPU(UINT_t *d_rowPtr, UINT_t *d_colInd_in, UINT_t *d_colInd_out,
                        const UINT_t num_vertices, const UINT_t num_edges, bool use_double_buffer) {
  std::uint8_t* d_temp_storage{};
  std::size_t temp_storage_bytes{};

  if (use_double_buffer) {
    cub::DoubleBuffer<UINT_t> d_keys(d_colInd_in, d_colInd_out);
    cub::DeviceSegmentedSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_edges, num_vertices, d_rowPtr, d_rowPtr + 1);
    checkCudaErrors(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes));
    cub::DeviceSegmentedSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_edges, num_vertices, d_rowPtr, d_rowPtr + 1);
    checkCudaErrors(cudaFree(d_temp_storage));
    return d_keys.Current();
  } else {
    cub::DeviceSegmentedSort::SortKeys(d_temp_storage, temp_storage_bytes, d_colInd_in, d_colInd_out, num_edges, num_vertices, d_rowPtr, d_rowPtr + 1);
    checkCudaErrors(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes));
    cub::DeviceSegmentedSort::SortKeys(d_temp_storage, temp_storage_bytes, d_colInd_in, d_colInd_out, num_edges, num_vertices, d_rowPtr, d_rowPtr + 1);
    checkCudaErrors(cudaFree(d_temp_storage));
    return d_colInd_out;
  }
}

/*********
 * Graph IO and preprocessing
 *********/

GRAPH_TYPE *read_graph(char *filename, bool matrix_market, bool zero_indexed, preprocess_t preprocess_style) {
  FILE *infile = fopen(filename, "r");
  if (infile == NULL) {
    fprintf(stderr, "ERROR: unable to open graph file.\n");
    exit(EXIT_FAILURE);
  }

  GRAPH_TYPE *graph = (GRAPH_TYPE *) malloc(sizeof(GRAPH_TYPE));
  assert_malloc(graph);
  char line[256];

  // Skip header lines
  do {
    if (fgets(line, sizeof(line), infile) == NULL) { fclose(infile); exit(EXIT_FAILURE); }
  } while (line[0] < '0' || line[0] > '9');

  // Skip Matrix Market dimensions line
  if (matrix_market) {
    if (fgets(line, sizeof(line), infile) == NULL) { fclose(infile); exit(EXIT_FAILURE); }
  }

  UINT_t vertex_count = 0;
  UINT_t edge_count = 0;
  size_t size = 10240;
  edge_t* edges = (edge_t*) malloc(size * sizeof(edge_t));
  assert_malloc(edges);

  UINT_t max_vertex = 0;
  UINT_t v, w;

  if (sscanf(line, "%d %d\n", &v, &w) == 2) {
    do {
      if (edge_count >= size) {
        size += 10240;
        edge_t *new_edges = (edge_t*) realloc(edges, size * sizeof(edge_t));
        assert_malloc(new_edges);
        edges = new_edges;
      }

      if ((!zero_indexed) && (v == 0 || w == 0)) {
        fprintf(stderr, "ERROR: zero vertex id detected but -z was not set.\n");
        exit(EXIT_FAILURE);
      }

      v -= (zero_indexed ? 0 : 1);
      w -= (zero_indexed ? 0 : 1);

      // Remove self-loops.
      if (v != w) {
        max_vertex = max2(max_vertex, max2(v, w));
        // v->w
        edges[edge_count].src = v;
        edges[edge_count].dst = w;
        edge_count++;
        // w->v
        edges[edge_count].src = w;
        edges[edge_count].dst = v;
        edge_count++;
      }
    } while (fscanf(infile, "%d %d\n", &v, &w) == 2);
  }
  fclose(infile);

  vertex_count = max_vertex + 1;

  // Sort edges to remove duplicates (GPU or CPU)
  if (preprocess_style != PREPROCESS_CPU) {
    edge_t *d_edges;
    edge_t *d_edges_alt;
    edge_t *d_out;

    checkCudaErrors(cudaMalloc((void **)&d_edges, edge_count * sizeof(edge_t)));
    checkCudaErrors(cudaMalloc((void **)&d_edges_alt, edge_count * sizeof(edge_t)));
    checkCudaErrors(cudaMemcpy(d_edges, edges, edge_count * sizeof(edge_t), cudaMemcpyHostToDevice));

    if (preprocess_style == PREPROCESS_GPU_CONSTRAINED)
      d_out = sort_edges_GPU(d_edges, d_edges_alt, edge_count, true);
    else
      d_out = sort_edges_GPU(d_edges, d_edges_alt, edge_count, false);

    checkCudaErrors(cudaMemcpy(edges, d_out, edge_count * sizeof(edge_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_edges));
    checkCudaErrors(cudaFree(d_edges_alt));
  } else {
    qsort(edges, edge_count, sizeof(edge_t), compareEdge_t);
  }

  // Build CSR (rowPtr and colInd), removing duplicates.
  UINT_t *rowPtr = (UINT_t *) calloc(vertex_count + 1, sizeof(UINT_t));
  assert_malloc(rowPtr);

  UINT_t edge_count_no_dup = 1;
  edge_t lastedge = edges[0];
  UINT_t *colInd = (UINT_t *) edges; // reuse buffer
  colInd[0] = lastedge.dst;
  rowPtr[lastedge.src + 1]++;

  for (UINT_t i = 1; i < edge_count; i++) {
    if (compareEdge_t(&lastedge, &edges[i]) != 0) {
      colInd[edge_count_no_dup++] = edges[i].dst;
      rowPtr[edges[i].src + 1]++;
      lastedge = edges[i];
    }
  }

  UINT_t *new_colInd = (UINT_t *) realloc(colInd, edge_count_no_dup * sizeof(UINT_t));
  assert_malloc(new_colInd);

  for (UINT_t vv = 1; vv <= vertex_count; vv++) rowPtr[vv] += rowPtr[vv - 1];

  graph->numVertices = vertex_count;
  graph->numEdges    = edge_count_no_dup;
  graph->rowPtr      = rowPtr;
  graph->colInd      = new_colInd;

  return graph;
}

GRAPH_TYPE *preprocess(const GRAPH_TYPE *original_graph, preprocess_t preprocess_style) {
  preprocess_vertex_t *vertices = (preprocess_vertex_t *) malloc(original_graph->numVertices * sizeof(preprocess_vertex_t));
  assert_malloc(vertices);

  for (UINT_t v = 0; v < original_graph->numVertices; v++) {
    vertices[v].id        = v;
    vertices[v].edges     = &original_graph->colInd[original_graph->rowPtr[v]];
    vertices[v].num_edges = original_graph->rowPtr[v+1] - original_graph->rowPtr[v];
  }

  if (preprocess_style != PREPROCESS_CPU) {
    preprocess_vertex_t *d_vertices;
    preprocess_vertex_t *d_vertices_alt;
    preprocess_vertex_t *d_out;

    checkCudaErrors(cudaMalloc((void **)&d_vertices, original_graph->numVertices * sizeof(preprocess_vertex_t)));
    checkCudaErrors(cudaMalloc((void **)&d_vertices_alt, original_graph->numVertices * sizeof(preprocess_vertex_t)));
    checkCudaErrors(cudaMemcpy(d_vertices, vertices, original_graph->numVertices * sizeof(preprocess_vertex_t), cudaMemcpyHostToDevice));

    if (preprocess_style == PREPROCESS_GPU_CONSTRAINED)
      d_out = sort_vertices_GPU(d_vertices, d_vertices_alt, original_graph->numVertices, true);
    else
      d_out = sort_vertices_GPU(d_vertices, d_vertices_alt, original_graph->numVertices, false);

    checkCudaErrors(cudaMemcpy(vertices, d_out, original_graph->numVertices * sizeof(preprocess_vertex_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_vertices));
    checkCudaErrors(cudaFree(d_vertices_alt));
  } else {
    qsort(vertices, original_graph->numVertices, sizeof(preprocess_vertex_t), compare_vertex_degree_ascending);
  }

  UINT_t *reverse = (UINT_t *) malloc(original_graph->numVertices * sizeof(UINT_t));
  assert_malloc(reverse);
  for (UINT_t v = 0; v < original_graph->numVertices; v++) reverse[vertices[v].id] = v;

  GRAPH_TYPE *graph = (GRAPH_TYPE *) malloc(sizeof(GRAPH_TYPE));
  assert_malloc(graph);

  graph->numVertices = original_graph->numVertices;
  graph->numEdges    = original_graph->numEdges / 2;

  graph->rowPtr = (UINT_t*) malloc((graph->numVertices + 1) * sizeof(UINT_t));
  graph->colInd = (UINT_t*) malloc(graph->numEdges * sizeof(UINT_t));
  assert_malloc(graph->rowPtr);
  assert_malloc(graph->colInd);

  UINT_t edge_count = 0;
  graph->rowPtr[0] = 0;

  for (UINT_t v = 0; v < original_graph->numVertices; v++) {
    UINT_t new_degree = 0;
    for (INT_t j = 0; j < vertices[v].num_edges; j++) {
      UINT_t w = vertices[v].edges[j];
      UINT_t w_new = reverse[w];
      if (w_new > v) {
        graph->colInd[edge_count++] = w_new;
        new_degree++;
      }
    }
    graph->rowPtr[v+1] = graph->rowPtr[v] + new_degree;
    if (preprocess_style == PREPROCESS_CPU) {
      qsort(&graph->colInd[graph->rowPtr[v]], new_degree, sizeof(UINT_t), compareInt_t);
    }
  }

  free(vertices);
  free(reverse);

  if (preprocess_style != PREPROCESS_CPU) {
    UINT_t *d_rowPtr;
    UINT_t *d_colInd;
    UINT_t *d_colInd_alt;
    UINT_t *d_colInd_out;

    checkCudaErrors(cudaMalloc((void **)&d_rowPtr, (graph->numVertices + 1) * sizeof(UINT_t)));
    checkCudaErrors(cudaMalloc((void **)&d_colInd, graph->numEdges * sizeof(UINT_t)));
    checkCudaErrors(cudaMalloc((void **)&d_colInd_alt, graph->numEdges * sizeof(UINT_t)));
    checkCudaErrors(cudaMemcpy(d_rowPtr, graph->rowPtr, (graph->numVertices + 1) * sizeof(UINT_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_colInd, graph->colInd, graph->numEdges * sizeof(UINT_t), cudaMemcpyHostToDevice));

    if (preprocess_style == PREPROCESS_GPU_CONSTRAINED)
      d_colInd_out = sort_colInd_GPU(d_rowPtr, d_colInd, d_colInd_alt, graph->numVertices, graph->numEdges, true);
    else
      d_colInd_out = sort_colInd_GPU(d_rowPtr, d_colInd, d_colInd_alt, graph->numVertices, graph->numEdges, false);

    checkCudaErrors(cudaMemcpy(graph->colInd, d_colInd_out, graph->numEdges * sizeof(UINT_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_rowPtr));
    checkCudaErrors(cudaFree(d_colInd));
    checkCudaErrors(cudaFree(d_colInd_alt));
  }

  return graph;
}

void free_graph(GRAPH_TYPE *graph) {
  free(graph->rowPtr);
  free(graph->colInd);
  free(graph);
}

/*********
 * Adaptive spread autotuning
 *********/

typedef struct {
  UINT_t spread;
  double time_s;
} SpreadTrial;

static UINT_t clamp_spread_by_smem(UINT_t candidate, UINT_t threads) {
  cudaDeviceProp prop{};
  checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
  size_t smem = smem_usage_bytes(threads, candidate);
  while (smem > prop.sharedMemPerBlock && candidate > 1) {
    candidate -= 1;
    smem = smem_usage_bytes(threads, candidate);
  }
  return max2(candidate, (UINT_t)1);
}

static UINT_t autotune_spread(const GRAPH_TYPE *graph,
                              UINT_t adjacency_matrix_len,
                              const char *graph_filename,
                              UINT_t threads) {
  // Candidate spreads around typical plateau region
  UINT_t candidates_raw[] = {1, 2, 3, 4, 6, 7, 8, 10, 12, 16};
  const int C = (int)(sizeof(candidates_raw)/sizeof(candidates_raw[0]));
  SpreadTrial trials[16];
  memset(trials, 0, sizeof(trials));

  for (int i = 0; i < C; i++) {
    UINT_t s = clamp_spread_by_smem(candidates_raw[i], threads);
    GPU_time t_gpu = { .copy=0.0, .exec=0.0 };
    (void) tc_GPU(graph, s, adjacency_matrix_len, &t_gpu);
    double exec_s = t_gpu.exec / 1000.0; // ms -> s
    trials[i].spread = s;
    trials[i].time_s = exec_s;
  }

  double best_t = 1e300;
  UINT_t best_s = 1;
  for (int i = 0; i < C; i++) {
    if (trials[i].time_s < best_t) {
      best_t = trials[i].time_s;
      best_s = trials[i].spread;
    }
  }
  fprintf(stderr, "[adaptive] graph=%s best_spread=%u (GPU exec %.6fs)\n", graph_filename, best_s, best_t);
  return best_s;
}

/*********
 * CLI
 *********/

static void usage() {
  printf("Wedge Parallel Triangle Counting (Adaptive Spread)\n\n");
  printf("Usage:\n\n");
  printf("Either one of these must be selected:\n");
  printf(" -m <filename>   [Input graph in Matrix Market format]\n");
  printf(" -e <filename>   [Input graph in edge list format]\n");
  printf("Required arguments:\n");
  printf(" -a <num>        [Adjacency matrix length] (must be divisible by 32)\n");
  printf("Optional arguments:\n");
  printf(" -s <num>        [Fixed spread (wedges/thread)]\n");
  printf(" -A              [Enable adaptive spread autotuning]\n");
  printf(" -l <num>        [Loop count] (default 1)\n");
  printf(" -z              [Input graph is zero-indexed]\n");
  printf(" -p <0|1|2>      [Preprocessing: 0=CPU, 1=GPU, 2=GPU low-memory (default)]\n");
  printf("\nExamples:\n");
  printf("  ./tc_adaptive -m Amazon0302.mtx -a 8192 -A -l 10\n");
  printf("  ./tc_adaptive -e edges.txt -a 32768 -s 7\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  char *graph_filename = NULL;
  bool graph_mm = false;
  bool graph_zero_indexed = false;
  UINT_t loop_cnt = 1;

  preprocess_t preprocess_style = PREPROCESS_GPU_CONSTRAINED;

  UINT_t spread = 0;
  UINT_t adjacency_matrix_len = 0;
  bool adaptive = false;

  while ((argc > 1) && (argv[1][0] == '-')) {
    switch (argv[1][1]) {
      case 'm':
        graph_mm = true;
        // fall-through to 'e' for filename
      case 'e':
        if (argc < 3) usage();
        graph_filename = argv[2];
        if (graph_filename == NULL) usage();
        argv += 2; argc -= 2;
        break;
      case 'z':
        graph_zero_indexed = true;
        argv++; argc--;
        break;
      case 'a':
        if (argc < 3) usage();
        adjacency_matrix_len = (UINT_t) atoi(argv[2]);
        if (adjacency_matrix_len % 32 != 0 || adjacency_matrix_len == 0) usage();
        argv += 2; argc -= 2;
        break;
      case 's':
        if (argc < 3) usage();
        spread = (UINT_t) atoi(argv[2]);
        if (spread <= 0) usage();
        argv += 2; argc -= 2;
        break;
      case 'A':
        adaptive = true;
        argv++; argc--;
        break;
      case 'l':
        if (argc < 3) usage();
        loop_cnt = (UINT_t) atoi(argv[2]);
        argv += 2; argc -= 2;
        break;
      case 'p':
        if (argc < 3) usage();
        {
          int p = atoi(argv[2]);
          if (p < PREPROCESS_CPU || p > PREPROCESS_GPU_CONSTRAINED) usage();
          preprocess_style = (preprocess_t) p;
        }
        argv += 2; argc -= 2;
        break;
      default:
        usage();
    }
  }

  if (graph_filename == NULL || adjacency_matrix_len == 0) usage();

  GRAPH_TYPE *original_graph = read_graph(graph_filename, graph_mm, graph_zero_indexed, preprocess_style);
  double t_preprocessing = get_seconds();
  GRAPH_TYPE *graph = preprocess(original_graph, preprocess_style);
  t_preprocessing = get_seconds() - t_preprocessing;
  free_graph(original_graph);

  // Select spread
  const UINT_t num_threads = 128;
  if (adaptive) {
    spread = autotune_spread(graph, adjacency_matrix_len, graph_filename, num_threads);
  } else {
    if (spread == 0) spread = 7; // good default near plateau
    spread = clamp_spread_by_smem(spread, num_threads);
  }

  printf("%-60s %16s %16s %16s %16s %16s %16s %16s %16s %16s %16s\n",
         "graph", "n", "m", "s", "a", "triangles", "prepro (s)",
         "GPU copy (s)", "GPU exec (s)", "GPU total (s)", "CPU+GPU (s)");

  bool warmed_up = false;

  for (UINT_t i = 0; i < (loop_cnt + 1); i++) {
    double t_cpu = get_seconds();
    GPU_time t_gpu = { .copy=0.0, .exec=0.0 };

    ULONG_t triangles = tc_GPU(graph, spread, adjacency_matrix_len, &t_gpu);

    t_cpu = get_seconds() - t_cpu;

    t_gpu.copy /= 1000.0;
    t_gpu.exec /= 1000.0;

    if (warmed_up) {
      printf("%-60s %16u %16u %16u %16u %16llu %16.6f %16.6f %16.6f %16.6f %16.6f\n",
             graph_filename, graph->numVertices, graph->numEdges, spread, adjacency_matrix_len,
             (unsigned long long) triangles,
             t_preprocessing, t_gpu.copy, t_gpu.exec, t_gpu.copy + t_gpu.exec, t_cpu);
    } else {
      warmed_up = true;
    }
  }

  free_graph(graph);
  return EXIT_SUCCESS;
}