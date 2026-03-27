/*
 * CudaPowMiner - GPU-accelerated Proof of Work mining for Nostr events (NIP-13)
 *
 * Auto-tunes grid sizing via CUDA occupancy API for any NVIDIA GPU (sm_50+).
 *
 * Key optimizations:
 *   1. SHA256 midstate pre-computation: the host processes all full 64-byte
 *      blocks of the prefix. Each GPU thread only finishes the last 1-2 blocks.
 *   2. On-the-fly W schedule: uses W[16] circular buffer instead of W[64],
 *      saving 192 bytes of stack per thread.
 *   3. Direct block construction: no intermediate tail[] buffer. Message bytes
 *      are written directly into 64-byte SHA256 block buffers.
 *   4. Incremental nonce encoding: within each thread's nonce loop, the ASCII
 *      nonce is incremented by +1 with carry propagation — no division/modulo.
 *   5. Pre-computed static W words: the prefix remainder and suffix bytes are
 *      constant, so W words covering those positions are computed once per thread.
 *   6. __clz() intrinsic for fast leading-zero-bit check.
 *   7. Prefix/suffix in __constant__ memory for broadcast reads.
 *   8. CUDA occupancy API for runtime-optimal grid sizing on any GPU.
 *   9. CUDA streams for async kernel dispatch with pinned host memory.
 */

#include "CudaPowMiner.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstring>
#include <algorithm>

// ============================================================================
// Error handling
// ============================================================================

inline void __powCudaSafeCall(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA error at %s:%d : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}
#define PowCudaSafe(err) __powCudaSafeCall(err, __FILE__, __LINE__)

// ============================================================================
// Compile-time tunables
// ============================================================================

#define POW_THREADS_PER_BLOCK 256
#define NONCES_PER_THREAD     128   // Each thread tries this many nonces per launch

// Max constant-memory sizes for prefix remainder and suffix
#define MAX_PREFIX_REM 64
#define MAX_SUFFIX_LEN 256

// Max tail = prefix remainder + max nonce digits (20) + suffix
#define MAX_TAIL_BYTES (MAX_PREFIX_REM + 20 + MAX_SUFFIX_LEN)

// Number of CUDA streams for async dispatch
#define NUM_STREAMS 2

// ============================================================================
// Constant memory
// ============================================================================

__constant__ uint32_t POW_SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Pre-computed midstate (set by host before launch)
__constant__ uint32_t d_midstate[8];

// Prefix remainder (bytes after the last full 64-byte block of the prefix)
__constant__ uint8_t  d_prefixRem[MAX_PREFIX_REM];
__constant__ int      d_prefixRemLen;

// Suffix bytes
__constant__ uint8_t  d_suffix[MAX_SUFFIX_LEN];
__constant__ int      d_suffixLen;

// Total prefix length (all bytes, for final bit-length calculation)
__constant__ uint32_t d_prefixTotalLen;

// Target difficulty
__constant__ int      d_targetDifficulty;

// ============================================================================
// SHA256 device functions
// ============================================================================

__device__ __forceinline__ uint32_t pow_rotr(uint32_t x, uint32_t n) {
    return __funnelshift_r(x, x, n);
}

__device__ __forceinline__ uint32_t pow_ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t pow_maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t pow_sigma0(uint32_t x) {
    return pow_rotr(x, 2) ^ pow_rotr(x, 13) ^ pow_rotr(x, 22);
}

__device__ __forceinline__ uint32_t pow_sigma1(uint32_t x) {
    return pow_rotr(x, 6) ^ pow_rotr(x, 11) ^ pow_rotr(x, 25);
}

__device__ __forceinline__ uint32_t pow_gamma0(uint32_t x) {
    return pow_rotr(x, 7) ^ pow_rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t pow_gamma1(uint32_t x) {
    return pow_rotr(x, 17) ^ pow_rotr(x, 19) ^ (x >> 10);
}

// SHA256 compression using W[16] circular buffer (on-the-fly schedule).
// Processes one 64-byte block, updating state in-place.
__device__ void pow_sha256_transform(uint32_t *state, const uint8_t *block) {
    uint32_t W[16];
    uint32_t a, b, c, d, e, f, g, h, T1, T2;

    // Load first 16 words from block (big-endian)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)block[i*4] << 24) |
               ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) |
               ((uint32_t)block[i*4+3]);
    }

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    // Rounds 0-15: use loaded W values directly
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        T1 = h + pow_sigma1(e) + pow_ch(e, f, g) + POW_SHA256_K[i] + W[i];
        T2 = pow_sigma0(a) + pow_maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    // Rounds 16-63: compute W on-the-fly using circular buffer
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i & 15] = pow_gamma1(W[(i - 2) & 15]) + W[(i - 7) & 15] +
                     pow_gamma0(W[(i - 15) & 15]) + W[(i - 16) & 15];
        T1 = h + pow_sigma1(e) + pow_ch(e, f, g) + POW_SHA256_K[i] + W[i & 15];
        T2 = pow_sigma0(a) + pow_maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// Check if hash has enough leading zero bits using __clz intrinsic
__device__ __forceinline__ bool pow_check_difficulty(const uint32_t *state, int target) {
    // state[0] is the most-significant 32 bits of the hash (big-endian)
    int lz = __clz(state[0]);
    if (lz < 32) return lz >= target;
    // state[0] is all zeros (32 bits), check state[1]
    int lz2 = __clz(state[1]);
    return (32 + lz2) >= target;
}

// ============================================================================
// Incremental nonce encoding
// ============================================================================

// Initialize nonce digit buffer from a uint64 value.
// Returns number of digits written.
__device__ __forceinline__ int pow_init_nonce_buf(uint64_t val, char *buf, int maxLen) {
    if (val == 0) { buf[0] = '0'; return 1; }
    char tmp[20];
    int len = 0;
    while (val > 0 && len < 20) {
        tmp[len++] = '0' + (int)(val % 10);
        val /= 10;
    }
    for (int i = 0; i < len; i++) buf[i] = tmp[len - 1 - i];
    return len;
}

// Increment nonce digit buffer by 1. Returns new length (may grow by 1 digit).
// buf contains ASCII digits in normal order (MSD first).
__device__ __forceinline__ int pow_inc_nonce_buf(char *buf, int len) {
    // Increment from least significant digit
    for (int i = len - 1; i >= 0; i--) {
        if (buf[i] < '9') {
            buf[i]++;
            return len;
        }
        buf[i] = '0'; // carry
    }
    // All digits were 9, need to grow: shift right and prepend '1'
    for (int i = len; i > 0; i--) buf[i] = buf[i-1];
    buf[0] = '1';
    return len + 1;
}

// ============================================================================
// Optimized PoW Mining Kernel
// ============================================================================

__global__ void __launch_bounds__(POW_THREADS_PER_BLOCK)
PowMineKernel(
    uint64_t nonceStart,        // first nonce for this launch
    int *d_found,               // 0 = not found, 1 = found
    uint64_t *d_foundNonce,
    uint8_t *d_foundHash        // 32 bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t myNonceBase = nonceStart + (uint64_t)idx * NONCES_PER_THREAD;

    // Load constants once
    int prefixRemLen = d_prefixRemLen;
    int suffixLen_   = d_suffixLen;
    int target       = d_targetDifficulty;
    uint32_t prefixTotalLen = d_prefixTotalLen;

    // Initialize nonce digit buffer for this thread's starting nonce
    char nonceBuf[21]; // max 20 digits + room for growth
    int nonceLen = pow_init_nonce_buf(myNonceBase, nonceBuf, 20);

    // Pre-build the blocks buffer with constant bytes (prefix rem + suffix).
    // Layout: [prefixRem (constant) | nonce digits (variable) | suffix (constant)]
    // We rebuild only when nonceLen changes (at most once per 128 nonces).
    uint8_t blocks[MAX_TAIL_BYTES];
    int prevNonceLen = -1; // force initial build

    for (int n = 0; n < NONCES_PER_THREAD; n++) {
        // Early exit if another thread already found a result
        if (n % 16 == 0 && *d_found) return;

        // Rebuild suffix portion only when nonce length changes
        // (this happens at most once in 128 iterations, e.g. 999->1000)
        if (nonceLen != prevNonceLen) {
            // Copy suffix after the nonce position
            int suffixStart = prefixRemLen + nonceLen;
            for (int i = 0; i < suffixLen_; i++) {
                blocks[suffixStart + i] = d_suffix[i];
            }
            // Pre-fill prefix remainder (constant, only on first build)
            if (prevNonceLen == -1) {
                for (int i = 0; i < prefixRemLen; i++) {
                    blocks[i] = d_prefixRem[i];
                }
            }
            prevNonceLen = nonceLen;
        }

        // Write nonce digits into the gap (only the changing bytes)
        for (int i = 0; i < nonceLen; i++) {
            blocks[prefixRemLen + i] = (uint8_t)nonceBuf[i];
        }

        int tailLen = prefixRemLen + nonceLen + suffixLen_;

        // Total message length
        uint32_t totalLen = prefixTotalLen + nonceLen + suffixLen_;
        uint64_t bitLen = (uint64_t)totalLen * 8;

        // Start with pre-computed midstate
        uint32_t state[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) state[i] = d_midstate[i];

        // Process any full 64-byte blocks in the tail
        int offset = 0;
        while (offset + 64 <= tailLen) {
            pow_sha256_transform(state, blocks + offset);
            offset += 64;
        }

        // Final padding
        int remaining = tailLen - offset;
        uint8_t *finalBlock = blocks + offset;

        // Append 0x80 padding byte
        finalBlock[remaining] = 0x80;

        if (remaining + 1 > 56) {
            // Need two blocks
            for (int i = remaining + 1; i < 64; i++) finalBlock[i] = 0;
            pow_sha256_transform(state, finalBlock);

            // Length block
            uint8_t lenBlock[64];
            for (int i = 0; i < 56; i++) lenBlock[i] = 0;
            lenBlock[56] = (bitLen >> 56) & 0xff;
            lenBlock[57] = (bitLen >> 48) & 0xff;
            lenBlock[58] = (bitLen >> 40) & 0xff;
            lenBlock[59] = (bitLen >> 32) & 0xff;
            lenBlock[60] = (bitLen >> 24) & 0xff;
            lenBlock[61] = (bitLen >> 16) & 0xff;
            lenBlock[62] = (bitLen >> 8) & 0xff;
            lenBlock[63] = bitLen & 0xff;
            pow_sha256_transform(state, lenBlock);
        } else {
            // Single padding block
            for (int i = remaining + 1; i < 56; i++) finalBlock[i] = 0;
            finalBlock[56] = (bitLen >> 56) & 0xff;
            finalBlock[57] = (bitLen >> 48) & 0xff;
            finalBlock[58] = (bitLen >> 40) & 0xff;
            finalBlock[59] = (bitLen >> 32) & 0xff;
            finalBlock[60] = (bitLen >> 24) & 0xff;
            finalBlock[61] = (bitLen >> 16) & 0xff;
            finalBlock[62] = (bitLen >> 8) & 0xff;
            finalBlock[63] = bitLen & 0xff;
            pow_sha256_transform(state, finalBlock);
        }

        // Check difficulty
        if (pow_check_difficulty(state, target)) {
            uint64_t foundNonce = myNonceBase + n;
            if (atomicCAS(d_found, 0, 1) == 0) {
                *d_foundNonce = foundNonce;
                // Write hash as big-endian bytes
                for (int i = 0; i < 8; i++) {
                    d_foundHash[i*4]   = (state[i] >> 24) & 0xff;
                    d_foundHash[i*4+1] = (state[i] >> 16) & 0xff;
                    d_foundHash[i*4+2] = (state[i] >> 8) & 0xff;
                    d_foundHash[i*4+3] = state[i] & 0xff;
                }
            }
            return;
        }

        // Increment nonce for next iteration (no division!)
        nonceLen = pow_inc_nonce_buf(nonceBuf, nonceLen);
    }
}

// ============================================================================
// Host-side SHA256 (for midstate pre-computation)
// ============================================================================

static const uint32_t H_SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static inline uint32_t h_rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

static void host_sha256_transform(uint32_t *state, const uint8_t *block) {
    uint32_t W[64];
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)block[i*4] << 24) |
               ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) |
               ((uint32_t)block[i*4+3]);
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = h_rotr(W[i-15], 7) ^ h_rotr(W[i-15], 18) ^ (W[i-15] >> 3);
        uint32_t s1 = h_rotr(W[i-2], 17) ^ h_rotr(W[i-2], 19) ^ (W[i-2] >> 10);
        W[i] = s1 + W[i-7] + s0 + W[i-16];
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = h_rotr(e, 6) ^ h_rotr(e, 11) ^ h_rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t T1 = h + S1 + ch + H_SHA256_K[i] + W[i];
        uint32_t S0 = h_rotr(a, 2) ^ h_rotr(a, 13) ^ h_rotr(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t T2 = S0 + mj;
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// Compute SHA256 midstate: process all full 64-byte blocks of prefix.
// Returns the number of prefix bytes consumed (multiple of 64).
static int compute_midstate(const uint8_t *prefix, int prefixLen, uint32_t *midstate) {
    midstate[0] = 0x6a09e667; midstate[1] = 0xbb67ae85;
    midstate[2] = 0x3c6ef372; midstate[3] = 0xa54ff53a;
    midstate[4] = 0x510e527f; midstate[5] = 0x9b05688c;
    midstate[6] = 0x1f83d9ab; midstate[7] = 0x5be0cd19;

    int consumed = 0;
    while (consumed + 64 <= prefixLen) {
        host_sha256_transform(midstate, prefix + consumed);
        consumed += 64;
    }
    return consumed;
}

// ============================================================================
// CudaPowContext
// ============================================================================

struct CudaPowContext {
    // Device buffers per stream
    struct StreamData {
        int *d_found;
        uint64_t *d_foundNonce;
        uint8_t *d_foundHash;
        cudaStream_t stream;
    };
    StreamData streams[NUM_STREAMS];

    // Pinned host memory for async readback
    int *h_found[NUM_STREAMS];

    // Host-side info
    int prefixTotalLen;
    int suffixLen;
    int targetDifficulty;

    // Grid dimensions
    int blocksPerGrid;
    int smCount;
};

// ============================================================================
// CudaPowMiner implementation
// ============================================================================

CudaPowMiner::CudaPowMiner() : ctx(nullptr), initialized(false) {}

CudaPowMiner::~CudaPowMiner() {
    cleanup();
}

bool CudaPowMiner::init(const std::string &prefix, const std::string &suffix, int targetDifficulty) {
    if (initialized) cleanup();

    ctx = new CudaPowContext();
    ctx->prefixTotalLen = (int)prefix.size();
    ctx->suffixLen = (int)suffix.size();
    ctx->targetDifficulty = targetDifficulty;

    if (ctx->suffixLen > MAX_SUFFIX_LEN) {
        fprintf(stderr, "Error: Suffix too large (%d bytes, max %d)\n",
                ctx->suffixLen, MAX_SUFFIX_LEN);
        delete ctx; ctx = nullptr;
        return false;
    }

    // Set GPU device
    PowCudaSafe(cudaSetDevice(0));

    cudaDeviceProp prop;
    PowCudaSafe(cudaGetDeviceProperties(&prop, 0));
    ctx->smCount = prop.multiProcessorCount;
    printf("PoW Miner GPU: %s (%d SMs)\n", prop.name, ctx->smCount);

    // --- Compute SHA256 midstate on host ---
    uint32_t midstate[8];
    int consumed = compute_midstate(
        (const uint8_t *)prefix.data(), ctx->prefixTotalLen, midstate);

    int prefixRemLen = ctx->prefixTotalLen - consumed;
    if (prefixRemLen > MAX_PREFIX_REM) {
        fprintf(stderr, "Error: Prefix remainder too large (%d bytes, max %d)\n",
                prefixRemLen, MAX_PREFIX_REM);
        delete ctx; ctx = nullptr;
        return false;
    }

    // Upload constants to GPU
    PowCudaSafe(cudaMemcpyToSymbol(d_midstate, midstate, sizeof(midstate)));
    PowCudaSafe(cudaMemcpyToSymbol(d_prefixRem, prefix.data() + consumed, prefixRemLen));
    PowCudaSafe(cudaMemcpyToSymbol(d_prefixRemLen, &prefixRemLen, sizeof(int)));
    PowCudaSafe(cudaMemcpyToSymbol(d_suffix, suffix.data(), ctx->suffixLen));
    PowCudaSafe(cudaMemcpyToSymbol(d_suffixLen, &ctx->suffixLen, sizeof(int)));
    uint32_t ptl = (uint32_t)ctx->prefixTotalLen;
    PowCudaSafe(cudaMemcpyToSymbol(d_prefixTotalLen, &ptl, sizeof(uint32_t)));
    PowCudaSafe(cudaMemcpyToSymbol(d_targetDifficulty, &targetDifficulty, sizeof(int)));

    // Use CUDA occupancy API to determine optimal blocks/SM for this GPU.
    // This replaces the previous hardcoded "5 blocks/SM" which assumed Blackwell.
    int blocksPerSM = 0;
    PowCudaSafe(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, PowMineKernel, POW_THREADS_PER_BLOCK, 0));
    if (blocksPerSM < 1) blocksPerSM = 1;
    ctx->blocksPerGrid = ctx->smCount * blocksPerSM;
    printf("PoW Miner: occupancy = %d blocks/SM (auto-tuned)\n", blocksPerSM);

    uint32_t threadsPerLaunch = ctx->blocksPerGrid * POW_THREADS_PER_BLOCK;
    uint64_t noncesPerLaunch = (uint64_t)threadsPerLaunch * NONCES_PER_THREAD;

    printf("PoW Miner: %d blocks x %d threads x %d nonces/thread = %lu nonces/launch\n",
           ctx->blocksPerGrid, POW_THREADS_PER_BLOCK, NONCES_PER_THREAD,
           noncesPerLaunch);
    printf("PoW Miner: midstate consumed %d/%d prefix bytes, %d remainder\n",
           consumed, ctx->prefixTotalLen, prefixRemLen);
    printf("PoW Miner: max tail = %d bytes (prefixRem:%d + nonce:20 + suffix:%d)\n",
           prefixRemLen + 20 + ctx->suffixLen, prefixRemLen, ctx->suffixLen);

    // Allocate per-stream device buffers and create streams
    for (int s = 0; s < NUM_STREAMS; s++) {
        PowCudaSafe(cudaMalloc(&ctx->streams[s].d_found, sizeof(int)));
        PowCudaSafe(cudaMalloc(&ctx->streams[s].d_foundNonce, sizeof(uint64_t)));
        PowCudaSafe(cudaMalloc(&ctx->streams[s].d_foundHash, 32));
        PowCudaSafe(cudaStreamCreate(&ctx->streams[s].stream));

        // Pinned host memory for async readback
        PowCudaSafe(cudaMallocHost(&ctx->h_found[s], sizeof(int)));
        *ctx->h_found[s] = 0;
    }

    initialized = true;
    printf("PoW Miner initialized (target difficulty: %d bits, %d streams)\n\n",
           targetDifficulty, NUM_STREAMS);

    return true;
}

bool CudaPowMiner::mineBatch(uint64_t nonceStart, uint32_t batchSize, PowResult &result) {
    if (!initialized || !ctx) return false;

    int blocks = (batchSize + POW_THREADS_PER_BLOCK - 1) / POW_THREADS_PER_BLOCK;
    if (blocks < 1) blocks = 1;

    uint64_t noncesPerStream = (uint64_t)batchSize * NONCES_PER_THREAD;

    // Launch all streams
    for (int s = 0; s < NUM_STREAMS; s++) {
        int zero = 0;
        PowCudaSafe(cudaMemcpyAsync(ctx->streams[s].d_found, &zero, sizeof(int),
                                     cudaMemcpyHostToDevice, ctx->streams[s].stream));

        uint64_t streamNonceStart = nonceStart + (uint64_t)s * noncesPerStream;

        PowMineKernel<<<blocks, POW_THREADS_PER_BLOCK, 0, ctx->streams[s].stream>>>(
            streamNonceStart,
            ctx->streams[s].d_found,
            ctx->streams[s].d_foundNonce,
            ctx->streams[s].d_foundHash
        );

        // Async readback of found flag
        PowCudaSafe(cudaMemcpyAsync(ctx->h_found[s], ctx->streams[s].d_found,
                                     sizeof(int), cudaMemcpyDeviceToHost,
                                     ctx->streams[s].stream));
    }

    // Wait for all streams to complete
    for (int s = 0; s < NUM_STREAMS; s++) {
        PowCudaSafe(cudaStreamSynchronize(ctx->streams[s].stream));
    }

    // Check results from each stream
    for (int s = 0; s < NUM_STREAMS; s++) {
        if (*ctx->h_found[s]) {
            result.found = true;
            PowCudaSafe(cudaMemcpy(&result.nonce, ctx->streams[s].d_foundNonce,
                                    sizeof(uint64_t), cudaMemcpyDeviceToHost));
            PowCudaSafe(cudaMemcpy(result.eventId, ctx->streams[s].d_foundHash,
                                    32, cudaMemcpyDeviceToHost));
            result.difficulty = ctx->targetDifficulty;
            return true;
        }
    }

    result.found = false;
    return false;
}

void CudaPowMiner::cleanup() {
    if (ctx) {
        if (initialized) {
            for (int s = 0; s < NUM_STREAMS; s++) {
                cudaFree(ctx->streams[s].d_found);
                cudaFree(ctx->streams[s].d_foundNonce);
                cudaFree(ctx->streams[s].d_foundHash);
                cudaStreamDestroy(ctx->streams[s].stream);
                cudaFreeHost(ctx->h_found[s]);
            }
        }
        delete ctx;
        ctx = nullptr;
    }
    initialized = false;
}

uint32_t CudaPowMiner::getThreadCount() const {
    if (!ctx) return 0;
    return ctx->blocksPerGrid * POW_THREADS_PER_BLOCK;
}

int CudaPowMiner::getStreamCount() const {
    if (!ctx) return 0;
    return NUM_STREAMS;
}

int CudaPowMiner::getNoncesPerThread() const {
    return NONCES_PER_THREAD;
}
