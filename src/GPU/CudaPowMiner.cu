/*
 * CudaPowMiner - GPU-accelerated Proof of Work mining for Nostr events (NIP-13)
 */

#include "CudaPowMiner.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

#ifndef POW_THREADS_PER_BLOCK
#define POW_THREADS_PER_BLOCK 256
#endif

#ifndef POW_LAUNCH_WAVES
#define POW_LAUNCH_WAVES 1
#endif

#define NONCES_PER_UNIT 1000
#define UNITS_PER_HIGH 10
#define LOW_DIGITS 4
#define MIN_NONCE_DIGITS 12
#define MAX_NONCE_DIGITS 20
#define POW_SLOTS 2

static const uint32_t H_SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static const uint32_t H_SHA256_IV[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

__constant__ uint32_t c_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

struct PowOutput {
    uint64_t nonce;
    uint32_t hash[8];
    int found;
    int pad;
};

#define POW_PARAM_TRAIL 6

struct PowParams {
    uint32_t midstate[8];
    uint32_t tpl[32];
    uint32_t kw[POW_PARAM_TRAIL * 64];
    const uint32_t *trailKW;
    PowOutput *out;
    uint64_t unitStart;
    uint64_t highBase;
    uint32_t unitCount;
    int nTrail;
    int highDigits;
    int target;
    int firstTarget;
};

__device__ __forceinline__ uint32_t sha_rotr(uint32_t x, uint32_t n) {
    return __funnelshift_r(x, x, n);
}

__device__ __forceinline__ uint32_t sha_bsig0(uint32_t x) { return sha_rotr(x, 2) ^ sha_rotr(x, 13) ^ sha_rotr(x, 22); }
__device__ __forceinline__ uint32_t sha_bsig1(uint32_t x) { return sha_rotr(x, 6) ^ sha_rotr(x, 11) ^ sha_rotr(x, 25); }
__device__ __forceinline__ uint32_t sha_ssig0(uint32_t x) { return sha_rotr(x, 7) ^ sha_rotr(x, 18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t sha_ssig1(uint32_t x) { return sha_rotr(x, 17) ^ sha_rotr(x, 19) ^ (x >> 10); }
__device__ __forceinline__ uint32_t sha_ch(uint32_t x, uint32_t y, uint32_t z) { return z ^ (x & (y ^ z)); }
__device__ __forceinline__ uint32_t sha_maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (z & (x | y)); }

__device__ __forceinline__ void sha_round(uint32_t (&v)[8], int i, uint32_t kw) {
    uint32_t &a = v[(0 - i) & 7], &b = v[(1 - i) & 7], &c = v[(2 - i) & 7], &d = v[(3 - i) & 7];
    uint32_t &e = v[(4 - i) & 7], &f = v[(5 - i) & 7], &g = v[(6 - i) & 7], &h = v[(7 - i) & 7];
    uint32_t t1 = h + sha_bsig1(e) + sha_ch(e, f, g) + kw;
    uint32_t t2 = sha_bsig0(a) + sha_maj(a, b, c);
    d += t1;
    h = t1 + t2;
}

__device__ __forceinline__ void compress(uint32_t (&st)[8], uint32_t (&w)[16]) {
    uint32_t v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = st[i];
    #pragma unroll
    for (int i = 0; i < 16; i++) sha_round(v, i, c_K[i] + w[i]);
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        w[i & 15] += sha_ssig1(w[(i - 2) & 15]) + w[(i - 7) & 15] + sha_ssig0(w[(i - 15) & 15]);
        sha_round(v, i, c_K[i] + w[i & 15]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) st[i] += v[i];
}

template <int B0, int WV>
__global__ void __launch_bounds__(POW_THREADS_PER_BLOCK)
PowKernel(const PowParams p) {
    const uint32_t gid = blockIdx.x * POW_THREADS_PER_BLOCK + threadIdx.x;
    if (gid >= p.unitCount) return;

    const uint64_t unit = p.unitStart + gid;
    const uint64_t high = p.highBase + unit / UNITS_PER_HIGH;
    const uint32_t thousands = (uint32_t)(unit % UNITS_PER_HIGH);

    uint32_t msg[(B0 + 1) * 16];
    #pragma unroll
    for (int i = 0; i < (B0 + 1) * 16; i++) msg[i] = p.tpl[i];
    uint64_t rest = high;
    #pragma unroll
    for (int j = 0; j < MAX_NONCE_DIGITS - LOW_DIGITS; j++) {
        const int word = B0 * 16 + WV - 1 - j / 4;
        if (word >= 0 && j < p.highDigits) {
            msg[word] |= (uint32_t)('0' + (int)(rest % 10)) << (8 * (j & 3));
            rest /= 10;
        }
    }

    uint32_t st[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) st[i] = p.midstate[i];
    if (B0 == 1) {
        uint32_t w0[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) w0[i] = msg[i];
        compress(st, w0);
    }

    uint32_t w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) w[i] = msg[B0 * 16 + i];

    uint32_t pre[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) pre[i] = st[i];
    #pragma unroll
    for (int i = 0; i < WV; i++) sha_round(pre, i, c_K[i] + w[i]);

    const uint32_t lowBase = 0x30303030u | (thousands << 24);
    volatile int *found = &p.out->found;

    for (uint32_t hd = 0; hd < 10; hd++) {
        if (*found) return;
        for (uint32_t td = 0; td < 10; td++) {
            for (uint32_t od = 0; od < 10; od++) {
                uint32_t m[16];
                #pragma unroll
                for (int i = 0; i < 16; i++) m[i] = w[i];
                m[WV] = lowBase | (hd << 16) | (td << 8) | od;

                uint32_t v[8];
                #pragma unroll
                for (int i = 0; i < 8; i++) v[i] = pre[i];
                #pragma unroll
                for (int i = WV; i < 16; i++) sha_round(v, i, c_K[i] + m[i]);
                #pragma unroll
                for (int i = 16; i < 64; i++) {
                    m[i & 15] += sha_ssig1(m[(i - 2) & 15]) + m[(i - 7) & 15] + sha_ssig0(m[(i - 15) & 15]);
                    sha_round(v, i, c_K[i] + m[i & 15]);
                }

                uint32_t s[8];
                #pragma unroll
                for (int i = 0; i < 8; i++) s[i] = st[i] + v[i];

                for (int b = 0; b < p.nTrail; b++) {
                    uint32_t y[8];
                    #pragma unroll
                    for (int i = 0; i < 8; i++) y[i] = s[i];
                    if (b < POW_PARAM_TRAIL) {
                        #pragma unroll
                        for (int i = 0; i < 64; i++) sha_round(y, i, p.kw[b * 64 + i]);
                    } else {
                        const uint32_t *kw = p.trailKW + b * 64;
                        #pragma unroll
                        for (int i = 0; i < 64; i++) sha_round(y, i, __ldg(kw + i));
                    }
                    #pragma unroll
                    for (int i = 0; i < 8; i++) s[i] += y[i];
                }

                if (__clz(s[0]) >= p.firstTarget) {
                    int bits = 0;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        if (s[i] != 0) { bits += __clz(s[i]); break; }
                        bits += 32;
                    }
                    if (bits >= p.target) {
                        if (atomicCAS(&p.out->found, 0, 1) == 0) {
                            p.out->nonce = high * 10000ull + thousands * 1000u + hd * 100u + td * 10u + od;
                            #pragma unroll
                            for (int i = 0; i < 8; i++) p.out->hash[i] = s[i];
                            __threadfence();
                        }
                        return;
                    }
                }
            }
        }
    }
}

typedef void (*PowLaunchFn)(dim3, cudaStream_t, const PowParams &);

template <int B0, int WV>
static void launchKernel(dim3 grid, cudaStream_t stream, const PowParams &p) {
    PowKernel<B0, WV><<<grid, POW_THREADS_PER_BLOCK, 0, stream>>>(p);
}

struct PowKernelEntry {
    PowLaunchFn launch;
    const void *fn;
};

#define POW_KERNEL_ENTRY(B0, WV) { launchKernel<B0, WV>, (const void *)PowKernel<B0, WV> }
#define POW_KERNEL_ROW(B0) \
    { POW_KERNEL_ENTRY(B0, 0), POW_KERNEL_ENTRY(B0, 1), POW_KERNEL_ENTRY(B0, 2), POW_KERNEL_ENTRY(B0, 3), \
      POW_KERNEL_ENTRY(B0, 4), POW_KERNEL_ENTRY(B0, 5), POW_KERNEL_ENTRY(B0, 6), POW_KERNEL_ENTRY(B0, 7), \
      POW_KERNEL_ENTRY(B0, 8), POW_KERNEL_ENTRY(B0, 9), POW_KERNEL_ENTRY(B0, 10), POW_KERNEL_ENTRY(B0, 11), \
      POW_KERNEL_ENTRY(B0, 12), POW_KERNEL_ENTRY(B0, 13), POW_KERNEL_ENTRY(B0, 14), POW_KERNEL_ENTRY(B0, 15) }

static const PowKernelEntry POW_KERNELS[2][16] = { POW_KERNEL_ROW(0), POW_KERNEL_ROW(1) };

static inline uint32_t h_rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

static void hostSchedule(const uint8_t *block, uint32_t *w) {
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) | (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = h_rotr(w[i - 15], 7) ^ h_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = h_rotr(w[i - 2], 17) ^ h_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = s1 + w[i - 7] + s0 + w[i - 16];
    }
}

static void hostCompress(uint32_t *state, const uint8_t *block) {
    uint32_t w[64];
    hostSchedule(block, w);
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + (h_rotr(e, 6) ^ h_rotr(e, 11) ^ h_rotr(e, 25)) + ((e & f) ^ (~e & g)) + H_SHA256_K[i] + w[i];
        uint32_t t2 = (h_rotr(a, 2) ^ h_rotr(a, 13) ^ h_rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

static void hostSha256(const std::string &msg, uint8_t out[32]) {
    uint32_t st[8];
    memcpy(st, H_SHA256_IV, sizeof(st));
    size_t len = msg.size();
    std::vector<uint8_t> buf(msg.begin(), msg.end());
    buf.push_back(0x80);
    while (buf.size() % 64 != 56) buf.push_back(0);
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 7; i >= 0; i--) buf.push_back((uint8_t)(bits >> (8 * i)));
    for (size_t off = 0; off < buf.size(); off += 64) hostCompress(st, buf.data() + off);
    for (int i = 0; i < 8; i++) {
        out[i * 4] = (uint8_t)(st[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(st[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(st[i] >> 8);
        out[i * 4 + 3] = (uint8_t)st[i];
    }
}

static int leadingZeroBits(const uint8_t *hash) {
    int bits = 0;
    for (int i = 0; i < 32; i++) {
        if (hash[i] == 0) { bits += 8; continue; }
        uint8_t b = hash[i];
        while (!(b & 0x80)) { bits++; b <<= 1; }
        break;
    }
    return bits;
}

static uint64_t pow10u(int n) {
    uint64_t v = 1;
    while (n-- > 0) v *= 10;
    return v;
}

struct PowLayout {
    int digits;
    int b0;
    int wv;
    int nTrail;
    int cost;
    uint64_t highBase;
    uint64_t units;
    uint64_t firstUnit;
    uint32_t tpl[32];
    std::vector<uint32_t> trailKW;
    uint32_t *d_trailKW;
    uint32_t launchUnits;
};

struct PowSlot {
    PowOutput *d_out;
    PowOutput *h_out;
    cudaEvent_t done;
    uint64_t units;
};

struct CudaPowContext {
    int device = 0;
    int smCount = 0;
    std::string prefix;
    std::string suffix;
    int target = 0;
    int prefixRem = 0;
    uint32_t midstate[8];
    std::vector<PowLayout> layouts;
    uint64_t totalUnits = 0;
    uint64_t cursor = 0;
    uint64_t limit = 0;
    cudaStream_t stream = nullptr;
    PowSlot slots[POW_SLOTS];
    int head = 0;
    int inFlight = 0;
};

static PowLayout buildLayout(const CudaPowContext &c, int digits) {
    PowLayout l;
    l.digits = digits;
    const int high = digits - LOW_DIGITS;
    const int lowPos = c.prefixRem + high;
    l.b0 = lowPos / 64;
    l.wv = (lowPos % 64) / 4;
    const size_t tailLen = (size_t)c.prefixRem + digits + c.suffix.size();
    const size_t blocks = (tailLen + 9 + 63) / 64;
    l.nTrail = (int)(blocks - l.b0 - 1);
    l.cost = (64 - l.wv) + 24 + 64 * l.nTrail;
    l.highBase = pow10u(high - 1);
    const uint64_t highMax = std::min(pow10u(high) - 1, (UINT64_MAX - 9999) / 10000);
    l.units = (highMax - l.highBase + 1) * UNITS_PER_HIGH;
    l.firstUnit = 0;
    l.d_trailKW = nullptr;
    l.launchUnits = 0;

    std::vector<uint8_t> tail(blocks * 64, 0);
    const size_t consumed = c.prefix.size() - c.prefixRem;
    memcpy(tail.data(), c.prefix.data() + consumed, c.prefixRem);
    memcpy(tail.data() + c.prefixRem + digits, c.suffix.data(), c.suffix.size());
    tail[tailLen] = 0x80;
    const uint64_t bits = (uint64_t)(c.prefix.size() + digits + c.suffix.size()) * 8;
    for (int i = 0; i < 8; i++) tail[blocks * 64 - 1 - i] = (uint8_t)(bits >> (8 * i));

    memset(l.tpl, 0, sizeof(l.tpl));
    for (int i = 0; i < (l.b0 + 1) * 16; i++) {
        l.tpl[i] = ((uint32_t)tail[i * 4] << 24) | ((uint32_t)tail[i * 4 + 1] << 16) |
                   ((uint32_t)tail[i * 4 + 2] << 8) | (uint32_t)tail[i * 4 + 3];
    }

    l.trailKW.resize((size_t)l.nTrail * 64);
    for (int b = 0; b < l.nTrail; b++) {
        uint32_t w[64];
        hostSchedule(tail.data() + (size_t)(l.b0 + 1 + b) * 64, w);
        for (int i = 0; i < 64; i++) l.trailKW[(size_t)b * 64 + i] = w[i] + H_SHA256_K[i];
    }
    return l;
}

#define POW_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) return fail(std::string(#call) + ": " + cudaGetErrorString(err_)); \
} while (0)

#define POW_CHECK_STATUS(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { fail(std::string(#call) + ": " + cudaGetErrorString(err_)); return POW_ERROR; } \
} while (0)

CudaPowMiner::CudaPowMiner() : ctx(nullptr) {}

CudaPowMiner::~CudaPowMiner() {
    cleanup();
}

bool CudaPowMiner::fail(const std::string &message) {
    error = message;
    return false;
}

bool CudaPowMiner::init(const std::string &prefix, const std::string &suffix, int targetDifficulty, int device) {
    cleanup();
    error.clear();
    if (setup(prefix, suffix, targetDifficulty, device)) return true;
    std::string reason = error;
    cleanup();
    error = reason;
    return false;
}

bool CudaPowMiner::setup(const std::string &prefix, const std::string &suffix, int targetDifficulty, int device) {
    if (targetDifficulty < 1 || targetDifficulty > 256) {
        return fail("target difficulty must be between 1 and 256");
    }

    const int devices = deviceCount();
    if (devices == 0) return fail("no CUDA device available");
    if (device < 0 || device >= devices) {
        return fail("CUDA device " + std::to_string(device) + " does not exist");
    }

    ctx = new CudaPowContext();
    for (int s = 0; s < POW_SLOTS; s++) ctx->slots[s] = PowSlot{nullptr, nullptr, nullptr, 0};
    ctx->device = device;
    ctx->prefix = prefix;
    ctx->suffix = suffix;
    ctx->target = targetDifficulty;

    POW_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    POW_CHECK(cudaGetDeviceProperties(&prop, device));
    ctx->smCount = prop.multiProcessorCount;
    gpuName = prop.name;

    memcpy(ctx->midstate, H_SHA256_IV, sizeof(ctx->midstate));
    size_t consumed = 0;
    while (consumed + 64 <= prefix.size()) {
        hostCompress(ctx->midstate, (const uint8_t *)prefix.data() + consumed);
        consumed += 64;
    }
    ctx->prefixRem = (int)(prefix.size() - consumed);

    for (int digits = MIN_NONCE_DIGITS; digits <= MAX_NONCE_DIGITS; digits++) {
        if ((ctx->prefixRem + digits - LOW_DIGITS) % 4 == 0) ctx->layouts.push_back(buildLayout(*ctx, digits));
    }
    std::stable_sort(ctx->layouts.begin(), ctx->layouts.end(), [](const PowLayout &a, const PowLayout &b) {
        if (a.cost != b.cost) return a.cost < b.cost;
        return a.digits > b.digits;
    });

    POW_CHECK(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking));

    for (auto &l : ctx->layouts) {
        l.firstUnit = ctx->totalUnits;
        ctx->totalUnits += l.units;
        ctx->limit = ctx->totalUnits;
        if (l.nTrail > 0) {
            POW_CHECK(cudaMalloc(&l.d_trailKW, l.trailKW.size() * sizeof(uint32_t)));
            POW_CHECK(cudaMemcpy(l.d_trailKW, l.trailKW.data(), l.trailKW.size() * sizeof(uint32_t),
                                 cudaMemcpyHostToDevice));
        }
        int blocksPerSM = 0;
        POW_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSM, POW_KERNELS[l.b0][l.wv].fn, POW_THREADS_PER_BLOCK, 0));
        if (blocksPerSM < 1) blocksPerSM = 1;
        l.launchUnits = (uint32_t)ctx->smCount * blocksPerSM * POW_THREADS_PER_BLOCK * POW_LAUNCH_WAVES;
    }

    for (int s = 0; s < POW_SLOTS; s++) {
        POW_CHECK(cudaMalloc(&ctx->slots[s].d_out, sizeof(PowOutput)));
        POW_CHECK(cudaMallocHost(&ctx->slots[s].h_out, sizeof(PowOutput)));
        POW_CHECK(cudaEventCreateWithFlags(&ctx->slots[s].done, cudaEventDisableTiming));
    }
    return true;
}

void CudaPowMiner::cleanup() {
    if (!ctx) return;
    cudaSetDevice(ctx->device);
    if (ctx->stream) cudaStreamSynchronize(ctx->stream);
    for (auto &l : ctx->layouts) {
        if (l.d_trailKW) cudaFree(l.d_trailKW);
    }
    for (int s = 0; s < POW_SLOTS; s++) {
        if (ctx->slots[s].d_out) cudaFree(ctx->slots[s].d_out);
        if (ctx->slots[s].h_out) cudaFreeHost(ctx->slots[s].h_out);
        if (ctx->slots[s].done) cudaEventDestroy(ctx->slots[s].done);
    }
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    cudaGetLastError();
    delete ctx;
    ctx = nullptr;
}

uint64_t CudaPowMiner::cursor() const {
    return ctx ? ctx->cursor : 0;
}

uint64_t CudaPowMiner::unitCount() const {
    return ctx ? ctx->totalUnits : 0;
}

uint64_t CudaPowMiner::noncesPerUnit() {
    return NONCES_PER_UNIT;
}

void CudaPowMiner::setRange(uint64_t start, uint64_t end) {
    if (!ctx) return;
    cudaSetDevice(ctx->device);
    cudaStreamSynchronize(ctx->stream);
    ctx->inFlight = 0;
    ctx->head = 0;
    ctx->limit = std::min(end, ctx->totalUnits);
    ctx->cursor = std::min(start, ctx->limit);
}

int CudaPowMiner::deviceCount() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return n;
}

static bool enqueue(CudaPowContext *c, uint64_t start, uint64_t limit, PowSlot &slot, std::string &err) {
    const PowLayout *layout = nullptr;
    for (const auto &l : c->layouts) {
        if (start >= l.firstUnit && start < l.firstUnit + l.units) { layout = &l; break; }
    }
    if (!layout) {
        err = "unit outside every nonce layout";
        return false;
    }

    uint64_t end = std::min(layout->firstUnit + layout->units, limit);
    uint32_t count = (uint32_t)std::min<uint64_t>(end - start, layout->launchUnits);

    PowParams p;
    memcpy(p.midstate, c->midstate, sizeof(p.midstate));
    memcpy(p.tpl, layout->tpl, sizeof(p.tpl));
    p.trailKW = layout->d_trailKW;
    memset(p.kw, 0, sizeof(p.kw));
    memcpy(p.kw, layout->trailKW.data(), std::min<size_t>(layout->trailKW.size(), POW_PARAM_TRAIL * 64) * sizeof(uint32_t));
    p.out = slot.d_out;
    p.unitStart = start - layout->firstUnit;
    p.highBase = layout->highBase;
    p.unitCount = count;
    p.nTrail = layout->nTrail;
    p.highDigits = layout->digits - LOW_DIGITS;
    p.target = c->target;
    p.firstTarget = std::min(c->target, 32);

    cudaError_t e = cudaMemsetAsync(slot.d_out, 0, sizeof(PowOutput), c->stream);
    if (e == cudaSuccess) {
        dim3 grid((count + POW_THREADS_PER_BLOCK - 1) / POW_THREADS_PER_BLOCK);
        POW_KERNELS[layout->b0][layout->wv].launch(grid, c->stream, p);
        e = cudaGetLastError();
    }
    if (e == cudaSuccess) e = cudaMemcpyAsync(slot.h_out, slot.d_out, sizeof(PowOutput), cudaMemcpyDeviceToHost, c->stream);
    if (e == cudaSuccess) e = cudaEventRecord(slot.done, c->stream);
    if (e != cudaSuccess) {
        err = cudaGetErrorString(e);
        return false;
    }
    slot.units = count;
    return true;
}

static std::string nonceString(uint64_t nonce) {
    char buf[24];
    snprintf(buf, sizeof(buf), "%llu", (unsigned long long)nonce);
    return buf;
}

PowStatus CudaPowMiner::mine(PowResult &result, uint64_t &attempts) {
    result.found = false;
    attempts = 0;
    if (!ctx) { fail("miner is not initialised"); return POW_ERROR; }
    POW_CHECK_STATUS(cudaSetDevice(ctx->device));

    while (ctx->inFlight < POW_SLOTS && ctx->cursor < ctx->limit) {
        PowSlot &slot = ctx->slots[(ctx->head + ctx->inFlight) % POW_SLOTS];
        std::string err;
        if (!enqueue(ctx, ctx->cursor, ctx->limit, slot, err)) { fail("kernel launch failed: " + err); return POW_ERROR; }
        ctx->cursor += slot.units;
        ctx->inFlight++;
    }
    if (ctx->inFlight == 0) return POW_EXHAUSTED;

    PowSlot &slot = ctx->slots[ctx->head];
    POW_CHECK_STATUS(cudaEventSynchronize(slot.done));
    ctx->head = (ctx->head + 1) % POW_SLOTS;
    ctx->inFlight--;
    attempts = slot.units * NONCES_PER_UNIT;

    if (!slot.h_out->found) return POW_NOT_FOUND;

    uint8_t hash[32];
    hostSha256(ctx->prefix + nonceString(slot.h_out->nonce) + ctx->suffix, hash);
    for (int i = 0; i < 8; i++) {
        uint32_t w = slot.h_out->hash[i];
        uint8_t be[4] = {(uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w};
        if (memcmp(be, hash + i * 4, 4) != 0) {
            fail("GPU hash for nonce " + nonceString(slot.h_out->nonce) + " does not match the CPU hash");
            return POW_ERROR;
        }
    }
    int bits = leadingZeroBits(hash);
    if (bits < ctx->target) {
        fail("GPU nonce " + nonceString(slot.h_out->nonce) + " has only " + std::to_string(bits) + " leading zero bits");
        return POW_ERROR;
    }
    result.found = true;
    result.nonce = slot.h_out->nonce;
    memcpy(result.eventId, hash, 32);
    result.difficulty = bits;
    return POW_FOUND;
}
