/*
 * CudaPowMiner - GPU-accelerated Proof of Work mining for Nostr events (NIP-13)
 *
 * This header is designed to be includable from .cpp files compiled with g++.
 * No CUDA headers are required.
 */

#ifndef CUDA_POW_MINER_H
#define CUDA_POW_MINER_H

#include <stdint.h>
#include <string>

// Result structure returned to the caller
struct PowResult {
    bool found;
    uint64_t nonce;
    uint8_t eventId[32];     // SHA256 hash (the event ID)
    int difficulty;          // actual leading zero bits achieved
};

enum PowStatus {
    POW_NOT_FOUND = 0,
    POW_FOUND = 1,
    POW_ERROR = -1,
    POW_EXHAUSTED = -2,
};

// Opaque handle to GPU resources (allocated in .cu, opaque to .cpp)
struct CudaPowContext;

class CudaPowMiner {
public:
    CudaPowMiner();
    ~CudaPowMiner();
    CudaPowMiner(const CudaPowMiner &) = delete;
    CudaPowMiner &operator=(const CudaPowMiner &) = delete;

    // Initialize GPU resources. Call once before mining.
    // prefix: everything before the nonce value in the serialized event
    // suffix: everything after the nonce value in the serialized event
    // targetDifficulty: required leading zero bits
    bool init(const std::string &prefix, const std::string &suffix, int targetDifficulty, int device = 0);

    PowStatus mine(PowResult &result, uint64_t &attempts);

    // Free GPU resources
    void cleanup();

    uint64_t cursor() const;
    void setRange(uint64_t start, uint64_t end);
    uint64_t unitCount() const;
    static uint64_t noncesPerUnit();
    static int deviceCount();

    const std::string &lastError() const { return error; }
    const std::string &deviceName() const { return gpuName; }

private:
    bool fail(const std::string &message);
    bool setup(const std::string &prefix, const std::string &suffix, int targetDifficulty, int device);

    CudaPowContext *ctx;
    std::string error;
    std::string gpuName;
};

#endif
