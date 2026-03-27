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

// Maximum event template size (prefix + suffix combined)
#define POW_MAX_TEMPLATE_SIZE 8192

// Result structure returned to the caller
struct PowResult {
    bool found;
    uint64_t nonce;
    uint8_t eventId[32];     // SHA256 hash (the event ID)
    int difficulty;          // actual leading zero bits achieved
};

// Opaque handle to GPU resources (allocated in .cu, opaque to .cpp)
struct CudaPowContext;

class CudaPowMiner {
public:
    CudaPowMiner();
    ~CudaPowMiner();

    // Initialize GPU resources. Call once before mining.
    // prefix: everything before the nonce value in the serialized event
    // suffix: everything after the nonce value in the serialized event
    // targetDifficulty: required leading zero bits
    bool init(const std::string &prefix, const std::string &suffix, int targetDifficulty);

    // Run one batch of nonce attempts on the GPU.
    // Returns true if a valid nonce was found.
    // nonceStart: first nonce value to try in this batch
    // batchSize: number of nonces to try (should be a multiple of blocks*threads)
    // result: output result if found
    bool mineBatch(uint64_t nonceStart, uint32_t batchSize, PowResult &result);

    // Free GPU resources
    void cleanup();

    // Get the number of CUDA threads used per launch (per stream)
    uint32_t getThreadCount() const;

    // Get the number of CUDA streams
    int getStreamCount() const;

    // Get the number of nonces each thread processes per launch
    int getNoncesPerThread() const;

private:
    CudaPowContext *ctx;
    bool initialized;
};

#endif // CUDA_POW_MINER_H
