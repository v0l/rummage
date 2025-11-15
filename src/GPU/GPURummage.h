/*
 * Rummage - GPU Nostr Vanity Key Miner - Header
 *
 * Copyright (c) 2025 rossbates
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef GPURUMMAGE_H
#define GPURUMMAGE_H

#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

//CUDA-specific parameters that determine occupancy and thread-count
//Adjust according to your GPU specs
// RTX 3060: 28 SMs, optimize for high occupancy
#define NOSTR_BLOCKS_PER_GRID 512    // Balanced for memory and performance
#define NOSTR_THREADS_PER_BLOCK 256  // Keep at 256 (optimal for most kernels)
#define KEYS_PER_THREAD_BATCH 64     // Each thread generates multiple keys per iteration

#define NOSTR_COUNT_CUDA_THREADS (NOSTR_BLOCKS_PER_GRID * NOSTR_THREADS_PER_BLOCK)
#define NOSTR_IDX_CUDA_THREAD ((blockIdx.x * blockDim.x) + threadIdx.x)

//Maximum vanity prefix/suffix length in characters
#define MAX_VANITY_HEX_LEN 16
#define MAX_VANITY_BECH32_LEN 52

//Size definitions
#define SIZE_PRIV_KEY_NOSTR 32  // 32-byte private key
#define SIZE_PUBKEY_NOSTR 32    // 32-byte x-only public key (Schnorr)
#define SIZE_LONG 8             // Each Long is 8 bytes

//GTable configuration (same as main secp)
#define NUM_GTABLE_CHUNK 16
#define NUM_GTABLE_VALUE 65536
#define SIZE_GTABLE_POINT 32
#define COUNT_GTABLE_POINTS (NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE)

//Contains the first element index for each chunk
__constant__ int NOSTR_CHUNK_FIRST_ELEMENT[NUM_GTABLE_CHUNK] = {
  65536*0,  65536*1,  65536*2,  65536*3,
  65536*4,  65536*5,  65536*6,  65536*7,
  65536*8,  65536*9,  65536*10, 65536*11,
  65536*12, 65536*13, 65536*14, 65536*15,
};

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

// Vanity pattern matching modes
enum VanityMode {
    VANITY_HEX_PREFIX = 0,
    VANITY_HEX_SUFFIX = 1,
    VANITY_HEX_BOTH = 2,
    VANITY_BECH32_PREFIX = 3,
    VANITY_BECH32_SUFFIX = 4,
    VANITY_BECH32_BOTH = 5
};

// Search modes
enum SearchMode {
    SEARCH_RANDOM = 0,      // Random key generation (default)
    SEARCH_SEQUENTIAL = 1   // Sequential exhaustive search
};

class GPURummage
{
public:
    GPURummage(
        const uint8_t *gTableXCPU,
        const uint8_t *gTableYCPU,
        const char *vanityPattern,
        VanityMode mode,
        const uint8_t *startOffset,
        SearchMode searchMode = SEARCH_RANDOM,
        int bech32PatternLen = 0  // Original bech32 pattern length (0 if not bech32)
    );

    // Run one iteration of vanity mining
    void doIteration(uint64_t iteration);

    // Check for and print any found keys
    bool checkAndPrintResults();

    // Free GPU memory
    void doFreeMemory();

    // Get statistics
    uint64_t getKeysGenerated() const { return keysGenerated; }
    uint64_t getMatchesFound() const { return matchesFound; }

    // Sequential search methods
    bool saveCheckpoint(const char *filename);
    bool loadCheckpoint(const char *filename);
    double getSearchProgress() const;  // Returns 0.0 to 1.0
    uint64_t getCurrentIteration() const { return currentIteration; }
    uint64_t getTotalIterations() const { return totalIterations; }

    // Bech32 verification setup (for hex-converted patterns)
    void setBech32Verification(const char *originalPattern, VanityMode originalMode) {
        this->needsBech32Verification = true;
        strncpy(this->originalBech32Pattern, originalPattern, MAX_VANITY_BECH32_LEN);
        this->originalBech32Pattern[MAX_VANITY_BECH32_LEN] = '\0';
        this->originalBech32Mode = originalMode;
    }

private:
    //GTable buffer containing ~1 million pre-computed points for Secp256k1 point multiplication
    uint8_t *gTableXGPU;
    uint8_t *gTableYGPU;

    //Vanity pattern buffer
    uint8_t *vanityPatternGPU;
    uint8_t vanityLen;
    VanityMode vanityMode;

    //Pre-initialized cuRAND states (one per thread)
    curandState *randStatesGPU;

    //Output buffer indicating success (1 if match found)
    uint8_t *outputFoundGPU;
    uint8_t *outputFoundCPU;

    //Output buffer for matched private keys
    uint8_t *outputPrivKeysGPU;
    uint8_t *outputPrivKeysCPU;

    //Output buffer for matched public keys (x-only)
    uint8_t *outputPubKeysGPU;
    uint8_t *outputPubKeysCPU;

    //Statistics
    uint64_t keysGenerated;
    uint64_t matchesFound;
    uint8_t startOffset[32];  // 256-bit starting offset for sequential search

    //Sequential search state
    SearchMode searchMode;
    uint64_t currentIteration;  // Current global iteration for sequential mode
    uint64_t totalIterations;   // Total iterations needed to exhaust space
    uint64_t searchSpaceSize;   // Total keys in search space

    //Bech32 verification (for patterns converted from bech32 to hex)
    bool needsBech32Verification;
    char originalBech32Pattern[MAX_VANITY_BECH32_LEN + 1];
    VanityMode originalBech32Mode;
};

#endif // GPURUMMAGE_H
