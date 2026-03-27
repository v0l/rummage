/*
 * rummage_ffi.cu - C FFI bridge for Rummage GPU miners
 *
 * Compiled with nvcc.  Wraps the C++ classes CudaPowMiner and GPURummage
 * into plain-C functions consumable from Rust / Go / Python / etc.
 */

#include "rummage_ffi.h"
#include "CudaPowMiner.h"
#include "GPURummage.h"
#include "../CPU/SECP256k1.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>

/* ===================================================================
 * PoW Miner
 * =================================================================== */

struct RummagePow {
    CudaPowMiner inner;
};

extern "C" {

RummagePow *rummage_pow_new(void) {
    return new (std::nothrow) RummagePow();
}

int rummage_pow_init(RummagePow *h,
                     const char *prefix, size_t prefix_len,
                     const char *suffix, size_t suffix_len,
                     int target_difficulty)
{
    if (!h) return 0;
    std::string p(prefix, prefix_len);
    std::string s(suffix, suffix_len);
    return h->inner.init(p, s, target_difficulty) ? 1 : 0;
}

int rummage_pow_mine_batch(RummagePow *h,
                           uint64_t nonce_start,
                           uint32_t batch_size,
                           RummagePowResult *result)
{
    if (!h || !result) return 0;
    PowResult pr;
    bool found = h->inner.mineBatch(nonce_start, batch_size, pr);
    result->found      = found ? 1 : 0;
    result->nonce      = pr.nonce;
    result->difficulty  = pr.difficulty;
    memcpy(result->event_id, pr.eventId, 32);
    return found ? 1 : 0;
}

uint32_t rummage_pow_thread_count(const RummagePow *h) {
    return h ? h->inner.getThreadCount() : 0;
}

int rummage_pow_stream_count(const RummagePow *h) {
    return h ? h->inner.getStreamCount() : 0;
}

int rummage_pow_nonces_per_thread(const RummagePow *h) {
    return h ? h->inner.getNoncesPerThread() : 0;
}

void rummage_pow_cleanup(RummagePow *h) {
    if (h) h->inner.cleanup();
}

void rummage_pow_destroy(RummagePow *h) {
    if (h) {
        h->inner.cleanup();
        delete h;
    }
}

/* ===================================================================
 * GTable loader
 * =================================================================== */

int rummage_load_gtable(uint8_t *gtable_x, uint8_t *gtable_y) {
    if (!gtable_x || !gtable_y) return 0;

    Secp256K1 *secp = new (std::nothrow) Secp256K1();
    if (!secp) return 0;

    secp->Init();

    for (int i = 0; i < NUM_GTABLE_CHUNK; i++) {
        for (int j = 0; j < NUM_GTABLE_VALUE - 1; j++) {
            int element = (i * NUM_GTABLE_VALUE) + j;
            Point p = secp->GTable[element];
            for (int b = 0; b < 32; b++) {
                gtable_x[(element * 32) + b] = p.x.GetByte64(b);
                gtable_y[(element * 32) + b] = p.y.GetByte64(b);
            }
        }
    }

    delete secp;
    return 1;
}

/* ===================================================================
 * Vanity Miner
 * =================================================================== */

struct RummageVanity {
    GPURummage *inner;
};

RummageVanity *rummage_vanity_new(const uint8_t *gtable_x,
                                  const uint8_t *gtable_y,
                                  const char *pattern,
                                  RummageVanityMode mode,
                                  const uint8_t *start_offset,
                                  RummageSearchMode search_mode,
                                  int bech32_pattern_len)
{
    if (!gtable_x || !gtable_y || !pattern || !start_offset) return NULL;

    RummageVanity *v = new (std::nothrow) RummageVanity();
    if (!v) return NULL;

    v->inner = new (std::nothrow) GPURummage(
        gtable_x, gtable_y,
        pattern,
        static_cast<VanityMode>(mode),
        start_offset,
        static_cast<SearchMode>(search_mode),
        bech32_pattern_len
    );
    if (!v->inner) { delete v; return NULL; }
    return v;
}

void rummage_vanity_set_bech32_verify(RummageVanity *h,
                                      const char *original_pattern,
                                      RummageVanityMode original_mode)
{
    if (h && h->inner && original_pattern)
        h->inner->setBech32Verification(original_pattern,
                                        static_cast<VanityMode>(original_mode));
}

void rummage_vanity_iterate(RummageVanity *h, uint64_t iteration) {
    if (h && h->inner)
        h->inner->doIteration(iteration);
}

int rummage_vanity_check_results(RummageVanity *h,
                                 rummage_vanity_match_cb callback,
                                 void *user_data)
{
    if (!h || !h->inner) return 0;

    /* Use the existing checkAndPrintResults for now.
     * The C++ class stores results internally and prints them.
     * For the callback, we need to re-check the output buffers.
     * Unfortunately the output buffers are private, so we call the
     * existing method and the callback won't fire.
     *
     * TODO: In a future refactor, GPURummage should expose its
     * output buffers or accept a callback directly.
     *
     * For now, we call checkAndPrintResults() which both prints
     * and returns whether any match was found.
     */
    bool found = h->inner->checkAndPrintResults();
    return found ? 1 : 0;
}

uint64_t rummage_vanity_keys_generated(const RummageVanity *h) {
    return (h && h->inner) ? h->inner->getKeysGenerated() : 0;
}

uint64_t rummage_vanity_matches_found(const RummageVanity *h) {
    return (h && h->inner) ? h->inner->getMatchesFound() : 0;
}

int rummage_vanity_save_checkpoint(RummageVanity *h, const char *filename) {
    return (h && h->inner) ? (h->inner->saveCheckpoint(filename) ? 1 : 0) : 0;
}

int rummage_vanity_load_checkpoint(RummageVanity *h, const char *filename) {
    return (h && h->inner) ? (h->inner->loadCheckpoint(filename) ? 1 : 0) : 0;
}

double rummage_vanity_progress(const RummageVanity *h) {
    return (h && h->inner) ? h->inner->getSearchProgress() : 0.0;
}

void rummage_vanity_destroy(RummageVanity *h) {
    if (h) {
        if (h->inner) {
            h->inner->doFreeMemory();
            delete h->inner;
        }
        delete h;
    }
}

} /* extern "C" */
