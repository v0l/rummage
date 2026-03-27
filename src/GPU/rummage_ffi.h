/*
 * rummage_ffi.h - C-compatible FFI interface for Rummage GPU miners
 *
 * This header exposes both CudaPowMiner and GPURummage through plain C
 * functions using opaque handles.  It deliberately has NO C++ or CUDA
 * dependencies so it can be consumed from Rust (bindgen), Go (cgo), etc.
 */

#ifndef RUMMAGE_FFI_H
#define RUMMAGE_FFI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * PoW Miner (NIP-13)
 * =================================================================== */

/** Opaque handle to a CudaPowMiner instance. */
typedef struct RummagePow RummagePow;

/** Result of a PoW mining batch. */
typedef struct {
    int      found;          /* non-zero if a valid nonce was found       */
    uint64_t nonce;          /* the winning nonce value                   */
    uint8_t  event_id[32];   /* SHA256 event ID                          */
    int      difficulty;     /* actual leading zero bits achieved         */
} RummagePowResult;

/** Create a new PoW miner.  Returns NULL on failure. */
RummagePow *rummage_pow_new(void);

/** Initialise the PoW miner with a split template.
 *  prefix/suffix are the NIP-01 serialisation split at the nonce position.
 *  Returns non-zero on success. */
int rummage_pow_init(RummagePow *handle,
                     const char *prefix, size_t prefix_len,
                     const char *suffix, size_t suffix_len,
                     int target_difficulty);

/** Run one batch of nonce mining on the GPU.
 *  Returns non-zero if a valid nonce was found (written to *result). */
int rummage_pow_mine_batch(RummagePow *handle,
                           uint64_t nonce_start,
                           uint32_t batch_size,
                           RummagePowResult *result);

/** Get the number of CUDA threads per launch (per stream). */
uint32_t rummage_pow_thread_count(const RummagePow *handle);

/** Get the number of CUDA streams. */
int rummage_pow_stream_count(const RummagePow *handle);

/** Get the number of nonces each thread processes per launch. */
int rummage_pow_nonces_per_thread(const RummagePow *handle);

/** Release GPU resources (can be re-init'd afterwards). */
void rummage_pow_cleanup(RummagePow *handle);

/** Destroy the miner and free all memory. */
void rummage_pow_destroy(RummagePow *handle);


/* ===================================================================
 * Vanity Key Miner
 * =================================================================== */

/** Opaque handle to a GPURummage instance. */
typedef struct RummageVanity RummageVanity;

/** Vanity pattern matching modes (mirrors VanityMode enum). */
typedef enum {
    RUMMAGE_VANITY_HEX_PREFIX     = 0,
    RUMMAGE_VANITY_HEX_SUFFIX     = 1,
    RUMMAGE_VANITY_HEX_BOTH       = 2,
    RUMMAGE_VANITY_BECH32_PREFIX  = 3,
    RUMMAGE_VANITY_BECH32_SUFFIX  = 4,
    RUMMAGE_VANITY_BECH32_BOTH    = 5
} RummageVanityMode;

/** Search modes (mirrors SearchMode enum). */
typedef enum {
    RUMMAGE_SEARCH_RANDOM     = 0,
    RUMMAGE_SEARCH_SEQUENTIAL = 1
} RummageSearchMode;

/** A single vanity match result. */
typedef struct {
    uint8_t private_key[32];
    uint8_t public_key[32];
} RummageVanityResult;

/** Callback invoked for each vanity match.
 *  Return non-zero to keep mining, zero to stop. */
typedef int (*rummage_vanity_match_cb)(const RummageVanityResult *result, void *user_data);

/** Load the secp256k1 GTable.
 *  Caller must provide two buffers of at least
 *  (16 * 65536 * 32) = 33,554,432 bytes each.
 *  Returns non-zero on success. */
int rummage_load_gtable(uint8_t *gtable_x, uint8_t *gtable_y);

/** Create a new vanity miner.
 *  gtable_x/gtable_y: pre-loaded via rummage_load_gtable().
 *  pattern:           hex or bech32 pattern (null-terminated).
 *  start_offset:      32-byte starting offset for key generation.
 *  bech32_pattern_len: original bech32 length (0 if hex mode).
 *  Returns NULL on failure. */
RummageVanity *rummage_vanity_new(const uint8_t *gtable_x,
                                  const uint8_t *gtable_y,
                                  const char *pattern,
                                  RummageVanityMode mode,
                                  const uint8_t *start_offset,
                                  RummageSearchMode search_mode,
                                  int bech32_pattern_len);

/** Enable bech32 verification for hex-converted patterns. */
void rummage_vanity_set_bech32_verify(RummageVanity *handle,
                                      const char *original_pattern,
                                      RummageVanityMode original_mode);

/** Run one iteration of vanity mining. */
void rummage_vanity_iterate(RummageVanity *handle, uint64_t iteration);

/** Check for results and invoke callback for each match.
 *  Returns the number of matches found in this check. */
int rummage_vanity_check_results(RummageVanity *handle,
                                 rummage_vanity_match_cb callback,
                                 void *user_data);

/** Get total keys generated so far. */
uint64_t rummage_vanity_keys_generated(const RummageVanity *handle);

/** Get total matches found so far. */
uint64_t rummage_vanity_matches_found(const RummageVanity *handle);

/** Sequential mode: save checkpoint to file.  Returns non-zero on success. */
int rummage_vanity_save_checkpoint(RummageVanity *handle, const char *filename);

/** Sequential mode: load checkpoint from file.  Returns non-zero on success. */
int rummage_vanity_load_checkpoint(RummageVanity *handle, const char *filename);

/** Sequential mode: progress 0.0 .. 1.0 */
double rummage_vanity_progress(const RummageVanity *handle);

/** Free GPU memory and destroy the miner. */
void rummage_vanity_destroy(RummageVanity *handle);

#ifdef __cplusplus
}
#endif

#endif /* RUMMAGE_FFI_H */
