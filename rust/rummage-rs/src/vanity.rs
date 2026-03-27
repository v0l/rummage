use std::ffi::CString;

use rummage_sys::{
    rummage_load_gtable, rummage_vanity_check_results, rummage_vanity_destroy,
    rummage_vanity_iterate, rummage_vanity_keys_generated, rummage_vanity_load_checkpoint,
    rummage_vanity_matches_found, rummage_vanity_new, rummage_vanity_progress,
    rummage_vanity_save_checkpoint, rummage_vanity_set_bech32_verify,
    RummageSearchMode_RUMMAGE_SEARCH_RANDOM, RummageSearchMode_RUMMAGE_SEARCH_SEQUENTIAL,
    RummageVanity, RummageVanityMode_RUMMAGE_VANITY_BECH32_BOTH,
    RummageVanityMode_RUMMAGE_VANITY_BECH32_PREFIX, RummageVanityMode_RUMMAGE_VANITY_BECH32_SUFFIX,
    RummageVanityMode_RUMMAGE_VANITY_HEX_BOTH, RummageVanityMode_RUMMAGE_VANITY_HEX_PREFIX,
    RummageVanityMode_RUMMAGE_VANITY_HEX_SUFFIX,
};

use crate::pow::Error;

/// Size of each GTable axis buffer in bytes: 16 * 65536 * 32 = 33,554,432.
pub const GTABLE_SIZE: usize = 16 * 65536 * 32;

/// Pre-computed secp256k1 generator table for vanity mining.
///
/// This is ~64 MB of data (32 MB per axis). It only needs to be computed
/// once and can be shared across multiple [`VanityMiner`] instances.
pub struct GTable {
    pub(crate) x: Vec<u8>,
    pub(crate) y: Vec<u8>,
}

impl GTable {
    /// Compute the GTable.  This takes a few seconds on first call.
    pub fn load() -> Result<Self, Error> {
        let mut x = vec![0u8; GTABLE_SIZE];
        let mut y = vec![0u8; GTABLE_SIZE];
        let ok = unsafe { rummage_load_gtable(x.as_mut_ptr(), y.as_mut_ptr()) };
        if ok != 0 {
            Ok(Self { x, y })
        } else {
            Err(Error::InitFailed)
        }
    }
}

/// Vanity pattern matching mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VanityMode {
    /// Match hex prefix of the raw public key.
    HexPrefix,
    /// Match hex suffix of the raw public key.
    HexSuffix,
    /// Match hex prefix AND suffix of the raw public key.
    HexBoth,
    /// Match bech32 prefix of the npub address.
    Bech32Prefix,
    /// Match bech32 suffix of the npub address.
    Bech32Suffix,
    /// Match bech32 prefix AND suffix of the npub address.
    Bech32Both,
}

impl VanityMode {
    fn to_raw(self) -> u32 {
        match self {
            Self::HexPrefix => RummageVanityMode_RUMMAGE_VANITY_HEX_PREFIX,
            Self::HexSuffix => RummageVanityMode_RUMMAGE_VANITY_HEX_SUFFIX,
            Self::HexBoth => RummageVanityMode_RUMMAGE_VANITY_HEX_BOTH,
            Self::Bech32Prefix => RummageVanityMode_RUMMAGE_VANITY_BECH32_PREFIX,
            Self::Bech32Suffix => RummageVanityMode_RUMMAGE_VANITY_BECH32_SUFFIX,
            Self::Bech32Both => RummageVanityMode_RUMMAGE_VANITY_BECH32_BOTH,
        }
    }
}

/// Key generation search mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Random key generation (default).
    Random,
    /// Sequential exhaustive search from a starting offset.
    Sequential,
}

impl SearchMode {
    fn to_raw(self) -> u32 {
        match self {
            Self::Random => RummageSearchMode_RUMMAGE_SEARCH_RANDOM,
            Self::Sequential => RummageSearchMode_RUMMAGE_SEARCH_SEQUENTIAL,
        }
    }
}

/// A vanity key match found by the GPU miner.
#[derive(Debug, Clone)]
pub struct VanityResult {
    /// The 32-byte private key (big-endian, ready for Nostr use).
    pub private_key: [u8; 32],
    /// The 32-byte x-only public key.
    pub public_key: [u8; 32],
}

/// GPU-accelerated vanity npub/hex key miner.
///
/// Generates secp256k1 keypairs at high speed on the GPU and checks
/// whether the public key matches a target pattern.
pub struct VanityMiner {
    handle: *mut RummageVanity,
}

// The GPU handle is not tied to a specific thread.
unsafe impl Send for VanityMiner {}

impl VanityMiner {
    /// Create a new vanity miner.
    ///
    /// - `gtable` — pre-computed generator table (see [`GTable::load`]).
    /// - `pattern` — the hex or bech32 pattern to search for.
    /// - `mode` — which part(s) of the key to match.
    /// - `start_offset` — 32-byte random starting point for key generation.
    /// - `search_mode` — random or sequential search.
    /// - `bech32_pattern_len` — length of the original bech32 pattern (0 if hex mode).
    pub fn new(
        gtable: &GTable,
        pattern: &str,
        mode: VanityMode,
        start_offset: &[u8; 32],
        search_mode: SearchMode,
        bech32_pattern_len: i32,
    ) -> Result<Self, Error> {
        let c_pattern = CString::new(pattern).map_err(|_| Error::InitFailed)?;
        let handle = unsafe {
            rummage_vanity_new(
                gtable.x.as_ptr(),
                gtable.y.as_ptr(),
                c_pattern.as_ptr(),
                mode.to_raw(),
                start_offset.as_ptr(),
                search_mode.to_raw(),
                bech32_pattern_len,
            )
        };
        if handle.is_null() {
            Err(Error::InitFailed)
        } else {
            Ok(Self { handle })
        }
    }

    /// Enable bech32 verification for patterns that were converted from
    /// bech32 to hex for fast GPU pre-filtering.
    pub fn set_bech32_verification(&mut self, original_pattern: &str, original_mode: VanityMode) {
        let c_pat = CString::new(original_pattern).expect("pattern must not contain NUL");
        unsafe {
            rummage_vanity_set_bech32_verify(self.handle, c_pat.as_ptr(), original_mode.to_raw());
        }
    }

    /// Run one iteration of vanity mining on the GPU.
    pub fn iterate(&mut self, iteration: u64) {
        unsafe { rummage_vanity_iterate(self.handle, iteration) }
    }

    /// Check for results from the last iteration.
    ///
    /// Returns `true` if at least one match was found.
    /// Currently, matches are printed to stdout and saved to `keys.txt`
    /// by the underlying C++ implementation.
    pub fn check_results(&mut self) -> bool {
        let found =
            unsafe { rummage_vanity_check_results(self.handle, None, std::ptr::null_mut()) };
        found != 0
    }

    /// Total keys generated across all iterations.
    pub fn keys_generated(&self) -> u64 {
        unsafe { rummage_vanity_keys_generated(self.handle) }
    }

    /// Total matches found so far.
    pub fn matches_found(&self) -> u64 {
        unsafe { rummage_vanity_matches_found(self.handle) }
    }

    /// Save a checkpoint for sequential search mode.
    pub fn save_checkpoint(&mut self, filename: &str) -> bool {
        let c_name = CString::new(filename).expect("filename must not contain NUL");
        unsafe { rummage_vanity_save_checkpoint(self.handle, c_name.as_ptr()) != 0 }
    }

    /// Load a checkpoint for sequential search mode.
    pub fn load_checkpoint(&mut self, filename: &str) -> bool {
        let c_name = CString::new(filename).expect("filename must not contain NUL");
        unsafe { rummage_vanity_load_checkpoint(self.handle, c_name.as_ptr()) != 0 }
    }

    /// Search progress for sequential mode (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        unsafe { rummage_vanity_progress(self.handle) }
    }
}

impl Drop for VanityMiner {
    fn drop(&mut self) {
        unsafe { rummage_vanity_destroy(self.handle) }
    }
}
