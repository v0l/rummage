use std::ffi::c_char;
use std::fmt;

use rummage_sys::{
    rummage_pow_cleanup, rummage_pow_destroy, rummage_pow_init, rummage_pow_mine_batch,
    rummage_pow_new, rummage_pow_stream_count, rummage_pow_thread_count, RummagePow,
    RummagePowResult,
};

/// Result of a successful PoW mining operation.
#[derive(Debug, Clone)]
pub struct PowResult {
    /// The nonce value that satisfies the difficulty target.
    pub nonce: u64,
    /// The SHA256 event ID (32 bytes).
    pub event_id: [u8; 32],
    /// The actual number of leading zero bits achieved.
    pub difficulty: i32,
}

/// GPU-accelerated NIP-13 Proof of Work miner.
///
/// Mines a nonce for a Nostr event such that the event ID (SHA256 of the
/// NIP-01 serialisation) has a target number of leading zero bits.
///
/// The event template is split into a `prefix` (everything before the nonce
/// value) and a `suffix` (everything after), following NIP-01 serialisation:
///
/// ```text
/// [0,"<pubkey>",<created_at>,<kind>,[...tags...,["nonce","<NONCE>","<target>"]],\"<content>\"]
///                                                        ^^^^^^^^
///                                                    split point
/// ```
pub struct PowMiner {
    handle: *mut RummagePow,
}

// The GPU handle is not tied to a specific thread.
unsafe impl Send for PowMiner {}

impl PowMiner {
    /// Create a new PoW miner.  Allocates the handle but does not touch the GPU.
    pub fn new() -> Option<Self> {
        let handle = unsafe { rummage_pow_new() };
        if handle.is_null() {
            None
        } else {
            Some(Self { handle })
        }
    }

    /// Initialise the GPU miner with a split event template.
    ///
    /// Must be called before [`mine_batch`](Self::mine_batch).
    /// Can be called again to reinitialise with a new template
    /// (e.g. after refreshing `created_at`), but call [`cleanup`](Self::cleanup) first.
    pub fn init(
        &mut self,
        prefix: &str,
        suffix: &str,
        target_difficulty: i32,
    ) -> Result<(), Error> {
        let ok = unsafe {
            rummage_pow_init(
                self.handle,
                prefix.as_ptr() as *const c_char,
                prefix.len(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                target_difficulty,
            )
        };
        if ok != 0 {
            Ok(())
        } else {
            Err(Error::InitFailed)
        }
    }

    /// Run one batch of nonce mining on the GPU.
    ///
    /// `nonce_start` — first nonce to try.  
    /// `batch_size` — should equal [`thread_count`](Self::thread_count) for optimal occupancy.
    ///
    /// Returns `Some(PowResult)` if a valid nonce was found, `None` otherwise.
    pub fn mine_batch(&mut self, nonce_start: u64, batch_size: u32) -> Option<PowResult> {
        let mut raw = RummagePowResult::default();
        let found =
            unsafe { rummage_pow_mine_batch(self.handle, nonce_start, batch_size, &mut raw) };
        if found != 0 {
            Some(PowResult {
                nonce: raw.nonce,
                event_id: raw.event_id,
                difficulty: raw.difficulty,
            })
        } else {
            None
        }
    }

    /// Number of CUDA threads launched per stream per batch.
    pub fn thread_count(&self) -> u32 {
        unsafe { rummage_pow_thread_count(self.handle) }
    }

    /// Number of CUDA streams used for async dispatch.
    pub fn stream_count(&self) -> i32 {
        unsafe { rummage_pow_stream_count(self.handle) }
    }

    /// Release GPU resources without destroying the handle.
    ///
    /// The miner can be re-initialised with [`init`](Self::init) afterwards.
    pub fn cleanup(&mut self) {
        unsafe { rummage_pow_cleanup(self.handle) }
    }
}

impl Drop for PowMiner {
    fn drop(&mut self) {
        unsafe { rummage_pow_destroy(self.handle) }
    }
}

/// Errors returned by the Rummage library.
#[derive(Debug, Clone)]
pub enum Error {
    /// GPU initialisation failed (no CUDA device, out of memory, etc.).
    InitFailed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InitFailed => write!(f, "GPU initialisation failed"),
        }
    }
}

impl std::error::Error for Error {}
