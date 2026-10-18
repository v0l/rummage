use std::ffi::{CStr, c_char};
use std::fmt;

use rummage_sys::{
    RummagePow, RummagePowResult, RummagePowStatus_RUMMAGE_POW_ERROR,
    RummagePowStatus_RUMMAGE_POW_EXHAUSTED, RummagePowStatus_RUMMAGE_POW_FOUND,
    RummagePowStatus_RUMMAGE_POW_NOT_FOUND, rummage_pow_cleanup, rummage_pow_cursor,
    rummage_pow_destroy, rummage_pow_device_count, rummage_pow_device_name, rummage_pow_init,
    rummage_pow_last_error, rummage_pow_mine, rummage_pow_new, rummage_pow_nonces_per_unit,
    rummage_pow_set_range, rummage_pow_unit_count,
};

/// Result of a successful PoW mining operation.
#[derive(Debug, Clone)]
pub struct PowResult {
    /// The nonce value that satisfies the difficulty target.
    pub nonce: u64,
    /// The SHA256 event ID (32 bytes).
    pub event_id: [u8; 32],
    /// The actual number of leading zero bits achieved.
    pub difficulty: u32,
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
/// [0,"<pubkey>",<created_at>,<kind>,[...tags...,["nonce","<NONCE>","<target>"]],"<content>"]
///                                                        ^^^^^^^^
///                                                    split point
/// ```
pub struct PowMiner {
    handle: *mut RummagePow,
    attempts: u64,
}

unsafe impl Send for PowMiner {}

impl PowMiner {
    /// Create a new PoW miner.  Allocates the handle but does not touch the GPU.
    pub fn new() -> Option<Self> {
        let handle = unsafe { rummage_pow_new() };
        if handle.is_null() {
            None
        } else {
            Some(Self {
                handle,
                attempts: 0,
            })
        }
    }

    pub fn device_count() -> u32 {
        unsafe { rummage_pow_device_count() }.max(0) as u32
    }

    pub fn nonces_per_unit() -> u64 {
        unsafe { rummage_pow_nonces_per_unit() }
    }

    /// Initialise the GPU miner with a split event template.
    ///
    /// Can be called again to start over with a new template (e.g. after
    /// refreshing `created_at`).
    pub fn init(
        &mut self,
        prefix: &str,
        suffix: &str,
        target_difficulty: u32,
    ) -> Result<(), Error> {
        self.init_on_device(prefix, suffix, target_difficulty, 0)
    }

    pub fn init_on_device(
        &mut self,
        prefix: &str,
        suffix: &str,
        target_difficulty: u32,
        device: u32,
    ) -> Result<(), Error> {
        if !(1..=256).contains(&target_difficulty) {
            return Err(Error::InvalidDifficulty(target_difficulty));
        }
        let device = i32::try_from(device)
            .map_err(|_| Error::Init(format!("CUDA device {device} does not exist")))?;
        let ok = unsafe {
            rummage_pow_init(
                self.handle,
                prefix.as_ptr() as *const c_char,
                prefix.len(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                target_difficulty as i32,
                device,
            )
        };
        self.attempts = 0;
        if ok != 0 {
            Ok(())
        } else {
            Err(Error::Init(self.last_error()))
        }
    }

    /// Run one batch of mining with automatic nonce advancement.
    pub fn mine(&mut self) -> Result<Option<PowResult>, Error> {
        let mut raw = RummagePowResult::default();
        let mut attempts = 0u64;
        let status = unsafe { rummage_pow_mine(self.handle, &mut raw, &mut attempts) };
        self.attempts += attempts;
        #[allow(non_upper_case_globals)]
        match status {
            RummagePowStatus_RUMMAGE_POW_FOUND => Ok(Some(PowResult {
                nonce: raw.nonce,
                event_id: raw.event_id,
                difficulty: raw.difficulty as u32,
            })),
            RummagePowStatus_RUMMAGE_POW_NOT_FOUND => Ok(None),
            RummagePowStatus_RUMMAGE_POW_EXHAUSTED => Err(Error::Exhausted),
            RummagePowStatus_RUMMAGE_POW_ERROR => Err(Error::Gpu(self.last_error())),
            other => Err(Error::Gpu(format!("unknown miner status {other}"))),
        }
    }

    pub fn device_name(&self) -> String {
        let ptr = unsafe { rummage_pow_device_name(self.handle) };
        if ptr.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    pub fn attempts(&self) -> u64 {
        self.attempts
    }

    pub fn set_range(&mut self, start: u64, end: u64) {
        unsafe { rummage_pow_set_range(self.handle, start, end) };
        self.attempts = 0;
    }

    pub fn cursor(&self) -> u64 {
        unsafe { rummage_pow_cursor(self.handle) }
    }

    pub fn unit_count(&self) -> u64 {
        unsafe { rummage_pow_unit_count(self.handle) }
    }

    /// Release GPU resources without destroying the handle.
    pub fn cleanup(&mut self) {
        unsafe { rummage_pow_cleanup(self.handle) }
    }

    fn last_error(&self) -> String {
        let ptr = unsafe { rummage_pow_last_error(self.handle) };
        if ptr.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
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
    InvalidDifficulty(u32),
    /// GPU initialisation failed (no CUDA device, out of memory, etc.).
    InitFailed,
    Init(String),
    Gpu(String),
    Exhausted,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidDifficulty(d) => write!(f, "difficulty {d} is outside 1..=256"),
            Error::InitFailed => write!(f, "GPU initialisation failed"),
            Error::Init(e) => write!(f, "GPU initialisation failed: {e}"),
            Error::Gpu(e) => write!(f, "GPU mining failed: {e}"),
            Error::Exhausted => write!(f, "nonce space exhausted for this template"),
        }
    }
}

impl std::error::Error for Error {}
