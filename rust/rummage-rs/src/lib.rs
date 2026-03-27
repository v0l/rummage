//! Safe Rust wrappers for the Rummage GPU Nostr mining library.
//!
//! This crate provides high-level, memory-safe wrappers around the Rummage
//! CUDA GPU mining library. All GPU resources are automatically cleaned up
//! via `Drop`, and the API uses Rust types instead of raw pointers.
//!
//! # Main Types
//!
//! - [`PowMiner`] — NIP-13 Proof of Work mining for Nostr events
//! - [`VanityMiner`] — Vanity npub/hex key generation
//! - [`GTable`] — Pre-computed secp256k1 generator table (required for vanity mining)
//!
//! # Example: PoW Mining
//!
//! ```no_run
//! use rummage_rs::PowMiner;
//!
//! let prefix = r#"[0,"ab..cd",1234567890,1,[["nonce",""#;
//! let suffix = r#"","21"]],"hello"]"#;
//!
//! let mut miner = PowMiner::new().expect("GPU init");
//! miner.init(prefix, suffix, 21).expect("init failed");
//!
//! loop {
//!     if let Some(result) = miner.mine() {
//!         println!("Found nonce {} with {} bits", result.nonce, result.difficulty);
//!         break;
//!     }
//! }
//! ```
//!
//! # Example: Vanity Key Mining
//!
//! ```no_run
//! use rummage_rs::{GTable, VanityMiner, VanityMode, SearchMode};
//!
//! // Load the generator table (one-time cost, ~64 MB)
//! let gtable = GTable::load().expect("failed to load GTable");
//!
//! // Search for a hex prefix "cafe" using random mode
//! let start_offset = [0u8; 32]; // or use a random seed
//! let mut miner = VanityMiner::new(
//!     &gtable,
//!     "cafe",
//!     VanityMode::HexPrefix,
//!     &start_offset,
//!     SearchMode::Random,
//!     0, // bech32_pattern_len (0 for hex mode)
//! ).expect("failed to create miner");
//!
//! for iteration in 0.. {
//!     miner.iterate(iteration);
//!     if miner.check_results() {
//!         println!("Found a match! Total keys searched: {}", miner.keys_generated());
//!         break;
//!     }
//! }
//! ```

mod pow;
mod vanity;

pub use pow::{Error, PowMiner, PowResult};
pub use vanity::{GTable, SearchMode, VanityMiner, VanityMode, VanityResult};
