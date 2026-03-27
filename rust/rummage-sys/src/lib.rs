//! Raw FFI bindings to the Rummage GPU Nostr mining library.
//!
//! This crate provides automatically generated bindings from `src/GPU/rummage_ffi.h`
//! using [`bindgen`](https://docs.rs/bindgen). It exposes opaque handles and
//! `extern "C"` functions for both the PoW miner (`RummagePow*`) and the vanity
//! key miner (`RummageVanity*`).
//!
//! **You should not use this crate directly.** Prefer the safe, idiomatic wrappers
//! in the [`rummage-rs`](../rummage_rs/index.html) crate instead.
//!
//! # Build Requirements
//!
//! The `build.rs` script compiles the CUDA/C++ source files with `nvcc` and `cc`,
//! then runs `bindgen` on the C header. This requires:
//!
//! - NVIDIA CUDA Toolkit (nvcc)
//! - GMP library (`-lgmp`)
//! - g++ compiler
//!
//! The CUDA compute capability defaults to 120 (Blackwell) and can be overridden
//! with the `CUDA_CCAP` environment variable.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
