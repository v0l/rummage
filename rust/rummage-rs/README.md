# rummage-rs

Safe Rust wrappers for the [Rummage](https://github.com/v0l/rummage) GPU Nostr mining library.

Rummage is a high-performance CUDA-accelerated tool for Nostr vanity key mining and NIP-13 Proof of Work. This crate provides memory-safe, idiomatic Rust wrappers with automatic GPU resource cleanup via `Drop`.

## Features

- **`PowMiner`** — NIP-13 Proof of Work mining for Nostr events (8,000+ MH/s on RTX PRO 6000)
- **`VanityMiner`** — Vanity npub/hex key generation (240M+ keys/sec on RTX PRO 6000)
- **`GTable`** — Pre-computed secp256k1 generator table for vanity mining

## Example: PoW Mining

```rust
use rummage_rs::PowMiner;

let prefix = r#"[0,"ab..cd",1234567890,1,[["nonce",""#;
let suffix = r#"","21"]],"hello"]"#;

let mut miner = PowMiner::new().expect("GPU init");
miner.init(prefix, suffix, 21).expect("init failed");

loop {
    if let Some(result) = miner.mine() {
        println!("Found nonce {} with {} bits", result.nonce, result.difficulty);
        break;
    }
}
```

## Example: Vanity Key Mining

```rust
use rummage_rs::{GTable, VanityMiner, VanityMode, SearchMode};

let gtable = GTable::load().expect("failed to load GTable");

let start_offset = [0u8; 32];
let mut miner = VanityMiner::new(
    &gtable, "cafe", VanityMode::HexPrefix,
    &start_offset, SearchMode::Random, 0,
).expect("failed to create miner");

for iteration in 0.. {
    miner.iterate(iteration);
    if miner.check_results() {
        println!("Match found! Keys searched: {}", miner.keys_generated());
        break;
    }
}
```

## Build Requirements

- NVIDIA CUDA Toolkit (`nvcc`)
- GMP library (`libgmp-dev`)
- g++ compiler

```bash
CUDA_CCAP=89 cargo build  # override compute capability (default: 120)
```
