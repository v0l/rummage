# rummage-cli

GPU-accelerated Nostr mining tool built on [Rummage](https://github.com/v0l/rummage). Supports vanity npub/hex key mining and NIP-13 Proof of Work, powered by NVIDIA CUDA.

## Install

```bash
cargo install rummage-cli
```

Requires NVIDIA CUDA Toolkit, GMP library, and g++ on the build machine.

Override the CUDA compute capability if needed (default: 120 / Blackwell):

```bash
CUDA_CCAP=89 cargo install rummage-cli
```

## Usage

### Vanity Key Mining

```bash
# Bech32 npub prefix
rummage vanity --npub-prefix alice

# Hex prefix
rummage vanity --prefix cafe

# Sequential exhaustive search (resumable via checkpoint)
rummage vanity --npub-prefix satoshi --sequential

# Exit after first match
rummage vanity --prefix dead --once
```

### Proof of Work (NIP-13)

```bash
# From a JSON file
rummage pow --file event.json --difficulty 32

# From an inline JSON string
rummage pow --event '{"pubkey":"<hex>","created_at":1735000000,"kind":1,"tags":[],"content":"hello"}' --difficulty 24
```

## Performance

On an RTX PRO 6000 (Blackwell):

- **PoW**: ~8,000 MH/s
- **Vanity**: ~240M keys/sec
