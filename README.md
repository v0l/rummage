<div align="center">
<img src="/assets/iconx128.png" alt="Rummage Logo" title="Rummage logo" width="64"/>
  
# Rummage
</div>

Rummage is a high-performance Nostr vanity address miner that searches the secp256k1 keyspace to generate custom prefixes or suffixes in either Bech32 npub format or raw hex. It supports both random sampling and exhaustive sequential search modes.

It's built for NVIDIA GPUs using CUDA and can sustain 42M+ keys/second on a consumer card (RTX 3070, Ampere architecture) and 170M+ keys/second on a datacenter card (H200, Hopper architecture).

There is also an experimental build for Apple Silicon in the Metal branch. This is a ground-up implementation in Metal Shading Language, as no existing secp256k1 libraries are available for Metal. Currently it can do about 9M+ keys/second on a 2021 M1.


## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8+
- GMP library
- g++ compiler

## Installation

```bash
make
```

If you need to configure for a different GPU or CUDA version, see [docs/BUILD.md](docs/BUILD.md).

## Usage

Search for npub prefix:
```bash
./rummage --npub-prefix alice
```

Search for hex prefix:
```bash
./rummage --prefix cafe
```

Sequential exhaustive search (resumable):
```bash
./rummage --npub-prefix satoshi --sequential
```

To learn more about how search modes work in Rummage, see [docs/SEARCH.md](docs/SEARCH.md).

## Options

**Bech32 Mode** (searches npub address):
- `--npub-prefix <str>` - Search for prefix in npub
- `--npub-suffix <str>` - Search for suffix in npub
- `--npub-both <str>` - Search for both prefix and suffix

**Hex Mode** (searches raw public key):
- `--prefix <hex>` - Search for hex prefix
- `--suffix <hex>` - Search for hex suffix
- `--both <hex>` - Search for both prefix and suffix

**Search Modes**:
- `--sequential` - Exhaustive search (resumable, guarantees completion)
- `--checkpoint <file>` - Checkpoint file path (default: checkpoint.txt)

By default, random search mode is used (faster for short patterns).


## Output

Found keys are saved to `keys.txt` with private key, public key, and npub address.

## What is Nostr?
Perhaps you stumbled upon this project and have no idea what on earth is going on here. 

Quick explanation: Nostr is a decentralized protocol where your identity is a cryptographic key pair, not an account on someone else's server. Your npub (public key) is your permanent identifier across the network. Think of it like a username that no company can take away, suspend, or ban.

Unlike traditional platforms where @yourname can disappear if the service shuts down or decides they don't like you, your Nostr identity is yours forever. A vanity npub is not necessary, but it lets you claim something readable and memorable (like npub1alice...) instead of a random string. It's a little nerdy, but suitable for those who perk up when they hear terms like sovereign identity.

An npub is not just an identifier, it's how you log in to a growing number of apps in the Nostr ecosystem. Think "Sign in with Google," except there is no Google. Just you and the math.



