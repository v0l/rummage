# PoW Miner Performance

## Benchmark events

Standard event (the `TEST_EVENT` in `benchmark.sh`):

```json
{"id":"","pubkey":"79c2cae114ea28a681e1ba5ebc76007ed87f86694f35f782483f4e4c2d45b96f","created_at":1234567890,"kind":1,"tags":[["e","abc123"],["p","def456"]],"content":"hello world","sig":""}
```

The long event (`LONG_EVENT`) adds two tags and 160 characters of content, which puts two
constant SHA-256 blocks after the nonce.

## Results

RTX PRO 6000 Blackwell Max-Q (188 SMs), Rust CLI, difficulty 60 so nothing is found during
the run. "Before" is commit `cf51544`.

| Event | Before | After |
|---|---|---|
| Standard (`hello world`) | 8,537 MH/s | 19,429 MH/s |
| Fast (`--fast`) | 15,370 MH/s | 20,206 MH/s |
| Long suffix | 4,982 MH/s | 6,213 MH/s |
| No tags, `hello world` | 8,001 MH/s | 11,291 MH/s |
| No tags, `hello world`, `--fast` | 7,793 MH/s | 17,744 MH/s |
| No tags, 213 chars | 3,730 MH/s | 4,936 MH/s |
| No tags, 213 chars, `--fast` | 14,289 MH/s | 18,620 MH/s |

Before this change, any template with a suffix block after the nonce returned wrong hashes
for nonces that were not exactly 10 digits, so the old long-suffix figures were measured on
a miner that skipped valid solutions and reported invalid ones.

## How the kernel works

The message hashed for a nonce is `prefix + decimal nonce + suffix`.

- **Midstate.** The host hashes every full 64-byte block of the prefix once.
- **Fixed-width layouts.** Nonces are mined in groups with the same digit count, 12 to 20
  digits. Within a group the template, padding and length are constant, so the host builds
  them once.
- **Aligned low digits.** A digit count is used only when its last 4 digits fill one
  32-bit SHA-256 message word. Each thread owns 1,000 consecutive nonces that differ only in
  that word.
- **Partial rounds.** Each thread hashes the rounds before that word once and reuses them
  for all 1,000 nonces. That saves up to 15 of the 64 rounds in the nonce block.
- **Constant trailing blocks.** Blocks after the nonce block never change, so the host
  precomputes `K[i] + W[i]` for all 64 rounds. The first six blocks are passed in the kernel
  parameters (constant bank), the rest are read from global memory.
- **Cheapest layout first.** Layouts are ordered by the number of rounds each nonce costs,
  so the miner starts with the digit count that puts the nonce word latest in its block and
  needs the fewest blocks.
- **Two launches in flight** on one stream, with pinned host memory, so the GPU never waits
  for the host between batches.
- **Early reject** on the first hash word with `__clz`. Only candidates that pass are checked
  against the full 256-bit target.
- **Host verification.** Every GPU result is hashed again on the CPU before it is returned.
  A mismatch is an error, never a result.

Each template has about 10^19 nonces. When they run out, `mine()` returns
`Exhausted` and the CLIs move `created_at` forward.

## Tunables

Set through `POW_DEFINES` (see `benchmark.sh`):

| Define | Default | Effect |
|---|---|---|
| `POW_THREADS_PER_BLOCK` | 256 | 128 and 512 measured within 1% |
| `POW_LAUNCH_WAVES` | 1 | Units per launch = SMs x occupancy x threads x waves. 4 measured 1% slower |
