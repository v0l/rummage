# Performance Optimization Findings

## Benchmark Test Event

All benchmarks used the following realistic Nostr event:

```json
{"id":"","pubkey":"79c2cae114ea28a681e1ba5ebc76007ed87f86694f35f782483f4e4c2d45b96f","created_at":1234567890,"kind":1,"tags":[["e","abc123"],["p","def456"]],"content":"hello world","sig":""}
```

**Template breakdown:**
- Prefix: 124 bytes (pubkey + created_at + kind + tags)
- Suffix: 23 bytes (content + sig)
- Standard mode tail: 103 bytes (60 prefix remainder + 10 digit nonce + 23 suffix)
- Fast mode tail: 42 bytes (20 prefix remainder + 10 digit nonce + 2 suffix)

---

## NONCES_PER_THREAD Tuning (Completed)

### Summary
Changing `NONCES_PER_THREAD` has **minimal impact** on performance (<3% variation). This parameter can be **ruled out** as a meaningful optimization target.

### Benchmark Results

**Test Setup:**
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (188 SMs)
- Event: 124-byte prefix, 23-byte suffix (realistic Nostr event)
- Tail size: 103 bytes (standard mode), 42 bytes (fast mode)

#### Standard Mode (23-byte suffix)

| Difficulty | NONCES=128 | NONCES=256 | Difference |
|------------|------------|------------|------------|
| 32-bit     | 8,514 MH/s | 8,530 MH/s | +0.2% |
| 34-bit     | 7,513 MH/s | 7,474 MH/s | -0.5% |
| 36-bit     | 7,365 MH/s | 7,387 MH/s | +0.3% |
| 38-bit     | 7,223 MH/s | 7,306 MH/s | +1.1% |

#### Fast Mode (2-byte suffix)

| Difficulty | NONCES=128 | NONCES=256 | Difference |
|------------|------------|------------|------------|
| 32-bit     | 14,352 MH/s | 14,640 MH/s | +2.0% |
| 34-bit     | 14,504 MH/s | 14,015 MH/s | -3.4% |
| 36-bit     | 13,962 MH/s | 13,904 MH/s | -0.4% |
| 38-bit     | 13,963 MH/s | 13,810 MH/s | -1.1% |

### Analysis

1. **Tail rebuild overhead is not the bottleneck**
   - Expected significant gains at high difficulty where nonces grow from 9→10 digits
   - Actual gains were negligible (<3%)
   - The SHA256 computation dominates, not the tail rebuild

2. **Fast mode benefits less from higher values**
   - Minimal tail (42 bytes) means fewer SHA256 transforms
   - Larger NONCES_PER_THREAD reduces parallelism slightly
   - `NONCES=128` is optimal for fast mode

3. **Standard mode sees marginal gains at high difficulty**
   - 1% improvement at 38-bit difficulty
   - Not worth the trade-off of reduced fast mode performance

### Conclusion

**Keep `NONCES_PER_THREAD = 128`** (baseline value in `CudaPowMiner.cu:48`)

This value provides:
- Best performance for fast mode
- Equivalent performance for standard mode
- Lower register/memory pressure per thread
- Better occupancy on all GPU architectures

---

## Fixed-Width ASCII Nonce Mode (Completed)

### Summary
Fixed-width nonce mode (10-digit, no divisions in loop) provides **modest gains at low difficulty** but **negligible impact at high difficulty**. Limited to nonces up to 9,999,999,999.

### Implementation

Toggle with `USE_FIXED_WIDTH_NONCE` define in `CudaPowMiner.cu:51`:
```cpp
#define USE_FIXED_WIDTH_NONCE 0  // Set to 1 for fixed-width mode
```

### Benchmark Results

**Test Setup:**
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (188 SMs)
- Event: 124-byte prefix, 23-byte suffix (realistic Nostr event)
- Fixed nonce width: 10 digits

| Difficulty | Standard Mode | Fixed-Width Mode | Speedup |
|------------|---------------|------------------|---------|
| 32-bit     | ~8,056 MH/s   | ~8,688 MH/s      | **+7.8%** |
| 34-bit     | ~7,411 MH/s   | ~7,421 MH/s      | +0.1% |
| 36-bit     | ~7,407 MH/s   | ~7,334 MH/s      | -1.0% |

### Analysis

1. **Low difficulty (32-bit) sees +7.8% gain**
   - Fixed-width avoids tail rebuild overhead
   - Nonce always fits in single 64-byte block

2. **Higher difficulties show minimal/no gain**
   - SHA256 transforms dominate performance
   - Larger tail (93 bytes) still requires 2 transforms regardless of nonce encoding

3. **Fixed-width has limitations**
   - Only supports nonces up to 9,999,999,999 (10 digits)
   - For >10 billion attempts, nonce wraps around
   - Not suitable for very high difficulty mining

### Conclusion

**Keep `USE_FIXED_WIDTH_NONCE = 0`** (disabled by default)

Fixed-width mode is only beneficial when:
- Mining at low difficulty (<34 bits)
- Nonces fit within 10 digits (<10 billion attempts)
- Tail size is significant (large prefix remainder + suffix)

For high-difficulty NIP-13 mining, the standard variable-length mode is preferred.

### Future Optimization Focus

Instead of the above, consider:

1. **Multiple mid-states** - Pre-compute mid-states for common prefix patterns
2. **W schedule optimization** - Explore unrolling or vectorization
3. **Shared memory caching** - Cache frequently accessed data
4. **Kernel fusion** - Combine nonce generation with hash computation

---

## SHA256 Micro-Optimizations (VanitySearch)

### Summary
Applied SHA256 optimizations from JeanLucPons' VanitySearch:
- **`__byte_perm`** for byte swapping (single PTX instruction vs 7 ops)
- **Optimized Ch/Maj formulas** (fewer operations per round)
- **4-block transform structure** (better instruction-level parallelism)

### Implementation

**File:** `src/GPU/CudaPowMiner.cu`

**Byte swapping optimization:**
```cpp
// Before (7 ops: 4 shifts + 3 ORs)
W[i] = ((uint32_t)block[i*4] << 24) |
       ((uint32_t)block[i*4+1] << 16) |
       ((uint32_t)block[i*4+2] << 8) |
       (uint32_t)block[i*4+3];

// After (1 PTX instruction)
W[i] = pow_bswap32(((uint32_t*)block)[i]);
// where pow_bswap32 uses __byte_perm(v, 0, 0x0123)
```

**Optimized Ch/Maj formulas:**
```cpp
// Ch: 4 ops → 3 ops
#define Ch(x,y,z) (z ^ (x & (y ^ z)))

// Maj: 5 ops → 4 ops  
#define Maj(x,y,z) ((x & y) | (z & (x | y)))
```

**4-block transform structure:**
```cpp
SHA256_RND(0);   // Rounds 0-15
WMIX();          // Expand W[16-31]
SHA256_RND(16);  // Rounds 16-31
WMIX();          // Expand W[32-47]
SHA256_RND(32);  // Rounds 32-47
WMIX();          // Expand W[48-63]
SHA256_RND(48);  // Rounds 48-63
```

### Benchmark Results

**Test Setup:**
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (188 SMs)
- Event: 124-byte prefix, 23-byte suffix (realistic Nostr event)
- Tail size: 103 bytes

| Difficulty | Vanilla SHA256 | + VanitySearch | + 4-Block | Total Gain |
|------------|----------------|----------------|-----------|------------|
| 32-bit     | 8,056 MH/s     | 8,801 MH/s     | 8,655 MH/s| **+7.4%**  |
| 34-bit     | 7,411 MH/s     | 7,666 MH/s     | 7,613 MH/s| **+2.7%**  |
| 36-bit     | 7,407 MH/s     | 7,651 MH/s     | 7,533 MH/s| **+1.7%**  |

### Analysis

1. **`__byte_perm` delivers significant gains**
   - Single PTX instruction vs 7 operations
   - Most impactful at lower difficulties (32-bit: +9.2%)
   - Word loading becomes less dominant at higher difficulties

2. **Optimized Ch/Maj provide consistent small gains**
   - Reduced operation count per round
   - ~3-4% improvement across all difficulties

3. **4-block structure showed minimal additional benefit**
   - Compiler's `#pragma unroll` already achieves good ILP
   - Modern GPUs optimize the single-loop structure well
   - Slight regression vs VanitySearch-only at 32-bit

4. **ASM_SIGMA did NOT help**
   - Inline PTX 64-bit sigma functions were significantly slower
   - 5,631 MH/s vs 7,666 MH/s at 34-bit (-27%)
   - NVCC compiler generates better code than manual PTX

### Conclusion

**VanitySearch optimizations are highly effective:**
- **3-9% speedup** depending on difficulty
- **`__byte_perm` is the key win** (single instruction)
- Keep optimized Ch/Maj formulas
- 4-block structure provides marginal benefit (optional)
- ASM_SIGMA is not recommended on modern GPUs

**Code location:** `src/GPU/CudaPowMiner.cu:117-195`

---

## Future Optimization Opportunities

### High Priority (Architectural)

1. **Constant tail block W schedules** (IMPLEMENTED 2026-03-30)
    - Pre-compute W schedules for suffix-only blocks (blocks after nonce position)
    - Skips W expansion (WMIX) for constant blocks
    - **Performance gains:**
        - 1 constant block (~70 byte suffix): ~12% speedup (7.6k → 8.5k MH/s)
        - 2 constant blocks (~110 byte suffix): ~18% speedup (5.7k → 6.8k MH/s)
    - Requires suffix >= 54 bytes to have at least 1 constant block
    - Code: `compute_constant_tail_w_schedules()` in CudaPowMiner.cu
    
2. **Fixed-width nonce mode** (7.8% at 32-bit)
    - Already implemented via `USE_FIXED_WIDTH_NONCE`
    - Only beneficial at low difficulty (<34 bits)

### Medium Priority

2. **Shared memory W schedule** (3-5% potential)
    - Reduce register pressure by storing W in shared memory
    - May improve occupancy on register-constrained GPUs

3. **Warp shuffle for broadcasts** (1-2% potential)
    - Use `__shfl_sync()` for intra-warp data sharing
    - Faster than constant memory for certain patterns

### Low Priority

4. **Register pressure reduction** (1-3% potential)
    - W[16] → W[8] circular buffer
    - More threads per SM due to lower register usage

5. **Instruction scheduling tuning** (1-2% potential)
    - Manual operation reordering for better ILP
    - Compiler already does excellent optimization

### Implemented ✅

- ✅ **Constant tail block W schedules** - 12-18% speedup for long suffixes
- ✅ **VanitySearch SHA256 optimizations** - 3-9% speedup (byte_perm, Ch/Maj, 4-block structure)
- ✅ **Fixed-width nonce mode** - 7.8% at 32-bit (toggle with USE_FIXED_WIDTH_NONCE)

### Not Recommended

- ❌ **PTX SHA256 intrinsics** - Not available via inline PTX (SASS-only)
- ❌ **ASM_SIGMA** - Manual PTX slower than compiler optimization (-27%)
- ❌ **NONCES_PER_THREAD tuning** - Minimal impact (<3%)

---
*Document created: 2026-03-30*
*Last updated: 2026-03-30 (Constant tail block W schedules implemented)*
*Benchmark tool: `./rummage --pow-event <json> --pow-difficulty <n>`*
*Benchmark event: 124-byte prefix + 23-byte suffix Nostr event*
