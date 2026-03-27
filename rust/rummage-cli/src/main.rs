use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand};
use rand::RngCore;
use rummage_rs::{GTable, PowMiner, SearchMode, VanityMiner, VanityMode};
use sha2::{Digest, Sha256};

const BANNER: &str = r#"
        ╔═══════╗
        ║       ║
        ║ ╰───╯ ║   R U M M A G E
        ║       ║   nostr mining tool
        ╚═══════╝
"#;

const NONCES_PER_THREAD: u64 = 128;
const BECH32_CHARSET: &[u8] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "rummage", about = "GPU-accelerated Nostr mining tool")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Mine a vanity npub/hex key on the GPU
    Vanity(VanityArgs),
    /// Mine NIP-13 Proof of Work for a Nostr event on the GPU
    Pow(PowArgs),
}

#[derive(Parser)]
struct VanityArgs {
    /// Hex prefix to search for in the raw public key
    #[arg(long)]
    prefix: Option<String>,

    /// Hex suffix to search for in the raw public key
    #[arg(long)]
    suffix: Option<String>,

    /// Hex pattern to match as both prefix and suffix
    #[arg(long)]
    both: Option<String>,

    /// Bech32 prefix to search for in the npub address
    #[arg(long)]
    npub_prefix: Option<String>,

    /// Bech32 suffix to search for in the npub address
    #[arg(long)]
    npub_suffix: Option<String>,

    /// Bech32 pattern to match as both prefix and suffix of npub
    #[arg(long)]
    npub_both: Option<String>,

    /// Use sequential exhaustive search instead of random
    #[arg(long)]
    sequential: bool,

    /// Checkpoint file for sequential mode
    #[arg(long, default_value = "checkpoint.txt")]
    checkpoint: String,

    /// Exit as soon as a matching key is found
    #[arg(long)]
    once: bool,
}

#[derive(Parser)]
struct PowArgs {
    /// Unsigned Nostr event as a JSON string
    #[arg(long)]
    event: Option<String>,

    /// Path to a file containing the unsigned Nostr event JSON
    #[arg(long)]
    file: Option<PathBuf>,

    /// Target difficulty in leading zero bits
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(1..=64))]
    difficulty: u32,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bytes_to_hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

fn count_leading_zero_bits(hash: &[u8]) -> u32 {
    let mut total = 0u32;
    for &b in hash {
        if b == 0 {
            total += 8;
        } else {
            total += b.leading_zeros();
            break;
        }
    }
    total
}

fn is_valid_hex(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_hexdigit())
}

fn is_valid_bech32(s: &str) -> bool {
    s.chars()
        .all(|c| BECH32_CHARSET.contains(&(c.to_ascii_lowercase() as u8)))
}

fn bech32_to_hex(bech32_pattern: &str) -> String {
    let mut bits = String::new();
    for c in bech32_pattern.chars() {
        let lc = c.to_ascii_lowercase() as u8;
        let value = BECH32_CHARSET.iter().position(|&b| b == lc).unwrap_or(0);
        for b in (0..5).rev() {
            bits.push(if (value >> b) & 1 == 1 { '1' } else { '0' });
        }
    }

    let mut hex_pattern = String::new();
    let mut i = 0;
    while i + 8 <= bits.len() {
        let byte_val = u8::from_str_radix(&bits[i..i + 8], 2).unwrap_or(0);
        hex_pattern.push_str(&format!("{:02x}", byte_val));
        i += 8;
    }
    hex_pattern
}

/// Extract the inner content of "tags": [...] from the raw JSON string,
/// returning only what is between the outer brackets (exclusive).
fn extract_tags_inner(json: &str) -> Option<String> {
    let tags_pos = json.find("\"tags\"")?;
    let after_key = &json[tags_pos + 6..];
    let arr_start = after_key.find('[')? + 1;
    let slice = &after_key[arr_start..];

    let mut depth = 1i32;
    for (i, ch) in slice.char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    let inner = slice[..i].trim();
                    return if inner.is_empty() {
                        None
                    } else {
                        Some(inner.to_string())
                    };
                }
            }
            _ => {}
        }
    }
    None
}

fn build_pow_template(
    pubkey: &str,
    created_at: u32,
    kind: i64,
    existing_tags: &Option<String>,
    content: &str,
    difficulty: u32,
) -> (String, String) {
    let mut prefix = format!("[0,\"{}\",{},{},[", pubkey, created_at, kind);
    if let Some(tags) = existing_tags {
        prefix.push_str(tags);
        prefix.push(',');
    }
    prefix.push_str("[\"nonce\",\"");

    let suffix = format!("\",\"{}\"]],\"{}\"]", difficulty, content);
    (prefix, suffix)
}

fn unix_now() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32
}

// ---------------------------------------------------------------------------
// PoW mining
// ---------------------------------------------------------------------------

fn run_pow(args: PowArgs) -> anyhow::Result<()> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })?;

    // Load event JSON
    let event_json = match (&args.event, &args.file) {
        (Some(json), _) => json.clone(),
        (None, Some(path)) => fs::read_to_string(path)?,
        _ => anyhow::bail!("Must specify --event or --file"),
    };

    let event: serde_json::Value = serde_json::from_str(&event_json)?;

    let pubkey = event["pubkey"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing pubkey"))?;
    let mut created_at = event["created_at"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("missing created_at"))? as u32;
    let kind = event["kind"]
        .as_i64()
        .ok_or_else(|| anyhow::anyhow!("missing kind"))?;
    let content = event["content"].as_str().unwrap_or("");

    // Extract existing tags as raw JSON substring (not via serde re-serialization,
    // to preserve exact formatting the same way the C++ version does).
    let existing_tags = extract_tags_inner(&event_json);

    let difficulty = args.difficulty;

    let (mut prefix, suffix) = build_pow_template(
        pubkey,
        created_at,
        kind,
        &existing_tags,
        content,
        difficulty,
    );

    println!("\nPoW Mining Configuration:");
    println!("  Pubkey:     {}", pubkey);
    println!("  Created at: {}", created_at);
    println!("  Kind:       {}", kind);
    println!("  Difficulty: {} leading zero bits", difficulty);
    println!(
        "  Template:   {} prefix + nonce + {} suffix bytes",
        prefix.len(),
        suffix.len()
    );

    // CPU verify with nonce=0
    let test_msg = format!("{}0{}", prefix, suffix);
    let test_hash = Sha256::digest(test_msg.as_bytes());
    println!(
        "  Verify ID:  {} (nonce=0, {} bits)",
        bytes_to_hex(&test_hash),
        count_leading_zero_bits(&test_hash)
    );
    println!();

    // Init GPU miner
    let mut miner = PowMiner::new().ok_or_else(|| anyhow::anyhow!("failed to create GPU miner"))?;
    miner.init(&prefix, &suffix, difficulty as i32)?;

    let batch_size = miner.thread_count();
    let num_streams = miner.stream_count();
    let nonces_per_batch = batch_size as u64 * NONCES_PER_THREAD * num_streams as u64;

    let mut nonce_start: u64 = 0;
    let mut total_attempts: u64 = 0;

    let ts_refresh_secs = 30u64;
    let mut last_ts_refresh = Instant::now();
    let start_time = Instant::now();
    let mut last_report = start_time;

    println!("Mining started! Press Ctrl+C to stop.");
    println!(
        "  (created_at will refresh every {} seconds)\n",
        ts_refresh_secs
    );

    while running.load(Ordering::SeqCst) {
        if let Some(result) = miner.mine_batch(nonce_start, batch_size) {
            total_attempts += nonces_per_batch;
            let elapsed_ms = start_time.elapsed().as_millis() as f64;

            println!("\n========== PoW FOUND ==========");
            println!("Nonce:      {}", result.nonce);
            println!("Event ID:   {}", bytes_to_hex(&result.event_id));
            println!("Difficulty: {} bits", result.difficulty);
            println!("Timestamp:  {}", created_at);
            println!("Attempts:   {}", total_attempts);
            println!("Time:       {:.2} seconds", elapsed_ms / 1000.0);
            if elapsed_ms > 0.0 {
                println!(
                    "Rate:       {:.2} MH/s",
                    total_attempts as f64 / (elapsed_ms / 1000.0) / 1e6
                );
            }

            // CPU verify
            let full_msg = format!("{}{}{}", prefix, result.nonce, suffix);
            let cpu_hash = Sha256::digest(full_msg.as_bytes());
            let cpu_bits = count_leading_zero_bits(&cpu_hash);
            println!(
                "CPU verify: {} ({} bits) {}",
                bytes_to_hex(&cpu_hash),
                cpu_bits,
                if cpu_bits >= difficulty {
                    "OK"
                } else {
                    "MISMATCH!"
                }
            );

            println!("\nAdd this to your event:");
            println!("  \"created_at\": {}", created_at);
            println!("  [\"nonce\",\"{}\",\"{}\"]", result.nonce, difficulty);
            println!("================================");
            return Ok(());
        }

        total_attempts += nonces_per_batch;
        nonce_start += nonces_per_batch;

        let now = Instant::now();

        // Refresh timestamp
        if now.duration_since(last_ts_refresh) >= Duration::from_secs(ts_refresh_secs) {
            let new_ts = unix_now();
            if new_ts != created_at {
                created_at = new_ts;
                miner.cleanup();

                let (new_prefix, _new_suffix) = build_pow_template(
                    pubkey,
                    created_at,
                    kind,
                    &existing_tags,
                    content,
                    difficulty,
                );
                prefix = new_prefix;
                // suffix never changes (doesn't contain created_at)

                miner.init(&prefix, &suffix, difficulty as i32)?;
                nonce_start = 0;
                println!("PoW: refreshed created_at to {}, reset nonce", created_at);
            }
            last_ts_refresh = now;
        }

        // Progress report
        if now.duration_since(last_report) > Duration::from_secs(5) {
            let total_ms = start_time.elapsed().as_millis() as f64;
            let rate = if total_ms > 0.0 {
                total_attempts as f64 / (total_ms / 1000.0)
            } else {
                0.0
            };
            println!(
                "PoW: {} attempts, {:.2} MH/s, nonce range [{}..{})",
                total_attempts,
                rate / 1e6,
                nonce_start.saturating_sub(nonces_per_batch),
                nonce_start
            );
            last_report = now;
        }
    }

    println!(
        "\nPoW mining stopped. No solution found after {} attempts.",
        total_attempts
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Vanity mining
// ---------------------------------------------------------------------------

fn run_vanity(args: VanityArgs) -> anyhow::Result<()> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })?;

    // Determine pattern and mode
    let (raw_pattern, mut vanity_mode, is_bech32) = if let Some(ref p) = args.prefix {
        (p.clone(), VanityMode::HexPrefix, false)
    } else if let Some(ref p) = args.suffix {
        (p.clone(), VanityMode::HexSuffix, false)
    } else if let Some(ref p) = args.both {
        (p.clone(), VanityMode::HexBoth, false)
    } else if let Some(ref p) = args.npub_prefix {
        (p.clone(), VanityMode::Bech32Prefix, true)
    } else if let Some(ref p) = args.npub_suffix {
        (p.clone(), VanityMode::Bech32Suffix, true)
    } else if let Some(ref p) = args.npub_both {
        (p.clone(), VanityMode::Bech32Both, true)
    } else {
        anyhow::bail!("must specify a vanity pattern (--prefix, --suffix, --both, --npub-prefix, --npub-suffix, or --npub-both)");
    };

    // Validate and convert pattern
    let original_bech32_pattern: Option<String>;
    let original_bech32_mode: VanityMode;
    let pattern: String;
    let bech32_pattern_len: i32;

    if is_bech32 {
        if !is_valid_bech32(&raw_pattern) {
            anyhow::bail!(
                "pattern must use valid bech32 characters (qpzry9x8gf2tvdw0s3jn54khce6mua7l)\n\
                 Note: characters '1', 'b', 'i', 'o' are NOT valid in bech32"
            );
        }
        original_bech32_pattern = Some(raw_pattern.clone());
        original_bech32_mode = vanity_mode;
        bech32_pattern_len = raw_pattern.len() as i32;

        // Convert to hex for fast GPU pre-filtering
        pattern = bech32_to_hex(&raw_pattern);
        vanity_mode = match vanity_mode {
            VanityMode::Bech32Prefix => VanityMode::HexPrefix,
            VanityMode::Bech32Suffix => VanityMode::HexSuffix,
            VanityMode::Bech32Both => VanityMode::HexBoth,
            other => other,
        };

        println!(
            "Converted npub pattern '{}' -> hex pattern '{}' (for fast pre-filtering)",
            raw_pattern, pattern
        );
        println!("Will verify full bech32 match after hex pre-filter");
    } else {
        if !is_valid_hex(&raw_pattern) {
            anyhow::bail!("pattern must be valid hexadecimal (0-9, a-f)");
        }
        original_bech32_pattern = None;
        original_bech32_mode = VanityMode::HexPrefix; // unused
        bech32_pattern_len = 0;
        pattern = raw_pattern.to_ascii_lowercase();
    }

    if pattern.is_empty() || (!is_bech32 && pattern.len() > 16) {
        anyhow::bail!("hex pattern length must be between 1 and 16 characters");
    }

    let search_mode = if args.sequential {
        SearchMode::Sequential
    } else {
        SearchMode::Random
    };

    // Display config
    let lower = pattern.to_ascii_lowercase();
    println!("\nConfiguration:");
    print!("  Mode:    ");
    match vanity_mode {
        VanityMode::HexPrefix => println!("Hex Prefix\n  Pattern: {}***...", lower),
        VanityMode::HexSuffix => println!("Hex Suffix\n  Pattern: ...***{}", lower),
        VanityMode::HexBoth => {
            let half = lower.len() / 2;
            println!(
                "Hex Both (prefix + suffix)\n  Pattern: {}***...***{}",
                &lower[..half],
                &lower[half..]
            );
        }
        _ => {} // bech32 modes already converted
    }
    println!();

    // Generate start offset
    let mut start_offset = [0u8; 32];
    let resuming_from_checkpoint;

    if search_mode == SearchMode::Sequential {
        // Try loading offset from checkpoint
        if let Ok(contents) = fs::read_to_string(&args.checkpoint) {
            for line in contents.lines() {
                if let Some(hex_str) = line.strip_prefix("startOffset=") {
                    if let Ok(bytes) = hex::decode(hex_str.trim()) {
                        if bytes.len() == 32 {
                            start_offset.copy_from_slice(&bytes);
                        }
                    }
                }
            }
            resuming_from_checkpoint = true;
            println!("Loaded starting offset from checkpoint file");
        } else {
            rand::rng().fill_bytes(&mut start_offset);
            resuming_from_checkpoint = true; // will be false, but set below
            println!("Generated random 256-bit starting offset for sequential search");
        }
        print!("Offset (hex): ");
        println!("{}", bytes_to_hex(&start_offset));
        println!("WARNING: This offset will be saved in the checkpoint file - protect it!\n");
    } else {
        let ts = unix_now() as u64;
        start_offset[24..32].copy_from_slice(&ts.to_le_bytes());
        resuming_from_checkpoint = false;
    }

    // Load GTable
    println!("Generating GTable (this may take a minute)...");
    let gtable = GTable::load()?;
    println!("GTable generation complete!");

    // Create miner
    let mut miner = VanityMiner::new(
        &gtable,
        &pattern,
        vanity_mode,
        &start_offset,
        search_mode,
        bech32_pattern_len,
    )?;

    // Enable bech32 verification if needed
    if let Some(ref orig) = original_bech32_pattern {
        miner.set_bech32_verification(orig, original_bech32_mode);
    }

    // Load checkpoint
    if search_mode == SearchMode::Sequential && resuming_from_checkpoint {
        if miner.load_checkpoint(&args.checkpoint) {
            println!("Resumed from checkpoint");
        }
    }

    println!("\nMining started! Press Ctrl+C to stop.");
    println!("Keys will be saved to: keys.txt");
    if search_mode == SearchMode::Sequential {
        println!("Checkpoints will be saved to: {}", args.checkpoint);
    }
    println!();

    let mut iteration: u64 = 0;
    let start_time = Instant::now();
    let mut last_report = start_time;
    let mut last_checkpoint = start_time;

    while running.load(Ordering::SeqCst) {
        miner.iterate(iteration);
        let found = miner.check_results();

        if iteration % 10 == 0 {
            let now = Instant::now();

            if now.duration_since(last_report) > Duration::from_secs(5) {
                let keys = miner.keys_generated();
                let elapsed_s = start_time.elapsed().as_secs();
                if elapsed_s > 0 {
                    let rate = keys as f64 / elapsed_s as f64;
                    if search_mode == SearchMode::Sequential {
                        println!(
                            "Progress: {:.2}% | {} keys searched, {:.2} keys/sec",
                            miner.progress() * 100.0,
                            keys,
                            rate
                        );
                    } else {
                        println!("Stats: {} keys searched, {:.2} keys/sec", keys, rate);
                    }
                }
                last_report = now;
            }

            if search_mode == SearchMode::Sequential
                && now.duration_since(last_checkpoint) >= Duration::from_secs(60)
            {
                miner.save_checkpoint(&args.checkpoint);
                last_checkpoint = now;
            }
        }

        // Check exhaustion
        if search_mode == SearchMode::Sequential && miner.progress() >= 1.0 {
            println!("\nSequential search complete! Exhausted entire search space.");
            if miner.matches_found() == 0 {
                println!("No matches found in entire search space.");
            }
            break;
        }

        if found && args.once {
            break;
        }

        if found {
            println!("\nMatch found! Continuing to search for more matches...");
            println!("(Press Ctrl+C to stop mining)\n");
        }

        iteration += 1;
    }

    println!("\nShutting down...");

    if search_mode == SearchMode::Sequential {
        miner.save_checkpoint(&args.checkpoint);
        println!("Final checkpoint saved");
    }

    let keys = miner.keys_generated();
    let matches = miner.matches_found();
    let elapsed_s = start_time.elapsed().as_secs();

    println!("\nFinal Statistics:");
    if search_mode == SearchMode::Sequential {
        println!("  Search progress: {:.2}%", miner.progress() * 100.0);
    }
    println!("  Total keys searched: {}", keys);
    println!("  Matches found: {}", matches);
    if elapsed_s > 0 {
        println!(
            "  Average rate: {:.2} keys/sec",
            keys as f64 / elapsed_s as f64
        );
    }
    println!("  Total time: {} seconds", elapsed_s);
    println!();
    println!("Mining stopped.");

    Ok(())
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    print!("{}", BANNER);

    let cli = Cli::parse();
    match cli.command {
        Command::Pow(args) => run_pow(args),
        Command::Vanity(args) => run_vanity(args),
    }
}
