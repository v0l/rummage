use std::num::NonZeroU8;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use nostr_sdk::prelude::{PowAdapter, Tag, TagKind, UnsignedEvent};

use crate::pow::{Error, PowMiner};

/// GPU-accelerated NIP-13 Proof of Work adapter for `nostr-sdk`.
///
/// Uses the Rummage CUDA miner to find nonces that satisfy a target
/// difficulty (leading zero bits in the event ID).
#[derive(Debug, Default, Clone)]
pub struct RummagePowAdapter {
    device: u32,
    cancel: Option<Arc<AtomicBool>>,
}

impl RummagePowAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn device(mut self, device: u32) -> Self {
        self.device = device;
        self
    }

    pub fn cancel_flag(mut self, cancel: Arc<AtomicBool>) -> Self {
        self.cancel = Some(cancel);
        self
    }

    fn cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(|c| c.load(Ordering::Relaxed))
    }
}

pub fn pow_template(unsigned: &UnsignedEvent, difficulty: u8) -> (String, String) {
    let mut tags = unsigned.tags.clone();
    tags.retain(|t| t.kind() != TagKind::Nonce);
    tags.push(Tag::pow(0, difficulty));

    let json = serde_json::json!([
        0,
        unsigned.pubkey,
        unsigned.created_at,
        unsigned.kind,
        tags,
        unsigned.content,
    ])
    .to_string();
    let content =
        serde_json::to_string(&unsigned.content).expect("string serialisation cannot fail");
    let suffix = format!("\",\"{difficulty}\"]],{content}]");
    let prefix_len = json.len() - suffix.len() - 1;
    assert_eq!(&json[prefix_len..], format!("0{suffix}"));
    (json[..prefix_len].to_string(), suffix)
}

impl PowAdapter for RummagePowAdapter {
    type Error = PowAdapterError;

    fn compute(
        &self,
        mut unsigned: UnsignedEvent,
        difficulty: NonZeroU8,
    ) -> Result<UnsignedEvent, Self::Error> {
        let difficulty = difficulty.get();
        let mut miner = PowMiner::new().ok_or(PowAdapterError::NoGpuAvailable)?;
        let (prefix, suffix) = pow_template(&unsigned, difficulty);
        miner.init_on_device(&prefix, &suffix, difficulty as u32, self.device)?;

        let result = loop {
            if self.cancelled() {
                return Err(PowAdapterError::Cancelled);
            }
            if let Some(result) = miner.mine()? {
                break result;
            }
        };
        miner.cleanup();

        unsigned.tags.retain(|t| t.kind() != TagKind::Nonce);
        unsigned
            .tags
            .push(Tag::pow(result.nonce as u128, difficulty));
        let event_id = unsigned.compute_id();
        if event_id.as_bytes() != &result.event_id || !event_id.check_pow(difficulty) {
            return Err(PowAdapterError::Mismatch);
        }
        unsigned.id = Some(event_id);
        Ok(unsigned)
    }
}

/// Errors that can occur during GPU-accelerated PoW mining.
#[derive(Debug, Clone)]
pub enum PowAdapterError {
    /// No CUDA-capable GPU was found.
    NoGpuAvailable,
    /// GPU initialisation failed (out of memory, CUDA error, etc.).
    GpuInitFailed(String),
    Gpu(String),
    Exhausted,
    Cancelled,
    Mismatch,
}

impl From<Error> for PowAdapterError {
    fn from(e: Error) -> Self {
        match e {
            Error::Init(m) if m.contains("no CUDA device") => PowAdapterError::NoGpuAvailable,
            Error::Init(m) => PowAdapterError::GpuInitFailed(m),
            Error::InitFailed => PowAdapterError::GpuInitFailed(String::new()),
            Error::InvalidDifficulty(d) => {
                PowAdapterError::GpuInitFailed(format!("difficulty {d} is outside 1..=256"))
            }
            Error::Gpu(m) => PowAdapterError::Gpu(m),
            Error::Exhausted => PowAdapterError::Exhausted,
        }
    }
}

impl std::fmt::Display for PowAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PowAdapterError::NoGpuAvailable => write!(f, "no CUDA-capable GPU available"),
            PowAdapterError::GpuInitFailed(e) => write!(f, "GPU initialisation failed: {e}"),
            PowAdapterError::Gpu(e) => write!(f, "GPU mining failed: {e}"),
            PowAdapterError::Exhausted => write!(f, "nonce space exhausted for this event"),
            PowAdapterError::Cancelled => write!(f, "mining cancelled"),
            PowAdapterError::Mismatch => write!(f, "mined nonce does not reproduce the event ID"),
        }
    }
}

impl std::error::Error for PowAdapterError {}

#[cfg(test)]
mod tests {
    use nostr_sdk::prelude::{EventBuilder, PublicKey, nip13};
    use sha2::{Digest, Sha256};

    use super::*;

    fn mine(content: &str, tags: Vec<Tag>, target: u8) -> UnsignedEvent {
        let pubkey = PublicKey::from_slice(&[7; 32]).unwrap();
        let unsigned = EventBuilder::text_note(content).tags(tags).build(pubkey);
        let mined = RummagePowAdapter::new()
            .compute(unsigned, NonZeroU8::new(target).unwrap())
            .expect("GPU mining failed");
        let event_id = mined.id.expect("event ID should be set after mining");
        assert_eq!(event_id, mined.compute_id());
        assert!(nip13::get_leading_zero_bits(event_id) >= target);
        let nonce_tags: Vec<_> = mined.tags.filter(TagKind::Nonce).collect();
        assert_eq!(nonce_tags.len(), 1);
        assert_eq!(nonce_tags[0].as_slice()[2], target.to_string());
        mined
    }

    #[test]
    fn mine_pow_21() {
        mine(
            "Why must I find leading zero bits? Is there no beauty in the ones?",
            vec![],
            21,
        );
    }

    #[test]
    fn mine_long_content() {
        let content = "Long notes used to break the GPU miner. ".repeat(60);
        mine(&content, vec![], 20);
    }

    #[test]
    fn mine_escaped_content_and_tags() {
        let tags = vec![
            Tag::parse(["t", "quote \" and \\ backslash"]).unwrap(),
            Tag::parse(["nonce", "999", "4"]).unwrap(),
            Tag::parse(["e", "[\"nonce\",\"0\"]"]).unwrap(),
        ];
        mine("line\nbreak \"quoted\" \u{1} 🤙 \",\"21\"]],", tags, 18);
    }

    #[test]
    fn mine_above_64_bits_is_accepted() {
        let pubkey = PublicKey::from_slice(&[1; 32]).unwrap();
        let unsigned = EventBuilder::text_note("x").build(pubkey);
        let (prefix, suffix) = pow_template(&unsigned, 200);
        let mut miner = PowMiner::new().unwrap();
        miner.init(&prefix, &suffix, 200).unwrap();
        assert!(miner.mine().unwrap().is_none());
    }

    #[test]
    fn cancel_stops_mining() {
        let pubkey = PublicKey::from_slice(&[2; 32]).unwrap();
        let unsigned = EventBuilder::text_note("x").build(pubkey);
        let cancel = Arc::new(AtomicBool::new(true));
        let err = RummagePowAdapter::new()
            .cancel_flag(cancel)
            .compute(unsigned, NonZeroU8::new(200).unwrap())
            .unwrap_err();
        assert!(matches!(err, PowAdapterError::Cancelled));
    }

    #[test]
    fn two_miners_do_not_share_state() {
        let pubkey = PublicKey::from_slice(&[3; 32]).unwrap();
        let a = EventBuilder::text_note("first").build(pubkey);
        let b = EventBuilder::text_note("second, with a different length").build(pubkey);
        let (pa, sa) = pow_template(&a, 16);
        let (pb, sb) = pow_template(&b, 16);
        let mut ma = PowMiner::new().unwrap();
        let mut mb = PowMiner::new().unwrap();
        ma.init(&pa, &sa, 16).unwrap();
        mb.init(&pb, &sb, 16).unwrap();
        let ra = loop {
            if let Some(r) = ma.mine().unwrap() {
                break r;
            }
        };
        let rb = loop {
            if let Some(r) = mb.mine().unwrap() {
                break r;
            }
        };
        assert_eq!(
            Sha256::digest(format!("{pa}{}{sa}", ra.nonce)).as_slice(),
            ra.event_id
        );
        assert_eq!(
            Sha256::digest(format!("{pb}{}{sb}", rb.nonce)).as_slice(),
            rb.event_id
        );
    }
}
