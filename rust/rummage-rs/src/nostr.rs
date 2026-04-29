use std::num::NonZeroU8;

use nostr_sdk::prelude::{PowAdapter, Tag, UnsignedEvent};

use crate::pow::{PowMiner, PowResult};

/// GPU-accelerated NIP-13 Proof of Work adapter for `nostr-sdk`.
///
/// Uses the Rummage CUDA miner to find nonces that satisfy a target
/// difficulty (leading zero bits in the event ID).
#[derive(Debug)]
pub struct RummagePowAdapter;

impl RummagePowAdapter {
    /// Build the NIP-01 prefix/suffix split around the nonce value for the
    /// GPU miner.
    ///
    /// The serialised event for ID computation looks like:
    ///
    /// ```text
    /// [0,"<pubkey>",<created_at>,<kind>,[...tags...,["nonce","<NONCE>","<difficulty>"]],"<content>"]
    /// ```
    ///
    /// We split at the nonce value so the GPU can substitute different nonces
    /// and hash each candidate in parallel.
    fn build_pow_template(
        unsigned: &UnsignedEvent,
        difficulty: u8,
    ) -> (String, String) {
        // Serialize with a dummy nonce of 0 to get the full JSON, then split.
        let mut tags = unsigned.tags.clone();
        tags.push(Tag::pow(0u128, difficulty));

        let json = serde_json::json!([
            0,
            unsigned.pubkey,
            unsigned.created_at,
            unsigned.kind,
            tags,
            unsigned.content,
        ]);
        let json_str = serde_json::to_string(&json).unwrap();

        // Find the placeholder "0" nonce value inside ["nonce","0","<difficulty>"]
        let nonce_tag_start = r#"["nonce",""#;
        let idx = json_str
            .find(nonce_tag_start)
            .expect("nonce tag should be present in serialized event");

        let prefix_end = idx + nonce_tag_start.len();
        let prefix = json_str[..prefix_end].to_string();

        // Find the suffix starting after the nonce "0" value.
        let after_nonce = &json_str[prefix_end..];
        let nonce_value_end = after_nonce
            .find(r#"","#)
            .expect("comma after nonce value");
        let suffix = after_nonce[nonce_value_end..].to_string();

        (prefix, suffix)
    }
}

impl PowAdapter for RummagePowAdapter {
    type Error = PowAdapterError;

    fn compute(
        &self,
        unsigned: UnsignedEvent,
        difficulty: NonZeroU8,
    ) -> Result<UnsignedEvent, Self::Error> {
        let difficulty_u8 = difficulty.get();

        let mut miner = PowMiner::new().ok_or(PowAdapterError::NoGpuAvailable)?;

        let (prefix, suffix) = Self::build_pow_template(&unsigned, difficulty_u8);
        miner.init(&prefix, &suffix, difficulty_u8 as i32).map_err(|_| PowAdapterError::GpuInitFailed)?;

        let mut result: Option<PowResult> = None;
        while result.is_none() {
            result = miner.mine();
        }

        miner.cleanup();

        let pow_result = result.unwrap();

        // Set the nonce tag and compute the final event ID
        let mut unsigned = unsigned;
        unsigned.tags.push(Tag::pow(pow_result.nonce as u128, difficulty_u8));

        // Verify by computing the ID ourselves (defence-in-depth)
        let event_id = unsigned.compute_id();
        debug_assert!(
            event_id.check_pow(difficulty_u8),
            "GPU miner returned a nonce that does not satisfy the difficulty target"
        );
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
    GpuInitFailed,
}

impl std::fmt::Display for PowAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PowAdapterError::NoGpuAvailable => write!(f, "no CUDA-capable GPU available"),
            PowAdapterError::GpuInitFailed => write!(f, "GPU initialisation failed"),
        }
    }
}

impl std::error::Error for PowAdapterError {}

#[cfg(test)]
mod tests {
    use nostr_sdk::prelude::{
        EventBuilder, PublicKey, TagKind, nip13,
        PowAdapter as _,
    };
    use super::*;

    #[test]
    fn mine_pow_21() {
        let pubkey = PublicKey::from_slice(&[0; 32]).unwrap();
        let unsigned = EventBuilder::text_note(
            "Why must I find leading zero bits? Is there no beauty in the ones?",
        )
        .build(pubkey);

        const POW_TARGET: u8 = 21;
        let adapter = RummagePowAdapter;
        let difficulty = NonZeroU8::new(POW_TARGET).unwrap();

        let mined = adapter.compute(unsigned, difficulty).expect("GPU mining failed");

        // Event ID must be set
        let event_id = mined.id.expect("event ID should be set after mining");

        // Must have a nonce tag
        let nonce_tag = mined.tags.find(TagKind::Nonce).expect("nonce tag should exist");
        assert_eq!(nonce_tag.as_slice()[2], POW_TARGET.to_string(), "difficulty in nonce tag should match");

        // Verify the PoW difficulty
        let bits = nip13::get_leading_zero_bits(event_id);
        assert!(bits >= POW_TARGET, "event ID must have >= 21 leading zero bits, got {bits}");
    }
}
