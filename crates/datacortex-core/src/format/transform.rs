//! Transform framework — chain of reversible preprocessing transforms.
//!
//! Each transform has an ID, produces transformed data + metadata.
//! The chain is serialized into the .dcx transform_metadata field.

use std::io;

/// Result of a forward transform.
pub struct TransformResult {
    pub data: Vec<u8>,
    pub metadata: Vec<u8>,
}

/// A single transform record in the chain.
pub struct TransformRecord {
    pub id: u8,
    pub metadata: Vec<u8>,
}

/// Ordered chain of transforms applied during preprocessing.
pub struct TransformChain {
    pub records: Vec<TransformRecord>,
}

// Transform IDs.
pub const TRANSFORM_JSON_KEY_INTERN: u8 = 1;
pub const TRANSFORM_NDJSON_COLUMNAR: u8 = 2;

impl TransformChain {
    pub fn new() -> Self {
        Self { records: vec![] }
    }

    pub fn push(&mut self, id: u8, metadata: Vec<u8>) {
        self.records.push(TransformRecord { id, metadata });
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Serialize the chain for storage in .dcx transform_metadata.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(self.records.len() as u8);
        for rec in &self.records {
            out.push(rec.id);
            out.extend_from_slice(&(rec.metadata.len() as u32).to_le_bytes());
            out.extend_from_slice(&rec.metadata);
        }
        out
    }

    /// Deserialize a chain from .dcx transform_metadata.
    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        if data.is_empty() {
            return Ok(Self::new());
        }
        let mut pos = 0;
        let num = data[pos] as usize;
        pos += 1;
        let mut records = Vec::with_capacity(num);
        for _ in 0..num {
            if pos + 5 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "transform record truncated",
                ));
            }
            let id = data[pos];
            pos += 1;
            let meta_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + meta_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "transform metadata truncated",
                ));
            }
            let metadata = data[pos..pos + meta_len].to_vec();
            pos += meta_len;
            records.push(TransformRecord { id, metadata });
        }
        Ok(Self { records })
    }
}

impl Default for TransformChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chain_roundtrip() {
        let chain = TransformChain::new();
        let serialized = chain.serialize();
        let deserialized = TransformChain::deserialize(&serialized).unwrap();
        assert!(deserialized.is_empty());
    }

    #[test]
    fn chain_roundtrip() {
        let mut chain = TransformChain::new();
        chain.push(1, vec![10, 20, 30]);
        chain.push(2, vec![40, 50]);

        let serialized = chain.serialize();
        let deserialized = TransformChain::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.records.len(), 2);
        assert_eq!(deserialized.records[0].id, 1);
        assert_eq!(deserialized.records[0].metadata, vec![10, 20, 30]);
        assert_eq!(deserialized.records[1].id, 2);
        assert_eq!(deserialized.records[1].metadata, vec![40, 50]);
    }
}
