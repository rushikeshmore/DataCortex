use datacortex_core::dcx::Mode;
use datacortex_core::format::{detect_format, preprocess, reverse_preprocess};

fn main() {
    let data = std::fs::read("corpus/test-ndjson.ndjson").unwrap();
    let format = detect_format(&data);
    eprintln!("Format: {:?}", format);
    eprintln!("Original: {} bytes", data.len());
    
    let (preprocessed, chain) = preprocess(&data, format, Mode::Balanced);
    eprintln!("Preprocessed: {} bytes", preprocessed.len());
    eprintln!("Transforms: {}", chain.records.len());
    for (i, rec) in chain.records.iter().enumerate() {
        eprintln!("  [{}] ID={} metadata={} bytes", i, rec.id, rec.metadata.len());
    }
    
    let restored = reverse_preprocess(&preprocessed, &chain);
    if restored == data {
        eprintln!("Roundtrip: OK");
    } else {
        eprintln!("Roundtrip: FAILED ({} vs {})", restored.len(), data.len());
    }
}
