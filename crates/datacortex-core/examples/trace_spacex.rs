use datacortex_core::dcx::Mode;
use datacortex_core::format::{detect_format, preprocess, reverse_preprocess};

fn main() {
    let data = std::fs::read("/tmp/spacex.json").expect("read spacex.json");
    let format = detect_format(&data);
    eprintln!("Format: {:?}", format);
    eprintln!("Original: {} bytes", data.len());

    // Test Fast mode (where the bug is)
    let (preprocessed, chain) = preprocess(&data, format, Mode::Fast);
    eprintln!("Preprocessed: {} bytes", preprocessed.len());
    eprintln!("Transforms: {}", chain.records.len());
    for (i, rec) in chain.records.iter().enumerate() {
        let name = match rec.id {
            7 => "JSON_ARRAY_COLUMNAR",
            15 => "NESTED_FLATTEN",
            14 => "TYPED_ENCODING",
            13 => "VALUE_DICT",
            _ => "?",
        };
        eprintln!(
            "  [{}] ID={} ({}) metadata={} bytes",
            i,
            rec.id,
            name,
            rec.metadata.len()
        );
    }

    let restored = reverse_preprocess(&preprocessed, &chain);
    if restored == data {
        eprintln!("Full pipeline roundtrip: OK");
    } else {
        eprintln!(
            "Full pipeline roundtrip: FAILED ({} vs {} bytes)",
            restored.len(),
            data.len()
        );
        // Find first difference
        for (i, (a, b)) in restored.iter().zip(data.iter()).enumerate() {
            if a != b {
                let start = i.saturating_sub(20);
                let end = (i + 20).min(restored.len()).min(data.len());
                eprintln!("  First diff at byte {}", i);
                eprintln!(
                    "  Restored: {:?}",
                    String::from_utf8_lossy(&restored[start..end])
                );
                eprintln!(
                    "  Original: {:?}",
                    String::from_utf8_lossy(&data[start..end])
                );
                break;
            }
        }
        if restored.len() != data.len() {
            eprintln!("  Length diff: {} vs {}", restored.len(), data.len());
        }
    }

    // Test Balanced mode (should pass)
    let (preprocessed_bal, chain_bal) = preprocess(&data, format, Mode::Balanced);
    let restored_bal = reverse_preprocess(&preprocessed_bal, &chain_bal);
    if restored_bal == data {
        eprintln!("Balanced pipeline roundtrip: OK");
    } else {
        eprintln!("Balanced pipeline roundtrip: FAILED");
    }
}
