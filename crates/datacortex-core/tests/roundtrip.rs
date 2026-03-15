//! Integration tests — roundtrip every corpus file through compress/decompress.

use std::fs;
use std::path::Path;

use datacortex_core::{
    codec::{compress_to_vec, decompress_from_slice},
    dcx::Mode,
};

fn corpus_dir() -> &'static Path {
    // Integration tests run from the workspace root.
    Path::new("corpus")
}

fn roundtrip_file(path: &Path, mode: Mode) {
    let original =
        fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let compressed = compress_to_vec(&original, mode, None)
        .unwrap_or_else(|e| panic!("compress failed for {}: {e}", path.display()));
    let decompressed = decompress_from_slice(&compressed)
        .unwrap_or_else(|e| panic!("decompress failed for {}: {e}", path.display()));

    assert_eq!(
        original,
        decompressed,
        "roundtrip mismatch for {} (mode={mode}): {} orig bytes, {} decompressed bytes",
        path.display(),
        original.len(),
        decompressed.len(),
    );
}

#[test]
fn roundtrip_corpus_balanced() {
    let dir = corpus_dir();
    if !dir.exists() {
        eprintln!("corpus/ not found — skipping roundtrip test (run from workspace root)");
        return;
    }

    let mut files: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .collect();
    files.sort_by_key(|e| e.path());

    assert!(!files.is_empty(), "corpus/ is empty");

    for entry in &files {
        roundtrip_file(&entry.path(), Mode::Balanced);
    }
}

#[test]
fn roundtrip_corpus_all_modes() {
    let dir = corpus_dir();
    if !dir.exists() {
        return;
    }

    // Test alice29.txt with all modes — it's the benchmark reference file.
    let alice = dir.join("alice29.txt");
    if alice.exists() {
        for mode in [Mode::Max, Mode::Balanced, Mode::Fast] {
            roundtrip_file(&alice, mode);
        }
    }
}

#[test]
fn roundtrip_synthetic_sizes() {
    // Test edge cases: empty, tiny, 1 byte, 256 bytes, 64KB.
    let cases: Vec<Vec<u8>> = vec![
        vec![],
        vec![0],
        vec![42],
        vec![0xFF; 256],
        (0..=255).collect(),
        vec![b'A'; 65536],
    ];

    for (i, original) in cases.iter().enumerate() {
        let compressed = compress_to_vec(original, Mode::Balanced, None)
            .unwrap_or_else(|e| panic!("compress failed for case {i}: {e}"));
        let decompressed = decompress_from_slice(&compressed)
            .unwrap_or_else(|e| panic!("decompress failed for case {i}: {e}"));
        assert_eq!(
            original, &decompressed,
            "roundtrip mismatch for synthetic case {i}"
        );
    }
}
