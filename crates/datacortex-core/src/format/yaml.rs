//! YAML columnar reorg — lossless transform that detects arrays of objects
//! in YAML (like Kubernetes manifests) and applies columnar layout.
//!
//! YAML multi-document files (---) with similar structure (K8s deployments,
//! services, etc.) benefit from grouping identical keys together.
//!
//! Approach: Flatten each YAML document into key-path/value pairs.
//! Group documents with identical key-path sets, then columnar reorg on values.
//!
//! For simplicity, we detect "---" document separators and flat key: value lines.
//! Each document's key-value lines are grouped by key path.
//!
//! Layout:
//!   [key_path_0 values joined by \x01] \x00 [key_path_1 values joined by \x01] \x00 ...
//!
//! Separators:
//!   \x00 = column separator
//!   \x01 = value separator within a column

use super::transform::TransformResult;

const COL_SEP: u8 = 0x00;
// VAL_SEP reserved for future key-path columnar within documents.
#[allow(dead_code)]
const VAL_SEP: u8 = 0x01;
const METADATA_VERSION: u8 = 1;

/// A line from a YAML document with its indentation and content.
/// Reserved for future key-path columnar within documents.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct YamlLine<'a> {
    indent: usize,
    raw: &'a [u8],
}

/// Split YAML data into documents (separated by "---").
fn split_documents(data: &[u8]) -> Vec<Vec<u8>> {
    let mut docs: Vec<Vec<u8>> = Vec::new();
    let mut current = Vec::new();
    let mut start = 0;

    for i in 0..data.len() {
        if data[i] == b'\n' {
            let line = &data[start..i];
            let trimmed = line
                .iter()
                .position(|&b| !b.is_ascii_whitespace())
                .map(|p| &line[p..])
                .unwrap_or(b"");

            if trimmed == b"---" {
                if !current.is_empty() || !docs.is_empty() {
                    docs.push(std::mem::take(&mut current));
                }
                // Include the --- line in the next doc's prefix.
                current.extend_from_slice(line);
                current.push(b'\n');
            } else {
                current.extend_from_slice(line);
                current.push(b'\n');
            }
            start = i + 1;
        }
    }
    // Handle last line without trailing newline.
    if start < data.len() {
        current.extend_from_slice(&data[start..]);
    }
    if !current.is_empty() {
        docs.push(current);
    }

    docs
}

/// Check if data looks like YAML (multi-document K8s style).
pub fn detect_yaml(data: &[u8]) -> bool {
    let sample = &data[..data.len().min(8192)];
    let text = String::from_utf8_lossy(sample);

    let mut doc_seps = 0;
    let mut kv_lines = 0;
    let mut total_lines = 0;

    for line in text.lines() {
        total_lines += 1;
        if total_lines > 50 {
            break;
        }
        let trimmed = line.trim();
        if trimmed == "---" {
            doc_seps += 1;
        } else if !trimmed.is_empty() && !trimmed.starts_with('#') {
            // Check for key: value pattern.
            if let Some(colon_pos) = trimmed.find(':') {
                let key_part = &trimmed[..colon_pos];
                // Key should be alphanumeric/dashes/dots/underscores (possibly with leading -)
                let key_check = key_part.trim().trim_start_matches("- ");
                if !key_check.is_empty()
                    && key_check.chars().all(|c| {
                        c.is_alphanumeric() || c == '_' || c == '-' || c == '.' || c == '/'
                    })
                {
                    kv_lines += 1;
                }
            }
        }
    }

    // Need document separators and key-value lines.
    doc_seps >= 2 && kv_lines >= 10
}

/// Forward transform: YAML document-level columnar reorg.
///
/// Strategy: Keep each document as a unit. Group all documents together.
/// Store each document as a column value. The grouping itself helps because
/// similar documents (same apiVersion, kind, etc.) compress well together.
///
/// For a more sophisticated approach, we flatten key paths and do proper columnar.
/// But for V1, we do document-level grouping: sort documents by kind+name,
/// then store them column-style.
pub fn preprocess(data: &[u8]) -> Option<TransformResult> {
    if data.is_empty() || data.len() < 100 {
        return None;
    }

    let has_trailing_newline = data.last() == Some(&b'\n');

    let docs = split_documents(data);
    if docs.len() < 3 {
        return None;
    }

    // Parse each document to extract kind and name for sorting.
    #[derive(Debug)]
    struct DocInfo {
        kind: String,
        name: String,
        #[allow(dead_code)]
        idx: usize,
    }

    let mut infos: Vec<DocInfo> = Vec::new();
    for (idx, doc) in docs.iter().enumerate() {
        let text = String::from_utf8_lossy(doc);
        let mut kind = String::new();
        let mut name = String::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("kind:") {
                kind = rest.trim().to_string();
            }
            if let Some(rest) = trimmed.strip_prefix("name:") {
                if name.is_empty() {
                    name = rest.trim().to_string();
                }
            }
        }
        infos.push(DocInfo { kind, name, idx });
    }

    // Sort documents by kind first, then by name.
    // This groups all Deployments together, all Services together, etc.
    let mut sort_order: Vec<usize> = (0..infos.len()).collect();
    sort_order.sort_by(|&a, &b| {
        infos[a]
            .kind
            .cmp(&infos[b].kind)
            .then_with(|| infos[a].name.cmp(&infos[b].name))
    });

    // Build output: sorted documents separated by \x00.
    let mut col_data = Vec::with_capacity(data.len());
    for (i, &si) in sort_order.iter().enumerate() {
        col_data.extend_from_slice(&docs[si]);
        if i < sort_order.len() - 1 {
            col_data.push(COL_SEP);
        }
    }

    // Check that reorganization is different from original (sorting actually changed order).
    let original_order: Vec<usize> = (0..docs.len()).collect();
    if sort_order == original_order {
        // Already sorted — no benefit from reordering.
        // Still apply the transform for the columnar separator structure.
    }

    // Build metadata.
    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION);
    metadata.extend_from_slice(&(docs.len() as u32).to_le_bytes());
    metadata.push(if has_trailing_newline { 1 } else { 0 });

    // Store the sort_order so we can reverse it.
    // inverse_order[i] = where document i in the sorted output came from in the original.
    // We store sort_order: sorted position -> original index.
    for &si in &sort_order {
        metadata.extend_from_slice(&(si as u16).to_le_bytes());
    }

    // Verify roundtrip before committing.
    let result = TransformResult {
        data: col_data,
        metadata,
    };
    let restored = reverse(&result.data, &result.metadata);
    if restored != data {
        return None;
    }

    Some(result)
}

/// Reverse transform: reconstruct YAML from sorted documents + metadata.
pub fn reverse(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    if metadata.len() < 6 {
        return data.to_vec();
    }

    let mut mpos = 0;
    let _version = metadata[mpos];
    mpos += 1;
    let num_docs = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
    mpos += 4;
    let _has_trailing_newline = metadata[mpos] != 0;
    mpos += 1;

    if num_docs == 0 {
        return data.to_vec();
    }

    // Read sort_order.
    let mut sort_order: Vec<usize> = Vec::with_capacity(num_docs);
    for _ in 0..num_docs {
        if mpos + 2 > metadata.len() {
            return data.to_vec();
        }
        let idx = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
        mpos += 2;
        sort_order.push(idx);
    }

    // Parse sorted documents from data.
    let sorted_docs: Vec<&[u8]> = data.split(|&b| b == COL_SEP).collect();
    if sorted_docs.len() != num_docs {
        return data.to_vec();
    }

    // Build inverse: original_docs[sort_order[i]] = sorted_docs[i].
    let mut original_docs: Vec<Option<&[u8]>> = vec![None; num_docs];
    for (sorted_pos, &orig_idx) in sort_order.iter().enumerate() {
        if orig_idx >= num_docs {
            return data.to_vec();
        }
        original_docs[orig_idx] = Some(sorted_docs[sorted_pos]);
    }

    // Reconstruct by concatenating in original order.
    let mut output = Vec::with_capacity(data.len());
    for doc in original_docs.iter() {
        match doc {
            Some(d) => {
                output.extend_from_slice(d);
            }
            None => return data.to_vec(),
        }
    }

    // Handle trailing newline: the last document's content may or may not end
    // with a newline. The has_trailing_newline flag tells us about the original file.
    // Since we preserve document content exactly, the trailing newline is already correct
    // as long as the last original document had the right ending.

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_documents_basic() {
        let data = b"---\nkind: Deployment\nname: web\n---\nkind: Service\nname: web\n";
        let docs = split_documents(data);
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn detect_yaml_positive() {
        let data = b"---\napiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: web-server\n  namespace: production\n---\napiVersion: v1\nkind: Service\nmetadata:\n  name: web-server\n  namespace: production\n---\napiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler\nmetadata:\n  name: web-server\n";
        assert!(detect_yaml(data));
    }

    #[test]
    fn detect_yaml_negative() {
        let data = b"just some plain text\nwithout YAML structure\n";
        assert!(!detect_yaml(data));
    }

    #[test]
    fn roundtrip_simple() {
        let data = b"---\napiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: web-server\nspec:\n  replicas: 3\n---\napiVersion: v1\nkind: Service\nmetadata:\n  name: web-server\nspec:\n  type: ClusterIP\n---\napiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler\nmetadata:\n  name: web-server\nspec:\n  minReplicas: 1\n";

        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(
            String::from_utf8_lossy(&restored),
            String::from_utf8_lossy(data),
        );
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn roundtrip_multiple_kinds() {
        // Multiple documents of different kinds — sorting groups them.
        let data = b"---\nkind: Deployment\nmetadata:\n  name: api\n---\nkind: Service\nmetadata:\n  name: api\n---\nkind: Deployment\nmetadata:\n  name: web\n---\nkind: Service\nmetadata:\n  name: web\n---\nkind: Deployment\nmetadata:\n  name: worker\n";

        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }

    #[test]
    fn too_few_docs_returns_none() {
        let data = b"---\nkind: Deployment\nname: web\n---\nkind: Service\nname: web\n";
        assert!(preprocess(data).is_none());
    }

    #[test]
    fn empty_returns_none() {
        assert!(preprocess(b"").is_none());
    }

    #[test]
    fn roundtrip_no_trailing_newline() {
        let data = b"---\nkind: Deployment\nmetadata:\n  name: a\n---\nkind: Service\nmetadata:\n  name: b\n---\nkind: HPA\nmetadata:\n  name: c";
        let result = preprocess(data).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data.to_vec());
    }
}
