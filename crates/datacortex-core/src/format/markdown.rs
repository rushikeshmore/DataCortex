//! Markdown structural transform — separates code blocks from prose.
//!
//! Markdown files contain two statistically different streams:
//! 1. **Prose** — natural language text, headers, lists, links
//! 2. **Code blocks** — fenced code (``` or ~~~), very different entropy profile
//!
//! Mixing these hurts CM predictions: code has semicolons, braces, keywords;
//! prose has articles, prepositions, punctuation. Separating them lets the
//! CM engine build specialized contexts for each.
//!
//! Transform output:
//!   [prose stream with placeholders] \x00 [code block 0] \x01 [code block 1] \x01 ...
//!
//! Placeholder in prose: \x02 marks where a code block was extracted from.
//!
//! Metadata stores:
//!   - version (1 byte)
//!   - num_blocks (u32 LE)
//!   - has_trailing_newline (1 byte)
//!   - for each block: fence_line bytes (the ``` line including lang tag) as u16 len + data
//!     and closing_fence bytes as u16 len + data
//!
//! The fence lines (```rust, ```json, etc.) are stored in metadata so they
//! don't pollute either stream. The code content is in the code stream.
//! The prose stream contains the natural language with \x02 placeholders.

use super::transform::TransformResult;

const STREAM_SEP: u8 = 0x00;
const CODE_BLOCK_SEP: u8 = 0x01;
const PLACEHOLDER: u8 = 0x02;
const METADATA_VERSION: u8 = 1;

/// A detected fenced code block in the markdown.
struct CodeBlock {
    /// The opening fence line (e.g., "```rust\n") — stored in metadata.
    opening_fence: Vec<u8>,
    /// The code content between fences (NOT including fence lines).
    content: Vec<u8>,
    /// The closing fence line (e.g., "```\n") — stored in metadata.
    closing_fence: Vec<u8>,
    /// Byte offset in original data where the block starts (opening fence).
    start: usize,
    /// Byte offset in original data after the block ends (after closing fence line).
    end: usize,
}

/// Detect if a line is a code fence opener/closer.
/// Returns Some(fence_chars) if line starts with ``` or ~~~, else None.
/// fence_chars is the fence prefix (e.g., "```" or "~~~~").
fn detect_fence(line: &[u8]) -> Option<(u8, usize)> {
    if line.is_empty() {
        return None;
    }

    // Skip leading whitespace (up to 3 spaces per CommonMark spec).
    let mut pos = 0;
    let mut spaces = 0;
    while pos < line.len() && line[pos] == b' ' && spaces < 3 {
        pos += 1;
        spaces += 1;
    }

    if pos >= line.len() {
        return None;
    }

    let fence_char = line[pos];
    if fence_char != b'`' && fence_char != b'~' {
        return None;
    }

    // Count consecutive fence chars.
    let fence_start = pos;
    while pos < line.len() && line[pos] == fence_char {
        pos += 1;
    }
    let fence_len = pos - fence_start;

    // Need at least 3 fence chars.
    if fence_len < 3 {
        return None;
    }

    Some((fence_char, fence_len))
}

/// Check if a line is a closing fence for the given opener.
fn is_closing_fence(line: &[u8], fence_char: u8, fence_len: usize) -> bool {
    // A closing fence must:
    // 1. Start with optional up to 3 spaces
    // 2. Have >= fence_len of the same fence_char
    // 3. Only have optional whitespace after the fence chars

    let mut pos = 0;
    let mut spaces = 0;
    while pos < line.len() && line[pos] == b' ' && spaces < 3 {
        pos += 1;
        spaces += 1;
    }

    if pos >= line.len() || line[pos] != fence_char {
        return false;
    }

    let start = pos;
    while pos < line.len() && line[pos] == fence_char {
        pos += 1;
    }

    if pos - start < fence_len {
        return false;
    }

    // After fence chars, only whitespace allowed.
    while pos < line.len() {
        if !line[pos].is_ascii_whitespace() {
            return false;
        }
        pos += 1;
    }

    true
}

/// Parse markdown and extract code blocks.
fn extract_code_blocks(data: &[u8]) -> Vec<CodeBlock> {
    let mut blocks = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        // Find the end of this line.
        let line_start = pos;
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        let line_end = pos;
        if pos < data.len() {
            pos += 1; // skip \n
        }

        let line = &data[line_start..line_end];

        // Check if this line is an opening fence.
        if let Some((fence_char, fence_len)) = detect_fence(line) {
            // Check it's an opener (not a closer) — openers can have info string after fence.
            // Closers must only have whitespace after fence chars.
            // At this point we're not inside a block, so this IS an opener.

            let block_start = line_start;
            let opening_fence = data[line_start..pos].to_vec(); // includes \n
            let content_start = pos;

            // Find the closing fence.
            let mut content_end = pos;
            let mut block_end = data.len(); // default: unclosed = rest of file
            let mut closing_fence = Vec::new();

            while pos < data.len() {
                let cl_start = pos;
                while pos < data.len() && data[pos] != b'\n' {
                    pos += 1;
                }
                let cl_end = pos;
                if pos < data.len() {
                    pos += 1; // skip \n
                }

                let cline = &data[cl_start..cl_end];
                if is_closing_fence(cline, fence_char, fence_len) {
                    content_end = cl_start;
                    closing_fence = data[cl_start..pos].to_vec();
                    block_end = pos;
                    break;
                }
            }

            if closing_fence.is_empty() {
                // Unclosed code block — treat rest as content, no closing fence.
                content_end = data.len();
                block_end = data.len();
            }

            let content = data[content_start..content_end].to_vec();

            blocks.push(CodeBlock {
                opening_fence,
                content,
                closing_fence,
                start: block_start,
                end: block_end,
            });
        }
    }

    blocks
}

/// Forward transform: separate code blocks from prose in markdown.
///
/// Returns None if:
/// - No code blocks found
/// - Data is too small to benefit
pub fn preprocess(data: &[u8]) -> Option<TransformResult> {
    if data.len() < 50 {
        return None;
    }

    let blocks = extract_code_blocks(data);
    if blocks.is_empty() {
        return None;
    }

    let has_trailing_newline = data.last() == Some(&b'\n');

    // Build prose stream: original data with code blocks replaced by \x02.
    let mut prose = Vec::with_capacity(data.len());
    let mut prev_end = 0;
    for block in &blocks {
        // Copy prose before this block.
        escape_special(&data[prev_end..block.start], &mut prose);
        // Insert placeholder.
        prose.push(PLACEHOLDER);
        prev_end = block.end;
    }
    // Copy remaining prose after last block.
    escape_special(&data[prev_end..], &mut prose);

    // Build code stream: code block contents separated by \x01.
    let mut code_stream = Vec::with_capacity(data.len());
    for (i, block) in blocks.iter().enumerate() {
        code_stream.extend_from_slice(&block.content);
        if i < blocks.len() - 1 {
            code_stream.push(CODE_BLOCK_SEP);
        }
    }

    // Build output: prose \x00 code_stream
    let mut output = Vec::with_capacity(prose.len() + 1 + code_stream.len());
    output.extend_from_slice(&prose);
    output.push(STREAM_SEP);
    output.extend_from_slice(&code_stream);

    // Build metadata.
    let mut metadata = Vec::new();
    metadata.push(METADATA_VERSION);
    metadata.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    metadata.push(if has_trailing_newline { 1 } else { 0 });

    for block in &blocks {
        // Opening fence.
        metadata.extend_from_slice(&(block.opening_fence.len() as u16).to_le_bytes());
        metadata.extend_from_slice(&block.opening_fence);
        // Closing fence.
        metadata.extend_from_slice(&(block.closing_fence.len() as u16).to_le_bytes());
        metadata.extend_from_slice(&block.closing_fence);
    }

    // Only apply if code blocks are substantial enough that stream separation
    // outweighs the metadata overhead. Requirements:
    // 1. At least 20 bytes of code content
    // 2. Code blocks make up at least 10% of the file
    // Note: unlike columnar transforms, this doesn't reduce raw byte count.
    // The win is in CM prediction — code and prose have different statistics.
    let total_code_bytes: usize = blocks.iter().map(|b| b.content.len()).sum();
    if total_code_bytes < 20 || total_code_bytes * 10 < data.len() {
        return None;
    }

    Some(TransformResult {
        data: output,
        metadata,
    })
}

/// Escape \x00, \x01, \x02 bytes in prose sections so they don't collide
/// with our separators. Since these bytes are extremely rare in real markdown,
/// we use a simple escape: \x03 + (original_byte + 1).
///   \x00 -> \x03\x01
///   \x01 -> \x03\x02
///   \x02 -> \x03\x03
///   \x03 -> \x03\x04  (escape the escape byte itself)
fn escape_special(src: &[u8], dst: &mut Vec<u8>) {
    for &b in src {
        match b {
            0x00..=0x03 => {
                dst.push(0x03);
                dst.push(b + 1);
            }
            _ => dst.push(b),
        }
    }
}

/// Unescape prose bytes.
fn unescape_special(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len());
    let mut i = 0;
    while i < src.len() {
        if src[i] == 0x03 && i + 1 < src.len() {
            out.push(src[i + 1] - 1);
            i += 2;
        } else {
            out.push(src[i]);
            i += 1;
        }
    }
    out
}

/// Reverse transform: reconstruct markdown from separated streams + metadata.
pub fn reverse(data: &[u8], metadata: &[u8]) -> Vec<u8> {
    if metadata.len() < 6 {
        return data.to_vec();
    }

    let mut mpos = 0;
    let _version = metadata[mpos];
    mpos += 1;
    let num_blocks = u32::from_le_bytes(metadata[mpos..mpos + 4].try_into().unwrap()) as usize;
    mpos += 4;
    let _has_trailing_newline = metadata[mpos] != 0;
    mpos += 1;

    if num_blocks == 0 {
        return data.to_vec();
    }

    // Read fence metadata for each block.
    let mut fences: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        if mpos + 2 > metadata.len() {
            return data.to_vec();
        }
        let open_len = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
        mpos += 2;
        if mpos + open_len > metadata.len() {
            return data.to_vec();
        }
        let opening = metadata[mpos..mpos + open_len].to_vec();
        mpos += open_len;

        if mpos + 2 > metadata.len() {
            return data.to_vec();
        }
        let close_len = u16::from_le_bytes(metadata[mpos..mpos + 2].try_into().unwrap()) as usize;
        mpos += 2;
        if mpos + close_len > metadata.len() {
            return data.to_vec();
        }
        let closing = metadata[mpos..mpos + close_len].to_vec();
        mpos += close_len;

        fences.push((opening, closing));
    }

    // Split data into prose stream and code stream by \x00.
    let sep_pos = match data.iter().position(|&b| b == STREAM_SEP) {
        Some(p) => p,
        None => return data.to_vec(),
    };

    let prose_escaped = &data[..sep_pos];
    let code_data = &data[sep_pos + 1..];

    // Unescape prose.
    let prose = unescape_special(prose_escaped);

    // Split code stream by \x01 into individual block contents.
    let code_blocks: Vec<&[u8]> = if num_blocks == 1 {
        vec![code_data]
    } else {
        code_data.split(|&b| b == CODE_BLOCK_SEP).collect()
    };

    if code_blocks.len() != num_blocks {
        return data.to_vec();
    }

    // Reconstruct: walk through prose, replacing \x02 placeholders with
    // opening_fence + code_content + closing_fence.
    let mut output = Vec::with_capacity(data.len() * 2);
    let mut block_idx = 0;

    for &b in &prose {
        if b == PLACEHOLDER && block_idx < num_blocks {
            let (ref opening, ref closing) = fences[block_idx];
            output.extend_from_slice(opening);
            output.extend_from_slice(code_blocks[block_idx]);
            output.extend_from_slice(closing);
            block_idx += 1;
        } else {
            output.push(b);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_fence_backtick() {
        assert_eq!(detect_fence(b"```rust"), Some((b'`', 3)));
        assert_eq!(detect_fence(b"````"), Some((b'`', 4)));
        assert_eq!(detect_fence(b"~~~"), Some((b'~', 3)));
        assert_eq!(detect_fence(b"``not enough"), None);
        assert_eq!(detect_fence(b"regular text"), None);
    }

    #[test]
    fn detect_fence_with_leading_spaces() {
        assert_eq!(detect_fence(b"   ```rust"), Some((b'`', 3)));
        assert_eq!(detect_fence(b"  ~~~python"), Some((b'~', 3)));
    }

    #[test]
    fn is_closing_fence_basic() {
        assert!(is_closing_fence(b"```", b'`', 3));
        assert!(is_closing_fence(b"````", b'`', 3));
        assert!(is_closing_fence(b"```  ", b'`', 3));
        assert!(!is_closing_fence(b"``", b'`', 3));
        assert!(!is_closing_fence(b"~~~", b'`', 3));
        assert!(!is_closing_fence(b"```text", b'`', 3)); // info string after = not a closer
    }

    #[test]
    fn extract_single_code_block() {
        let md = b"# Hello\n\nSome text.\n\n```rust\nfn main() {}\n```\n\nMore text.\n";
        let blocks = extract_code_blocks(md);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].opening_fence, b"```rust\n");
        assert_eq!(blocks[0].content, b"fn main() {}\n");
        assert_eq!(blocks[0].closing_fence, b"```\n");
    }

    #[test]
    fn extract_multiple_code_blocks() {
        let md = b"# Doc\n\n```python\nprint('hello')\n```\n\nText between.\n\n```json\n{\"a\":1}\n```\n";
        let blocks = extract_code_blocks(md);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].content, b"print('hello')\n");
        assert_eq!(blocks[1].content, b"{\"a\":1}\n");
    }

    #[test]
    fn roundtrip_simple() {
        let md = b"# Title\n\nSome prose text here.\n\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```\n\nMore prose after code.\n";
        let result = preprocess(md).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(
            String::from_utf8_lossy(&restored),
            String::from_utf8_lossy(md),
        );
        assert_eq!(restored, md.to_vec());
    }

    #[test]
    fn roundtrip_multiple_blocks() {
        let md = b"# Guide\n\n## Section 1\n\nHere is some Python code:\n\n```python\ndef hello():\n    print('world')\n\nhello()\n```\n\nAnd here is some JSON:\n\n```json\n{\n  \"name\": \"test\",\n  \"value\": 42\n}\n```\n\n## Section 2\n\nFinal paragraph with **bold** and [links](url).\n";
        let result = preprocess(md).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, md.to_vec());
    }

    #[test]
    fn roundtrip_tilde_fences() {
        let md = b"# Doc\n\nSome text with more content here.\n\n~~~bash\necho hello world\nls -la /tmp\ncat README.md\n~~~\n\nDone with the document.\n";
        let result = preprocess(md).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, md.to_vec());
    }

    #[test]
    fn roundtrip_no_trailing_newline() {
        let md = b"# Title\n\nSome intro text here.\n\n```rust\nlet x = 1;\nlet y = x + 2;\nprintln!(\"{}\", y);\n```\n\nEnd of document";
        let result = preprocess(md).expect("should produce transform");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, md.to_vec());
    }

    #[test]
    fn no_code_blocks_returns_none() {
        let md =
            b"# Title\n\nJust prose text without any code blocks. This has no fences at all.\n";
        assert!(preprocess(md).is_none());
    }

    #[test]
    fn too_small_returns_none() {
        let md = b"# Hi\n```\na\n```\n";
        assert!(preprocess(md).is_none());
    }

    #[test]
    fn escape_roundtrip() {
        // Verify escape/unescape for bytes 0x00-0x03.
        let src = vec![0x00, 0x01, 0x02, 0x03, b'A', 0x00, b'B'];
        let mut escaped = Vec::new();
        escape_special(&src, &mut escaped);
        let unescaped = unescape_special(&escaped);
        assert_eq!(unescaped, src);
    }

    #[test]
    fn streams_are_separated() {
        let md = b"# Title\n\nProse paragraph with normal text.\n\n```rust\nfn main() {\n    println!(\"hello world\");\n}\n```\n\nMore prose.\n";
        let result = preprocess(md).unwrap();

        // Output should have exactly one \x00 separating prose from code.
        let sep_count = result.data.iter().filter(|&&b| b == STREAM_SEP).count();
        assert_eq!(sep_count, 1, "should have exactly one stream separator");

        // Prose stream should contain placeholder.
        let sep_pos = result.data.iter().position(|&b| b == STREAM_SEP).unwrap();
        let prose = &result.data[..sep_pos];
        assert!(
            prose.contains(&PLACEHOLDER),
            "prose should contain placeholder"
        );

        // Code stream should contain the actual code.
        let code = &result.data[sep_pos + 1..];
        assert!(
            code.windows(7).any(|w| w == b"fn main"),
            "code stream should contain the code"
        );
    }

    #[test]
    fn roundtrip_with_special_bytes() {
        // Markdown with content that contains bytes we use as separators.
        // This is unlikely in real markdown but tests our escaping.
        let mut md = b"# Title\n\nText with special bytes: ".to_vec();
        md.push(0x03); // Our escape byte in normal prose.
        md.extend_from_slice(b" end.\n\n```rust\nfn test() { /* code */ }\n```\n\nDone.\n");

        let result = preprocess(&md).expect("should handle special bytes");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, md);
    }

    #[test]
    fn roundtrip_large_doc() {
        // Build a realistic large markdown document.
        let mut md = String::new();
        md.push_str("# Architecture Guide\n\n");
        md.push_str("> A comprehensive guide to the system internals.\n\n");
        md.push_str("## Overview\n\n");
        md.push_str("This system uses multiple layers of processing. ");
        md.push_str("Each layer transforms the input progressively.\n\n");
        md.push_str("### Processing Pipeline\n\n");
        md.push_str("The pipeline consists of three stages:\n\n");
        md.push_str("1. **Detection** -- identify the input format\n");
        md.push_str("2. **Preprocessing** -- apply format-specific transforms\n");
        md.push_str("3. **Compression** -- entropy coding\n\n");
        md.push_str("```rust\nfn pipeline(data: &[u8]) -> Vec<u8> {\n");
        md.push_str("    let format = detect(data);\n");
        md.push_str("    let preprocessed = preprocess(data, format);\n");
        md.push_str("    compress(preprocessed)\n");
        md.push_str("}\n```\n\n");
        md.push_str("## Configuration\n\n");
        md.push_str("Configuration is done via TOML:\n\n");
        md.push_str("```toml\n[compression]\nmode = \"balanced\"\nmax_memory = \"4GB\"\n```\n\n");
        md.push_str("## API Example\n\n");
        md.push_str("```python\nimport datacortex\n\nresult = datacortex.compress(data, mode='balanced')\nprint(f'Ratio: {result.ratio:.2f}')\n```\n\n");
        md.push_str("## References\n\n");
        md.push_str("- [PAQ compressor](http://mattmahoney.net/dc/)\n");
        md.push_str("- [Arithmetic coding](https://en.wikipedia.org/wiki/Arithmetic_coding)\n");

        let data = md.as_bytes();
        let result = preprocess(data).expect("should transform large doc");
        let restored = reverse(&result.data, &result.metadata);
        assert_eq!(restored, data);
    }
}
