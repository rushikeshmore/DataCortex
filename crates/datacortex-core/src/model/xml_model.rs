//! XML State Tracker — provides context bits for XML-heavy content (e.g. enwik8).
//!
//! Tracks XML structural state via a simple FSM updated byte-by-byte.
//! Does NOT predict — only provides context bits to the mixer/APM.
//! This gives specialized weight sets for tag names vs prose vs attributes.

/// XML structural states (3 bits = 8 states).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum XmlState {
    /// Normal text content between tags.
    Content = 0,
    /// Just saw '<', deciding what kind of tag.
    TagOpen = 1,
    /// Reading tag name after '<'.
    TagName = 2,
    /// Reading after '</'.
    TagClose = 3,
    /// Inside tag, reading attributes (after tag name, before '>').
    Attribute = 4,
    /// Inside quoted attribute value (after '=' and '"').
    AttributeValue = 5,
    /// Inside <!-- ... --> comment.
    Comment = 6,
    /// Inside &...; entity reference.
    Entity = 7,
}

/// XML state tracker FSM.
/// Updated byte-by-byte after each completed byte in the CM engine.
pub struct XmlTracker {
    /// Current XML state (3 bits).
    pub state: XmlState,
    /// Current tag nesting depth (saturates at 255).
    depth: u8,
    /// Comment state machine: tracks '--' sequence inside '<!'.
    comment_dashes: u8,
    /// Whether we've seen the tag name yet in current tag.
    tag_name_seen: bool,
}

impl XmlTracker {
    pub fn new() -> Self {
        XmlTracker {
            state: XmlState::Content,
            depth: 0,
            comment_dashes: 0,
            tag_name_seen: false,
        }
    }

    /// Update state after observing a completed byte.
    /// Call this once per byte, after the byte is fully decoded.
    #[inline]
    pub fn update(&mut self, byte: u8) {
        match self.state {
            XmlState::Content => {
                if byte == b'<' {
                    self.state = XmlState::TagOpen;
                    self.comment_dashes = 0;
                    self.tag_name_seen = false;
                } else if byte == b'&' {
                    self.state = XmlState::Entity;
                }
            }
            XmlState::TagOpen => {
                if byte == b'/' {
                    self.state = XmlState::TagClose;
                } else if byte == b'!' {
                    // Could be comment <!-- or CDATA/DOCTYPE.
                    // We'll track comment_dashes to detect <!--
                    self.comment_dashes = 0;
                    self.state = XmlState::Comment; // assume comment-like
                } else if byte == b'?' {
                    // Processing instruction <?...?>
                    self.state = XmlState::TagName;
                } else if byte == b'>' {
                    // Empty <> — unusual but handle gracefully.
                    self.state = XmlState::Content;
                } else {
                    self.state = XmlState::TagName;
                    self.tag_name_seen = false;
                }
            }
            XmlState::TagName => {
                if byte == b'>' {
                    self.state = XmlState::Content;
                    self.depth = self.depth.saturating_add(1);
                } else if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
                    self.state = XmlState::Attribute;
                    self.tag_name_seen = true;
                } else if byte == b'/' {
                    // Self-closing tag like <br/>
                    self.state = XmlState::Attribute;
                }
            }
            XmlState::TagClose => {
                if byte == b'>' {
                    self.state = XmlState::Content;
                    self.depth = self.depth.saturating_sub(1);
                }
                // Otherwise keep reading closing tag name.
            }
            XmlState::Attribute => {
                if byte == b'>' {
                    self.state = XmlState::Content;
                    self.depth = self.depth.saturating_add(1);
                } else if byte == b'"' {
                    self.state = XmlState::AttributeValue;
                } else if byte == b'/' {
                    // Self-closing: next should be '>'.
                    // Stay in Attribute state.
                }
            }
            XmlState::AttributeValue => {
                if byte == b'"' {
                    self.state = XmlState::Attribute;
                }
                // Otherwise stay in AttributeValue.
            }
            XmlState::Comment => {
                // Track for comment end: -->
                if byte == b'-' {
                    self.comment_dashes += 1;
                } else if byte == b'>' && self.comment_dashes >= 2 {
                    self.state = XmlState::Content;
                    self.comment_dashes = 0;
                } else {
                    self.comment_dashes = 0;
                }
            }
            XmlState::Entity => {
                if byte == b';' {
                    self.state = XmlState::Content;
                } else if !byte.is_ascii_alphanumeric() && byte != b'#' {
                    // Malformed entity — go back to content.
                    self.state = XmlState::Content;
                }
            }
        }
    }

    /// Return the XML state as a 3-bit value (0-7).
    #[inline(always)]
    pub fn state_bits(&self) -> u8 {
        self.state as u8
    }

    /// Return the tag depth quantized to 2 bits (0-3).
    /// 0=top level, 1=depth 1-2, 2=depth 3-6, 3=depth 7+
    #[inline(always)]
    pub fn depth_quantized(&self) -> u8 {
        match self.depth {
            0 => 0,
            1..=2 => 1,
            3..=6 => 2,
            _ => 3,
        }
    }
}

impl Default for XmlTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_is_content() {
        let tracker = XmlTracker::new();
        assert_eq!(tracker.state, XmlState::Content);
        assert_eq!(tracker.state_bits(), 0);
        assert_eq!(tracker.depth_quantized(), 0);
    }

    #[test]
    fn tracks_simple_tag() {
        let mut t = XmlTracker::new();
        // <title>
        t.update(b'<');
        assert_eq!(t.state, XmlState::TagOpen);
        t.update(b't');
        assert_eq!(t.state, XmlState::TagName);
        t.update(b'i');
        t.update(b't');
        t.update(b'l');
        t.update(b'e');
        assert_eq!(t.state, XmlState::TagName);
        t.update(b'>');
        assert_eq!(t.state, XmlState::Content);
        assert_eq!(t.depth_quantized(), 1);
    }

    #[test]
    fn tracks_closing_tag() {
        let mut t = XmlTracker::new();
        // <p>text</p>
        for b in b"<p>" {
            t.update(*b);
        }
        assert_eq!(t.state, XmlState::Content);
        for b in b"text" {
            t.update(*b);
        }
        t.update(b'<');
        assert_eq!(t.state, XmlState::TagOpen);
        t.update(b'/');
        assert_eq!(t.state, XmlState::TagClose);
        t.update(b'p');
        t.update(b'>');
        assert_eq!(t.state, XmlState::Content);
    }

    #[test]
    fn tracks_attribute() {
        let mut t = XmlTracker::new();
        // <a href="url">
        for b in b"<a" {
            t.update(*b);
        }
        t.update(b' ');
        assert_eq!(t.state, XmlState::Attribute);
        for b in b"href=" {
            t.update(*b);
        }
        t.update(b'"');
        assert_eq!(t.state, XmlState::AttributeValue);
        for b in b"url" {
            t.update(*b);
        }
        t.update(b'"');
        assert_eq!(t.state, XmlState::Attribute);
        t.update(b'>');
        assert_eq!(t.state, XmlState::Content);
    }

    #[test]
    fn tracks_comment() {
        let mut t = XmlTracker::new();
        // <!-- comment -->
        t.update(b'<');
        t.update(b'!');
        assert_eq!(t.state, XmlState::Comment);
        for b in b"-- comment --" {
            t.update(*b);
        }
        t.update(b'>');
        assert_eq!(t.state, XmlState::Content);
    }

    #[test]
    fn tracks_entity() {
        let mut t = XmlTracker::new();
        // &amp;
        t.update(b'&');
        assert_eq!(t.state, XmlState::Entity);
        for b in b"amp" {
            t.update(*b);
        }
        t.update(b';');
        assert_eq!(t.state, XmlState::Content);
    }

    #[test]
    fn state_bits_range() {
        let mut t = XmlTracker::new();
        assert!(t.state_bits() < 8);
        t.update(b'<');
        assert!(t.state_bits() < 8);
        t.update(b'a');
        assert!(t.state_bits() < 8);
    }

    #[test]
    fn depth_quantization() {
        let mut t = XmlTracker::new();
        assert_eq!(t.depth_quantized(), 0);
        // Nest 10 tags deep
        for _ in 0..10 {
            for b in b"<x>" {
                t.update(*b);
            }
        }
        assert_eq!(t.depth_quantized(), 3);
    }

    #[test]
    fn enwik8_snippet() {
        let mut t = XmlTracker::new();
        let snippet = b"<mediawiki><page><title>Some Title</title><text>Hello &amp; world</text></page></mediawiki>";
        for &b in snippet {
            t.update(b);
            assert!(t.state_bits() < 8);
        }
        // After all closing tags, should be back to content with depth near 0.
        assert_eq!(t.state, XmlState::Content);
    }
}
