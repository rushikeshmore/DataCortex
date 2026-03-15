//! Logistic transforms — squash and stretch for probability <-> log-odds.
//!
//! stretch(p) maps 12-bit probability [1, 4095] to log-odds (scaled integer).
//! squash(d)  maps log-odds back to 12-bit probability [1, 4095].
//!
//! Both use lookup tables computed at compile time.
//!
//! Formula pair:
//!   squash(d) = 2048 + d * 2047 / (K + |d|)
//!   stretch(p) = c * K / (2047 - |c|)  where c = p - 2048
//!
//! These are analytical inverses of each other.
//! K=64 gives a steep sigmoid covering nearly all of [1, 4095].

/// Steepness parameter. K=64 gives a steep sigmoid covering nearly all of [1, 4095].
const K: i32 = 64;

/// Squash table. 16384 entries covering d in [-8192, 8191].
/// Formula: p = 2048 + d * 2047 / (K + |d|)
const SQUASH_SIZE: usize = 16384;
const SQUASH_OFFSET: i32 = 8192;

static SQUASH_TABLE: [u16; SQUASH_SIZE] = {
    let mut table = [0u16; SQUASH_SIZE];
    let mut i = 0usize;
    while i < SQUASH_SIZE {
        let d = i as i32 - SQUASH_OFFSET;
        let abs_d = if d < 0 { -d } else { d };
        let p = 2048 + (d * 2047) / (K + abs_d);
        table[i] = if p < 1 {
            1
        } else if p > 4095 {
            4095
        } else {
            p as u16
        };
        i += 1;
    }
    table
};

/// Stretch table: 4097 entries for p in [0, 4096].
/// Formula: d = c * K / (2047 - |c|) where c = p - 2048.
/// This is the analytical inverse of squash.
static STRETCH_TABLE: [i16; 4097] = {
    let mut table = [0i16; 4097];
    let mut p = 0usize;
    while p <= 4096 {
        let c = p as i32 - 2048;
        let abs_c = if c < 0 { -c } else { c };

        let d = if abs_c >= 2047 {
            if c >= 0 { 8191i32 } else { -8191i32 }
        } else {
            (c * K) / (2047 - abs_c)
        };

        table[p] = if d > 8191 {
            8191
        } else if d < -8191 {
            -8191
        } else {
            d as i16
        };
        p += 1;
    }
    table
};

/// Convert 12-bit probability to log-odds.
/// Input: p in [1, 4095].
/// Output: log-odds as scaled integer.
#[inline(always)]
pub fn stretch(p: u32) -> i32 {
    STRETCH_TABLE[p.min(4096) as usize] as i32
}

/// Convert log-odds to 12-bit probability.
/// Input: d as scaled integer (any range, clamped internally).
/// Output: probability in [1, 4095].
#[inline(always)]
pub fn squash(d: i32) -> u32 {
    let idx = (d + SQUASH_OFFSET).clamp(0, (SQUASH_SIZE - 1) as i32) as usize;
    SQUASH_TABLE[idx] as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squash_at_zero_is_half() {
        let p = squash(0);
        assert_eq!(p, 2048, "squash(0) should be exactly 2048, got {p}");
    }

    #[test]
    fn stretch_at_half_is_zero() {
        let d = stretch(2048);
        assert_eq!(d, 0, "stretch(2048) should be 0, got {d}");
    }

    #[test]
    fn squash_output_in_range() {
        for d in -10000..=10000 {
            let p = squash(d);
            assert!(
                (1..=4095).contains(&p),
                "squash({d}) = {p}, out of [1, 4095]"
            );
        }
    }

    #[test]
    fn stretch_output_bounded() {
        for p in 1..=4095u32 {
            let d = stretch(p);
            assert!(
                (-8191..=8191).contains(&d),
                "stretch({p}) = {d}, out of bounds"
            );
        }
    }

    #[test]
    fn squash_is_monotonic() {
        let mut prev = squash(-10000);
        for d in -9999..=10000 {
            let p = squash(d);
            assert!(p >= prev, "squash not monotonic at d={d}: {prev} > {p}");
            prev = p;
        }
    }

    #[test]
    fn stretch_is_monotonic() {
        let mut prev = stretch(1);
        for p in 2..=4095u32 {
            let d = stretch(p);
            assert!(d >= prev, "stretch not monotonic at p={p}: {prev} > {d}");
            prev = d;
        }
    }

    #[test]
    fn roundtrip_squash_stretch() {
        // squash(stretch(p)) should be approximately p.
        // The steep sigmoid (K=64) creates large quantization steps in the
        // stretch table near the extremes, causing rounding errors. This is
        // acceptable — these transforms are used in logistic mixing (Phase 3)
        // where 1-2% error in probability space is fine.
        let mut max_diff = 0u32;
        for p in 100..=3996u32 {
            let d = stretch(p);
            let p2 = squash(d);
            let diff = (p2 as i32 - p as i32).unsigned_abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        // Error comes from integer division in both tables.
        assert!(
            max_diff <= 35,
            "max roundtrip error {max_diff} in range [100, 3996]"
        );
    }

    #[test]
    fn roundtrip_stretch_squash() {
        // stretch(squash(d)) should be approximately d.
        // Near extremes the sigmoid flattens, amplifying quantization error.
        for d in -1500..=1500 {
            let p = squash(d);
            let d2 = stretch(p);
            let diff = (d2 - d).unsigned_abs();
            assert!(
                diff <= 30,
                "roundtrip error: d={d}, squash={p}, stretch(squash)={d2}, diff={diff}"
            );
        }
    }

    #[test]
    fn symmetry() {
        // stretch(p) should equal -stretch(4096 - p)
        for p in 1..=4095u32 {
            let d1 = stretch(p);
            let d2 = stretch(4096 - p);
            assert_eq!(
                d1,
                -d2,
                "asymmetry at p={p}: stretch({p})={d1}, stretch({})={d2}",
                4096 - p,
            );
        }
    }

    #[test]
    fn squash_extremes() {
        assert!(squash(-10000) <= 20, "squash(-10000) = {}", squash(-10000));
        assert!(squash(10000) >= 4076, "squash(10000) = {}", squash(10000));
    }

    #[test]
    fn stretch_extremes() {
        assert!(stretch(1) < -60, "stretch(1) = {}", stretch(1));
        assert!(stretch(4095) > 60, "stretch(4095) = {}", stretch(4095));
    }
}
