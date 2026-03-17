//! GRU (Gated Recurrent Unit) byte-level predictor for dual-path compression.
//!
//! A byte-level neural predictor that provides a DIFFERENT signal from the
//! bit-level CM engine. The GRU captures cross-bit correlations within bytes
//! and sequential byte patterns via its hidden state.
//!
//! Architecture:
//!   Input: one-hot byte embedding (256 → 32 via embedding matrix)
//!   GRU: 64 hidden cells, 1 layer
//!   Output: 64 → 256 linear → softmax → byte probabilities
//!
//! Online training: after each byte is observed, backprop through current
//! step only (no BPTT). Hidden state carries forward context.
//!
//! ~43K parameters (~170KB at f32).
//!
//! CRITICAL: Encoder and decoder must maintain IDENTICAL GRU state.
//! sigmoid/tanh implementations must be EXACT same in both paths.

const EMBED_DIM: usize = 32;
const HIDDEN_DIM: usize = 128;
const VOCAB_SIZE: usize = 256;

// Total parameter count for reference: ~43,456
// - Embedding: 256 * 32 = 8,192
// - W_z, W_r, W_h: 3 * (32 * 64) = 6,144
// - U_z, U_r, U_h: 3 * (64 * 64) = 12,288
// - b_z, b_r, b_h: 3 * 64 = 192
// - W_o: 64 * 256 = 16,384
// - b_o: 256

/// Learning rate for online SGD.
const LEARNING_RATE: f32 = 0.01;

/// Gradient clipping threshold.
const GRAD_CLIP: f32 = 5.0;

/// GRU byte-level predictor with online learning.
pub struct GruModel {
    // --- Parameters ---
    /// Embedding matrix: [VOCAB_SIZE][EMBED_DIM]
    embedding: Vec<f32>,

    /// Update gate weights: W_z [HIDDEN_DIM][EMBED_DIM]
    w_z: Vec<f32>,
    /// Update gate recurrent: U_z [HIDDEN_DIM][HIDDEN_DIM]
    u_z: Vec<f32>,
    /// Update gate bias: [HIDDEN_DIM]
    b_z: Vec<f32>,

    /// Reset gate weights: W_r [HIDDEN_DIM][EMBED_DIM]
    w_r: Vec<f32>,
    /// Reset gate recurrent: U_r [HIDDEN_DIM][HIDDEN_DIM]
    u_r: Vec<f32>,
    /// Reset gate bias: [HIDDEN_DIM]
    b_r: Vec<f32>,

    /// Candidate weights: W_h [HIDDEN_DIM][EMBED_DIM]
    w_h: Vec<f32>,
    /// Candidate recurrent: U_h [HIDDEN_DIM][HIDDEN_DIM]
    u_h: Vec<f32>,
    /// Candidate bias: [HIDDEN_DIM]
    b_h: Vec<f32>,

    /// Output weights: W_o [VOCAB_SIZE][HIDDEN_DIM]
    w_o: Vec<f32>,
    /// Output bias: [VOCAB_SIZE]
    b_o: Vec<f32>,

    // --- State ---
    /// Hidden state: [HIDDEN_DIM]
    h: Vec<f32>,

    // --- Cached forward pass values (for backprop) ---
    /// Last input embedding: [EMBED_DIM]
    last_x: Vec<f32>,
    /// Last hidden state before this step: [HIDDEN_DIM]
    last_h_prev: Vec<f32>,
    /// Last update gate output: [HIDDEN_DIM]
    last_z: Vec<f32>,
    /// Last reset gate output: [HIDDEN_DIM]
    last_r: Vec<f32>,
    /// Last candidate activation: [HIDDEN_DIM]
    last_h_tilde: Vec<f32>,

    /// Cached byte probabilities after softmax: [VOCAB_SIZE]
    byte_probs: Vec<f32>,
    /// Whether byte_probs is valid.
    probs_valid: bool,
    /// Whether we have processed at least one byte (have valid probs).
    has_context: bool,
}

impl GruModel {
    /// Create a new GRU model with Xavier-initialized weights.
    pub fn new() -> Self {
        let mut model = GruModel {
            embedding: vec![0.0; VOCAB_SIZE * EMBED_DIM],
            w_z: vec![0.0; HIDDEN_DIM * EMBED_DIM],
            u_z: vec![0.0; HIDDEN_DIM * HIDDEN_DIM],
            b_z: vec![0.0; HIDDEN_DIM],
            w_r: vec![0.0; HIDDEN_DIM * EMBED_DIM],
            u_r: vec![0.0; HIDDEN_DIM * HIDDEN_DIM],
            b_r: vec![0.0; HIDDEN_DIM],
            w_h: vec![0.0; HIDDEN_DIM * EMBED_DIM],
            u_h: vec![0.0; HIDDEN_DIM * HIDDEN_DIM],
            b_h: vec![0.0; HIDDEN_DIM],
            w_o: vec![0.0; VOCAB_SIZE * HIDDEN_DIM],
            b_o: vec![0.0; VOCAB_SIZE],
            h: vec![0.0; HIDDEN_DIM],
            last_x: vec![0.0; EMBED_DIM],
            last_h_prev: vec![0.0; HIDDEN_DIM],
            last_z: vec![0.0; HIDDEN_DIM],
            last_r: vec![0.0; HIDDEN_DIM],
            last_h_tilde: vec![0.0; HIDDEN_DIM],
            byte_probs: vec![1.0 / VOCAB_SIZE as f32; VOCAB_SIZE],
            probs_valid: false,
            has_context: false,
        };
        model.init_weights();
        model
    }

    /// Initialize weights using a deterministic pseudo-random scheme.
    /// Xavier/Glorot initialization scaled by fan_in + fan_out.
    fn init_weights(&mut self) {
        // Deterministic PRNG for reproducibility (encoder = decoder).
        let mut seed: u64 = 0xDEAD_BEEF_CAFE_1234;

        // Xavier scale for embedding: sqrt(2 / (256 + 32))
        let embed_scale = (2.0 / (VOCAB_SIZE + EMBED_DIM) as f32).sqrt();
        fill_xavier(&mut self.embedding, embed_scale, &mut seed);

        // Xavier scale for input weights: sqrt(2 / (32 + 64))
        let wx_scale = (2.0 / (EMBED_DIM + HIDDEN_DIM) as f32).sqrt();
        fill_xavier(&mut self.w_z, wx_scale, &mut seed);
        fill_xavier(&mut self.w_r, wx_scale, &mut seed);
        fill_xavier(&mut self.w_h, wx_scale, &mut seed);

        // Xavier scale for recurrent weights: sqrt(2 / (64 + 64))
        let uh_scale = (2.0 / (HIDDEN_DIM + HIDDEN_DIM) as f32).sqrt();
        fill_xavier(&mut self.u_z, uh_scale, &mut seed);
        fill_xavier(&mut self.u_r, uh_scale, &mut seed);
        fill_xavier(&mut self.u_h, uh_scale, &mut seed);

        // Xavier scale for output weights: sqrt(2 / (64 + 256))
        let wo_scale = (2.0 / (HIDDEN_DIM + VOCAB_SIZE) as f32).sqrt();
        fill_xavier(&mut self.w_o, wo_scale, &mut seed);

        // Biases: initialize gate biases to slightly positive for update gate
        // (helps gradient flow) and zero for others.
        for b in self.b_z.iter_mut() {
            *b = 1.0; // Bias update gate to "remember" (z → 1 → keep old state).
        }
        // Reset gate and candidate biases stay at 0.
    }

    /// Forward pass: process one byte, update hidden state, compute output probs.
    /// Call this after observing a complete byte.
    #[inline(never)]
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&mut self, byte: u8) {
        // Save previous hidden state for backprop.
        self.last_h_prev.copy_from_slice(&self.h);

        // Get embedding for the input byte (just a row lookup, not full matmul).
        let byte_idx = byte as usize;
        let embed_start = byte_idx * EMBED_DIM;
        self.last_x
            .copy_from_slice(&self.embedding[embed_start..embed_start + EMBED_DIM]);

        // Compute all three gates in a fused manner for better cache locality.
        // Pre-compute x contribution for all gates at once.
        for i in 0..HIDDEN_DIM {
            let w_off = i * EMBED_DIM;
            let wz_row = &self.w_z[w_off..w_off + EMBED_DIM];
            let wr_row = &self.w_r[w_off..w_off + EMBED_DIM];
            let wh_row = &self.w_h[w_off..w_off + EMBED_DIM];

            let mut val_z = self.b_z[i];
            let mut val_r = self.b_r[i];
            let mut val_h = self.b_h[i];

            // W @ x for all three gates (auto-vectorizable).
            for j in 0..EMBED_DIM {
                let xj = self.last_x[j];
                val_z += wz_row[j] * xj;
                val_r += wr_row[j] * xj;
                val_h += wh_row[j] * xj;
            }

            // U_z @ h_{t-1} and U_r @ h_{t-1}
            let u_off = i * HIDDEN_DIM;
            let uz_row = &self.u_z[u_off..u_off + HIDDEN_DIM];
            let ur_row = &self.u_r[u_off..u_off + HIDDEN_DIM];

            for j in 0..HIDDEN_DIM {
                let hj = self.last_h_prev[j];
                val_z += uz_row[j] * hj;
                val_r += ur_row[j] * hj;
            }

            let z_i = sigmoid(val_z);
            let r_i = sigmoid(val_r);
            self.last_z[i] = z_i;
            self.last_r[i] = r_i;

            // U_h @ (r_t * h_{t-1}) for candidate
            let uh_row = &self.u_h[u_off..u_off + HIDDEN_DIM];
            for j in 0..HIDDEN_DIM {
                val_h += uh_row[j] * (r_i * self.last_h_prev[j]);
            }

            let h_tilde_i = tanh_approx(val_h);
            self.last_h_tilde[i] = h_tilde_i;

            // h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
            self.h[i] = (1.0 - z_i) * self.last_h_prev[i] + z_i * h_tilde_i;
        }

        // Output: y_t = softmax(W_o @ h_t + b_o)
        self.compute_output_probs();
        self.probs_valid = true;
        self.has_context = true;
    }

    /// Compute softmax output probabilities from current hidden state.
    #[inline(never)]
    #[allow(clippy::needless_range_loop)]
    fn compute_output_probs(&mut self) {
        // Compute logits: W_o @ h + b_o
        let mut max_logit: f32 = f32::NEG_INFINITY;
        for i in 0..VOCAB_SIZE {
            let w_row = &self.w_o[i * HIDDEN_DIM..(i + 1) * HIDDEN_DIM];
            let mut logit = self.b_o[i];
            for j in 0..HIDDEN_DIM {
                logit += w_row[j] * self.h[j];
            }
            self.byte_probs[i] = logit;
            if logit > max_logit {
                max_logit = logit;
            }
        }

        // Softmax with numerical stability (subtract max).
        let mut sum: f32 = 0.0;
        for p in self.byte_probs.iter_mut() {
            let e = (*p - max_logit).exp();
            *p = e;
            sum += e;
        }

        // Normalize. Add tiny epsilon to avoid division by zero.
        let inv_sum = 1.0 / (sum + 1e-30);
        for p in self.byte_probs.iter_mut() {
            *p *= inv_sum;
            // Clamp to avoid log(0) in training.
            if *p < 1e-8 {
                *p = 1e-8;
            }
        }
    }

    /// Convert byte probabilities to a bit prediction.
    /// Same logic as PPM: sum byte probs where the target bit is 1,
    /// conditioned on the already-decoded bits.
    ///
    /// `bpos`: bit position 0-7 (0 = MSB).
    /// `c0`: partial byte being built (starts at 1, accumulates bits MSB-first).
    ///
    /// Returns: 12-bit probability [1, 4095] of next bit being 1.
    #[inline]
    pub fn predict_bit(&self, bpos: u8, c0: u32) -> u32 {
        if !self.has_context {
            return 2048; // uniform before first byte
        }

        let bit_pos = 7 - bpos;
        let mask = 1u8 << bit_pos;

        let mut sum_one: f64 = 0.0;
        let mut sum_zero: f64 = 0.0;

        if bpos == 0 {
            // No bits decoded yet for this byte — sum over all 256.
            for b in 0..VOCAB_SIZE {
                let p = self.byte_probs[b] as f64;
                if (b as u8) & mask != 0 {
                    sum_one += p;
                } else {
                    sum_zero += p;
                }
            }
        } else {
            // Some bits decoded. Only consider bytes matching the partial prefix.
            // c0 has format: 1 followed by bpos decoded bits (MSB first).
            // Extract the decoded bits.
            let partial = (c0 & ((1u32 << bpos) - 1)) as u8;
            let shift = 8 - bpos;
            let base = (partial as usize) << shift;
            let count = 1usize << shift;

            for i in 0..count {
                let b = base | i;
                let p = self.byte_probs[b] as f64;
                if (b as u8) & mask != 0 {
                    sum_one += p;
                } else {
                    sum_zero += p;
                }
            }
        }

        let total = sum_one + sum_zero;
        if total < 1e-15 {
            return 2048;
        }

        let p = ((sum_one * 4096.0) / total) as u32;
        p.clamp(1, 4095)
    }

    /// Online training: backprop through current step after observing actual byte.
    /// No BPTT -- gradients don't flow through time. The hidden state carries
    /// sequential context from the forward pass.
    #[inline(never)]
    #[allow(clippy::needless_range_loop)]
    pub fn train(&mut self, actual_byte: u8) {
        if !self.has_context {
            return;
        }

        let target = actual_byte as usize;

        // --- Output layer gradient ---
        // d_logits[i] = probs[i] - (1 if i == target else 0)  [softmax + cross-entropy]
        let mut d_logits = [0.0f32; VOCAB_SIZE];
        d_logits.copy_from_slice(&self.byte_probs);
        d_logits[target] -= 1.0;

        // --- Gradient for W_o, b_o + accumulate d_h ---
        let mut d_h = [0.0f32; HIDDEN_DIM];

        for i in 0..VOCAB_SIZE {
            let dl = clip_grad(d_logits[i]);
            if dl.abs() < 1e-7 {
                continue; // Skip near-zero gradients (most of the 256 outputs).
            }
            let w_row = &mut self.w_o[i * HIDDEN_DIM..(i + 1) * HIDDEN_DIM];
            let lr_dl = LEARNING_RATE * dl;
            for j in 0..HIDDEN_DIM {
                d_h[j] += dl * w_row[j];
                w_row[j] -= lr_dl * self.h[j];
            }
            self.b_o[i] -= LEARNING_RATE * dl;
        }

        // --- GRU backward (single step, no BPTT) ---
        let mut d_h_tilde = [0.0f32; HIDDEN_DIM];
        let mut d_z = [0.0f32; HIDDEN_DIM];
        let mut d_pre_z = [0.0f32; HIDDEN_DIM];
        let mut d_pre_h = [0.0f32; HIDDEN_DIM];

        for i in 0..HIDDEN_DIM {
            let dhi = clip_grad(d_h[i]);
            d_h_tilde[i] = dhi * self.last_z[i];
            d_z[i] = dhi * (self.last_h_tilde[i] - self.last_h_prev[i]);
            d_pre_z[i] = clip_grad(d_z[i] * self.last_z[i] * (1.0 - self.last_z[i]));
            d_pre_h[i] =
                clip_grad(d_h_tilde[i] * (1.0 - self.last_h_tilde[i] * self.last_h_tilde[i]));
        }

        // --- Update gate: W_z, U_z, b_z ---
        // --- Candidate: W_h, U_h, b_h ---
        // Fused loop for better cache locality.
        let mut d_rh = [0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            let dpz = d_pre_z[i];
            let dph = d_pre_h[i];
            let r_i = self.last_r[i];

            let w_off = i * EMBED_DIM;
            let u_off = i * HIDDEN_DIM;

            // W_z and W_h updates (input weights)
            let lr_dpz = LEARNING_RATE * dpz;
            let lr_dph = LEARNING_RATE * dph;
            for j in 0..EMBED_DIM {
                let xj = self.last_x[j];
                self.w_z[w_off + j] -= lr_dpz * xj;
                self.w_h[w_off + j] -= lr_dph * xj;
            }

            // U_z and U_h updates (recurrent weights) + accumulate d_rh
            for j in 0..HIDDEN_DIM {
                let hj = self.last_h_prev[j];
                self.u_z[u_off + j] -= lr_dpz * hj;
                d_rh[j] += dph * self.u_h[u_off + j];
                self.u_h[u_off + j] -= lr_dph * r_i * hj;
            }

            self.b_z[i] -= LEARNING_RATE * dpz;
            self.b_h[i] -= LEARNING_RATE * dph;
        }

        // --- Reset gate backward ---
        let mut d_pre_r = [0.0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let dr = clip_grad(d_rh[j] * self.last_h_prev[j]);
            d_pre_r[j] = clip_grad(dr * self.last_r[j] * (1.0 - self.last_r[j]));
        }

        // W_r, U_r, b_r updates
        for i in 0..HIDDEN_DIM {
            let dp = d_pre_r[i];
            let w_off = i * EMBED_DIM;
            let u_off = i * HIDDEN_DIM;
            let lr_dp = LEARNING_RATE * dp;
            for j in 0..EMBED_DIM {
                self.w_r[w_off + j] -= lr_dp * self.last_x[j];
            }
            for j in 0..HIDDEN_DIM {
                self.u_r[u_off + j] -= lr_dp * self.last_h_prev[j];
            }
            self.b_r[i] -= LEARNING_RATE * dp;
        }

        // --- Embedding gradient ---
        let embed_start = target * EMBED_DIM;
        for j in 0..EMBED_DIM {
            let mut d_xj: f32 = 0.0;
            for i in 0..HIDDEN_DIM {
                let off = i * EMBED_DIM + j;
                d_xj += d_pre_z[i] * self.w_z[off];
                d_xj += d_pre_r[i] * self.w_r[off];
                d_xj += d_pre_h[i] * self.w_h[off];
            }
            self.embedding[embed_start + j] -= LEARNING_RATE * clip_grad(d_xj);
        }
    }
}

impl Default for GruModel {
    fn default() -> Self {
        Self::new()
    }
}

// --- Deterministic activation functions ---
// CRITICAL: These must produce IDENTICAL results in encoder and decoder.
// Using the same f32 operations guarantees this.

/// Sigmoid activation: 1 / (1 + exp(-x)).
/// Clamped input to [-15, 15] to avoid overflow.
#[inline]
fn sigmoid(x: f32) -> f32 {
    let x = x.clamp(-15.0, 15.0);
    1.0 / (1.0 + (-x).exp())
}

/// Tanh approximation using the identity tanh(x) = 2*sigmoid(2x) - 1.
/// This ensures tanh and sigmoid use the SAME exp() implementation.
#[inline]
fn tanh_approx(x: f32) -> f32 {
    let x = x.clamp(-7.5, 7.5);
    2.0 * sigmoid(2.0 * x) - 1.0
}

/// Clip gradient to prevent explosion.
#[inline]
fn clip_grad(g: f32) -> f32 {
    g.clamp(-GRAD_CLIP, GRAD_CLIP)
}

/// Fill a weight vector with deterministic pseudo-random Xavier initialization.
fn fill_xavier(weights: &mut [f32], scale: f32, seed: &mut u64) {
    for w in weights.iter_mut() {
        // Simple xorshift64 PRNG — deterministic and fast.
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        // Map to [-1, 1] then scale.
        let r = (*seed as f32 / u64::MAX as f32) * 2.0 - 1.0;
        *w = r * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_basic() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(15.0) > 0.999);
        assert!(sigmoid(-15.0) < 0.001);
    }

    #[test]
    fn tanh_basic() {
        assert!((tanh_approx(0.0)).abs() < 1e-6);
        assert!(tanh_approx(7.0) > 0.99);
        assert!(tanh_approx(-7.0) < -0.99);
    }

    #[test]
    fn deterministic_init() {
        let m1 = GruModel::new();
        let m2 = GruModel::new();
        assert_eq!(m1.embedding, m2.embedding);
        assert_eq!(m1.w_z, m2.w_z);
        assert_eq!(m1.w_o, m2.w_o);
    }

    #[test]
    fn initial_predict_bit_uniform() {
        let model = GruModel::new();
        let p = model.predict_bit(0, 1);
        assert_eq!(p, 2048, "before any forward pass, should return 2048");
    }

    #[test]
    fn forward_produces_valid_probs() {
        let mut model = GruModel::new();
        model.forward(b'A');
        let sum: f64 = model.byte_probs.iter().map(|&p| p as f64).sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "byte_probs should sum to ~1.0, got {sum}"
        );
        for &p in &model.byte_probs {
            assert!(p >= 0.0, "negative probability: {p}");
        }
    }

    #[test]
    fn predict_bit_in_range() {
        let mut model = GruModel::new();
        model.forward(b'A');
        for bpos in 0..8u8 {
            let c0 = if bpos == 0 {
                1u32
            } else {
                let mut p = 1u32;
                for prev in 0..bpos {
                    p = (p << 1) | ((b'B' >> (7 - prev)) & 1) as u32;
                }
                p
            };
            let p = model.predict_bit(bpos, c0);
            assert!(
                (1..=4095).contains(&p),
                "predict_bit out of range at bpos {bpos}: {p}"
            );
        }
    }

    #[test]
    fn train_does_not_crash() {
        let mut model = GruModel::new();
        model.forward(b'A');
        model.train(b'B');
        // Should still produce valid output.
        model.forward(b'B');
        let sum: f64 = model.byte_probs.iter().map(|&p| p as f64).sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "probs after training should sum to ~1.0, got {sum}"
        );
    }

    #[test]
    fn encoder_decoder_identical() {
        // Simulate encoder and decoder processing same bytes.
        let mut enc = GruModel::new();
        let mut dec = GruModel::new();
        let data = b"Hello, World!";

        for &byte in data {
            enc.forward(byte);
            dec.forward(byte);

            // Check predictions match.
            for bpos in 0..8u8 {
                let c0 = if bpos == 0 {
                    1u32
                } else {
                    let mut p = 1u32;
                    for prev in 0..bpos {
                        p = (p << 1) | ((byte >> (7 - prev)) & 1) as u32;
                    }
                    p
                };
                let pe = enc.predict_bit(bpos, c0);
                let pd = dec.predict_bit(bpos, c0);
                assert_eq!(pe, pd, "encoder/decoder diverged at bpos {bpos}");
            }

            // Train both with the same byte.
            enc.train(byte);
            dec.train(byte);
        }

        // Hidden states must match.
        assert_eq!(enc.h, dec.h, "hidden states diverged");
    }

    #[test]
    fn adapts_to_pattern() {
        let mut model = GruModel::new();
        // Simulate the codec flow: for each byte, train on it then forward it.
        // This matches codec.rs: train(byte) then forward(byte) at end of each byte.
        // After forward(byte_N), byte_probs predict byte_{N+1}.
        let pattern: Vec<u8> = b"ab".repeat(500);
        for &byte in &pattern {
            model.train(byte); // learn from this byte
            model.forward(byte); // update hidden state, predict next
        }
        // After forward(b'b'), model should predict the next byte.
        // In the pattern, after 'b' comes 'a'. Let's check after 'a':
        model.train(b'a');
        model.forward(b'a');
        // After seeing 'a', the next byte should be 'b' with high prob.
        let p_b = model.byte_probs[b'b' as usize];
        // 'b' should be the dominant prediction after 'a'.
        assert!(
            p_b > 0.1,
            "after 'a' in ab pattern, P('b')={p_b} should be significant"
        );
    }
}
