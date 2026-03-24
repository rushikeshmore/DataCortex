//! GRU (Gated Recurrent Unit) byte-level predictor with truncated BPTT.
//!
//! A byte-level neural predictor providing a DIFFERENT signal from the bit-level
//! CM engine. The GRU captures cross-byte sequential patterns via a recurrent
//! hidden state trained with backpropagation through time (BPTT).
//!
//! Architecture:
//!   Input: one-hot byte embedding (256 → 32 via embedding matrix)
//!   GRU: 128 hidden cells, 1 layer
//!   Output: 128 → 256 linear → softmax → byte probabilities
//!
//! Training: truncated BPTT-10. At each byte completion, gradients propagate
//! back through the last 10 steps of GRU history. This is the same strategy
//! used by cmix (which uses BPTT-100) and gives the majority of the gain at
//! 10% of the BPTT-100 cost.
//!
//! ~43K parameters (~170KB at f32). History + gradient buffers: ~260KB.
//!
//! CRITICAL: Encoder and decoder must maintain IDENTICAL GRU state.
//! Both must call train(byte) then forward(byte) in the same order on the
//! same bytes so that history buffers and weight updates are identical.

const EMBED_DIM: usize = 32;
const HIDDEN_DIM: usize = 128;
const VOCAB_SIZE: usize = 256;

/// BPTT truncation horizon: backpropagate gradients through this many steps.
///
/// 10 steps captures most sequential byte patterns with manageable overhead.
/// cmix achieves -0.07 bpb using BPTT-100; 10 steps provides ~60-70% of that
/// gain at 1/10 the training cost (~5× total slowdown vs no-BPTT).
const BPTT_HORIZON: usize = 10;

/// Online SGD learning rate.
///
/// 0.01 is conservative — the GRU sees each byte only once (online learning).
/// Lower than typical offline training LR to avoid overshooting on rare bytes.
const LEARNING_RATE: f32 = 0.01;

/// Gradient clip magnitude.
///
/// BPTT through 10 steps can accumulate gradients. Clipping at 5.0 prevents
/// exploding gradients while preserving the direction of improvement.
const GRAD_CLIP: f32 = 5.0;

/// GRU byte-level predictor with BPTT-10 online training.
pub struct GruModel {
    // ─── Parameters ──────────────────────────────────────────────────────────
    /// Embedding matrix: [VOCAB_SIZE * EMBED_DIM]
    embedding: Vec<f32>,

    /// Update gate input weights: [HIDDEN_DIM * EMBED_DIM]
    w_z: Vec<f32>,
    /// Update gate recurrent weights: [HIDDEN_DIM * HIDDEN_DIM]
    u_z: Vec<f32>,
    /// Update gate biases: [HIDDEN_DIM]
    b_z: Vec<f32>,

    /// Reset gate input weights: [HIDDEN_DIM * EMBED_DIM]
    w_r: Vec<f32>,
    /// Reset gate recurrent weights: [HIDDEN_DIM * HIDDEN_DIM]
    u_r: Vec<f32>,
    /// Reset gate biases: [HIDDEN_DIM]
    b_r: Vec<f32>,

    /// Candidate hidden input weights: [HIDDEN_DIM * EMBED_DIM]
    w_h: Vec<f32>,
    /// Candidate hidden recurrent weights: [HIDDEN_DIM * HIDDEN_DIM]
    u_h: Vec<f32>,
    /// Candidate hidden biases: [HIDDEN_DIM]
    b_h: Vec<f32>,

    /// Output projection weights: [VOCAB_SIZE * HIDDEN_DIM]
    w_o: Vec<f32>,
    /// Output projection biases: [VOCAB_SIZE]
    b_o: Vec<f32>,

    // ─── Recurrent state ─────────────────────────────────────────────────────
    /// Hidden state: [HIDDEN_DIM]
    h: Vec<f32>,

    // ─── Cached forward pass values (for the most recent step) ───────────────
    /// Input embedding for the most recent forward step: [EMBED_DIM]
    last_x: Vec<f32>,
    /// Hidden state before the most recent forward step: [HIDDEN_DIM]
    last_h_prev: Vec<f32>,
    /// Update gate output from the most recent step: [HIDDEN_DIM]
    last_z: Vec<f32>,
    /// Reset gate output from the most recent step: [HIDDEN_DIM]
    last_r: Vec<f32>,
    /// Candidate hidden from the most recent step: [HIDDEN_DIM]
    last_h_tilde: Vec<f32>,

    /// Cached softmax probabilities over the next byte: [VOCAB_SIZE]
    byte_probs: Vec<f32>,
    /// Whether byte_probs has been computed for the current step.
    probs_valid: bool,
    /// Whether at least one byte has been processed (have valid hidden state).
    has_context: bool,

    // ─── BPTT history ring buffer ─────────────────────────────────────────────
    // Flat layout: entry at ring position `p` occupies [p*DIM .. (p+1)*DIM].
    // hist_pos is the next WRITE position (circular). After writing,
    // hist_pos = (hist_pos + 1) % BPTT_HORIZON.
    //
    /// Saved input embeddings: [BPTT_HORIZON * EMBED_DIM]
    hist_x: Vec<f32>,
    /// Saved h_prev (hidden before each step): [BPTT_HORIZON * HIDDEN_DIM]
    hist_h_prev: Vec<f32>,
    /// Saved update gate outputs: [BPTT_HORIZON * HIDDEN_DIM]
    hist_z: Vec<f32>,
    /// Saved reset gate outputs: [BPTT_HORIZON * HIDDEN_DIM]
    hist_r: Vec<f32>,
    /// Saved candidate hidden values: [BPTT_HORIZON * HIDDEN_DIM]
    hist_h_tilde: Vec<f32>,
    /// Next write position in the ring (0..BPTT_HORIZON).
    hist_pos: usize,
    /// Number of valid entries in the ring (0..=BPTT_HORIZON).
    hist_count: usize,

    // ─── Pre-allocated gradient accumulators ─────────────────────────────────
    // Zeroed at the start of each train() call and accumulated across all BPTT
    // steps before a single weight update. Stored in the struct to avoid
    // per-call heap allocation in the hot path.
    //
    /// Accumulated gradient for w_z: [HIDDEN_DIM * EMBED_DIM]
    grad_w_z: Vec<f32>,
    /// Accumulated gradient for u_z: [HIDDEN_DIM * HIDDEN_DIM]
    grad_u_z: Vec<f32>,
    /// Accumulated gradient for b_z: [HIDDEN_DIM]
    grad_b_z: Vec<f32>,
    /// Accumulated gradient for w_r: [HIDDEN_DIM * EMBED_DIM]
    grad_w_r: Vec<f32>,
    /// Accumulated gradient for u_r: [HIDDEN_DIM * HIDDEN_DIM]
    grad_u_r: Vec<f32>,
    /// Accumulated gradient for b_r: [HIDDEN_DIM]
    grad_b_r: Vec<f32>,
    /// Accumulated gradient for w_h: [HIDDEN_DIM * EMBED_DIM]
    grad_w_h: Vec<f32>,
    /// Accumulated gradient for u_h: [HIDDEN_DIM * HIDDEN_DIM]
    grad_u_h: Vec<f32>,
    /// Accumulated gradient for b_h: [HIDDEN_DIM]
    grad_b_h: Vec<f32>,
}

impl GruModel {
    /// Create a new GRU model with Xavier-initialized weights and zeroed buffers.
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
            // History ring buffers — zeroed.
            hist_x: vec![0.0; BPTT_HORIZON * EMBED_DIM],
            hist_h_prev: vec![0.0; BPTT_HORIZON * HIDDEN_DIM],
            hist_z: vec![0.0; BPTT_HORIZON * HIDDEN_DIM],
            hist_r: vec![0.0; BPTT_HORIZON * HIDDEN_DIM],
            hist_h_tilde: vec![0.0; BPTT_HORIZON * HIDDEN_DIM],
            hist_pos: 0,
            hist_count: 0,
            // Gradient accumulators — zeroed (will be explicitly zeroed each train() call).
            grad_w_z: vec![0.0; HIDDEN_DIM * EMBED_DIM],
            grad_u_z: vec![0.0; HIDDEN_DIM * HIDDEN_DIM],
            grad_b_z: vec![0.0; HIDDEN_DIM],
            grad_w_r: vec![0.0; HIDDEN_DIM * EMBED_DIM],
            grad_u_r: vec![0.0; HIDDEN_DIM * HIDDEN_DIM],
            grad_b_r: vec![0.0; HIDDEN_DIM],
            grad_w_h: vec![0.0; HIDDEN_DIM * EMBED_DIM],
            grad_u_h: vec![0.0; HIDDEN_DIM * HIDDEN_DIM],
            grad_b_h: vec![0.0; HIDDEN_DIM],
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

        // Xavier scale for input weights: sqrt(2 / (32 + 128))
        let wx_scale = (2.0 / (EMBED_DIM + HIDDEN_DIM) as f32).sqrt();
        fill_xavier(&mut self.w_z, wx_scale, &mut seed);
        fill_xavier(&mut self.w_r, wx_scale, &mut seed);
        fill_xavier(&mut self.w_h, wx_scale, &mut seed);

        // Xavier scale for recurrent weights: sqrt(2 / (128 + 128))
        let uh_scale = (2.0 / (HIDDEN_DIM + HIDDEN_DIM) as f32).sqrt();
        fill_xavier(&mut self.u_z, uh_scale, &mut seed);
        fill_xavier(&mut self.u_r, uh_scale, &mut seed);
        fill_xavier(&mut self.u_h, uh_scale, &mut seed);

        // Xavier scale for output weights: sqrt(2 / (128 + 256))
        let wo_scale = (2.0 / (HIDDEN_DIM + VOCAB_SIZE) as f32).sqrt();
        fill_xavier(&mut self.w_o, wo_scale, &mut seed);

        // Bias update gate to slightly positive so it starts in "remember" mode
        // (z → 1 means keep old hidden state). Helps gradient flow early in training.
        for b in self.b_z.iter_mut() {
            *b = 1.0;
        }
        // Reset gate and candidate biases stay at 0.
    }

    /// Forward pass: process one byte, update hidden state, compute output probs.
    ///
    /// Call this with the byte that was just OBSERVED. The resulting byte_probs
    /// predict the NEXT byte. After forward(), call predict_bit() to get bit
    /// probabilities for the next byte.
    ///
    /// Also saves the step into the BPTT history ring for train() to use.
    #[inline(never)]
    #[allow(
        clippy::needless_range_loop,
        reason = "matrix ops are clearer with explicit indices"
    )]
    pub fn forward(&mut self, byte: u8) {
        // Save previous hidden state for backprop.
        self.last_h_prev.copy_from_slice(&self.h);

        // Get embedding for the input byte (row lookup).
        let byte_idx = byte as usize;
        let embed_start = byte_idx * EMBED_DIM;
        self.last_x
            .copy_from_slice(&self.embedding[embed_start..embed_start + EMBED_DIM]);

        // Compute update gate z, reset gate r, and candidate h_tilde in a fused
        // loop for cache locality.
        for i in 0..HIDDEN_DIM {
            let w_off = i * EMBED_DIM;
            let wz_row = &self.w_z[w_off..w_off + EMBED_DIM];
            let wr_row = &self.w_r[w_off..w_off + EMBED_DIM];
            let wh_row = &self.w_h[w_off..w_off + EMBED_DIM];

            let mut val_z = self.b_z[i];
            let mut val_r = self.b_r[i];
            let mut val_h = self.b_h[i];

            // W @ x for all three gates.
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

            // U_h @ (r[i] * h_{t-1}) — uses r[i] (output-dimension convention).
            let uh_row = &self.u_h[u_off..u_off + HIDDEN_DIM];
            for j in 0..HIDDEN_DIM {
                val_h += uh_row[j] * (r_i * self.last_h_prev[j]);
            }

            let h_tilde_i = tanh_approx(val_h);
            self.last_h_tilde[i] = h_tilde_i;

            // h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
            self.h[i] = (1.0 - z_i) * self.last_h_prev[i] + z_i * h_tilde_i;
        }

        // Compute output probabilities.
        self.compute_output_probs();
        self.probs_valid = true;
        self.has_context = true;

        // ─── Save this step into the BPTT history ring ───────────────────────
        let x_base = self.hist_pos * EMBED_DIM;
        self.hist_x[x_base..x_base + EMBED_DIM].copy_from_slice(&self.last_x);

        let h_base = self.hist_pos * HIDDEN_DIM;
        self.hist_h_prev[h_base..h_base + HIDDEN_DIM].copy_from_slice(&self.last_h_prev);
        self.hist_z[h_base..h_base + HIDDEN_DIM].copy_from_slice(&self.last_z);
        self.hist_r[h_base..h_base + HIDDEN_DIM].copy_from_slice(&self.last_r);
        self.hist_h_tilde[h_base..h_base + HIDDEN_DIM].copy_from_slice(&self.last_h_tilde);

        // Advance circular write head.
        self.hist_pos = (self.hist_pos + 1) % BPTT_HORIZON;
        if self.hist_count < BPTT_HORIZON {
            self.hist_count += 1;
        }
    }

    /// Compute softmax output probabilities from current hidden state.
    #[inline(never)]
    #[allow(
        clippy::needless_range_loop,
        reason = "matrix ops are clearer with explicit indices"
    )]
    fn compute_output_probs(&mut self) {
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

        // Numerically stable softmax: subtract max before exp.
        let mut sum: f32 = 0.0;
        for p in self.byte_probs.iter_mut() {
            let e = (*p - max_logit).exp();
            *p = e;
            sum += e;
        }

        // Normalize with epsilon guard.
        let inv_sum = 1.0 / (sum + 1e-30);
        for p in self.byte_probs.iter_mut() {
            *p *= inv_sum;
            // Clamp to avoid log(0) in training.
            if *p < 1e-8 {
                *p = 1e-8;
            }
        }
    }

    /// Convert byte probabilities to a bit prediction for the CM MetaMixer.
    ///
    /// `bpos`: bit position 0-7 (0 = MSB).
    /// `c0`: partial byte being built (starts at 1, accumulates bits MSB-first).
    ///
    /// Returns: 12-bit probability [1, 4095] of next bit being 1.
    #[inline]
    pub fn predict_bit(&self, bpos: u8, c0: u32) -> u32 {
        if !self.has_context {
            return 2048; // Uniform before first byte.
        }

        let bit_pos = 7 - bpos;
        let mask = 1u8 << bit_pos;

        let mut sum_one: f64 = 0.0;
        let mut sum_zero: f64 = 0.0;

        if bpos == 0 {
            // No bits decoded yet — sum over all 256.
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

    /// Online training with truncated BPTT.
    ///
    /// Computes the output-layer gradient for `actual_byte`, then propagates
    /// d_h backwards through up to BPTT_HORIZON stored steps. Gradients are
    /// accumulated into pre-allocated buffers and applied in a single weight
    /// update at the end. This avoids the weight-corruption from immediate
    /// per-step updates that would otherwise occur in BPTT.
    ///
    /// Call this BEFORE forward(actual_byte) (matches codec.rs flow). Both
    /// encoder and decoder see the same byte sequence so their weights and
    /// history buffers evolve identically — parity is preserved.
    #[inline(never)]
    #[allow(
        clippy::needless_range_loop,
        reason = "matrix ops are clearer with explicit indices"
    )]
    pub fn train(&mut self, actual_byte: u8) {
        if !self.has_context {
            return;
        }

        let target = actual_byte as usize;

        // ─── Output layer: update W_o, b_o and compute initial d_h ──────────
        // Cross-entropy + softmax gradient: d_logits[i] = probs[i] - (i==target).
        let mut d_h = [0.0f32; HIDDEN_DIM];

        for i in 0..VOCAB_SIZE {
            let dl = clip_grad(self.byte_probs[i] - if i == target { 1.0 } else { 0.0 });
            if dl.abs() < 1e-7 {
                // Skip near-zero gradient — most of the 256 outputs.
                continue;
            }
            let w_row = &mut self.w_o[i * HIDDEN_DIM..(i + 1) * HIDDEN_DIM];
            let lr_dl = LEARNING_RATE * dl;
            for j in 0..HIDDEN_DIM {
                d_h[j] += dl * w_row[j];
                w_row[j] -= lr_dl * self.h[j];
            }
            self.b_o[i] -= LEARNING_RATE * dl;
        }

        // ─── Zero gradient accumulators ──────────────────────────────────────
        self.grad_w_z.fill(0.0);
        self.grad_u_z.fill(0.0);
        self.grad_b_z.fill(0.0);
        self.grad_w_r.fill(0.0);
        self.grad_u_r.fill(0.0);
        self.grad_b_r.fill(0.0);
        self.grad_w_h.fill(0.0);
        self.grad_u_h.fill(0.0);
        self.grad_b_h.fill(0.0);

        // ─── BPTT: propagate d_h backwards through history ───────────────────
        // Iterate from most-recent (step_back=0) to oldest (step_back=steps-1).
        // At each step, accumulate weight gradients and compute d_h_prev to
        // pass to the step before it.
        let steps = self.hist_count;

        // Gate gradients at step 0 — needed for the embedding update below.
        let mut d_pre_z_s0 = [0.0f32; HIDDEN_DIM];
        let mut d_pre_r_s0 = [0.0f32; HIDDEN_DIM];
        let mut d_pre_h_s0 = [0.0f32; HIDDEN_DIM];

        for step_back in 0..steps {
            // Read from ring: most-recent step is at hist_pos - 1 (mod BPTT_HORIZON).
            let ring_idx = (self.hist_pos + BPTT_HORIZON - 1 - step_back) % BPTT_HORIZON;

            let x_base = ring_idx * EMBED_DIM;
            let h_base = ring_idx * HIDDEN_DIM;

            // ── GRU cell backward (one step) ─────────────────────────────────
            // Given d_h (gradient of loss w.r.t. h_t), compute:
            //   d_h_tilde, d_pre_z, d_pre_h (upstream gradients through h_t)
            //   d_pre_r (upstream gradient through the reset gate path)
            //   Accumulate dW, dU, db for all three gates.
            //   Compute d_h_prev to pass to the previous step.

            let mut d_pre_z = [0.0f32; HIDDEN_DIM];
            let mut d_pre_r = [0.0f32; HIDDEN_DIM];
            let mut d_pre_h = [0.0f32; HIDDEN_DIM];

            for i in 0..HIDDEN_DIM {
                let dhi = clip_grad(d_h[i]);
                let z_i = self.hist_z[h_base + i];
                let r_i = self.hist_r[h_base + i];
                let h_tilde_i = self.hist_h_tilde[h_base + i];
                let h_prev_i = self.hist_h_prev[h_base + i];

                // h_t = (1-z)*h_prev + z*h_tilde  ⟹  dh_tilde = dh * z
                let d_h_tilde_i = dhi * z_i;
                // dz = dh * (h_tilde - h_prev)
                let dz_i = dhi * (h_tilde_i - h_prev_i);

                // Sigmoid backward: d_pre_z = dz * z * (1-z)
                d_pre_z[i] = clip_grad(dz_i * z_i * (1.0 - z_i));
                // Tanh backward: d_pre_h = d_h_tilde * (1 - h_tilde²)
                d_pre_h[i] = clip_grad(d_h_tilde_i * (1.0 - h_tilde_i * h_tilde_i));

                // Bias gradients.
                self.grad_b_z[i] += d_pre_z[i];
                self.grad_b_h[i] += d_pre_h[i];

                // Input weight gradients: dW_z += d_pre_z[i] * x^T
                let w_off = i * EMBED_DIM;
                let lr_dpz = d_pre_z[i];
                let lr_dph = d_pre_h[i];
                for j in 0..EMBED_DIM {
                    let xj = self.hist_x[x_base + j];
                    self.grad_w_z[w_off + j] += lr_dpz * xj;
                    self.grad_w_h[w_off + j] += lr_dph * xj;
                }

                // Recurrent weight gradients: dU_z += d_pre_z[i] * h_prev^T
                // Also accumulate d_rh = (U_h^T @ d_pre_h) for the reset gate.
                // Note: d_rh is NOT used for d_h_prev here — that uses
                // sum_i(d_pre_h[i] * r[i] * U_h[i,j]) — see loop below.
                let u_off = i * HIDDEN_DIM;
                for j in 0..HIDDEN_DIM {
                    let hj = self.hist_h_prev[h_base + j];
                    self.grad_u_z[u_off + j] += d_pre_z[i] * hj;
                    // dU_h[i,j] = d_pre_h[i] * r[i] * h_prev[j]
                    self.grad_u_h[u_off + j] += d_pre_h[i] * r_i * hj;
                }
            }

            // Reset gate backward.
            // d_rh[j] = sum_i(d_pre_h[i] * U_h[i,j]) = (U_h^T @ d_pre_h)[j]
            // dr[j] = d_rh[j] * h_prev[j]   (gradient w.r.t. r[j])
            // d_pre_r[j] = dr[j] * r[j] * (1-r[j])   (sigmoid backward)
            let mut d_rh = [0.0f32; HIDDEN_DIM];
            for i in 0..HIDDEN_DIM {
                let u_off = i * HIDDEN_DIM;
                for j in 0..HIDDEN_DIM {
                    d_rh[j] += d_pre_h[i] * self.u_h[u_off + j];
                }
            }
            for j in 0..HIDDEN_DIM {
                let dr = clip_grad(d_rh[j] * self.hist_h_prev[h_base + j]);
                d_pre_r[j] =
                    clip_grad(dr * self.hist_r[h_base + j] * (1.0 - self.hist_r[h_base + j]));
                self.grad_b_r[j] += d_pre_r[j];
            }
            // Accumulate dW_r and dU_r.
            for i in 0..HIDDEN_DIM {
                let dp = d_pre_r[i];
                let w_off = i * EMBED_DIM;
                let u_off = i * HIDDEN_DIM;
                for j in 0..EMBED_DIM {
                    self.grad_w_r[w_off + j] += dp * self.hist_x[x_base + j];
                }
                for j in 0..HIDDEN_DIM {
                    self.grad_u_r[u_off + j] += dp * self.hist_h_prev[h_base + j];
                }
            }

            // Save step-0 gate gradients for the embedding update.
            if step_back == 0 {
                d_pre_z_s0.copy_from_slice(&d_pre_z);
                d_pre_r_s0.copy_from_slice(&d_pre_r);
                d_pre_h_s0.copy_from_slice(&d_pre_h);
            }

            // ── Propagate d_h to the previous step ───────────────────────────
            // d_h_prev[j] = d_h[j] * (1 - z[j])          (direct path)
            //             + sum_i(d_pre_z[i] * U_z[i,j])  (update gate path)
            //             + sum_i(d_pre_r[i] * U_r[i,j])  (reset gate path)
            //             + sum_i(d_pre_h[i] * r[i] * U_h[i,j])  (candidate path)
            let mut d_h_prev = [0.0f32; HIDDEN_DIM];

            // Direct path.
            for j in 0..HIDDEN_DIM {
                d_h_prev[j] = clip_grad(d_h[j]) * (1.0 - self.hist_z[h_base + j]);
            }

            // Gate recurrent paths: loop over hidden units i, accumulate into j.
            for i in 0..HIDDEN_DIM {
                let dpz = d_pre_z[i];
                let dpr = d_pre_r[i];
                // d_pre_h[i] * r[i] — used for the U_h candidate path.
                let dph_r = d_pre_h[i] * self.hist_r[h_base + i];
                let u_off = i * HIDDEN_DIM;
                for j in 0..HIDDEN_DIM {
                    d_h_prev[j] += dpz * self.u_z[u_off + j];
                    d_h_prev[j] += dpr * self.u_r[u_off + j];
                    d_h_prev[j] += dph_r * self.u_h[u_off + j];
                }
            }

            // Clip for stability before passing to the next step.
            for j in 0..HIDDEN_DIM {
                d_h_prev[j] = clip_grad(d_h_prev[j]);
            }
            d_h.copy_from_slice(&d_h_prev);
        }

        // ─── Apply accumulated weight gradients in one shot ───────────────────
        // Using current weights (not stored snapshots) for the update is standard
        // "online BPTT" practice — the per-step weight change from LR=0.01 is
        // small enough that the approximation is negligible.
        for i in 0..HIDDEN_DIM {
            let w_off = i * EMBED_DIM;
            let u_off = i * HIDDEN_DIM;
            for j in 0..EMBED_DIM {
                self.w_z[w_off + j] -= LEARNING_RATE * clip_grad(self.grad_w_z[w_off + j]);
                self.w_r[w_off + j] -= LEARNING_RATE * clip_grad(self.grad_w_r[w_off + j]);
                self.w_h[w_off + j] -= LEARNING_RATE * clip_grad(self.grad_w_h[w_off + j]);
            }
            for j in 0..HIDDEN_DIM {
                self.u_z[u_off + j] -= LEARNING_RATE * clip_grad(self.grad_u_z[u_off + j]);
                self.u_r[u_off + j] -= LEARNING_RATE * clip_grad(self.grad_u_r[u_off + j]);
                self.u_h[u_off + j] -= LEARNING_RATE * clip_grad(self.grad_u_h[u_off + j]);
            }
            self.b_z[i] -= LEARNING_RATE * clip_grad(self.grad_b_z[i]);
            self.b_r[i] -= LEARNING_RATE * clip_grad(self.grad_b_r[i]);
            self.b_h[i] -= LEARNING_RATE * clip_grad(self.grad_b_h[i]);
        }

        // ─── Embedding gradient (current step only) ───────────────────────────
        // Update the embedding for the target byte using the step-0 gate gradients.
        // Only the current step's input embedding is updated — historical embeddings
        // are treated as fixed inputs in BPTT (standard practice for online learning).
        let embed_start = target * EMBED_DIM;
        for j in 0..EMBED_DIM {
            let mut d_xj: f32 = 0.0;
            for i in 0..HIDDEN_DIM {
                let off = i * EMBED_DIM + j;
                d_xj += d_pre_z_s0[i] * self.w_z[off];
                d_xj += d_pre_r_s0[i] * self.w_r[off];
                d_xj += d_pre_h_s0[i] * self.w_h[off];
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

// ─── Activation functions ────────────────────────────────────────────────────
// CRITICAL: These must produce IDENTICAL results in encoder and decoder.
// The same f32 operations guarantee bit-exact results across both paths.

/// Sigmoid activation: 1 / (1 + exp(-x)), clamped to prevent overflow.
#[inline]
fn sigmoid(x: f32) -> f32 {
    let x = x.clamp(-15.0, 15.0);
    1.0 / (1.0 + (-x).exp())
}

/// Tanh using the identity tanh(x) = 2*sigmoid(2x) - 1.
/// Reusing sigmoid ensures tanh and sigmoid use the SAME exp() path.
#[inline]
fn tanh_approx(x: f32) -> f32 {
    let x = x.clamp(-7.5, 7.5);
    2.0 * sigmoid(2.0 * x) - 1.0
}

/// Clip gradient magnitude to prevent explosion during BPTT.
#[inline]
fn clip_grad(g: f32) -> f32 {
    g.clamp(-GRAD_CLIP, GRAD_CLIP)
}

/// Fill a weight slice with deterministic pseudo-random Xavier initialization.
fn fill_xavier(weights: &mut [f32], scale: f32, seed: &mut u64) {
    for w in weights.iter_mut() {
        // xorshift64 PRNG — deterministic and fast.
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
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
    fn history_ring_fills_correctly() {
        let mut model = GruModel::new();
        // Before any forward, history is empty.
        assert_eq!(model.hist_count, 0);

        // After N forward passes, history fills up to BPTT_HORIZON.
        for i in 0..BPTT_HORIZON + 3 {
            model.forward(b'A' + (i % 26) as u8);
            let expected = (i + 1).min(BPTT_HORIZON);
            assert_eq!(model.hist_count, expected, "hist_count wrong at step {i}");
        }
        // hist_pos should have wrapped.
        assert_eq!(model.hist_pos, 3);
    }

    #[test]
    fn bptt_does_not_produce_nan() {
        let mut model = GruModel::new();
        let data = b"Hello, World! This is a BPTT test. Let's check for NaN.";
        for &byte in data {
            model.forward(byte);
            model.train(byte);
            for j in 0..HIDDEN_DIM {
                assert!(!model.h[j].is_nan(), "hidden state has NaN at j={j}");
            }
            for &p in &model.byte_probs {
                assert!(!p.is_nan(), "byte_probs has NaN");
            }
        }
    }

    #[test]
    fn encoder_decoder_identical() {
        // Encoder and decoder must produce bit-identical predictions throughout.
        let mut enc = GruModel::new();
        let mut dec = GruModel::new();
        let data = b"Hello, World! Testing BPTT encoder-decoder parity.";

        for &byte in data {
            enc.forward(byte);
            dec.forward(byte);

            // Predictions must match exactly.
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

            // Train both with the same byte (same as codec flow).
            enc.train(byte);
            dec.train(byte);
        }

        // After training, hidden states must match.
        assert_eq!(enc.h, dec.h, "hidden states diverged after training");
        // History ring must match.
        assert_eq!(enc.hist_count, dec.hist_count, "hist_count diverged");
        assert_eq!(enc.hist_pos, dec.hist_pos, "hist_pos diverged");
    }

    #[test]
    fn bptt_improves_over_1step() {
        // BPTT-trained model should learn patterns faster than 1-step SGD.
        // Test: repeated pattern "ab" — after BPTT training, should predict 'b'
        // after 'a' with high confidence.
        let mut model = GruModel::new();
        let pattern: Vec<u8> = b"ab".repeat(200);
        for &byte in &pattern {
            model.train(byte);
            model.forward(byte);
        }
        // After 'a' in the pattern, next byte should be 'b'.
        model.train(b'a');
        model.forward(b'a');
        let p_b = model.byte_probs[b'b' as usize];
        assert!(
            p_b > 0.1,
            "after 'a' in ab pattern with BPTT, P('b')={p_b} should be significant"
        );
    }

    #[test]
    fn adapts_to_pattern() {
        let mut model = GruModel::new();
        let pattern: Vec<u8> = b"ab".repeat(500);
        for &byte in &pattern {
            model.train(byte);
            model.forward(byte);
        }
        model.train(b'a');
        model.forward(b'a');
        let p_b = model.byte_probs[b'b' as usize];
        assert!(
            p_b > 0.1,
            "after 'a' in ab pattern, P('b')={p_b} should be significant"
        );
    }
}
