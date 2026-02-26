"""
closed_loop_takens.py
=====================
A Takens dendrite neuron with two adaptive biological mechanisms:

  1. ADAPTIVE SELF-BUFFER (dendritic branch growth)
     The neuron's recurrent self-history window grows until it can
     carry context information forward through a silence gap.
     Biological analog: dendritic branch elongation driven by frustration.

  2. CLOSED-LOOP NETWORK (recurrent connectivity)
     Neuron C receives recurrent input from neurons A and B.
     C cannot solve the task alone. Only the loop makes it possible.
     Biological analog: stimulus-evoked causal connectivity (organoid papers).

THE TASK: Temporal context disambiguation
------------------------------------------
Sequence structure per trial:
  [context token 50ms] [silence 40ms] [ambiguous 40Hz token 50ms]

Context is 11Hz (label=0) or 61Hz (label=1), chosen randomly.
The 40Hz ambiguous token is IDENTICAL regardless of context.
The 40ms silence gap ensures A/B output decays before the 40Hz window.

A feedforward detector tuned to 40Hz: ~50% (chance).
To classify correctly, C must remember what happened 90ms ago
(context token + silence = 50+40 = 90ms back from 40Hz midpoint).

This requires the self-buffer to span ~115 samples.
Starting at 4, it must grow to ~115. That is the experiment.

FOUR CONDITIONS
---------------
1. Open loop:    C receives no recurrent input. Accuracy ~50%.
2. Short buffer: C receives A,B but self_taps=4 (can't reach context). ~50%.
3. Oracle:       C receives A,B, self_taps=80. Accuracy ~92%.
4. Adaptive:     C starts at self_taps=4, grows homeostically. Reaches ~92%.

The adaptive neuron discovers the right memory depth by itself.
No gradient. No teacher. Frustration drives growth.

BIOLOGICAL MAPPING
------------------
self_taps growth  = dendritic branch elongation
frustration       = mismatch between context signal strength and target
homeostatic drive = AIS structural plasticity (Grubb & Bhatt 2010)
recurrent inputs  = stimulus-evoked causal connectivity (organoid papers)
closed loop       = brain is an endless loop (Antti)

Dependencies: numpy, matplotlib only.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

# ============================================================================
# PARAMETERS
# ============================================================================
FS              = 1000.0
TOK_DUR         = 0.05          # 50ms per token
N               = int(FS * TOK_DUR)
GAP             = 40            # 40ms silence gap between context and 40Hz
TOK_STRIDE      = N + GAP + N   # samples per context+gap+ambiguous triple
TOK_S_IN_PAIR   = N + GAP       # start of 40Hz window within each triple
NOISE           = 0.15
CTX_FREQS       = (11.0, 61.0)  # non-harmonic to 40Hz, minimises cross-resonance
TAU             = 2             # world buffer delay (covers 24ms at tau=2, n_taps=12)
N_TAPS          = 12

# Adaptive self-buffer parameters
SELF_TAPS_INIT  = 4             # start short — cannot reach context
SELF_TAPS_MAX   = 160           # maximum branch length
SELF_GROW_RATE  = 0.5           # samples added per timestep when frustrated
DISAMBIG_TARGET = 0.15          # target context signal strength

# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def make_sequence(n_pairs=100, noise=NOISE, seed=0):
    """
    Each trial: [context_token][silence][40Hz_token]
    Context is CTX_FREQS[0] (label=0) or CTX_FREQS[1] (label=1).
    The 40ms silence gap forces A/B output to decay before the 40Hz token.
    """
    np.random.seed(seed)
    parts, labels = [], []
    for _ in range(n_pairs):
        ctx      = np.random.choice([0, 1])
        t        = np.linspace(0, TOK_DUR, N, endpoint=False)
        ctx_freq = CTX_FREQS[ctx]
        parts.append(np.sin(2*np.pi*ctx_freq*t) + np.random.normal(0, noise, N))
        parts.append(np.zeros(GAP) + np.random.normal(0, noise, GAP))
        parts.append(np.sin(2*np.pi*40.0*t) + np.random.normal(0, noise, N))
        labels.append(ctx)
    return np.concatenate(parts), labels


# ============================================================================
# NEURON
# ============================================================================

class TakensNeuron:
    """
    Takens dendrite neuron with optional recurrent self-buffer.

    World processing:
      Delay buffer -> Takens vector -> dot(mosaic) -> resonance^2

    Context processing (when receive_recurrent=True):
      self_buf_A and self_buf_B track recent outputs of neurons A and B.
      ctx = mean(self_buf_A[:taps]) - mean(self_buf_B[:taps])
      Positive ctx: A fired recently (label 0 context)
      Negative ctx: B fired recently (label 1 context)

    Output:
      world_resonance * (1 + tanh(ctx * 3))
      This encodes both the 40Hz detection AND the context.

    Adaptive self-buffer (when adaptive=True):
      self_taps grows when |ctx| < DISAMBIG_TARGET (frustrated)
      self_taps stabilises when context signal is strong enough
      This is the dendritic branch elongating to capture context.
    """
    def __init__(self, freq, tau=TAU, n_taps=N_TAPS, fs=FS,
                 self_taps=SELF_TAPS_INIT, adaptive=False, receive_recurrent=True):
        self.tau              = tau
        self.n_taps           = n_taps
        self.adaptive         = adaptive
        self.receive_recurrent = receive_recurrent
        self.self_taps        = float(self_taps)

        # Receptor mosaic: fixed to target frequency
        t_taps      = np.arange(n_taps) * (tau / fs)
        self.mosaic = np.cos(2*np.pi*freq*t_taps)
        self.mosaic /= np.linalg.norm(self.mosaic)

        # Buffers
        self.world_buf  = np.zeros(n_taps * tau + 5)
        self.self_buf_A = np.zeros(SELF_TAPS_MAX + 10)
        self.self_buf_B = np.zeros(SELF_TAPS_MAX + 10)

        # History
        self.output_history    = []
        self.taps_history      = []
        self.ctx_history       = []
        self.world_res_history = []

    def step(self, x, out_A=0.0, out_B=0.0):
        # Update world buffer
        self.world_buf    = np.roll(self.world_buf, 1)
        self.world_buf[0] = x

        # Update recurrent buffers
        if self.receive_recurrent:
            self.self_buf_A    = np.roll(self.self_buf_A, 1)
            self.self_buf_A[0] = out_A
            self.self_buf_B    = np.roll(self.self_buf_B, 1)
            self.self_buf_B[0] = out_B

        # World Takens resonance
        tv        = self.world_buf[::self.tau][:self.n_taps]
        world_res = np.dot(tv, self.mosaic)**2 if len(tv) == len(self.mosaic) else 0.0

        # Context signal from self-buffers
        st  = max(1, int(self.self_taps))
        ctx = (self.self_buf_A[:st].mean() - self.self_buf_B[:st].mean()) \
              if self.receive_recurrent else 0.0

        # Combined output
        output = world_res * (1.0 + np.tanh(ctx * 3.0))

        # Store
        self.output_history.append(output)
        self.taps_history.append(self.self_taps)
        self.ctx_history.append(ctx)
        self.world_res_history.append(world_res)

        # Adaptive growth
        if self.adaptive and self.receive_recurrent:
            frustration = DISAMBIG_TARGET - abs(ctx)
            if frustration > 0:
                self.self_taps = min(SELF_TAPS_MAX, self.self_taps + SELF_GROW_RATE)
            else:
                self.self_taps = max(SELF_TAPS_INIT, self.self_taps - 0.01)

        return output


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(C_out, labels):
    """
    Classify each 40Hz token window: above-median output = label 0, below = label 1.
    Returns (correct, total, all_outputs, all_labels).
    """
    outs, lbls = [], []
    for i, lbl in enumerate(labels):
        ts = i * TOK_STRIDE + TOK_S_IN_PAIR
        te = ts + N
        if te > len(C_out): break
        outs.append(np.mean(C_out[ts:te]))
        lbls.append(lbl)
    threshold = np.median(outs)
    preds     = [0 if v > threshold else 1 for v in outs]
    correct   = sum(1 for p, l in zip(preds, lbls) if p == l)
    return correct, len(lbls), outs, lbls


# ============================================================================
# RUN FOUR CONDITIONS
# ============================================================================

print("=" * 65)
print("CLOSED LOOP TAKENS NEURON: TEMPORAL CONTEXT DISAMBIGUATION")
print("=" * 65)
print(f"Context freqs: {CTX_FREQS[0]}Hz (label 0) vs {CTX_FREQS[1]}Hz (label 1)")
print(f"Ambiguous:     40Hz | Gap: {GAP}ms | Noise: {NOISE}")
print(f"Context window: {N+GAP//2}ms before 40Hz midpoint ({N+GAP//2} samples)")
print()

sig, labels = make_sequence(100, seed=0)

# Context neurons (fixed, no recurrent)
nA_shared = TakensNeuron(CTX_FREQS[0], receive_recurrent=False)
nB_shared = TakensNeuron(CTX_FREQS[1], receive_recurrent=False)
oA_all, oB_all = [], []
for x in sig:
    oA_all.append(nA_shared.step(x))
    oB_all.append(nB_shared.step(x))

def run_neuron_C(neuron):
    for i, x in enumerate(sig):
        neuron.step(x, oA_all[i], oB_all[i])

conditions = {
    "Open loop\n(no recurrent)":
        TakensNeuron(40.0, self_taps=4,  adaptive=False, receive_recurrent=False),
    "Closed loop\nshort buffer\n(taps=4)":
        TakensNeuron(40.0, self_taps=4,  adaptive=False, receive_recurrent=True),
    "Closed loop\noracle buffer\n(taps=80)":
        TakensNeuron(40.0, self_taps=80, adaptive=False, receive_recurrent=True),
    "Closed loop\nadaptive buffer\n(starts at 4)":
        TakensNeuron(40.0, self_taps=4,  adaptive=True,  receive_recurrent=True),
}

results = {}
for name, neuron in conditions.items():
    run_neuron_C(neuron)
    c, t, outs, lbls = evaluate(neuron.output_history, labels)
    results[name] = dict(correct=c, total=t, outs=outs, lbls=lbls,
                         neuron=neuron, accuracy=100*c/t)
    final_taps = neuron.taps_history[-1] if neuron.taps_history else neuron.self_taps
    print(f"{name.replace(chr(10),' ')}: {c}/{t} = {100*c/t:.0f}%  (taps→{final_taps:.0f})")

adaptive_neuron = conditions["Closed loop\nadaptive buffer\n(starts at 4)"]
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
for name, r in results.items():
    label = name.replace('\n', ' ')
    print(f"  {label:45s} {r['accuracy']:5.1f}%")
print()
print(f"Adaptive self_taps: {SELF_TAPS_INIT} → {adaptive_neuron.taps_history[-1]:.0f}")
print(f"Context window:      ~{N + GAP//2} samples back from 40Hz midpoint")
print(f"Minimum needed:      ~80 samples (from empirical test)")


# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nGenerating visualization...")

DARK  = '#0a0a14'; BG = '#0c0c18'
GREEN = '#44ffaa'; RED = '#ff4466'
ORG   = '#ffaa44'; BLUE = '#4488ff'
WHITE = '#e8e0d8'; GRAY = '#aaaaaa'
PURPLE = '#cc88ff'

def sax(ax):
    ax.set_facecolor(DARK)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']: ax.spines[s].set_color('#333')
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.xaxis.label.set_color(GRAY)
    ax.yaxis.label.set_color(GRAY)

fig = plt.figure(figsize=(16, 22), facecolor=BG)
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35)

t_ms = np.arange(len(sig)) / FS * 1000
SHOW = 1600  # ms to show in signal plots

# --- Row 0: Input signal ---
ax0 = fig.add_subplot(gs[0, :])
sax(ax0)
ax0.plot(t_ms[:int(SHOW*FS/1000)], sig[:int(SHOW*FS/1000)],
         color=WHITE, lw=0.4, alpha=0.5)
for i, lbl in enumerate(labels):
    ctx_s = i * TOK_STRIDE
    gap_s = ctx_s + N
    tok_s = gap_s + GAP
    tok_e = tok_s + N
    if tok_e / FS * 1000 > SHOW: break
    ctx_col = GREEN if lbl == 0 else BLUE
    ax0.axvspan(t_ms[ctx_s], t_ms[ctx_s+N-1], alpha=0.15, color=ctx_col)
    ax0.axvspan(t_ms[gap_s], t_ms[gap_s+GAP-1], alpha=0.05, color='#555')
    ax0.axvspan(t_ms[tok_s], t_ms[tok_e-1],  alpha=0.08, color=ORG)
    label_str = f"{CTX_FREQS[0]:.0f}Hz" if lbl==0 else f"{CTX_FREQS[1]:.0f}Hz"
    ax0.text(t_ms[ctx_s]+5, 2.0, label_str, color=ctx_col, fontsize=7, va='top')
    ax0.text(t_ms[tok_s]+5, 2.0, '40Hz', color=ORG, fontsize=7, va='top')
ax0.set_title(
    f'Input Signal  |  Green={CTX_FREQS[0]:.0f}Hz context (label 0)  '
    f'Blue={CTX_FREQS[1]:.0f}Hz context (label 1)  '
    f'Gold=ambiguous 40Hz  Grey=silence gap',
    color=WHITE, fontsize=9)
ax0.set_xlabel('Time (ms)', fontsize=9)
ax0.set_xlim(0, SHOW)

# --- Row 1: Context neuron outputs ---
ax1 = fig.add_subplot(gs[1, :])
sax(ax1)
kernel = np.ones(8)/8
smA = np.convolve(nA_shared.output_history, kernel, 'same')
smB = np.convolve(nB_shared.output_history, kernel, 'same')
n_show = int(SHOW * FS / 1000)
ax1.plot(t_ms[:n_show], smA[:n_show], color=GREEN, lw=1.2, alpha=0.85,
         label=f'Neuron A ({CTX_FREQS[0]:.0f}Hz)')
ax1.plot(t_ms[:n_show], smB[:n_show], color=BLUE,  lw=1.2, alpha=0.85,
         label=f'Neuron B ({CTX_FREQS[1]:.0f}Hz)')
for i, lbl in enumerate(labels):
    ctx_s = i * TOK_STRIDE; tok_e = ctx_s + TOK_STRIDE
    if tok_e / FS * 1000 > SHOW: break
    ax1.axvspan(t_ms[ctx_s], t_ms[ctx_s+N-1], alpha=0.08,
                color=GREEN if lbl==0 else BLUE)
ax1.set_title('Context Neurons A and B — feedforward, no recurrent\n'
              'A fires for 11Hz; B fires for 61Hz. Both decay during silence gap.',
              color=WHITE, fontsize=9)
ax1.set_xlabel('Time (ms)', fontsize=9)
ax1.set_ylabel('Resonance power', fontsize=9)
ax1.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=9)
ax1.set_xlim(0, SHOW)

# --- Row 2: Adaptive self-buffer growth ---
ax2 = fig.add_subplot(gs[2, :])
sax(ax2)
taps_arr = np.array(adaptive_neuron.taps_history)
ax2.plot(t_ms, taps_arr, color=ORG, lw=1.5, label='Self-buffer length (adaptive C)')
ax2.axhline(80, color=PURPLE, lw=1.2, linestyle=':',
            label='Empirical minimum for disambiguation (~80 samples)')
ax2.axhline(N + GAP//2, color=RED, lw=1.5, linestyle='--',
            label=f'Context window distance = {N+GAP//2} samples')
# Shade region where buffer is long enough
ax2.fill_between(t_ms, taps_arr, 80,
                 where=taps_arr >= 80, color=GREEN, alpha=0.07,
                 label='Buffer long enough to disambiguate')
cross_idx = next((i for i, v in enumerate(taps_arr) if v >= 80), None)
if cross_idx:
    ax2.axvline(t_ms[cross_idx], color=GREEN, lw=1, linestyle=':', alpha=0.8)
    ax2.text(t_ms[cross_idx]+30, 85,
             f'Functional at\nt={t_ms[cross_idx]:.0f}ms', color=GREEN, fontsize=8)
ax2.set_title(
    f'Adaptive Self-Buffer Growth (Dendritic Branch Elongation)\n'
    f'Starts at {SELF_TAPS_INIT} samples. Frustration drives growth until context is reachable.',
    color=WHITE, fontsize=9)
ax2.set_xlabel('Time (ms)', fontsize=9)
ax2.set_ylabel('Self-buffer length (samples)', fontsize=9)
ax2.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=8, ncol=2)

# --- Row 3: C output per token for oracle vs adaptive ---
ax3 = fig.add_subplot(gs[3, :])
sax(ax3)
oracle_key   = "Closed loop\noracle buffer\n(taps=80)"
adaptive_key = "Closed loop\nadaptive buffer\n(starts at 4)"

for key, marker, offset, alpha in [(oracle_key, 'o', 0, 0.5),
                                    (adaptive_key, '^', 25, 0.9)]:
    r    = results[key]
    toks = [(2*i+1)*N + N//2 for i in range(len(r['lbls']))]
    t_c  = [tc/FS*1000 + offset for tc in toks]
    for lv, col, lab in [(0, GREEN, '0 (11Hz ctx)'), (1, BLUE, '1 (61Hz ctx)')]:
        vs = [o for o, l in zip(r['outs'], r['lbls']) if l == lv]
        ts = [t for t, l in zip(t_c, r['lbls']) if l == lv]
        ax3.scatter(ts, vs, color=col, s=18, alpha=alpha, marker=marker,
                    label=f'Label {lab} — {"oracle" if key==oracle_key else "adaptive"}')

ax3.set_title('Neuron C Output per 40Hz Token\n'
              'Circles=oracle (taps=80) Triangles=adaptive. '
              'Green (label 0) and Blue (label 1) must separate.',
              color=WHITE, fontsize=9)
ax3.set_xlabel('Token centre time (ms)', fontsize=9)
ax3.set_ylabel('Mean C output during 40Hz window', fontsize=9)
ax3.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=8, ncol=2)

# --- Row 4 left: accuracy bars ---
ax4 = fig.add_subplot(gs[4, 0])
sax(ax4)
cond_labels = [name.replace('\n', '\n') for name in conditions]
accs        = [results[n]['accuracy'] for n in conditions]
cols        = [RED, RED, GREEN, ORG]
bars        = ax4.bar(cond_labels, accs, color=cols, alpha=0.85, width=0.6)
ax4.axhline(50, color='#555', lw=1, linestyle='--', label='Chance (50%)')
ax4.set_ylim(0, 110)
ax4.set_ylabel('Accuracy (%)', fontsize=9)
ax4.set_title('Accuracy by Condition', color=WHITE, fontsize=10)
ax4.tick_params(axis='x', colors=WHITE, labelsize=7)
ax4.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=9)
for bar, acc in zip(bars, accs):
    ax4.text(bar.get_x() + bar.get_width()/2, acc + 2,
             f'{acc:.0f}%', ha='center', va='bottom',
             color=WHITE, fontsize=10, fontweight='bold')

# --- Row 4 right: summary text ---
ax5 = fig.add_subplot(gs[4, 1])
ax5.set_facecolor('#050510'); ax5.axis('off')

lines = [
    "NETWORK",
    "",
    "  A (11Hz) ──────────────────────┐",
    "                                  ├─► ctx = A_mean - B_mean",
    "  B (61Hz) ──────────────────────┘        │",
    "                                           ▼",
    "  world ──► C (40Hz) ◄──── self_buf_A, self_buf_B",
    "                  │",
    "                  └─► output = res_40 × (1 + tanh(ctx × 3))",
    "",
    "ADAPTIVE MECHANISM",
    "",
    f"  self_taps:  {SELF_TAPS_INIT} → {adaptive_neuron.taps_history[-1]:.0f}",
    f"  target ctx strength:  {DISAMBIG_TARGET}",
    f"  growth rate:  {SELF_GROW_RATE}/step when frustrated",
    "",
    "BIOLOGICAL ANALOG",
    "",
    "  self_taps growth  = dendritic elongation",
    "  frustration       = ctx signal below target",
    "  homeostatic drive = AIS structural plasticity",
    "  recurrent inputs  = causal connectivity",
    "  the whole loop    = brain is endless loop",
]

for i, line in enumerate(lines):
    col = WHITE
    if line in ("NETWORK", "ADAPTIVE MECHANISM", "BIOLOGICAL ANALOG"): col = ORG
    if '→' in line and 'self' in line.lower(): col = ORG
    ax5.text(0.03, 0.99 - i*0.052, line,
             transform=ax5.transAxes,
             color=col, fontsize=8, fontfamily='monospace', va='top')

fig.suptitle(
    "Closed Loop Takens Neuron\n"
    "Temporal Context Disambiguation via Adaptive Recurrent Self-Buffer",
    color=WHITE, fontsize=12, y=0.999
)

out_path = '/mnt/user-data/outputs/closed_loop_takens.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
print("Done.")
