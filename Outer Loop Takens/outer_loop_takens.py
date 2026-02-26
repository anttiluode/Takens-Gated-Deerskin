"""
outer_loop_takens.py
====================
Closing the outer loop: C's predictions feed back to modulate
what A and B detect. The network now responds to its own responses.

INNER LOOP (previous experiment):
  World -> A, B -> C
  C uses A, B history as context.

OUTER LOOP (this experiment):
  C's prediction feeds back to excite A and suppress B (or vice versa).
  The circuit is now fully recurrent:

    World ──┬──► A (11Hz) ◄── amplified by +alpha*C_pred
            │         │
            │         └──────────────────────┐
            │                                ▼
            ├──► B (61Hz) ◄── suppressed by -alpha*C_pred
            │         │                      │
            │         └──────────┐           │
            │                   ctx = A_mean - B_mean
            │                   ▼
            └──► C (40Hz) ◄── ctx modulates output
                      │
                      └──► C_pred ──────────────► feeds back to A, B

This is genuine predictive coding: C's current belief about context
amplifies the evidence that supports that belief (A if label 0)
and suppresses the counter-evidence (B).

THREE DYNAMICAL REGIMES (found by sweeping alpha):

  alpha 0.05-0.25  SENSORY:      100% accuracy
                                  C correctly tracks context tokens.
                                  Feedback amplifies signal slightly.
                                  World drives perception.

  alpha 0.30-0.55  LIMIT CYCLE:  ~35-65% accuracy
                                  C oscillates - can't commit.
                                  Feedback is strong enough to
                                  compete with the signal but not
                                  dominate it.

  alpha 0.60+      HALLUCINATION: 35% accuracy (= base rate for label 0)
                                  ctx locked at +1.0 regardless of input.
                                  C predicts label 0 for every token.
                                  C's feedback to A has overwhelmed
                                  the world signal. The network hears
                                  itself, not the world.

BIOLOGICAL INTERPRETATION:
  Sensory regime   = normal perception
  Limit cycle      = active inference / uncertainty / attention search
  Hallucination    = prediction error too high, prediction overrides sensation
                   = psychosis model (aberrant precision, Friston 2010)

VERIFIED: at alpha=1.5, feeding all label-1 context (61Hz only):
  -> network still predicts label 0 for every token (locked attractor)
  -> cannot be broken by sensory evidence without disrupting the loop

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
FS          = 1000.0
TOK         = 50            # token duration (ms = samples at 1kHz)
GAP         = 40            # silence gap (ms)
STRIDE      = TOK + GAP + TOK
TOK_START   = TOK + GAP     # offset of 40Hz token within each stride
NOISE       = 0.1
N_PAIRS     = 20
SELF_TAPS   = 80
CTX_GAIN    = 3.0

# Alpha values to sweep
ALPHAS      = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
# Three canonical regimes to show in detail
ALPHA_SENSORY      = 0.10
ALPHA_LIMIT_CYCLE  = 0.40
ALPHA_HALLUCIN     = 1.50

# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def make_sequence(n_pairs=N_PAIRS, noise=NOISE, seed=0):
    np.random.seed(seed)
    parts, labels = [], []
    for _ in range(n_pairs):
        ctx     = np.random.choice([0, 1])
        cf      = 11.0 if ctx == 0 else 61.0
        t       = np.arange(TOK) / FS
        parts.append(np.sin(2*np.pi*cf*t) + np.random.normal(0, noise, TOK))
        parts.append(np.zeros(GAP) + np.random.normal(0, noise*0.5, GAP))
        parts.append(np.sin(2*np.pi*40*t) + np.random.normal(0, noise, TOK))
        labels.append(ctx)
    return np.concatenate(parts), labels


def make_all_label1(n_pairs=N_PAIRS, noise=NOISE, seed=1):
    """All context tokens are 61Hz (label 1). Tests hallucination robustness."""
    np.random.seed(seed)
    parts, labels = [], []
    for _ in range(n_pairs):
        t = np.arange(TOK) / FS
        parts.append(np.sin(2*np.pi*61.0*t) + np.random.normal(0, noise, TOK))
        parts.append(np.zeros(GAP) + np.random.normal(0, noise*0.5, GAP))
        parts.append(np.sin(2*np.pi*40*t) + np.random.normal(0, noise, TOK))
        labels.append(1)
    return np.concatenate(parts), labels


# ============================================================================
# NETWORK
# ============================================================================

def run_network(alpha, signal, self_taps=SELF_TAPS, ctx_gain=CTX_GAIN):
    """
    Run the fully recurrent 3-neuron network.

    Outer loop: C_pred feeds back to A (+alpha) and B (-alpha).
    Positive C_pred -> excite A (evidence for label 0) -> suppress B.
    Negative C_pred -> excite B (evidence for label 1) -> suppress A.

    This is winner-take-all competition between context hypotheses,
    driven by C's current belief.

    Returns: (A_output, B_output, C_output, ctx_signal)
    """
    # --- Neuron A (11Hz context detector) ---
    tau_A, n_A = 2, 12
    t_tap = np.arange(n_A) * (tau_A / FS)
    m_A   = np.cos(2*np.pi*11.0*t_tap); m_A /= np.linalg.norm(m_A)
    buf_A = np.zeros(n_A * tau_A + 5)
    out_A = np.zeros(len(signal))

    # --- Neuron B (61Hz context detector) ---
    tau_B, n_B = 2, 12
    m_B = np.cos(2*np.pi*61.0*t_tap); m_B /= np.linalg.norm(m_B)
    buf_B = np.zeros(n_B * tau_B + 5)
    out_B = np.zeros(len(signal))

    # --- Neuron C (40Hz ambiguous token detector + context reader) ---
    tau_C, n_C = 2, 12
    m_C   = np.cos(2*np.pi*40.0*np.arange(n_C)*(tau_C/FS)); m_C /= np.linalg.norm(m_C)
    buf_C = np.zeros(n_C * tau_C + 5)
    self_buf_A = np.zeros(self_taps + 20)
    self_buf_B = np.zeros(self_taps + 20)
    out_C   = np.zeros(len(signal))
    ctx_sig = np.zeros(len(signal))

    C_pred = 0.0  # C's current signed prediction [-1, 1]

    for i, x in enumerate(signal):
        # Outer loop: modulate what A and B hear
        xA = x + alpha * C_pred    # positive pred excites A
        xB = x - alpha * C_pred    # positive pred suppresses B

        # A step
        buf_A = np.roll(buf_A, 1); buf_A[0] = xA
        tv    = buf_A[::tau_A][:n_A]
        rA    = np.dot(tv, m_A)**2 if len(tv) == n_A else 0.0
        out_A[i] = rA

        # B step
        buf_B = np.roll(buf_B, 1); buf_B[0] = xB
        tv    = buf_B[::tau_B][:n_B]
        rB    = np.dot(tv, m_B)**2 if len(tv) == n_B else 0.0
        out_B[i] = rB

        # Update C's recurrent buffers
        self_buf_A = np.roll(self_buf_A, 1); self_buf_A[0] = rA
        self_buf_B = np.roll(self_buf_B, 1); self_buf_B[0] = rB

        # C step: world resonance + context from self-buffers
        buf_C = np.roll(buf_C, 1); buf_C[0] = x
        tv    = buf_C[::tau_C][:n_C]
        rC    = np.dot(tv, m_C)**2 if len(tv) == n_C else 0.0

        ctx   = self_buf_A[:self_taps].mean() - self_buf_B[:self_taps].mean()
        C_pred = np.tanh(ctx * ctx_gain)   # signed prediction

        out_C[i]   = rC * (1.0 + C_pred)
        ctx_sig[i] = C_pred

    return out_A, out_B, out_C, ctx_sig


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(ctx_sig, labels, n_pairs=N_PAIRS, threshold=0.0):
    """
    Classify by sign of mean ctx signal in each 40Hz window.
    Uses fixed threshold=0: positive ctx = label 0, negative = label 1.
    (Median threshold breaks when ctx is locked at +1 or -1.)
    """
    outs, lbls = [], []
    for i in range(min(n_pairs, len(labels))):
        ts = i * STRIDE + TOK_START
        te = ts + TOK
        if te > len(ctx_sig): break
        outs.append(float(np.mean(ctx_sig[ts:te])))
        lbls.append(labels[i])
    preds   = [0 if v > threshold else 1 for v in outs]
    correct = sum(1 for p, l in zip(preds, lbls) if p == l)
    return correct / len(preds) if preds else 0.0, preds, outs, lbls


# ============================================================================
# RUN
# ============================================================================
print("=" * 65)
print("OUTER LOOP TAKENS: THREE DYNAMICAL REGIMES")
print("=" * 65)
print(f"Task: {N_PAIRS} context-disambiguation pairs (11Hz/61Hz -> 40Hz)")
print(f"Self-taps: {SELF_TAPS} | ctx_gain: {CTX_GAIN}")
print()

sig, labels = make_sequence(seed=0)

print("Alpha sweep:")
print(f"{'alpha':>8}  {'accuracy':>10}  {'ctx_mean':>10}  {'ctx_std':>10}  {'regime':>15}")
print("-" * 65)

sweep_results = []
for alpha in ALPHAS:
    _, _, _, ctx = run_network(alpha, sig)
    acc, _, _, _ = evaluate(ctx, labels)
    ctx_m = float(np.mean(ctx[500:]))
    ctx_s = float(np.std(ctx[500:]))
    if acc > 0.7 and abs(ctx_m) < 0.7:
        regime = "SENSORY"
    elif ctx_m > 0.8:
        regime = "HALLUCINATION"
    elif acc < 0.6:
        regime = "LIMIT CYCLE"
    else:
        regime = "TRANSITION"
    sweep_results.append((alpha, acc, ctx_m, ctx_s, regime))
    print(f"{alpha:>8.2f}  {100*acc:>9.1f}%  {ctx_m:>10.3f}  {ctx_s:>10.3f}  {regime}")

print()

# Hallucination robustness test
print("Hallucination test: all context tokens are 61Hz (true label = 1)")
sig_all1, labels_all1 = make_all_label1(seed=1)
for alpha, name in [(ALPHA_SENSORY, "Sensory"), (ALPHA_HALLUCIN, "Hallucination")]:
    _, _, _, ctx = run_network(alpha, sig_all1)
    acc, preds, _, _ = evaluate(ctx, labels_all1)
    ctx_final = float(np.mean(ctx[-200:]))
    locked = "CORRECT (sensory)" if acc > 0.7 else f"LOCKED (predicts 0={sum(1 for p in preds if p==0)}/{len(preds)} times)"
    print(f"  alpha={alpha}: accuracy={100*acc:.0f}%  ctx={ctx_final:.3f}  -> {locked}")

print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
for alpha, acc, ctx_m, ctx_s, regime in sweep_results:
    bar = "█" * int(acc * 20)
    print(f"  alpha={alpha:.2f}  {regime:15s}  {100*acc:5.1f}%  {bar}")


# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nGenerating visualization...")

DARK   = '#0a0a14'; BG = '#0c0c18'
GREEN  = '#44ffaa'; RED = '#ff4466'
ORG    = '#ffaa44'; BLUE = '#4488ff'
WHITE  = '#e8e0d8'; GRAY = '#aaaaaa'
PURPLE = '#cc88ff'

def sax(ax):
    ax.set_facecolor(DARK)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']: ax.spines[s].set_color('#333')
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.xaxis.label.set_color(GRAY); ax.yaxis.label.set_color(GRAY)

fig = plt.figure(figsize=(17, 23), facecolor=BG)
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35)

t_ms  = np.arange(len(sig)) / FS * 1000
SHOW  = 2100   # ms to show

# Get detailed traces for three regimes
regime_data = {}
for alpha, name, col in [
    (ALPHA_SENSORY,     "Sensory",      GREEN),
    (ALPHA_LIMIT_CYCLE, "Limit Cycle",  ORG),
    (ALPHA_HALLUCIN,    "Hallucination",RED),
]:
    A, B, C, CTX = run_network(alpha, sig)
    acc, preds, outs, lbls = evaluate(CTX, labels)
    regime_data[name] = dict(A=A, B=B, C=C, CTX=CTX, acc=acc,
                              preds=preds, outs=outs, lbls=lbls,
                              alpha=alpha, col=col)

# --- Row 0: Input signal ---
ax0 = fig.add_subplot(gs[0, :])
sax(ax0)
ax0.plot(t_ms[:SHOW], sig[:SHOW], color=WHITE, lw=0.4, alpha=0.5)
for i, lbl in enumerate(labels):
    cs = i * STRIDE; gs_ = cs + TOK; ts = gs_ + GAP; te = ts + TOK
    if te / FS * 1000 > SHOW: break
    cc = GREEN if lbl == 0 else BLUE
    ax0.axvspan(t_ms[cs], t_ms[cs+TOK-1], alpha=0.15, color=cc)
    ax0.axvspan(t_ms[gs_], t_ms[gs_+GAP-1], alpha=0.05, color='#555')
    ax0.axvspan(t_ms[ts], t_ms[te-1], alpha=0.08, color=ORG)
    label_str = "11Hz" if lbl == 0 else "61Hz"
    ax0.text(t_ms[cs]+3, 1.9, label_str, color=cc, fontsize=7, va='top')
    ax0.text(t_ms[ts]+3, 1.9, "40Hz", color=ORG, fontsize=7, va='top')
ax0.set_title(
    "Input Signal  |  Green=11Hz context (label 0)  Blue=61Hz context (label 1)\n"
    "Gold=ambiguous 40Hz token  Grey=silence gap  (All 40Hz tokens are identical)",
    color=WHITE, fontsize=9)
ax0.set_xlabel("Time (ms)", fontsize=9); ax0.set_xlim(0, SHOW)
ax0.set_ylim(-2.5, 2.5)

# --- Row 1: Three regime ctx signals ---
ax1 = fig.add_subplot(gs[1, :])
sax(ax1)
kernel = np.ones(5) / 5
offsets = {'Sensory': 2.5, 'Limit Cycle': 0.0, 'Hallucination': -2.5}
for name, rd in regime_data.items():
    offset = offsets[name]
    smoothed = np.convolve(rd['CTX'], kernel, 'same')
    ax1.plot(t_ms[:SHOW], smoothed[:SHOW] + offset,
             color=rd['col'], lw=1.2, alpha=0.9,
             label=f"α={rd['alpha']:.2f} {name} ({100*rd['acc']:.0f}%)")
    ax1.axhline(offset, color=rd['col'], lw=0.4, linestyle=':', alpha=0.4)

for i, lbl in enumerate(labels):
    ts = i * STRIDE + TOK_START
    if ts / FS * 1000 > SHOW: break
    for offset in offsets.values():
        ax1.axvspan(t_ms[ts], t_ms[ts+TOK-1], alpha=0.04, color=ORG)

ax1.set_title(
    "C's Context Signal (ctx) Across the Three Regimes\n"
    "Sensory tracks tokens • Limit Cycle oscillates • Hallucination locks regardless of input",
    color=WHITE, fontsize=9)
ax1.set_xlabel("Time (ms)", fontsize=9)
ax1.set_ylabel("ctx signal (offset per regime)", fontsize=9)
ax1.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=9)
ax1.set_xlim(0, SHOW)

# --- Row 2: Accuracy vs alpha ---
ax2 = fig.add_subplot(gs[2, :])
sax(ax2)
alphas_plot = [r[0] for r in sweep_results]
accs_plot   = [r[1]*100 for r in sweep_results]
regimes_r   = [r[4] for r in sweep_results]

cols_bar = [GREEN if r == 'SENSORY' else
            ORG   if r == 'LIMIT CYCLE' else RED
            for r in regimes_r]
bars = ax2.bar(alphas_plot, accs_plot, color=cols_bar, alpha=0.8, width=0.03)
ax2.axhline(50, color='#555', lw=1, linestyle='--', label='Chance (50%)')

# Shade regions
sensory_alphas = [r[0] for r in sweep_results if r[4] == 'SENSORY']
halluc_alphas  = [r[0] for r in sweep_results if r[4] == 'HALLUCINATION']
if sensory_alphas:
    ax2.axvspan(min(ALPHAS), max(sensory_alphas)+0.02, alpha=0.05, color=GREEN)
    ax2.text((min(ALPHAS)+max(sensory_alphas))/2, 108,
             'SENSORY', color=GREEN, ha='center', fontsize=9, fontweight='bold')
if halluc_alphas:
    ax2.axvspan(min(halluc_alphas)-0.02, max(ALPHAS)+0.1, alpha=0.05, color=RED)
    ax2.text((min(halluc_alphas)+max(ALPHAS))/2, 108,
             'HALLUCINATION', color=RED, ha='center', fontsize=9, fontweight='bold')
ax2.text((max(sensory_alphas if sensory_alphas else [0.25]) +
          min(halluc_alphas if halluc_alphas else [0.6])) / 2, 108,
         'LIMIT\nCYCLE', color=ORG, ha='center', fontsize=9, fontweight='bold')

ax2.set_xlabel("Outer loop coupling strength (α)", fontsize=9)
ax2.set_ylabel("Accuracy (%)", fontsize=9)
ax2.set_ylim(0, 118)
ax2.set_title(
    "Accuracy vs Outer Loop Coupling Strength\n"
    "Three dynamical regimes: signal drives perception → oscillation → prediction overrides sensation",
    color=WHITE, fontsize=9)
ax2.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=9)

# --- Row 3: Per-token classification scatter for each regime ---
for col_idx, (name, rd) in enumerate(regime_data.items()):
    ax = fig.add_subplot(gs[3, col_idx] if col_idx < 2 else gs[4, 0])
    sax(ax)
    tok_centers_ms = [(i * STRIDE + TOK_START + TOK//2) / FS * 1000
                      for i in range(len(rd['lbls']))]
    for lv, lc, ll in [(0, GREEN, 'Label 0 (11Hz)'), (1, BLUE, 'Label 1 (61Hz)')]:
        vs = [o for o, l in zip(rd['outs'], rd['lbls']) if l == lv]
        ts = [t for t, l in zip(tok_centers_ms, rd['lbls']) if l == lv]
        ax.scatter(ts, vs, color=lc, s=25, alpha=0.85, label=ll)
    ax.axhline(np.median(rd['outs']), color=WHITE, lw=1, linestyle='--',
               alpha=0.5, label='Decision threshold')
    ax.set_title(
        f"α={rd['alpha']}: {name}\n{100*rd['acc']:.0f}% accuracy",
        color=rd['col'], fontsize=9)
    ax.set_xlabel("Token time (ms)", fontsize=9)
    ax.set_ylabel("Mean ctx during 40Hz window", fontsize=9)
    ax.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=8)

# --- Row 4 right: architecture + summary text ---
ax_txt = fig.add_subplot(gs[4, 1])
ax_txt.set_facecolor('#050510'); ax_txt.axis('off')

lines = [
    "OUTER LOOP ARCHITECTURE",
    "",
    "  World ──┬──► A ◄── +α·C_pred (excited if pred>0)",
    "          │    │",
    "          │    └───────────────────────┐",
    "          │                            ▼",
    "          ├──► B ◄── -α·C_pred    ctx = A_mean - B_mean",
    "          │    │                       │",
    "          │    └──────────┐            │",
    "          │               └────► C ◄──┘",
    "          └────────────────────► C",
    "                                  │",
    "                              C_pred ──────►",
    "",
    "REGIME TRANSITIONS",
    "",
    f"  α≤0.25   SENSORY       {100*regime_data['Sensory']['acc']:.0f}%",
    f"  α≈0.3-0.5 LIMIT CYCLE  oscillating",
    f"  α≥0.6    HALLUCINATION {100*regime_data['Hallucination']['acc']:.0f}% (= base rate)",
    "",
    "HALLUCINATION TEST:",
    "  All inputs: 61Hz (true label=1)",
    "  Sensory: correctly predicts 1",
    "  Hallucin: locked at label 0",
    "  Cannot be broken by evidence.",
    "",
    "BIOLOGICAL ANALOG:",
    "  Sensory     = normal perception",
    "  Limit cycle = active inference",
    "  Hallucin    = aberrant precision",
    "                (Friston 2010)",
]

for i, line in enumerate(lines):
    col = WHITE
    if "OUTER" in line or "REGIME" in line or "HALLUCIN" in line[:10]: col = ORG
    if "SENSORY" in line and "%" in line: col = GREEN
    if "HALLUCINATION" in line and "%" in line: col = RED
    if "LIMIT" in line and "%" not in line and "≈" not in line: col = ORG
    ax_txt.text(0.03, 0.99 - i*0.051, line,
                transform=ax_txt.transAxes,
                color=col, fontsize=7.8, fontfamily='monospace', va='top')

fig.suptitle(
    "Outer Loop Takens: Three Dynamical Regimes\n"
    "Sensory → Limit Cycle → Hallucination as Outer Coupling α Increases",
    color=WHITE, fontsize=12, y=0.999
)

out_path = '/mnt/user-data/outputs/outer_loop_takens.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
print("Done.")
