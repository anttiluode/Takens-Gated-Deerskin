"""
active_inference_takens.py
==========================
Closing the outer loop on the physical world.

Previous experiments:
  Level 1: Open loop  — decode frequencies with no memory
  Level 2: Inner loop — context disambiguation via recurrent self-buffer
  Level 3: Outer loop — C's prediction modulates what A/B detect
                       → three dynamical regimes (sensory, limit cycle, hallucination)

This experiment (Level 4):
  The agent can ACTIVELY DETECT prediction errors against the world state.
  When detected: the belief buffer is cleared (reset) + outer loop suspended (refractory).
  This converts a 'dead attractor' hallucination into a 'live' oscillating search.

THE KEY PHENOMENON: Belief Revision After Belief Switch
-------------------------------------------------------
Sequence structure:
  - Trials 0–9:  all context = 11Hz (label 0)
  - Trials 10–29: all context = 61Hz (label 1) ← BELIEF SWITCH

Three conditions compared:

  1. Low alpha (sensory regime)
     Outer coupling α=0.1. World signal dominates.
     Pre: 100%, Post: 100%.  No problems to solve.

  2. High alpha, no reset (locked hallucination)
     Outer coupling α=0.8. Prediction overrides sensation.
     Pre: 100%, Post: 0%.
     After label switch: C_pred stays locked at +1.0.
     Variance = 0.00. Dead attractor. No error awareness.

  3. High alpha, WITH active error detection (this experiment)
     Outer coupling α=0.8. Same lock potential.
     Every TOK/2 samples: compare predicted frequency vs world state.
     If mismatch AND confidence high: RESET (clear belief buffers) + REFRACTORY.
     Pre: 100%, Post: ~15%.
     C_pred variance = 0.33 (alive, oscillating, searching).
     The hallucination has been converted to a limit cycle.

The 15% post-switch accuracy is NOT a success by most metrics.
But it represents something qualitatively different from 0%:
the network is AWARE of its mismatch.

BIOLOGICAL MAPPING
------------------
Outer loop strength α    = top-down prediction gain / precision weighting
Active error detection   = surprise signal / free energy gradient
Buffer reset             = neuromodulator-driven depotentiation (ACh, NE)
Refractory period        = acetylcholine-gated exploration phase
Locked hallucination     = full false prior / delusion
Reset to limit cycle     = confabulation: the network is wrong but searching
Low α, sensory           = normal perception: bottom-up dominates

The conversion from dead attractor to limit cycle is the computationally
meaningful effect. A dead attractor is unreachable for correction.
A limit cycle is reachable: given enough prediction errors, it can be
pushed toward the correct attractor.

"A network that knows it might be wrong can potentially be corrected.
A network that has stopped checking cannot." — the active inference insight

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
TOK         = 50
GAP         = 40
STRIDE      = TOK + GAP + TOK
SELF_TAPS   = 80
CTX_GAIN    = 3.0
OUTER_LOW   = 0.1
OUTER_HIGH  = 0.8
REFRACTORY  = 60    # samples of suspended outer loop after reset
N_PAIRS     = 30    # 10 label-0, then 20 label-1
SWITCH_AT   = 10    # trial index where labels switch


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def make_sequence(n_pairs=N_PAIRS, switch_at=SWITCH_AT, noise=0.1, seed=42):
    """
    First `switch_at` pairs: label 0 (11Hz context).
    Remaining pairs: label 1 (61Hz context).
    """
    np.random.seed(seed)
    parts, labels = [], []
    for i in range(n_pairs):
        c  = 0 if i < switch_at else 1
        cf = 11.0 if c == 0 else 61.0
        t  = np.arange(TOK) / FS
        parts.append(np.sin(2*np.pi*cf*t) + np.random.normal(0, noise, TOK))
        parts.append(np.zeros(GAP) + np.random.normal(0, noise*0.5, GAP))
        parts.append(np.sin(2*np.pi*40*t) + np.random.normal(0, noise, TOK))
        labels.append(c)
    return np.concatenate(parts), labels


# ============================================================================
# NETWORK
# ============================================================================

def run_network(signal, labels, outer_alpha, use_reset=False,
                self_taps=SELF_TAPS, ctx_gain=CTX_GAIN, refractory=REFRACTORY):
    """
    Run the outer-loop network with optional active error detection.

    Parameters
    ----------
    outer_alpha : float
        Coupling strength: C_pred modulates A/B input.
    use_reset : bool
        If True, check for prediction errors every TOK/2 steps.
        On mismatch: clear belief buffers + enter refractory period.

    Returns
    -------
    dict with pred_hist, reset_events, labels
    """
    def make_mosaic(freq, tau=2, n=12):
        tt = np.arange(n) * (tau / FS)
        m  = np.cos(2*np.pi*freq*tt)
        return m / np.linalg.norm(m)

    mA, mB = make_mosaic(11.0), make_mosaic(61.0)
    bA  = np.zeros(12*2 + 5)
    bB  = np.zeros(12*2 + 5)
    sA  = np.zeros(self_taps + 20)
    sB  = np.zeros(self_taps + 20)

    C_pred    = 0.0
    ref_count = 0

    pred_hist    = []
    reset_events = []   # (sample_index, C_pred_before, world_freq)
    world_freq   = 40.0

    for i, x in enumerate(signal):
        # Track world's current context frequency
        trial    = i // STRIDE
        in_trial = i  % STRIDE
        if trial < len(labels):
            if in_trial < TOK:
                world_freq = 11.0 if labels[trial] == 0 else 61.0
            else:
                world_freq = 40.0

        # Effective outer loop strength (0 during refractory)
        eff_alpha = outer_alpha if ref_count == 0 else 0.0
        if ref_count > 0:
            ref_count -= 1

        # A and B see outer-loop-modulated signal
        xAB = x + eff_alpha * C_pred
        bA  = np.roll(bA, 1); bA[0] = xAB
        tv  = bA[::2][:12]
        rA  = np.dot(tv, mA)**2 if len(tv) == 12 else 0.0

        bB  = np.roll(bB, 1); bB[0] = xAB
        tv  = bB[::2][:12]
        rB  = np.dot(tv, mB)**2 if len(tv) == 12 else 0.0

        sA  = np.roll(sA, 1); sA[0] = rA
        sB  = np.roll(sB, 1); sB[0] = rB

        # Active error detection
        if use_reset and (i % (TOK // 2) == 0) and i > TOK and ref_count == 0:
            expected_freq = 11.0 if C_pred > 0.3 else (61.0 if C_pred < -0.3 else 40.0)
            mismatch      = abs(expected_freq - world_freq) > 15.0
            confident     = abs(C_pred) > 0.4
            if mismatch and confident:
                # Prediction error: clear belief, enter refractory
                reset_events.append((i, float(C_pred), world_freq))
                sA[:]      = 0.0
                sB[:]      = 0.0
                C_pred     = 0.0
                ref_count  = refractory

        ctx    = sA[:self_taps].mean() - sB[:self_taps].mean()
        C_pred = np.tanh(ctx * ctx_gain)
        pred_hist.append(C_pred)

    return {
        'pred_hist':    np.array(pred_hist),
        'reset_events': reset_events,
        'labels':       labels,
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(pred_hist, labels, n=N_PAIRS):
    """Per-token classification: positive C_pred → label 0, negative → label 1."""
    preds, lbls = [], []
    for i in range(min(n, len(labels))):
        ts = i * STRIDE + TOK + GAP
        te = ts + TOK
        if te > len(pred_hist): break
        preds.append(0 if np.mean(pred_hist[ts:te]) > 0 else 1)
        lbls.append(labels[i])
    correct = [1 if p == l else 0 for p, l in zip(preds, lbls)]
    return correct, preds


# ============================================================================
# RUN
# ============================================================================
sig, labels = make_sequence()

print("=" * 65)
print("ACTIVE INFERENCE TAKENS: BELIEF REVISION AFTER BELIEF SWITCH")
print("=" * 65)
print(f"Trials 0–{SWITCH_AT-1}: label 0 (11Hz context)")
print(f"Trials {SWITCH_AT}–{N_PAIRS-1}: label 1 (61Hz context)  ← SWITCH")
print()

conditions = [
    ("Low α sensory (α=0.10, no reset)",  OUTER_LOW,  False),
    ("High α locked (α=0.80, no reset)",  OUTER_HIGH, False),
    ("High α active (α=0.80, with reset)", OUTER_HIGH, True),
]

results = {}
for name, alpha, use_reset in conditions:
    r         = run_network(sig, labels, outer_alpha=alpha, use_reset=use_reset)
    correct, preds = evaluate(r['pred_hist'], labels)
    pre  = sum(correct[:SWITCH_AT]) / SWITCH_AT
    post = sum(correct[SWITCH_AT:]) / max(1, len(correct) - SWITCH_AT)
    var  = float(np.var(r['pred_hist'][SWITCH_AT * STRIDE:]))
    mean = float(np.mean(r['pred_hist'][SWITCH_AT * STRIDE:]))
    n_resets = len(r['reset_events'])

    results[name] = dict(
        r=r, correct=correct, preds=preds,
        pre=pre, post=post, var=var, mean=mean, n_resets=n_resets,
        alpha=alpha, use_reset=use_reset,
    )
    print(f"{name}")
    print(f"  Pre-switch:  {100*pre:.0f}%  |  Post-switch: {100*post:.0f}%")
    print(f"  C_pred after switch: mean={mean:.3f}, var={var:.4f}")
    print(f"  Resets triggered: {n_resets}")
    print(f"  Per-token: {correct}")
    print()

print("=" * 65)
print("KEY RESULT: HALLUCINATION vs ACTIVE CONFABULATION")
print("=" * 65)
low_r  = results["Low α sensory (α=0.10, no reset)"]
lock_r = results["High α locked (α=0.80, no reset)"]
act_r  = results["High α active (α=0.80, with reset)"]

print(f"  Sensory     post={100*low_r['post']:.0f}%   var={low_r['var']:.4f}   <- world drives belief")
print(f"  Locked      post={100*lock_r['post']:.0f}%    var={lock_r['var']:.4f}   <- dead attractor")
print(f"  Active      post={100*act_r['post']:.0f}%   var={act_r['var']:.4f}   <- live oscillation")
print()
print(f"  Locked hallucination: C_pred variance = {lock_r['var']:.4f}")
print(f"  Active error detection: variance = {act_r['var']:.4f}")
print(f"  Conversion: dead attractor → limit cycle")
print(f"  'A network that knows it might be wrong can potentially be corrected.'")


# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nGenerating visualization...")

DARK  = '#0a0a14'; BG = '#0c0c18'
GREEN = '#44ffaa'; RED = '#ff4466'
ORG   = '#ffaa44'; BLUE = '#4488ff'
WHITE = '#e8e0d8'; GRAY = '#aaaaaa'
PURPLE= '#cc88ff'

def sax(ax):
    ax.set_facecolor(DARK)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']: ax.spines[s].set_color('#333')
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.xaxis.label.set_color(GRAY); ax.yaxis.label.set_color(GRAY)

fig = plt.figure(figsize=(17, 23), facecolor=BG)
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35)

t_ms    = np.arange(len(sig)) / FS * 1000
switch_ms = SWITCH_AT * STRIDE / FS * 1000

# --- Row 0: Input signal ---
ax0 = fig.add_subplot(gs[0, :])
sax(ax0)
ax0.plot(t_ms, sig, color=WHITE, lw=0.3, alpha=0.5)
for i, lbl in enumerate(labels):
    cs = i*STRIDE; tok_s = cs+TOK+GAP; tok_e = tok_s+TOK
    cc = GREEN if lbl==0 else BLUE
    ax0.axvspan(t_ms[cs], t_ms[cs+TOK-1], alpha=0.12, color=cc)
    ax0.axvspan(t_ms[tok_s], t_ms[tok_e-1], alpha=0.07, color=ORG)
ax0.axvline(switch_ms, color=WHITE, lw=2, linestyle='--', alpha=0.7,
            label=f'Belief switch at trial {SWITCH_AT}')
ax0.text(switch_ms+30, 2.1, '← Label 0 (11Hz)  |  Label 1 (61Hz) →',
         color=WHITE, fontsize=9, va='top')
ax0.set_title(
    f"Input Signal | Green=11Hz context (label 0)  Blue=61Hz context (label 1)\n"
    f"Gold=ambiguous 40Hz | All 40Hz tokens identical | Switch at trial {SWITCH_AT}",
    color=WHITE, fontsize=9)
ax0.set_xlabel("Time (ms)", fontsize=9)
ax0.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=9, loc='lower right')

# --- Row 1: C_pred timeseries for three conditions ---
ax1 = fig.add_subplot(gs[1, :])
sax(ax1)
cond_plot = [
    ("Low α sensory (α=0.10, no reset)",   GREEN,  2.5, "Sensory"),
    ("High α locked (α=0.80, no reset)",   RED,    0.0, "Locked"),
    ("High α active (α=0.80, with reset)", ORG,   -2.5, "Active"),
]
kernel = np.ones(5)/5
for name, col, offset, short in cond_plot:
    r   = results[name]
    ph  = np.convolve(r['r']['pred_hist'], kernel, 'same')
    ax1.plot(t_ms, ph + offset, color=col, lw=1.1, alpha=0.9,
             label=f'{short} ({100*r["pre"]:.0f}%→{100*r["post"]:.0f}%)')
    ax1.axhline(offset, color=col, lw=0.4, linestyle=':', alpha=0.4)
    # Mark reset events
    if r['n_resets'] > 0:
        for (idx, pred_before, wf) in r['r']['reset_events']:
            if idx < len(t_ms):
                ax1.axvline(t_ms[idx], color=ORG, lw=0.8, alpha=0.5, linestyle=':')

ax1.axvline(switch_ms, color=WHITE, lw=2, linestyle='--', alpha=0.7)
ax1.axvspan(0, switch_ms, alpha=0.03, color=GREEN)
ax1.axvspan(switch_ms, t_ms[-1], alpha=0.03, color=BLUE)
ax1.set_title(
    "C's Prediction Signal — Three Conditions (offset for clarity)\n"
    "After switch: Sensory corrects immediately, Locked stays at +1, Active oscillates",
    color=WHITE, fontsize=9)
ax1.set_xlabel("Time (ms)", fontsize=9)
ax1.set_ylabel("C_pred (offset per condition)", fontsize=9)
ax1.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=9)

# --- Row 2: Per-token accuracy over time (running window) ---
ax2 = fig.add_subplot(gs[2, :])
sax(ax2)
for name, col, _, short in cond_plot:
    r = results[name]
    c = r['correct']
    n_tok = len(c)
    tok_times = [(i * STRIDE + TOK + GAP + TOK//2) / FS * 1000 for i in range(n_tok)]
    # Running accuracy (window=5)
    W = 5
    running = [sum(c[max(0,i-W):i+1])/min(i+1,W) if i<W else
               sum(c[i-W+1:i+1])/W for i in range(n_tok)]
    ax2.plot(tok_times, [100*v for v in running], color=col, lw=1.8,
             alpha=0.9, marker='o', markersize=3, label=f'{short}')
    ax2.scatter(tok_times, [100*v for v in r['correct']],
                color=col, s=12, alpha=0.5)

ax2.axvline(switch_ms, color=WHITE, lw=2, linestyle='--', alpha=0.7,
            label='Belief switch')
ax2.axhline(50, color='#555', lw=1, linestyle='--', label='Chance (50%)')
ax2.set_xlabel("Token centre time (ms)", fontsize=9)
ax2.set_ylabel("Running accuracy % (window=5)", fontsize=9)
ax2.set_ylim(-5, 115)
ax2.set_title(
    "Per-Token Accuracy Over Time\n"
    "Active condition: accuracy drops but doesn't permanently lock — network remains correctable",
    color=WHITE, fontsize=9)
ax2.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=9)

# --- Row 3: C_pred histogram and phase portrait (post-switch only) ---
ax3 = fig.add_subplot(gs[3, 0])
sax(ax3)
bins = np.linspace(-1.1, 1.1, 30)
switch_sample = SWITCH_AT * STRIDE
for name, col, _, short in cond_plot:
    ph_after = results[name]['r']['pred_hist'][switch_sample:]
    counts, edges = np.histogram(ph_after, bins=bins, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    ax3.plot(centres, counts, color=col, lw=1.5, alpha=0.9, label=short)
    ax3.fill_between(centres, counts, alpha=0.1, color=col)
ax3.set_title(
    "C_pred Distribution (post-switch only)\n"
    "Locked=spike at +1  |  Active=spread across values  |  Sensory=centered at -1",
    color=WHITE, fontsize=9)
ax3.set_xlabel("C_pred value", fontsize=9)
ax3.set_ylabel("Density", fontsize=9)
ax3.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=8)

# --- Row 3 right: variance comparison bar ---
ax3b = fig.add_subplot(gs[3, 1])
sax(ax3b)
short_names = ["Sensory\n(α=0.10)", "Locked\n(α=0.80)", "Active\n(α=0.80)"]
vars_   = [low_r['var'], lock_r['var'], act_r['var']]
accs_   = [low_r['post'], lock_r['post'], act_r['post']]
bar_cols = [GREEN, RED, ORG]
x_pos = np.array([0, 1, 2])
bars = ax3b.bar(x_pos - 0.2, [100*a for a in accs_], width=0.35,
                color=bar_cols, alpha=0.7, label='Post-switch accuracy %')
ax3b_twin = ax3b.twinx()
ax3b_twin.set_facecolor(DARK)
ax3b_twin.tick_params(colors=GRAY, labelsize=8)
ax3b_twin.yaxis.label.set_color(PURPLE)
bars2 = ax3b_twin.bar(x_pos + 0.2, vars_, width=0.35,
                       color=PURPLE, alpha=0.5, label='C_pred variance')
ax3b.set_xticks(x_pos); ax3b.set_xticklabels(short_names, color=WHITE, fontsize=8)
ax3b.set_ylabel("Post-switch accuracy (%)", fontsize=9)
ax3b_twin.set_ylabel("C_pred variance (after switch)", fontsize=9, color=PURPLE)
ax3b.axhline(50, color='#555', lw=1, linestyle='--')
ax3b.set_title(
    "Accuracy vs C_pred Variance\n"
    "High variance = alive/searching  |  Zero variance = dead attractor",
    color=WHITE, fontsize=9)
for bar, acc in zip(bars, accs_):
    ax3b.text(bar.get_x()+bar.get_width()/2, 100*acc+2, f'{100*acc:.0f}%',
              ha='center', va='bottom', color=WHITE, fontsize=9, fontweight='bold')
for bar, v in zip(bars2, vars_):
    ax3b_twin.text(bar.get_x()+bar.get_width()/2, v+0.003, f'{v:.3f}',
                   ha='center', va='bottom', color=PURPLE, fontsize=8)

# --- Row 4: Summary text ---
ax5 = fig.add_subplot(gs[4, :])
ax5.set_facecolor('#050510'); ax5.axis('off')

lines = [
    ("THE FOUR LEVELS OF LOOP CLOSURE", ORG, True),
    ("", WHITE, False),
    ("  Level 1: Open loop          — decode frequencies, no memory, no recurrent", WHITE, False),
    ("  Level 2: Inner loop         — context disambiguation, adaptive self-buffer", WHITE, False),
    ("  Level 3: Outer loop         — three regimes: sensory / limit cycle / hallucination", WHITE, False),
    ("  Level 4: Active inference   — error detection breaks hallucination lock", WHITE, False),
    ("", WHITE, False),
    ("KEY RESULT", ORG, True),
    ("", WHITE, False),
    (f"  Locked hallucination (α=0.8, no reset):  post-switch acc={100*lock_r['post']:.0f}%  var={lock_r['var']:.4f}", RED, False),
    (f"  Active detection    (α=0.8, with reset): post-switch acc={100*act_r['post']:.0f}%  var={act_r['var']:.4f}", ORG, False),
    ("", WHITE, False),
    ("  The active condition converts dead attractor → limit cycle.", ORG, False),
    ("  Accuracy is still poor (15%) but the network is ALIVE — it can be reached.", ORG, False),
    ("", WHITE, False),
    ("BIOLOGICAL MAPPING", ORG, True),
    ("", WHITE, False),
    ("  Outer loop strength α       = top-down precision weighting", WHITE, False),
    ("  Active error detection      = surprise signal / free energy gradient", WHITE, False),
    ("  Buffer reset                = neuromodulator-driven depotentiation (ACh, NE)", WHITE, False),
    ("  Refractory period           = acetylcholine-gated exploration phase", WHITE, False),
    ("  Locked hallucination        = aberrant precision (Friston 2010)", RED, False),
    ("  Reset → limit cycle         = confabulation: wrong but searching", ORG, False),
    ("", WHITE, False),
    ("WHAT'S NOT DONE", ORG, True),
    ("", WHITE, False),
    ("  → Full correction would require reducing effective α during refractory (antipsychotic effect)", WHITE, False),
    ("  → Agent could change world input via real motor action (proprioceptive loop)", WHITE, False),
    ("  → With lateral inhibition A↔B, the competition sharpens further", WHITE, False),
    ("  → Each of these is the next experiment.", GREEN, False),
]

y = 0.98
for text, col, bold in lines:
    weight = 'bold' if bold else 'normal'
    ax5.text(0.02, y, text, transform=ax5.transAxes,
             color=col, fontsize=7.8, fontfamily='monospace', va='top',
             fontweight=weight)
    y -= 0.038

fig.suptitle(
    "Active Inference Takens\n"
    "Belief Revision via Prediction Error Detection: Hallucination → Limit Cycle",
    color=WHITE, fontsize=12, y=0.999
)

out_path = '/mnt/user-data/outputs/active_inference_takens.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
print("Done.")
