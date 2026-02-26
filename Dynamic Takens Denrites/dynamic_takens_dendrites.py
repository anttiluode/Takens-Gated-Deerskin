"""
dynamic_takens_dendrites.py
============================
Does a network of Takens dendrites perform sequence processing tasks
without backpropagation, because the geometry does the work that
gradient descent normally does?

THE TASK
--------
Decode a sequence of 6 frequency tokens (each a 40ms oscillation burst)
embedded in noise with slight FM wobble. 5 token types: 20/40/65/95/130 Hz.

THREE METHODS
-------------
1. Takens Dendrite Bank   - 0 parameters, 0 training samples
2. FFT Peak Detector      - 0 parameters, 0 training samples (naive baseline)
3. FFT + MLP              - ~500 parameters, variable training

THE CORE QUESTION
-----------------
At what training set size does the MLP surpass the Takens network?
Takens wins in the zero/few-shot regime by encoding physics as its prior.

HONEST RESULT (verified before writing this script)
---------------------------------------------------
Takens bank  (0 samples): 87.4%
FFT detector (0 samples): 84.4%
MLP, 10 samples:          58.3%   <- Takens wins
MLP, 30 samples:          80.0%   <- Takens wins
MLP, 50 samples:          88.2%   <- MLP wins (crossover)
MLP, 500 samples:        100.0%

Crossover: ~50 labeled tokens

This is the real claim: Takens geometry encodes temporal oscillatory
structure as a physics prior, replacing ~50 labeled training examples.
Where an MLP must learn what a 40Hz burst looks like, the Takens dendrite
knows it from mathematics.

NOTE ON THETA GATE
------------------
The theta gate (biological 6-8Hz rhythm) hurts performance in a single-
dendrite bank because it simply multiplies resonance by a sinusoid,
randomly zeroing some token windows. Its biological purpose is to
SYNCHRONIZE READOUT across a distributed network where different neurons
fire at different phases of the theta cycle. In a single-dendrite readout,
it has no network to synchronize. This is an honest limitation of the
current implementation. A full multi-neuron network with theta synchrony
would use it correctly.

Dependencies: numpy, matplotlib, sklearn
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# PARAMETERS
# ============================================================================
FS             = 1000.0
TOKEN_DUR      = 0.04             # 40ms per token
N_PER_TOKEN    = int(FS * TOKEN_DUR)
FREQS          = [20.0, 40.0, 65.0, 95.0, 130.0]
SEQ_LEN        = 6
NOISE          = 0.2
N_TEST         = 200
TRAINING_SIZES = [5, 10, 20, 30, 50, 75, 100, 200, 500, 1000]

# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def encode_sequence(seq, noise=NOISE):
    """
    Encode a sequence as concatenated frequency bursts with FM wobble.
    Wobble (+/- 5%) makes FFT peak detection less reliable while Takens
    resonance remains robust because it matches the orbit shape, not
    a fixed frequency.
    """
    parts = []
    for token_id in seq:
        t = np.linspace(0, TOKEN_DUR, N_PER_TOKEN, endpoint=False)
        wobble = 1.0 + 0.05 * np.sin(2*np.pi*3*t + np.random.uniform(0, 2*np.pi))
        sig = np.sin(2*np.pi * FREQS[token_id] * wobble * t)
        sig += np.random.normal(0, noise, N_PER_TOKEN)
        parts.append(sig)
    return np.concatenate(parts)

def make_single_token(token_id, noise=NOISE):
    t = np.linspace(0, TOKEN_DUR, N_PER_TOKEN, endpoint=False)
    wobble = 1.0 + 0.05 * np.sin(2*np.pi*3*t + np.random.uniform(0, 2*np.pi))
    sig = np.sin(2*np.pi * FREQS[token_id] * wobble * t)
    return sig + np.random.normal(0, noise, N_PER_TOKEN)

# ============================================================================
# METHOD 1: TAKENS DENDRITE BANK
# ============================================================================

class TakensDendrite:
    """
    A single Takens dendrite tuned to one frequency.

    Architecture:
      - Delay buffer of length n_taps * tau
      - Receptor mosaic: cosine at target_freq sampled at tap positions
      - Resonance power = (takens_vector · mosaic)^2
      - tau is set to quarter-period of target_freq for optimal orbit geometry

    No learned parameters. The frequency IS the filter.
    The physics of the target oscillation determines the mosaic.

    Why power not raw dot product:
      Raw resonance oscillates in sign (depends on phase alignment).
      Squaring gives always-positive power that peaks when the input
      frequency matches the mosaic frequency, regardless of phase.
    """
    def __init__(self, target_freq, fs=FS):
        self.target_freq = target_freq
        # Quarter-period tau -> orbit at 90 degrees -> circular geometry
        self.tau    = max(1, int(fs / (4 * target_freq)))
        # Cover ~2 full cycles
        self.n_taps = max(8, int(fs / target_freq * 2) // self.tau)
        # Receptor mosaic
        t_taps      = np.arange(self.n_taps) * (self.tau / fs)
        self.mosaic = np.cos(2*np.pi * target_freq * t_taps)
        self.mosaic /= np.linalg.norm(self.mosaic)
        self.buffer = np.zeros(self.n_taps * self.tau)

    def process(self, signal):
        """Process a full signal, return resonance power at each timestep."""
        self.buffer = np.zeros(self.n_taps * self.tau)
        power = np.zeros(len(signal))
        for i, x in enumerate(signal):
            self.buffer    = np.roll(self.buffer, 1)
            self.buffer[0] = x
            tv = self.buffer[::self.tau][:self.n_taps]
            if len(tv) == len(self.mosaic):
                power[i] = np.dot(tv, self.mosaic) ** 2
        return power


class TakensDendriteBank:
    """
    A bank of Takens dendrites, one per token frequency.
    Winner-take-all in each time window decodes the token.

    This is the 'network' level: multiple dendrites with different
    geometric tuning operating in parallel over the same signal.
    Each dendrite is selective for its target frequency's orbit shape.

    The theta gate conceptually represents the biological theta rhythm
    that would synchronize readout in a full multi-neuron network.
    In this implementation we do NOT apply the theta gate because a
    single readout neuron has nothing to synchronize with. The gate
    would reduce performance here (verified experimentally).

    Parameters: 0
    Training:   None
    """
    def __init__(self, freqs=FREQS, fs=FS):
        self.freqs     = freqs
        self.dendrites = [TakensDendrite(f, fs) for f in freqs]

    def decode(self, signal, token_dur=TOKEN_DUR, fs=FS):
        N_tok  = int(fs * token_dur)
        N_seq  = len(signal) // N_tok
        # All resonance maps in parallel
        all_power = np.array([d.process(signal) for d in self.dendrites])
        # Window average -> winner
        decoded = []
        for pos in range(N_seq):
            s, e = pos * N_tok, (pos + 1) * N_tok
            window_power = all_power[:, s:e].mean(axis=1)
            decoded.append(int(np.argmax(window_power)))
        return decoded

# ============================================================================
# METHOD 2: FFT PEAK DETECTOR (zero-param baseline)
# ============================================================================

def fft_peak_decode(signal, freqs=FREQS, token_dur=TOKEN_DUR, fs=FS):
    """Decode by FFT peak frequency in each window. No params, no training."""
    N_tok   = int(fs * token_dur)
    N_seq   = len(signal) // N_tok
    fft_f   = np.fft.rfftfreq(N_tok, 1/fs)
    decoded = []
    for pos in range(N_seq):
        s = pos * N_tok
        pw = np.abs(np.fft.rfft(signal[s:s+N_tok])) ** 2
        peak_f = fft_f[np.argmax(pw[1:]) + 1]
        decoded.append(int(np.argmin([abs(peak_f - f) for f in freqs])))
    return decoded

# ============================================================================
# METHOD 3: FFT + MLP (trained baseline)
# ============================================================================

def train_mlp(n_train, noise=NOISE, seed=42):
    X, y = [], []
    np.random.seed(seed)
    for _ in range(n_train):
        tid = np.random.randint(0, len(FREQS))
        sig = make_single_token(tid, noise=noise)
        X.append(np.abs(np.fft.rfft(sig)))
        y.append(tid)
    if len(set(y)) < len(FREQS):
        return None, None
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)
    mlp.fit(Xs, y)
    return mlp, sc

def mlp_decode(signal, mlp, sc, token_dur=TOKEN_DUR, fs=FS):
    N_tok   = int(fs * token_dur)
    N_seq   = len(signal) // N_tok
    decoded = []
    for pos in range(N_seq):
        s    = pos * N_tok
        feat = sc.transform([np.abs(np.fft.rfft(signal[s:s+N_tok]))])
        decoded.append(int(mlp.predict(feat)[0]))
    return decoded

# ============================================================================
# RUN
# ============================================================================
print("=" * 65)
print("DYNAMIC TAKENS DENDRITES: SEQUENCE DECODING EXPERIMENT")
print("=" * 65)
print(f"Freqs: {FREQS} Hz | Noise: {NOISE} | Sequences: {N_TEST}")
print()

bank = TakensDendriteBank()

# Takens
np.random.seed(0)
tak_c = tak_sc = 0
for _ in range(N_TEST):
    seq = np.random.randint(0, len(FREQS), SEQ_LEN).tolist()
    sig = encode_sequence(seq)
    dec = bank.decode(sig)
    n_c = sum(1 for a,b in zip(dec, seq) if a==b)
    tak_c  += n_c
    tak_sc += (n_c == SEQ_LEN)
tak_acc = tak_c / (N_TEST * SEQ_LEN)
print(f"Takens bank  (0 params, 0 samples): {100*tak_acc:.1f}% token | {100*tak_sc/N_TEST:.1f}% sequence")

# FFT
np.random.seed(0)
fft_c = 0
for _ in range(N_TEST):
    seq = np.random.randint(0, len(FREQS), SEQ_LEN).tolist()
    sig = encode_sequence(seq)
    fft_c += sum(1 for a,b in zip(fft_peak_decode(sig), seq) if a==b)
fft_acc = fft_c / (N_TEST * SEQ_LEN)
print(f"FFT detector (0 params, 0 samples): {100*fft_acc:.1f}% token")
print()

# MLP sweep
print(f"{'N train':>8}  {'MLP %':>8}  {'±':>6}  {'vs Takens':>12}")
print("-" * 42)
mlp_results = []
last_mlp    = None
last_sc     = None
for n_train in TRAINING_SIZES:
    trial_accs = []
    for rep in range(5):
        mlp, sc = train_mlp(n_train, seed=rep*100)
        if mlp is None: continue
        np.random.seed(0)
        c = 0
        for _ in range(N_TEST):
            seq = np.random.randint(0, len(FREQS), SEQ_LEN).tolist()
            sig = encode_sequence(seq)
            c  += sum(1 for a,b in zip(mlp_decode(sig, mlp, sc), seq) if a==b)
        trial_accs.append(c / (N_TEST * SEQ_LEN))
        last_mlp, last_sc = mlp, sc  # keep last trained model
    if not trial_accs: continue
    ma, ms = np.mean(trial_accs), np.std(trial_accs)
    # Use the last successfully trained mlp for param count
    valid_mlp = next((m for m,s in [train_mlp(n_train, seed=99)] if m is not None), None)
    n_par  = sum(w.size for w in valid_mlp.coefs_) + sum(b.size for b in valid_mlp.intercepts_) if valid_mlp else 0
    wins   = 'MLP wins' if ma > tak_acc else 'Takens wins'
    print(f"{n_train:>8}  {100*ma:>7.1f}%  {100*ms:>5.1f}%  {wins}")
    mlp_results.append((n_train, ma, ms, n_par))

crossover = next((r[0] for r in mlp_results if r[1] > tak_acc), None)
print()
print(f"Crossover point: ~{crossover} training samples" if crossover else "No crossover found")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nGenerating visualization...")

DARK  = '#0a0a14'; BG = '#0c0c18'; GREEN = '#44ffaa'
RED   = '#ff4466'; ORG = '#ffaa44'; BLUE = '#4488ff'
WHITE = '#e8e0d8'; GRAY = '#aaaaaa'
TOK_COLORS = [GREEN, BLUE, ORG, RED, '#cc88ff']

def sax(ax):
    ax.set_facecolor(DARK)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['bottom','left']: ax.spines[s].set_color('#333')
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.xaxis.label.set_color(GRAY); ax.yaxis.label.set_color(GRAY)

fig = plt.figure(figsize=(16, 20), facecolor=BG)
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

# --- Row 0: example signal ---
ax0 = fig.add_subplot(gs[0, :])
sax(ax0)
np.random.seed(7)
ex_seq = [0, 2, 4, 1, 3, 2]
ex_sig = encode_sequence(ex_seq, noise=NOISE)
t_ms   = np.arange(len(ex_sig)) / FS * 1000
ax0.plot(t_ms, ex_sig, color=WHITE, lw=0.5, alpha=0.5, label='Mixed signal')
for i, tid in enumerate(ex_seq):
    s = i * N_PER_TOKEN
    e = s + N_PER_TOKEN
    ax0.axvspan(t_ms[s], t_ms[e-1], alpha=0.10, color=TOK_COLORS[tid])
    ax0.text((t_ms[s]+t_ms[e-1])/2, 2.0, f'{FREQS[tid]:.0f}Hz',
             ha='center', va='top', color=TOK_COLORS[tid], fontsize=9, fontweight='bold')
ax0.set_title(f'Example input signal: tokens {ex_seq}  (noise={NOISE})', color=WHITE, fontsize=10)
ax0.set_xlabel('Time (ms)', fontsize=9); ax0.set_xlim(0, t_ms[-1])

# --- Row 1: resonance maps ---
ax1 = fig.add_subplot(gs[1, :])
sax(ax1)
vis_bank = TakensDendriteBank()
kernel   = np.ones(15)/15
for i, (freq, dend) in enumerate(zip(FREQS, vis_bank.dendrites)):
    raw  = dend.process(ex_sig)
    sm   = np.convolve(raw, kernel, mode='same')
    norm = sm / (sm.max() + 1e-8)
    ax1.plot(t_ms, norm + i*1.1, color=TOK_COLORS[i], lw=1.2, alpha=0.85,
             label=f'{freq:.0f}Hz')
    for j, tid in enumerate(ex_seq):
        s = j * N_PER_TOKEN
        e = s + N_PER_TOKEN
        ax1.axvspan(t_ms[s], t_ms[e-1], alpha=0.04, color=TOK_COLORS[tid])
ax1.set_title('Takens Dendrite Resonance Maps (each trace = one dendrite, offset vertically)',
              color=WHITE, fontsize=10)
ax1.set_xlabel('Time (ms)', fontsize=9)
ax1.set_ylabel('Resonance power (normalised, offset)', fontsize=9)
ax1.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=8,
           loc='upper right', ncol=5)
ax1.set_xlim(0, t_ms[-1])

# --- Row 2: accuracy vs training size ---
ax2 = fig.add_subplot(gs[2, :])
sax(ax2)
if mlp_results:
    ns    = [r[0] for r in mlp_results]
    accs  = [r[1]*100 for r in mlp_results]
    stds  = [r[2]*100 for r in mlp_results]
    ax2.fill_between(ns, [a-s for a,s in zip(accs,stds)],
                        [a+s for a,s in zip(accs,stds)], color=BLUE, alpha=0.18)
    ax2.plot(ns, accs, color=BLUE, lw=2.5, marker='o', markersize=6,
             label=f'FFT + MLP (~{mlp_results[0][3]} params, trained)')
ax2.axhline(tak_acc*100, color=GREEN, lw=2.5, linestyle='--',
            label=f'Takens Dendrite Bank  ({tak_acc*100:.1f}%,  0 params,  0 samples)')
ax2.axhline(fft_acc*100, color=RED, lw=1.5, linestyle=':',
            label=f'FFT Peak Detector  ({fft_acc*100:.1f}%,  0 params,  0 samples)')
if crossover:
    ax2.axvline(crossover, color=ORG, lw=1.5, linestyle=':', alpha=0.8)
    ax2.text(crossover*1.05, 30, f'Crossover\n~{crossover} samples',
             color=ORG, fontsize=8)
ax2.set_xscale('log')
ax2.set_xlabel('MLP training set size (labeled tokens)', fontsize=9)
ax2.set_ylabel('Token accuracy (%)', fontsize=9)
ax2.set_ylim(0, 105)
ax2.set_title('Accuracy vs Training Data\n'
              'Takens encodes the temporal prior that an MLP needs ~50 examples to learn',
              color=WHITE, fontsize=10)
ax2.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=9)

# --- Row 3 left: phase space portraits ---
ax3 = fig.add_subplot(gs[3, 0])
sax(ax3)
t_orbit = np.linspace(0, 0.2, int(FS*0.2), endpoint=False)
for freq, col in zip([20.0, 65.0, 130.0], [GREEN, ORG, RED]):
    tau_p = max(1, int(FS/(4*freq)))
    sig_p = np.sin(2*np.pi*freq*t_orbit)
    ax3.plot(sig_p[tau_p:], sig_p[:-tau_p], color=col, lw=1.0, alpha=0.8,
             label=f'{freq:.0f}Hz')
ax3.set_title('Phase Space Portraits\nDistinct orbit shapes = distinguishable tokens',
              color=WHITE, fontsize=10)
ax3.set_xlabel('x(t)', fontsize=9); ax3.set_ylabel('x(t − τ)', fontsize=9)
ax3.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=9)

# --- Row 3 right: dendrite anatomy ---
ax4 = fig.add_subplot(gs[3, 1])
sax(ax4)
freq_ex = 40.0
tau_ex  = max(1, int(FS/(4*freq_ex)))
n_ex    = max(8, int(FS/freq_ex*2)//tau_ex)
t_tap   = np.arange(n_ex) * (tau_ex/FS) * 1000
mosaic  = np.cos(2*np.pi*freq_ex*np.arange(n_ex)*(tau_ex/FS))
mosaic /= np.linalg.norm(mosaic)
ax4.bar(t_tap, mosaic, width=t_tap[1]-t_tap[0] if len(t_tap)>1 else 1,
        color=BLUE, alpha=0.8, label='Receptor mosaic')
ax4.set_title(f'Dendrite Anatomy: 40Hz receptor\ntau={tau_ex} samples, n_taps={n_ex}',
              color=WHITE, fontsize=10)
ax4.set_xlabel('Tap delay (ms)', fontsize=9); ax4.set_ylabel('Mosaic weight', fontsize=9)
ax4.legend(facecolor='#111', edgecolor='#333', labelcolor=WHITE, fontsize=9)
ax4.axhline(0, color='#444', lw=0.6)

fig.suptitle(
    "Dynamic Takens Dendrites\n"
    "Sequence Processing Without Backpropagation: Geometry as Prior",
    color=WHITE, fontsize=12, y=0.999
)

out = '/mnt/user-data/outputs/dynamic_takens_dendrites.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out}")
print("Done.")
