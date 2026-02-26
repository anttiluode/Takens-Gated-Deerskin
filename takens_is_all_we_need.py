"""
takens_is_all_we_need.py
========================
Replacing Transformer Attention (Q K^T V) with Biological Physics.

This script processes a continuous sequence of overlapping "tokens" 
(bursts of data) and selectively extracts specific tokens using 
only Takens delay embeddings (Keys) and Theta phase-shifting (Queries).
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. THE BIOLOGICAL HARDWARE
# ============================================================================

class TakensDendrite:
    """The 'Key' Mechanism: Filters incoming data by its topological shape."""
    def __init__(self, n_taps=20, tau=3, target_freq=40.0, fs=1000.0):
        self.n_taps = n_taps
        self.tau = tau
        self.buffer = np.zeros(n_taps * tau)
        
        # The Receptor Mosaic (The Deerskin Tuning Fork)
        t_taps = np.arange(n_taps) * (tau / fs)
        self.mosaic = np.cos(2 * np.pi * target_freq * t_taps)
        self.mosaic /= np.sum(np.abs(self.mosaic)) # Normalize

    def step(self, x):
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = x
        
        # Extract the Takens Vector (Delay Embedding)
        takens_vector = self.buffer[::self.tau]
        
        # Moiré Resonance (Topological Dot Product)
        resonance = np.sum(takens_vector * self.mosaic)
        return resonance

class ThetaSoma:
    """The 'Query' Mechanism: Filters data by its temporal arrival phase."""
    def __init__(self, theta_freq=2.0, fs=1000.0):
        self.theta_freq = theta_freq
        self.fs = fs
        self.time_step = 0

    def get_gate(self, query_phase):
        t = self.time_step / self.fs
        # The rigid pacemaker gate
        theta_wave = np.sin(2 * np.pi * self.theta_freq * t + query_phase)
        gate = np.maximum(0, theta_wave) # Only open on the positive peak
        self.time_step += 1
        return gate

# ============================================================================
# 2. THE SEQUENCE DECODING EXPERIMENT
# ============================================================================
fs = 1000.0
duration = 2.0 # 2 seconds of sequence data
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

print("Encoding the Sequence: [Word A] -> [Word B] -> [Word A] -> [Word B]")
# Token A (40Hz) pulses twice: at 0.25s and 1.25s
env_A = np.exp(-((t - 0.25)**2) / 0.005) + np.exp(-((t - 1.25)**2) / 0.005)
# Token B (65Hz) pulses twice: at 0.75s and 1.75s
env_B = np.exp(-((t - 0.75)**2) / 0.005) + np.exp(-((t - 1.75)**2) / 0.005)

word_A_signal = np.sin(2 * np.pi * 40 * t) * env_A
word_B_signal = np.sin(2 * np.pi * 65 * t) * env_B

# The Input Stream is a noisy, overlapping mess
input_stream = word_A_signal + word_B_signal + np.random.normal(0, 0.15, len(t))

# Initialize the Network (Tuned to Geometry A, 40Hz)
dendrite = TakensDendrite(target_freq=40.0, fs=fs)
soma = ThetaSoma(theta_freq=2.0, fs=fs) # 2Hz clock = 0.5s cycle

decoded_word_A = []
decoded_word_B = []

print("Running Takens-Gated Attention...")
for i in range(len(t)):
    # 1. The Key Match (Topological filtering via Takens Embedding)
    resonance = dendrite.step(input_stream[i])
    
    # 2. Query 1: Shift phase to decode Word A
    # Peak of sin(x) is at Pi/2. To peak at t=0.25s with 2Hz: phase = 0
    query_phase_A = 0.0
    gate_A = soma.get_gate(query_phase_A)
    decoded_word_A.append(resonance * gate_A)
    
    # 3. Query 2: Shift phase to decode Word B (ignoring Word A completely)
    # To peak at t=0.75s with 2Hz, we shift the phase by exactly Pi
    soma.time_step -= 1 # Rewind clock to compare the exact same timestep
    query_phase_B = np.pi 
    gate_B = soma.get_gate(query_phase_B)
    decoded_word_B.append(resonance * gate_B)

# ============================================================================
# 3. VISUALIZATION
# ============================================================================
fig, axs = plt.subplots(3, 1, figsize=(12, 10), facecolor="#0c0c18")
plt.rcParams['text.color'] = '#e8e0d8'

# Plot 1: The Input Sequence
axs[0].set_facecolor('#0a0a14')
axs[0].plot(t, input_stream, color='white', alpha=0.3, label="Noisy Sequence Stream")
axs[0].plot(t, word_A_signal, color='#44ffaa', alpha=0.8, label="Token A (40Hz)")
axs[0].plot(t, word_B_signal, color='#ff4466', alpha=0.8, label="Token B (65Hz)")
axs[0].set_title("Input Sequence: [ A ] --- [ B ] --- [ A ] --- [ B ]", color='white')
axs[0].legend(loc="upper right", facecolor="black")

# Plot 2: Querying Word A
axs[1].set_facecolor('#0a0a14')
theta_wave_A = np.maximum(0, np.sin(2 * np.pi * 2.0 * t + 0.0))
axs[1].fill_between(t, 0, theta_wave_A, color='#44ffaa', alpha=0.15, label="Query Phase A")
axs[1].plot(t, decoded_word_A, color='#44ffaa', lw=1.5)
axs[1].set_title("Attention Output: Decoding Token A (Query Phase = 0)", color='white')
axs[1].legend(loc="upper right", facecolor="black")

# Plot 3: Querying Word B (Notice it's silent, because the Dendrite Key filtered out 65Hz)
axs[2].set_facecolor('#0a0a14')
theta_wave_B = np.maximum(0, np.sin(2 * np.pi * 2.0 * t + np.pi))
axs[2].fill_between(t, 0, theta_wave_B, color='#ff4466', alpha=0.15, label="Query Phase B")
axs[2].plot(t, decoded_word_B, color='#ff4466', lw=1.5)
axs[2].set_title("Attention Output: Decoding Token B (Query Phase = $\pi$, Key Mismatch)", color='white')
axs[2].legend(loc="upper right", facecolor="black")

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.tick_params(colors='#aaa')

plt.tight_layout()
plt.show()