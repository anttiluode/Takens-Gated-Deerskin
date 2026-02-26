"""
takens_gated_deerskin.py
========================
The Biological Alternative to "Attention Is All You Need".

Demonstrates how a Takens Delay-Embedding (Dendrite) combined with 
an Exact Theta Gate (Soma) performs selective attention through 
pure physics (Resonance and Phase-Locking), without any weight matrices.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. THE TAKENS-GATED DEERSKIN UNIT
# ============================================================================

class TakensDendrite:
    """
    Expands a 1D scalar signal into a high-dimensional geometric trajectory 
    using delay lines, then filters it through a spatial Receptor Mosaic.
    """
    def __init__(self, n_taps=16, tau=3, target_freq=40.0, fs=1000.0):
        self.n_taps = n_taps
        self.tau = tau
        self.buffer = np.zeros(n_taps * tau)
        
        # The Receptor Mosaic (The Deerskin "Skin")
        # We physically shape the receptor to geometrically match a specific 
        # frequency's phase-space trajectory (e.g., 40 Hz Gamma).
        t_taps = np.arange(n_taps) * (tau / fs)
        self.mosaic = np.cos(2 * np.pi * target_freq * t_taps)
        # Normalize the mosaic
        self.mosaic /= np.sum(np.abs(self.mosaic))

    def step(self, x):
        # Shift buffer and insert new value (The Delay Line)
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = x
        
        # Extract the Takens Vector (Sampling the delay line)
        takens_vector = self.buffer[::self.tau]
        
        # Moiré Interference: Dot product of Takens geometry and Receptor Mosaic
        resonance = np.sum(takens_vector * self.mosaic)
        return resonance, takens_vector

class ThetaSoma:
    """
    The exact, rigid pacemaker. 
    Only allows information to pass if it arrives during the positive peak.
    """
    def __init__(self, theta_freq=6.0, fs=1000.0):
        self.theta_freq = theta_freq
        self.fs = fs
        self.time_step = 0

    def get_gate(self, phase_shift):
        # Calculate current time
        t = self.time_step / self.fs
        
        # The Exact Theta Wave
        theta_wave = np.sin(2 * np.pi * self.theta_freq * t + phase_shift)
        
        # The Gate: Half-wave rectified (only passes signal when > 0)
        gate = np.maximum(0, theta_wave)
        
        self.time_step += 1
        return gate

# ============================================================================
# 2. THE EXPERIMENT
# ============================================================================
print("Building the physical environment...")
fs = 1000.0
duration = 1.0
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Create Environment: Bursts of Target (40Hz) and Distractor (65Hz)
# They alternate in time.
envelope_A = np.maximum(0, np.sin(2 * np.pi * 6 * t))          # Target arrives
envelope_B = np.maximum(0, np.sin(2 * np.pi * 6 * t + np.pi))  # Distractor arrives

target_signal = np.sin(2 * np.pi * 40 * t) * envelope_A
distract_signal = np.sin(2 * np.pi * 65 * t) * envelope_B

# The raw mixed input entering the brain
mixed_input = target_signal + distract_signal + np.random.normal(0, 0.2, len(t))

print("Running Takens-Gated Deerskin...")

# Initialize our biological hardware
dendrite = TakensDendrite(n_taps=16, tau=4, target_freq=40.0, fs=fs)
soma = ThetaSoma(theta_freq=6.0, fs=fs)

# Arrays to store results
resonances =[]
outputs_focused_on_A = []
outputs_focused_on_B = []
takens_x =[]
takens_y =[]

# SIMULATION LOOP
for i in range(len(t)):
    # 1. Dendritic Processing (Takens Delay + Mosaic)
    res, takens_vec = dendrite.step(mixed_input[i])
    resonances.append(res)
    
    # Store the first two taps for the Phase Space Plot (like your GUI node)
    takens_x.append(takens_vec[0])
    takens_y.append(takens_vec[2]) # Delayed by 2 * tau
    
    # 2. Attention State 1: Phase aligned to Target A (Phase = 0)
    gate_A = soma.get_gate(phase_shift=0.0)
    outputs_focused_on_A.append(res * gate_A)
    
    # 3. Attention State 2: Phase shifted to Distractor B (Phase = Pi)
    # Notice we rewind the soma clock so we can compare the EXACT same timestep
    soma.time_step -= 1 
    gate_B = soma.get_gate(phase_shift=np.pi)
    outputs_focused_on_B.append(res * gate_B)

# ============================================================================
# 3. VISUALIZE THE PHYSICS
# ============================================================================
print("Rendering the physics...")
fig = plt.figure(figsize=(14, 10), facecolor="#0c0c18")
plt.rcParams['text.color'] = '#e8e0d8'
plt.rcParams['axes.labelcolor'] = '#aaa'

# Plot 1: The Raw Environment
ax1 = plt.subplot(3, 2, (1, 2))
ax1.set_facecolor('#0a0a14')
ax1.plot(t, target_signal, color='#44ffaa', alpha=0.8, label='Target (40Hz)')
ax1.plot(t, distract_signal, color='#ff4466', alpha=0.8, label='Distractor (65Hz)')
ax1.plot(t, mixed_input, color='white', alpha=0.2, lw=1)
ax1.set_title("The Environment (Raw Mixed Signal + Noise)", fontsize=12)
ax1.legend(loc='upper right', facecolor='#0c0c18', edgecolor='#333')
ax1.set_xticks([])

# Plot 2: Takens Phase Space (Replicating your PhaseSpaceNode2)
ax2 = plt.subplot(3, 2, 3)
ax2.set_facecolor('#0a0a14')
# Plot just a segment of the phase space to see the strange attractor
ax2.plot(takens_x[200:300], takens_y[200:300], color='#ffaa44', lw=1.5, alpha=0.8)
ax2.set_title("Takens Dendrite Phase Space ($X_t$ vs $X_{t-2\\tau}$)", fontsize=10)
ax2.set_xlabel("Current Voltage")
ax2.set_ylabel("Delayed Voltage")
ax2.grid(color='#333', alpha=0.5)

# Plot 3: Moiré Resonance Output
ax3 = plt.subplot(3, 2, 4)
ax3.set_facecolor('#0a0a14')
ax3.plot(t, resonances, color='#4488bb', lw=1)
ax3.set_title("Dendritic Moiré Interference (Filtering out non-40Hz shapes)", fontsize=10)
ax3.set_xticks([])

# Plot 4: Gated Output (Focused on A)
ax4 = plt.subplot(3, 2, 5)
ax4.set_facecolor('#0a0a14')
theta_wave_A = np.maximum(0, np.sin(2 * np.pi * 6.0 * t))
ax4.fill_between(t, 0, theta_wave_A, color='#44ffaa', alpha=0.15, label="Theta Gate")
ax4.plot(t, outputs_focused_on_A, color='#44ffaa', lw=1.5)
ax4.set_title("Attention on Target (Theta Gate Phase = 0)", fontsize=10)

# Plot 5: Gated Output (Focused on B)
ax5 = plt.subplot(3, 2, 6)
ax5.set_facecolor('#0a0a14')
theta_wave_B = np.maximum(0, np.sin(2 * np.pi * 6.0 * t + np.pi))
ax5.fill_between(t, 0, theta_wave_B, color='#ff4466', alpha=0.15, label="Theta Gate")
ax5.plot(t, outputs_focused_on_B, color='#ff4466', lw=1.5)
ax5.set_title("Attention Shifted (Theta Gate Phase = $\pi$)", fontsize=10)

for ax in[ax1, ax2, ax3, ax4, ax5]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')

plt.tight_layout()
plt.show()