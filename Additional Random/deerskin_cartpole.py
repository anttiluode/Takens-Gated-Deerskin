"""
DEERSKIN CARTPOLE
=================
A morphologically diverse 2D cortical sheet, coupled only through
ephaptic fields, embodied in a cartpole balancing task.

Based on: Robbins et al. (2026) "Goal-directed learning in cortical organoids"

Architecture:
- 8x8 = 64 neurons with 5 morphological types
- No synaptic connections. Only ephaptic field coupling.
- 2 input neurons: receive pole angle as frequency-coded stimulation
- 2 output neurons: their firing rate difference = force on cart
- ~10 training neurons: RL selects pairs to stimulate at episode end
- Remaining neurons: the computational medium

The RL doesn't learn the task. The FIELD learns the task.
The RL just searches for which perturbations push the field toward
a Moiré fixed point that naturally balances the pole.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(42)

# ============================================================
# CARTPOLE PHYSICS (standard OpenAI gym formulation)
# ============================================================
class CartPole:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # half-pole length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.dt = 0.02  # 50 Hz physics
        self.theta_threshold = 16.0 * np.pi / 180  # ±16 degrees (from paper)
        self.x_threshold = 2.4
        self.reset()
    
    def reset(self):
        # Random init near vertical (from paper: random initialization)
        self.x = np.random.uniform(-0.05, 0.05)
        self.x_dot = np.random.uniform(-0.05, 0.05)
        self.theta = np.random.uniform(-0.05, 0.05)
        self.theta_dot = np.random.uniform(-0.05, 0.05)
        return self._state()
    
    def _state(self):
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot])
    
    def step(self, force):
        """Apply force, return (state, done)."""
        force = np.clip(force, -self.force_mag, self.force_mag)
        
        costheta = np.cos(self.theta)
        sintheta = np.sin(self.theta)
        
        temp = (force + self.polemass_length * self.theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # Euler integration
        self.x += self.dt * self.x_dot
        self.x_dot += self.dt * xacc
        self.theta += self.dt * self.theta_dot
        self.theta_dot += self.dt * thetaacc
        
        done = (abs(self.theta) > self.theta_threshold or abs(self.x) > self.x_threshold)
        return self._state(), done


# ============================================================
# DEERSKIN NEURON (simplified for speed, keeping geometry)
# ============================================================
MOSAIC_SIZE = 5  # Smaller for speed
N_TYPES = 5

def make_templates():
    s = MOSAIC_SIZE
    x, y = np.meshgrid(np.linspace(-1, 1, s), np.linspace(-1, 1, s))
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    templates = []
    t = np.cos(4 * theta) * np.exp(-r)
    templates.append(t / (np.linalg.norm(t) + 1e-10))
    t = np.exp(-x**2 / 0.1) * np.cos(3 * np.pi * y) * np.exp(-y**2)
    templates.append(t / (np.linalg.norm(t) + 1e-10))
    t = np.exp(-((x-0.3)**2 + y**2) / 0.3) - 0.5 * np.exp(-((x+0.5)**2 + y**2) / 0.2)
    templates.append(t / (np.linalg.norm(t) + 1e-10))
    t = np.exp(-(r - 0.6)**2 / 0.08) * np.cos(2 * theta)
    templates.append(t / (np.linalg.norm(t) + 1e-10))
    t = np.cos(3 * theta) * r * np.exp(-r**2 / 0.5) + 0.3 * np.sin(5 * theta) * np.exp(-r**2 / 0.3)
    templates.append(t / (np.linalg.norm(t) + 1e-10))
    return templates

TEMPLATES = make_templates()
TYPE_NAMES = ["Stellate", "Chandelier", "Pyramidal", "Basket", "Martinotti"]

class DeerskinNeuron:
    def __init__(self, gx, gy, mtype):
        self.gx, self.gy = gx, gy
        self.mosaic_type = mtype
        self.base_mosaic = TEMPLATES[mtype].copy()
        # Add individual variation
        self.base_mosaic += 0.08 * np.random.randn(*self.base_mosaic.shape)
        self.base_mosaic /= (np.linalg.norm(self.base_mosaic) + 1e-10)
        
        self.phase = np.random.uniform(0, 2 * np.pi)
        self.natural_freq = 1.0 + 0.2 * np.random.randn()
        self.firing_rate = 0.1
        self.resonance = 0.0
        self.gain = 1.0
        
        # Smoothed output for decoding
        self.smoothed_rate = 0.0
        
        # Role
        self.role = "medium"  # "input", "output", "training", "medium"


# ============================================================
# FAST EPHAPTIC FIELD (vectorized for speed)
# ============================================================
class FastField:
    """
    Instead of pixel-level field, compute neuron-to-neuron coupling
    directly through geometric compatibility × distance decay.
    
    This IS the Moiré computation: each neuron pair has a coupling
    strength that depends on (a) distance and (b) geometric overlap
    between their mosaics.
    """
    def __init__(self, neurons, sigma=3.0):
        N = len(neurons)
        self.N = N
        
        # Precompute distance matrix
        self.dist_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dx = neurons[i].gx - neurons[j].gx
                dy = neurons[i].gy - neurons[j].gy
                self.dist_matrix[i, j] = np.sqrt(dx**2 + dy**2)
        
        # Distance decay
        self.spatial_coupling = np.exp(-self.dist_matrix**2 / (2 * sigma**2))
        np.fill_diagonal(self.spatial_coupling, 0)  # no self-coupling
        
        # Precompute geometric compatibility matrix (Moiré overlap)
        # This is the KEY: how well does neuron i's mosaic align with neuron j's?
        self.geometric_compat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                mi = neurons[i].base_mosaic
                mj = neurons[j].base_mosaic
                # Moiré = dot product of flattened mosaics
                dot = np.sum(mi * mj)
                # Can be positive (constructive) or negative (destructive)
                self.geometric_compat[i, j] = dot
        
        # Full coupling: spatial × geometric
        self.coupling = self.spatial_coupling * np.abs(self.geometric_compat)
        
        # Normalize rows
        row_sums = self.coupling.sum(axis=1, keepdims=True)
        self.coupling_norm = self.coupling / (row_sums + 1e-10)
        
        # Sign of geometric interaction (constructive vs destructive)
        self.compat_sign = np.sign(self.geometric_compat)
    
    def compute_field_influence(self, neurons):
        """
        For each neuron, compute the weighted sum of other neurons'
        firing rates, modulated by geometric compatibility.
        
        Returns: field_influence[i] = sum_j coupling[i,j] * sign[i,j] * rate_j * sin(phase_j)
        This is the Moiré: the field at neuron i is the superposition of
        all other neurons' phase-modulated emissions, filtered through
        geometric compatibility.
        """
        N = self.N
        rates = np.array([n.firing_rate for n in neurons])
        phases = np.array([n.phase for n in neurons])
        gains = np.array([n.gain for n in neurons])
        
        # Each neuron emits: rate * gain * sin(phase)
        emissions = rates * gains * np.sin(phases)
        
        # Field at each neuron = weighted sum of compatible emissions
        field_vals = self.coupling_norm @ (self.compat_sign * emissions[:, None]).sum(axis=1)
        
        # Actually: for each neuron i, sum over j of coupling[i,j] * sign[i,j] * emission[j]
        signed_coupling = self.coupling_norm * self.compat_sign
        field_vals = signed_coupling @ emissions
        
        return field_vals
    
    def apply_training_plasticity(self, pair_indices, neurons, strength=0.1):
        """
        KEY MECHANISM: Training stimulation induces lasting changes in the
        geometric coupling matrix. This is the Deerskin analog of LTP/LTD.
        
        When neurons i,j are co-stimulated:
        - Their mutual geometric compatibility increases (LTP between them)
        - Their coupling to input/output neurons shifts (network reconfiguration)
        - The change is PERMANENT (doesn't decay back)
        
        This means the Moiré interference pattern changes shape.
        Different training pairs produce different field geometries.
        """
        i, j = pair_indices
        
        # Direct LTP: increase coupling between stimulated pair
        self.geometric_compat[i, j] += strength
        self.geometric_compat[j, i] += strength
        
        # Neighborhood spreading: neurons near the pair also shift
        for k in range(self.N):
            if k == i or k == j:
                continue
            # Spread to neighbors proportional to spatial proximity
            prox_i = self.spatial_coupling[k, i]
            prox_j = self.spatial_coupling[k, j]
            
            # Hebb-like: if k is close to both i and j, strengthen its coupling to both
            if prox_i > 0.1 and prox_j > 0.1:
                delta = strength * 0.3 * prox_i * prox_j
                self.geometric_compat[k, i] += delta
                self.geometric_compat[i, k] += delta
                self.geometric_compat[k, j] += delta
                self.geometric_compat[j, k] += delta
        
        self.compat_sign = np.sign(self.geometric_compat)
    
    def compute_resonances(self, neurons, field_vals):
        """
        Each neuron's resonance = how well the local field matches its geometry.
        Neurons with high geometric compatibility with their active neighbors
        will resonate strongly.
        """
        for i, n in enumerate(neurons):
            n.resonance = np.clip(field_vals[i]**2, 0, 1)


# ============================================================
# RATE CODING (from organoid paper)
# ============================================================
def angle_to_rate(theta, max_angle=0.28):
    """
    Encode pole angle as stimulation rate for input neurons.
    Input 1: excited by positive angle (pole tilting right)
    Input 2: excited by negative angle (pole tilting left)
    Returns (rate1, rate2) in [0, 1]
    """
    # Normalize angle to [-1, 1]
    norm_angle = np.clip(theta / max_angle, -1, 1)
    
    # Sigmoid-like encoding (from paper: rate-coding scheme)
    rate1 = 1.0 / (1.0 + np.exp(-5 * norm_angle))   # high when tilting right
    rate2 = 1.0 / (1.0 + np.exp(5 * norm_angle))     # high when tilting left
    return rate1, rate2

def rates_to_force(rate_left, rate_right, force_mag=10.0):
    """
    Decode force from output neuron firing rates.
    Force = force_mag * (rate_right - rate_left)
    (from paper: smoothed spike count difference)
    """
    return force_mag * (rate_right - rate_left)


# ============================================================
# RL TRAINING (eligibility trace, from organoid paper)
# ============================================================
class AdaptiveTrainer:
    """
    Reinforcement learning with eligibility traces for selecting
    training neuron pairs (from Robbins et al. 2026).
    
    Maintains value estimates for each possible training pair.
    After each episode, if performance dropped, stimulate the
    highest-value pair. Update values based on improvement.
    """
    def __init__(self, training_indices, lr=0.1, decay=0.9, epsilon=0.15):
        self.training_indices = training_indices
        self.n_training = len(training_indices)
        
        # All possible pairs
        self.pairs = []
        for i in range(self.n_training):
            for j in range(i+1, self.n_training):
                self.pairs.append((training_indices[i], training_indices[j]))
        
        self.n_pairs = len(self.pairs)
        self.values = np.zeros(self.n_pairs)
        self.eligibility = np.zeros(self.n_pairs)
        
        self.lr = lr
        self.decay = decay
        self.epsilon = epsilon
        
        self.last_selected = None
        self.episode_history = []
        self.selection_history = []
    
    def select_pair(self):
        """Select training pair using epsilon-greedy on value estimates."""
        if np.random.random() < self.epsilon or np.all(self.values == 0):
            idx = np.random.randint(self.n_pairs)
        else:
            # Softmax selection weighted by values
            probs = np.exp(self.values - self.values.max())
            probs /= probs.sum()
            idx = np.random.choice(self.n_pairs, p=probs)
        
        self.last_selected = idx
        return self.pairs[idx]
    
    def update(self, duration, prev_duration):
        """Update value estimates based on performance change."""
        reward = duration - prev_duration  # positive if improved
        
        if self.last_selected is not None:
            # Eligibility trace update
            self.eligibility *= self.decay
            self.eligibility[self.last_selected] = 1.0
            
            # TD-like update
            self.values += self.lr * reward * self.eligibility
        
        self.episode_history.append(duration)
    
    def should_train(self):
        """Train when 5-episode mean doesn't exceed 20-episode mean."""
        if len(self.episode_history) < 5:
            return True
        recent_5 = np.mean(self.episode_history[-5:])
        recent_20 = np.mean(self.episode_history[-min(20, len(self.episode_history)):])
        return recent_5 <= recent_20


# ============================================================
# MAIN SIMULATION
# ============================================================
def run_deerskin_cartpole():
    print("=" * 60)
    print("DEERSKIN CARTPOLE")
    print("Morphological Computation via Ephaptic Field")
    print("=" * 60)
    
    # Build cortical sheet
    SHEET_W, SHEET_H = 8, 8
    neurons = []
    for gy in range(SHEET_H):
        for gx in range(SHEET_W):
            mtype = (gx + gy * 3) % N_TYPES
            if np.random.random() < 0.25:
                mtype = np.random.randint(N_TYPES)
            neurons.append(DeerskinNeuron(gx, gy, mtype))
    
    N = len(neurons)
    print(f"Neurons: {N} ({SHEET_W}x{SHEET_H})")
    type_counts = [0] * N_TYPES
    for n in neurons:
        type_counts[n.mosaic_type] += 1
    for t in range(N_TYPES):
        print(f"  {TYPE_NAMES[t]:15s}: {type_counts[t]}")
    
    # Assign roles (from paper: select based on position/connectivity)
    # Input neurons: corners (far from each other for diverse field influence)
    input_indices = [0, SHEET_W - 1]  # top-left, top-right
    # Output neurons: opposite corners
    output_indices = [N - SHEET_W, N - 1]  # bottom-left, bottom-right
    # Training neurons: ring around center
    training_indices = []
    for i, n in enumerate(neurons):
        dist_to_center = np.sqrt((n.gx - SHEET_W/2)**2 + (n.gy - SHEET_H/2)**2)
        if 1.5 < dist_to_center < 3.5 and i not in input_indices + output_indices:
            training_indices.append(i)
    training_indices = training_indices[:10]  # cap at 10
    
    for i in input_indices:
        neurons[i].role = "input"
    for i in output_indices:
        neurons[i].role = "output"
    for i in training_indices:
        neurons[i].role = "training"
    
    print(f"\nInput neurons: {input_indices} (types: {[TYPE_NAMES[neurons[i].mosaic_type] for i in input_indices]})")
    print(f"Output neurons: {output_indices} (types: {[TYPE_NAMES[neurons[i].mosaic_type] for i in output_indices]})")
    print(f"Training neurons: {len(training_indices)}")
    
    # Build ephaptic field
    field = FastField(neurons, sigma=3.0)
    print(f"\nGeometric compatibility matrix built.")
    print(f"  Max coupling: {field.coupling.max():.4f}")
    print(f"  Mean coupling: {field.coupling.mean():.6f}")
    
    # Initialize RL trainer
    trainer = AdaptiveTrainer(training_indices)
    print(f"Training pairs: {trainer.n_pairs}")
    
    # Cartpole
    env = CartPole()
    
    # Simulation parameters
    N_EPISODES = 300
    DT_NEURAL = 0.02  # neural update rate = physics rate
    EPHAPTIC_STRENGTH = 0.3
    STIM_STRENGTH = 0.8  # training pulse strength
    
    # Recording
    episode_durations = []
    episode_conditions = []  # 'null', 'adaptive'
    force_history = []
    
    # Run three conditions like the paper
    conditions = [
        ("null", 100),       # 100 episodes, no training
        ("adaptive", 200),   # 200 episodes with RL training
    ]
    
    total_ep = 0
    
    for condition_name, n_eps in conditions:
        print(f"\n{'='*40}")
        print(f"CONDITION: {condition_name.upper()} ({n_eps} episodes)")
        print(f"{'='*40}")
        
        if condition_name == "adaptive":
            trainer = AdaptiveTrainer(training_indices)
        
        for ep in range(n_eps):
            state = env.reset()
            
            # Reset neural state mildly (not fully — persistence matters)
            for n in neurons:
                n.phase += 0.1 * np.random.randn()
                n.phase %= (2 * np.pi)
                n.smoothed_rate *= 0.5  # partial reset
            
            done = False
            steps = 0
            ep_forces = []
            
            while not done and steps < 5000:  # max 100 seconds
                # --- ENCODE: pole angle → input neuron stimulation ---
                theta = state[2]
                rate1, rate2 = angle_to_rate(theta)
                
                # Stimulate input neurons (override their firing rate)
                neurons[input_indices[0]].firing_rate = 0.1 + 0.9 * rate1
                neurons[input_indices[0]].gain = 1.0 + 0.5 * rate1
                neurons[input_indices[1]].firing_rate = 0.1 + 0.9 * rate2
                neurons[input_indices[1]].gain = 1.0 + 0.5 * rate2
                
                # --- EPHAPTIC FIELD UPDATE ---
                field_vals = field.compute_field_influence(neurons)
                field.compute_resonances(neurons, field_vals)
                
                # --- UPDATE ALL NEURONS ---
                for i, n in enumerate(neurons):
                    if n.role == "input":
                        # Input neurons: phase locked to input
                        n.phase += n.natural_freq * 2 * np.pi * DT_NEURAL
                        n.phase %= (2 * np.pi)
                        continue
                    
                    # Phase advance
                    dphase = n.natural_freq * 2 * np.pi * DT_NEURAL
                    
                    # Ephaptic coupling: field pulls neuron
                    # Coupling MODULATED BY RESONANCE (geometry matters!)
                    coupling = EPHAPTIC_STRENGTH * (0.05 + n.resonance)
                    dphase += coupling * field_vals[i]
                    
                    # Noise
                    dphase += 0.01 * np.random.randn()
                    
                    n.phase += dphase
                    n.phase %= (2 * np.pi)
                    
                    # Firing rate: baseline + field-driven
                    baseline = 0.1 + 0.1 * (0.5 + 0.5 * np.sin(n.phase))
                    n.firing_rate = baseline + 0.5 * n.resonance * n.gain
                    n.firing_rate = np.clip(n.firing_rate, 0.01, 2.0)
                    
                    # Smooth for decoding
                    n.smoothed_rate = 0.85 * n.smoothed_rate + 0.15 * n.firing_rate
                
                # --- DECODE: output neuron rates → force ---
                rate_out1 = neurons[output_indices[0]].smoothed_rate
                rate_out2 = neurons[output_indices[1]].smoothed_rate
                force = rates_to_force(rate_out1, rate_out2)
                ep_forces.append(force)
                
                # --- PHYSICS STEP ---
                state, done = env.step(force)
                steps += 1
            
            duration = steps * env.dt
            episode_durations.append(duration)
            episode_conditions.append(condition_name)
            
            # --- TRAINING (at episode end) ---
            if condition_name == "adaptive" and trainer.should_train():
                pair = trainer.select_pair()
                
                # Deliver training pulse: high-frequency stimulation of the pair
                # This perturbs their gain and phase AND induces lasting plasticity
                for stim_idx in pair:
                    neurons[stim_idx].gain *= (1.0 + STIM_STRENGTH)
                    neurons[stim_idx].firing_rate = min(2.0, neurons[stim_idx].firing_rate + STIM_STRENGTH)
                    neurons[stim_idx].phase += 0.5 * np.random.randn()
                
                # PERMANENT PLASTICITY: reshape the geometric coupling matrix
                # This is the Deerskin analog of LTP induced by co-stimulation
                field.apply_training_plasticity(pair, neurons, strength=0.08)
                
                # Let perturbation propagate for a few ticks
                for _ in range(10):
                    fv = field.compute_field_influence(neurons)
                    field.compute_resonances(neurons, fv)
                    for i, n in enumerate(neurons):
                        n.phase += n.natural_freq * 2 * np.pi * DT_NEURAL
                        coupling = EPHAPTIC_STRENGTH * (0.05 + n.resonance)
                        n.phase += coupling * fv[i]
                        n.phase %= (2 * np.pi)
                        baseline = 0.1 + 0.1 * (0.5 + 0.5 * np.sin(n.phase))
                        n.firing_rate = baseline + 0.5 * n.resonance * n.gain
                        n.firing_rate = np.clip(n.firing_rate, 0.01, 2.0)
                    
                    # Gain decays back toward 1
                    for n in neurons:
                        n.gain = 0.9 * n.gain + 0.1 * 1.0
                
                trainer.selection_history.append((total_ep, pair))
            
            prev_dur = episode_durations[-2] if len(episode_durations) > 1 else 0
            if condition_name == "adaptive":
                trainer.update(duration, prev_dur)
            
            total_ep += 1
            
            if (ep + 1) % 25 == 0:
                recent = episode_durations[-25:]
                p90 = np.percentile(recent, 90)
                print(f"  Ep {ep+1:3d}/{n_eps}: last={duration:.2f}s  "
                      f"mean25={np.mean(recent):.2f}s  p90={p90:.2f}s")
    
    # ============================================================
    # ANALYSIS
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    durations = np.array(episode_durations)
    conditions_arr = np.array(episode_conditions)
    
    null_durations = durations[conditions_arr == "null"]
    adaptive_durations = durations[conditions_arr == "adaptive"]
    
    print(f"\nNull condition ({len(null_durations)} episodes):")
    print(f"  Mean: {null_durations.mean():.2f}s")
    print(f"  P90:  {np.percentile(null_durations, 90):.2f}s")
    print(f"  Max:  {null_durations.max():.2f}s")
    
    print(f"\nAdaptive condition ({len(adaptive_durations)} episodes):")
    print(f"  Mean: {adaptive_durations.mean():.2f}s")
    print(f"  P90:  {np.percentile(adaptive_durations, 90):.2f}s")
    print(f"  Max:  {adaptive_durations.max():.2f}s")
    
    # Check for improvement over time in adaptive
    if len(adaptive_durations) >= 40:
        first_quarter = adaptive_durations[:len(adaptive_durations)//4]
        last_quarter = adaptive_durations[-len(adaptive_durations)//4:]
        print(f"\n  First quarter mean: {first_quarter.mean():.2f}s")
        print(f"  Last quarter mean:  {last_quarter.mean():.2f}s")
        improvement = last_quarter.mean() - first_quarter.mean()
        print(f"  Improvement: {improvement:+.2f}s")
        
        if improvement > 0:
            print(f"  → LEARNING DETECTED: performance improved over training")
        else:
            print(f"  → No clear learning trend (honest result)")
    
    # Compare conditions
    if len(null_durations) > 0 and len(adaptive_durations) > 0:
        diff = adaptive_durations.mean() - null_durations.mean()
        print(f"\n  Adaptive vs Null difference: {diff:+.2f}s")
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    print("\nGenerating visualization...")
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("DEERSKIN CARTPOLE\n"
                 "Morphological Computation Embodied in a Control Task",
                 fontsize=14, fontweight='bold')
    
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Episode duration over time
    ax1 = fig.add_subplot(gs[0, :])
    null_mask = conditions_arr == "null"
    adapt_mask = conditions_arr == "adaptive"
    
    ep_nums = np.arange(len(durations))
    ax1.scatter(ep_nums[null_mask], durations[null_mask], c='steelblue', s=10, alpha=0.5, label='Null')
    ax1.scatter(ep_nums[adapt_mask], durations[adapt_mask], c='forestgreen', s=10, alpha=0.5, label='Adaptive')
    
    # Smoothed trend
    window = 15
    if len(durations) > window:
        smoothed = np.convolve(durations, np.ones(window)/window, mode='valid')
        ax1.plot(np.arange(window-1, len(durations)), smoothed, 'k-', linewidth=2, alpha=0.8, label=f'{window}-ep smooth')
    
    ax1.axvline(x=len(null_durations), color='red', linestyle='--', alpha=0.5, label='Training starts')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Time Balanced (s)')
    ax1.set_title('Episode Duration Over Time')
    ax1.legend()
    
    # Distribution comparison
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.boxplot([null_durations, adaptive_durations], labels=['Null', 'Adaptive'])
    ax2.set_ylabel('Time Balanced (s)')
    ax2.set_title('Performance Distribution')
    
    # Rolling P90
    ax3 = fig.add_subplot(gs[1, 1:])
    cycle_size = 25
    null_p90s = []
    adapt_p90s = []
    for i in range(0, len(null_durations), cycle_size):
        chunk = null_durations[i:i+cycle_size]
        if len(chunk) >= 5:
            null_p90s.append(np.percentile(chunk, 90))
    for i in range(0, len(adaptive_durations), cycle_size):
        chunk = adaptive_durations[i:i+cycle_size]
        if len(chunk) >= 5:
            adapt_p90s.append(np.percentile(chunk, 90))
    
    if null_p90s:
        ax3.bar(np.arange(len(null_p90s)), null_p90s, color='steelblue', alpha=0.7, label='Null cycles')
    if adapt_p90s:
        ax3.bar(np.arange(len(null_p90s), len(null_p90s) + len(adapt_p90s)), adapt_p90s,
               color='forestgreen', alpha=0.7, label='Adaptive cycles')
    ax3.set_xlabel('Cycle (25 episodes)')
    ax3.set_ylabel('P90 Time Balanced (s)')
    ax3.set_title('Per-Cycle 90th Percentile Performance')
    ax3.legend()
    
    # Morphological layout
    ax4 = fig.add_subplot(gs[2, 0])
    colors_type = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for n in neurons:
        color = colors_type[n.mosaic_type]
        marker = 'o'
        size = 30
        if n.role == "input":
            marker = '^'
            size = 120
        elif n.role == "output":
            marker = 's'
            size = 120
        elif n.role == "training":
            marker = 'D'
            size = 60
        ax4.scatter(n.gx, n.gy, c=color, marker=marker, s=size, edgecolors='black', linewidth=0.5)
    
    # Legend
    for t in range(N_TYPES):
        ax4.scatter([], [], c=colors_type[t], label=TYPE_NAMES[t], s=30)
    ax4.scatter([], [], marker='^', c='gray', s=80, label='Input', edgecolors='black')
    ax4.scatter([], [], marker='s', c='gray', s=80, label='Output', edgecolors='black')
    ax4.scatter([], [], marker='D', c='gray', s=40, label='Training', edgecolors='black')
    ax4.legend(fontsize=6, loc='center left', bbox_to_anchor=(1, 0.5))
    ax4.set_title('Cortical Sheet Layout')
    ax4.set_xlim(-0.5, SHEET_W - 0.5)
    ax4.set_ylim(-0.5, SHEET_H - 0.5)
    ax4.set_aspect('equal')
    ax4.invert_yaxis()
    
    # Summary
    ax5 = fig.add_subplot(gs[2, 1:])
    ax5.axis('off')
    
    lines = [
        "DEERSKIN CARTPOLE: RESULTS",
        "=" * 40,
        f"Neurons: {N} ({SHEET_W}×{SHEET_H}), {N_TYPES} morphological types",
        f"Coupling: ephaptic field only (NO synapses)",
        f"Input: 2 neurons (angle → frequency coding)",
        f"Output: 2 neurons (rate difference → force)",
        f"Training: {len(training_indices)} neurons, {trainer.n_pairs} pairs",
        "",
        f"NULL ({len(null_durations)} eps):",
        f"  Mean: {null_durations.mean():.2f}s  P90: {np.percentile(null_durations, 90):.2f}s",
        "",
        f"ADAPTIVE ({len(adaptive_durations)} eps):",
        f"  Mean: {adaptive_durations.mean():.2f}s  P90: {np.percentile(adaptive_durations, 90):.2f}s",
    ]
    
    if len(adaptive_durations) >= 40:
        lines += [
            f"  First quarter: {first_quarter.mean():.2f}s",
            f"  Last quarter:  {last_quarter.mean():.2f}s",
            f"  Improvement:   {improvement:+.2f}s",
        ]
    
    lines += [
        "",
        "INTERPRETATION:",
        "The ephaptic field transforms input stimulation",
        "into output firing through Moiré interference.",
        "RL training perturbs the field geometry,",
        "searching for the standing wave that naturally",
        "maps pole angle to corrective force.",
    ]
    
    ax5.text(0.05, 0.95, "\n".join(lines), transform=ax5.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig('/home/claude/deerskin_cartpole.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("Saved: deerskin_cartpole.png")

if __name__ == "__main__":
    run_deerskin_cartpole()
