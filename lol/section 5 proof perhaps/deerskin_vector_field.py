"""
DEERSKIN VECTOR FIELD SIMULATION
=================================
The Moiré surface as a vector field, sampled sharply by Takens neurons
and diffusely by the ephaptic field.

Key design decisions:
- Neurons live on a 2D cortical sheet (not 1D)
- Each neuron has a UNIQUE 2D receptor mosaic (small NxN patch with distinct geometry)
  Not a single rotation angle. Actual different shapes.
- The ephaptic field is the spatial superposition of all neurons' emissions
- Each neuron's emission has geometric structure (shaped by its mosaic)
- The Moiré = interference between the global field and each neuron's local mosaic
- Resonance = how well the local field matches the local geometry
- Firing updates the field. Field changes resonance. Loop closes.

What we measure:
- Scalar coherence: are neurons firing in phase? (Kuramoto-like)
- Geometric coherence: are neurons' Moiré patterns compatible? (genuinely new)
- The DIVERGENCE between these two after perturbation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
np.random.seed(SEED)

# Cortical sheet
SHEET_W, SHEET_H = 10, 10          # grid of neurons
N_NEURONS = SHEET_W * SHEET_H      # 100 neurons

# Receptor mosaic (each neuron's internal geometry)
MOSAIC_SIZE = 7                     # 7x7 pixel receptor mosaic per neuron
N_MOSAIC_TYPES = 5                  # 5 fundamentally different morphological types

# Ephaptic field resolution
FIELD_RES = 70                      # 70x70 pixel field (each neuron covers ~7x7 area)

# Simulation
N_TICKS = 300
DT = 0.05

# Coupling
EPHAPTIC_RANGE = 4.0               # Gaussian decay sigma for field coupling (wider = more interaction)
EPHAPTIC_STRENGTH = 0.4            # How much field influences neurons
INTERNAL_STRENGTH = 0.5            # How much internal geometry resists change
NOISE_LEVEL = 0.02                 # Thermal noise

# Trauma
TRAUMA_CENTER = (5, 5)
TRAUMA_RADIUS = 2.0
TRAUMA_START = 80
TRAUMA_END = 120
TRAUMA_STRENGTH = 2.0

# ============================================================
# GENERATE DISTINCT RECEPTOR MOSAICS
# ============================================================
def make_mosaic_templates():
    """
    Create genuinely different 2D geometric templates.
    These represent fundamentally different dendritic morphologies.
    Not rotations of the same thing - actually different shapes.
    """
    templates = []
    s = MOSAIC_SIZE
    x, y = np.meshgrid(np.linspace(-1, 1, s), np.linspace(-1, 1, s))
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Type 0: Radial star (stellate cell - isotropic)
    t0 = np.cos(4 * theta) * np.exp(-r)
    templates.append(t0 / (np.linalg.norm(t0) + 1e-10))
    
    # Type 1: Bipolar (elongated along one axis - chandelier cell)
    t1 = np.exp(-x**2 / 0.1) * np.cos(3 * np.pi * y) * np.exp(-y**2)
    templates.append(t1 / (np.linalg.norm(t1) + 1e-10))
    
    # Type 2: Asymmetric apical (pyramidal - strong one direction)
    t2 = np.exp(-((x-0.3)**2 + y**2) / 0.3) - 0.5 * np.exp(-((x+0.5)**2 + y**2) / 0.2)
    templates.append(t2 / (np.linalg.norm(t2) + 1e-10))
    
    # Type 3: Ring/annular (basket cell - surrounds but doesn't overlap center)
    t3 = np.exp(-(r - 0.6)**2 / 0.08) * np.cos(2 * theta)
    templates.append(t3 / (np.linalg.norm(t3) + 1e-10))
    
    # Type 4: Multi-lobe (complex branching - Martinotti cell)
    t4 = np.cos(3 * theta) * r * np.exp(-r**2 / 0.5) + 0.3 * np.sin(5 * theta) * np.exp(-r**2 / 0.3)
    templates.append(t4 / (np.linalg.norm(t4) + 1e-10))
    
    return templates

# ============================================================
# NEURON CLASS
# ============================================================
class DeerskinNeuron:
    def __init__(self, grid_x, grid_y, mosaic_type, mosaic_template):
        # Position on cortical sheet
        self.gx = grid_x
        self.gy = grid_y
        # Position in field coordinates
        self.fx = grid_x * MOSAIC_SIZE + MOSAIC_SIZE // 2
        self.fy = grid_y * MOSAIC_SIZE + MOSAIC_SIZE // 2
        
        # Internal geometry: the receptor mosaic
        self.mosaic_type = mosaic_type
        self.base_mosaic = mosaic_template.copy()
        
        # Dynamic state: phase (oscillatory) and gain (excitability)
        self.phase = np.random.uniform(0, 2 * np.pi)
        self.gain = 1.0
        self.natural_freq = 1.0 + 0.3 * np.random.randn()  # Hz, varies per neuron
        
        # Firing state
        self.firing_rate = 0.0
        self.resonance = 0.0  # How well local field matches mosaic
        
    def current_mosaic(self):
        """The mosaic modulated by current phase - this is what the neuron EMITS."""
        return self.base_mosaic * (1.0 + 0.5 * np.sin(self.phase)) * self.gain
    
    def compute_resonance(self, local_field_patch):
        """
        SHARP SAMPLING: project the local field through this neuron's geometry.
        This is the Takens dot product - the Moiré computation.
        """
        mosaic = self.base_mosaic
        # Ensure sizes match
        if local_field_patch.shape != mosaic.shape:
            return 0.0
        # Dot product = geometric projection of field onto receptor
        dot = np.sum(local_field_patch * mosaic)
        # Squared for phase-invariance (energy matching)
        self.resonance = dot ** 2 / (np.sum(mosaic**2) * np.sum(local_field_patch**2) + 1e-10)
        return self.resonance
    
    def update(self, resonance, diffuse_phase, dt):
        """
        Update neuron state based on:
        - resonance: sharp geometric match with local field
        - diffuse_phase: average phase from ephaptic neighborhood
        - internal dynamics: natural frequency
        """
        # Phase advances by natural frequency
        dphase = self.natural_freq * 2 * np.pi * dt
        
        # Ephaptic pull: diffuse field pulls phase (Kuramoto-like but modulated by resonance)
        phase_diff = diffuse_phase - self.phase
        # Wrap to [-pi, pi]
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        
        # KEY: coupling strength depends on GEOMETRIC resonance, not just proximity
        # High resonance = strong coupling (field matches my geometry, I listen to it)
        # Low resonance = weak coupling (field doesn't match, I ignore it)
        geometric_coupling = EPHAPTIC_STRENGTH * (0.1 + resonance)
        
        dphase += geometric_coupling * np.sin(phase_diff)
        
        # Noise
        dphase += NOISE_LEVEL * np.random.randn()
        
        self.phase += dphase
        self.phase = self.phase % (2 * np.pi)
        
        # Firing rate: baseline + resonance modulation (never goes to zero)
        baseline_rate = 0.15 + 0.1 * (0.5 + 0.5 * np.sin(self.phase))  # oscillatory baseline
        self.firing_rate = baseline_rate + 0.6 * resonance
        self.firing_rate = np.clip(self.firing_rate, 0.05, 2.0)

# ============================================================
# EPHAPTIC FIELD
# ============================================================
class EphapticField:
    def __init__(self, res=FIELD_RES):
        self.res = res
        self.field = np.zeros((res, res))
        self.neuron_contributions = {}  # store each neuron's contribution for exclusion
        
    def update_from_neurons(self, neurons):
        """
        Build the field from all neurons' emissions.
        Each neuron emits a field that extends well beyond its mosaic footprint.
        The mosaic modulates the angular pattern; distance provides decay.
        """
        self.field = np.zeros((self.res, self.res))
        self.neuron_contributions = {}
        
        # Precompute coordinate grids
        yy, xx = np.mgrid[0:self.res, 0:self.res]
        
        for n in neurons:
            # Distance from this neuron to every field point
            dx = xx - n.fx
            dy = yy - n.fy
            dist = np.sqrt(dx**2 + dy**2 + 0.1)
            
            # Gaussian decay envelope (extends ~3*EPHAPTIC_RANGE*MOSAIC_SIZE pixels)
            emission_range = EPHAPTIC_RANGE * MOSAIC_SIZE
            envelope = np.exp(-dist**2 / (2 * emission_range**2)) * n.firing_rate
            
            # Angular modulation from mosaic: what "shape" does this neuron emit?
            # Map field coordinates to mosaic coordinates
            ms = MOSAIC_SIZE
            half = ms // 2
            # Angle and radius from neuron center
            angle = np.arctan2(dy, dx)
            # Sample mosaic pattern at this angle (radial projection of geometry)
            # Map angle to mosaic row/col
            mx = np.clip(((np.cos(angle) + 1) / 2 * (ms - 1)).astype(int), 0, ms - 1)
            my = np.clip(((np.sin(angle) + 1) / 2 * (ms - 1)).astype(int), 0, ms - 1)
            
            # The emission pattern: envelope * mosaic-modulated angular pattern * phase
            mosaic_vals = n.base_mosaic[my, mx]
            phase_mod = 1.0 + 0.5 * np.sin(n.phase)
            
            contrib = envelope * mosaic_vals * phase_mod * n.gain
            
            self.neuron_contributions[id(n)] = contrib
            self.field += contrib
        
    def get_local_patch_excluding(self, neuron):
        """
        Extract the field patch at a neuron's location, EXCLUDING that neuron's
        own contribution. This is what the neuron actually senses from others.
        """
        # Field minus self-contribution
        self_contrib = self.neuron_contributions.get(id(neuron), np.zeros((self.res, self.res)))
        other_field = self.field - self_contrib
        
        ms = MOSAIC_SIZE
        half = ms // 2
        patch = np.zeros((ms, ms))
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                fx = neuron.fx + dx
                fy = neuron.fy + dy
                if 0 <= fx < self.res and 0 <= fy < self.res:
                    patch[dy + half, dx + half] = other_field[fy, fx]
        return patch
    
    def get_local_patch(self, neuron):
        """Extract the full field patch (including self) for visualization."""
        ms = MOSAIC_SIZE
        half = ms // 2
        patch = np.zeros((ms, ms))
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                fx = neuron.fx + dx
                fy = neuron.fy + dy
                if 0 <= fx < self.res and 0 <= fy < self.res:
                    patch[dy + half, dx + half] = self.field[fy, fx]
        return patch

    def get_diffuse_phase(self, neuron, neurons):
        """
        DIFFUSE SAMPLING: weighted average phase from the field neighborhood.
        This is what Kuramoto would use. It's the scalar approximation.
        """
        total_sin = 0.0
        total_cos = 0.0
        total_w = 0.0
        for other in neurons:
            if other is neuron:
                continue
            dist = np.sqrt((neuron.gx - other.gx)**2 + (neuron.gy - other.gy)**2)
            w = np.exp(-dist**2 / (2 * EPHAPTIC_RANGE**2)) * other.firing_rate
            total_sin += w * np.sin(other.phase)
            total_cos += w * np.cos(other.phase)
            total_w += w
        if total_w < 1e-10:
            return neuron.phase
        return np.arctan2(total_sin / total_w, total_cos / total_w)

# ============================================================
# MEASUREMENT
# ============================================================
def measure_scalar_coherence(neurons):
    """Kuramoto order parameter: how aligned are phases?"""
    phases = np.array([n.phase for n in neurons])
    r = np.abs(np.mean(np.exp(1j * phases)))
    return r

def measure_geometric_coherence(neurons, field):
    """
    Mean resonance across all neurons: how well does the field
    (from others) match each neuron's geometry?
    Uses the resonance already computed in the main loop.
    """
    return np.mean([n.resonance for n in neurons])

def measure_type_coherence(neurons, field):
    """
    Per-morphological-type mean resonance.
    """
    type_resonances = {}
    for n in neurons:
        if n.mosaic_type not in type_resonances:
            type_resonances[n.mosaic_type] = []
        type_resonances[n.mosaic_type].append(n.resonance)
    
    type_means = {}
    for t, vals in type_resonances.items():
        type_means[t] = np.mean(vals)
    return type_means

# ============================================================
# SIMULATION
# ============================================================
def run_simulation():
    print("=" * 60)
    print("DEERSKIN VECTOR FIELD SIMULATION")
    print("=" * 60)
    print(f"Cortical sheet: {SHEET_W}x{SHEET_H} = {N_NEURONS} neurons")
    print(f"Receptor mosaic: {MOSAIC_SIZE}x{MOSAIC_SIZE} pixels, {N_MOSAIC_TYPES} morphological types")
    print(f"Field resolution: {FIELD_RES}x{FIELD_RES}")
    print(f"Simulation: {N_TICKS} ticks, dt={DT}")
    print()
    
    # Create mosaic templates
    templates = make_mosaic_templates()
    print("Mosaic types created:")
    type_names = ["Stellate (radial)", "Chandelier (bipolar)", "Pyramidal (asymmetric)", 
                  "Basket (annular)", "Martinotti (multi-lobe)"]
    for i, name in enumerate(type_names):
        print(f"  Type {i}: {name}")
    
    # Create neurons with assigned morphologies
    neurons = []
    type_counts = [0] * N_MOSAIC_TYPES
    for gy in range(SHEET_H):
        for gx in range(SHEET_W):
            # Assign type: clustered but mixed (like real cortex)
            # Use spatial noise + bias to create heterogeneous distribution
            base_type = (gx // 3 + gy // 3) % N_MOSAIC_TYPES
            if np.random.random() < 0.3:  # 30% chance of random type
                mtype = np.random.randint(N_MOSAIC_TYPES)
            else:
                mtype = base_type
            
            # Add individual variation to the template (no two neurons identical)
            template = templates[mtype].copy()
            template += 0.1 * np.random.randn(*template.shape)
            template /= (np.linalg.norm(template) + 1e-10)
            
            n = DeerskinNeuron(gx, gy, mtype, template)
            neurons.append(n)
            type_counts[mtype] += 1
    
    print(f"\nType distribution: {type_counts}")
    
    # Initialize field
    field = EphapticField()
    
    # Initialize firing rates with a gentle push
    for n in neurons:
        n.firing_rate = 0.3 + 0.1 * np.random.randn()
        n.firing_rate = np.clip(n.firing_rate, 0.05, 1.0)
    
    # Recording arrays
    scalar_coherence_history = []
    geometric_coherence_history = []
    type_coherence_history = {t: [] for t in range(N_MOSAIC_TYPES)}
    mean_resonance_history = []
    field_energy_history = []
    phase_snapshots = []
    field_snapshots = []
    
    # Snapshot ticks
    snapshot_ticks = [0, 40, 79, 100, 140, 200, 299]
    
    print(f"\nRunning simulation...")
    print(f"  Trauma: center=({TRAUMA_CENTER}), radius={TRAUMA_RADIUS}")
    print(f"  Trauma window: ticks {TRAUMA_START}-{TRAUMA_END}")
    print()
    
    for tick in range(N_TICKS):
        # ---- TRAUMA ----
        if TRAUMA_START <= tick <= TRAUMA_END:
            for n in neurons:
                dist = np.sqrt((n.gx - TRAUMA_CENTER[0])**2 + 
                              (n.gy - TRAUMA_CENTER[1])**2)
                if dist < TRAUMA_RADIUS:
                    # Scramble phase and boost gain (excitotoxic)
                    trauma_intensity = TRAUMA_STRENGTH * (1 - dist / TRAUMA_RADIUS)
                    n.phase += trauma_intensity * np.random.randn() * DT
                    n.gain = 1.0 + trauma_intensity * 0.5
                elif dist < TRAUMA_RADIUS * 2:
                    # Penumbra: mild disruption
                    trauma_intensity = TRAUMA_STRENGTH * 0.3 * (1 - (dist - TRAUMA_RADIUS) / TRAUMA_RADIUS)
                    n.phase += trauma_intensity * 0.3 * np.random.randn() * DT
        else:
            # Gain recovers slowly outside trauma
            for n in neurons:
                n.gain = n.gain * 0.99 + 1.0 * 0.01
        
        # ---- UPDATE FIELD FROM ALL NEURONS ----
        field.update_from_neurons(neurons)
        
        # ---- EACH NEURON SAMPLES THE FIELD ----
        resonances = []
        for n in neurons:
            # Sharp sampling: geometric projection EXCLUDING self-contribution
            local_patch = field.get_local_patch_excluding(n)
            resonance = n.compute_resonance(local_patch)
            resonances.append(resonance)
            
            # Diffuse sampling: neighborhood phase average
            diffuse_phase = field.get_diffuse_phase(n, neurons)
            
            # Update neuron state
            n.update(resonance, diffuse_phase, DT)
        
        # ---- MEASUREMENTS ----
        sc = measure_scalar_coherence(neurons)
        gc = measure_geometric_coherence(neurons, field)
        tc = measure_type_coherence(neurons, field)
        
        scalar_coherence_history.append(sc)
        geometric_coherence_history.append(gc)
        mean_resonance_history.append(np.mean(resonances))
        field_energy_history.append(np.sum(field.field**2))
        
        for t in range(N_MOSAIC_TYPES):
            type_coherence_history[t].append(tc.get(t, 0.0))
        
        if tick in snapshot_ticks:
            phases = np.zeros((SHEET_H, SHEET_W))
            types = np.zeros((SHEET_H, SHEET_W))
            res_map = np.zeros((SHEET_H, SHEET_W))
            for n in neurons:
                phases[n.gy, n.gx] = n.phase
                types[n.gy, n.gx] = n.mosaic_type
                res_map[n.gy, n.gx] = n.resonance
            phase_snapshots.append((tick, phases.copy(), res_map.copy()))
            field_snapshots.append((tick, field.field.copy()))
        
        if tick % 50 == 0 or tick == N_TICKS - 1:
            status = "TRAUMA" if TRAUMA_START <= tick <= TRAUMA_END else "RECOVERY" if tick > TRAUMA_END else "BASELINE"
            print(f"  t={tick:3d} [{status:8s}] scalar={sc:.4f} geometric={gc:.6f} "
                  f"mean_res={np.mean(resonances):.6f} field_E={np.sum(field.field**2):.2f}")
    
    # ============================================================
    # ANALYSIS
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    sc_arr = np.array(scalar_coherence_history)
    gc_arr = np.array(geometric_coherence_history)
    
    # Phases
    baseline = slice(0, TRAUMA_START)
    trauma = slice(TRAUMA_START, TRAUMA_END)
    recovery = slice(TRAUMA_END + 50, N_TICKS)  # skip transient
    
    print(f"\nScalar coherence (Kuramoto):")
    print(f"  Baseline:  {np.mean(sc_arr[baseline]):.4f} ± {np.std(sc_arr[baseline]):.4f}")
    print(f"  Trauma:    {np.mean(sc_arr[trauma]):.4f} ± {np.std(sc_arr[trauma]):.4f}")
    print(f"  Recovery:  {np.mean(sc_arr[recovery]):.4f} ± {np.std(sc_arr[recovery]):.4f}")
    
    print(f"\nGeometric coherence (Moiré):")
    print(f"  Baseline:  {np.mean(gc_arr[baseline]):.6f} ± {np.std(gc_arr[baseline]):.6f}")
    print(f"  Trauma:    {np.mean(gc_arr[trauma]):.6f} ± {np.std(gc_arr[trauma]):.6f}")
    print(f"  Recovery:  {np.mean(gc_arr[recovery]):.6f} ± {np.std(gc_arr[recovery]):.6f}")
    
    # THE KEY MEASUREMENT: correlation between scalar and geometric
    if len(sc_arr[recovery]) > 5:
        # Normalize both to [0,1] range for fair comparison
        sc_norm = (sc_arr[recovery] - sc_arr[recovery].min()) / (sc_arr[recovery].max() - sc_arr[recovery].min() + 1e-10)
        gc_norm = (gc_arr[recovery] - gc_arr[recovery].min()) / (gc_arr[recovery].max() - gc_arr[recovery].min() + 1e-10)
        
        if np.std(sc_norm) > 1e-10 and np.std(gc_norm) > 1e-10:
            correlation = np.corrcoef(sc_norm, gc_norm)[0, 1]
        else:
            correlation = float('nan')
        
        divergence = np.mean(np.abs(sc_norm - gc_norm))
        
        print(f"\n*** KEY RESULT ***")
        print(f"  Scalar-Geometric correlation (recovery): r = {correlation:.4f}")
        print(f"  Mean divergence (recovery): {divergence:.4f}")
        
        if not np.isnan(correlation) and correlation < 0.90:
            print(f"  → SIGNIFICANT DIVERGENCE: geometry is doing independent work!")
        elif not np.isnan(correlation):
            print(f"  → Moderate correlation: geometry partially tracks phase")
    
    # Per-type analysis
    print(f"\nPer-morphological-type mean resonance (recovery phase):")
    for t in range(N_MOSAIC_TYPES):
        tc_arr = np.array(type_coherence_history[t])
        print(f"  Type {t} ({type_names[t]:25s}): {np.mean(tc_arr[recovery]):.6f}")
    
    # Type-specific divergence: do different morphologies respond differently to same field?
    type_recovery_means = []
    for t in range(N_MOSAIC_TYPES):
        tc_arr = np.array(type_coherence_history[t])
        type_recovery_means.append(np.mean(tc_arr[recovery]))
    type_spread = np.std(type_recovery_means)
    print(f"\n  Cross-type resonance spread: {type_spread:.6f}")
    print(f"  (Higher = different geometries compute differently in same field)")
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    print("\nGenerating visualization...")
    
    fig = plt.figure(figsize=(24, 28))
    fig.suptitle("DEERSKIN VECTOR FIELD SIMULATION\n"
                 "Moiré Interference as Spatially Integrated Computation",
                 fontsize=16, fontweight='bold', y=0.98)
    
    gs = GridSpec(6, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # Row 0: The 5 mosaic templates
    for i in range(N_MOSAIC_TYPES):
        ax = fig.add_subplot(gs[0, i] if i < 4 else gs[1, 0])
        ax.imshow(templates[i], cmap='RdBu_r', interpolation='nearest')
        ax.set_title(f"Type {i}: {type_names[i]}", fontsize=8)
        ax.axis('off')
    
    # Row 1: Coherence time series
    ax1 = fig.add_subplot(gs[1, 1:4])
    t_axis = np.arange(N_TICKS)
    ax1.plot(t_axis, sc_arr, 'b-', alpha=0.8, linewidth=1.5, label='Scalar (Kuramoto)')
    ax1.axvspan(TRAUMA_START, TRAUMA_END, alpha=0.3, color='red', label='Trauma')
    ax1.set_ylabel('Scalar Coherence', color='b')
    ax1.set_xlabel('Tick')
    ax1.set_title('Scalar vs Geometric Coherence Over Time')
    ax1.legend(loc='upper left', fontsize=8)
    
    ax1b = ax1.twinx()
    ax1b.plot(t_axis, gc_arr, 'g-', alpha=0.8, linewidth=1.5, label='Geometric (Moiré)')
    ax1b.set_ylabel('Geometric Coherence', color='g')
    ax1b.legend(loc='upper right', fontsize=8)
    
    # Row 2: Field snapshots
    snap_indices = [0, 2, 3, 5]  # pre, trauma-start, mid-trauma, late recovery
    for idx, si in enumerate(snap_indices):
        if si < len(field_snapshots):
            tick_val, fld = field_snapshots[si]
            ax = fig.add_subplot(gs[2, idx])
            im = ax.imshow(fld, cmap='inferno', interpolation='bilinear')
            status = "BASELINE" if tick_val < TRAUMA_START else "TRAUMA" if tick_val <= TRAUMA_END else "RECOVERY"
            ax.set_title(f"Field t={tick_val} [{status}]", fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 3: Phase maps at same snapshots
    for idx, si in enumerate(snap_indices):
        if si < len(phase_snapshots):
            tick_val, phases, res_map = phase_snapshots[si]
            ax = fig.add_subplot(gs[3, idx])
            im = ax.imshow(phases, cmap='hsv', interpolation='nearest', vmin=0, vmax=2*np.pi)
            ax.set_title(f"Phase t={tick_val}", fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 4: Resonance maps at same snapshots
    for idx, si in enumerate(snap_indices):
        if si < len(phase_snapshots):
            tick_val, phases, res_map = phase_snapshots[si]
            ax = fig.add_subplot(gs[4, idx])
            im = ax.imshow(res_map, cmap='magma', interpolation='nearest')
            ax.set_title(f"Resonance t={tick_val}", fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 5: Per-type coherence evolution + summary
    ax5a = fig.add_subplot(gs[5, 0:2])
    colors_type = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for t in range(N_MOSAIC_TYPES):
        tc_arr = np.array(type_coherence_history[t])
        ax5a.plot(t_axis, tc_arr, color=colors_type[t], alpha=0.7, linewidth=1.2,
                  label=f"T{t}: {type_names[t][:12]}")
    ax5a.axvspan(TRAUMA_START, TRAUMA_END, alpha=0.2, color='red')
    ax5a.set_xlabel('Tick')
    ax5a.set_ylabel('Mean Resonance')
    ax5a.set_title('Per-Type Resonance: Different Geometries, Different Responses')
    ax5a.legend(fontsize=6, loc='lower right')
    
    # Summary text box
    ax5b = fig.add_subplot(gs[5, 2:4])
    ax5b.axis('off')
    
    summary_lines = [
        "DEERSKIN VECTOR FIELD: KEY RESULTS",
        "=" * 40,
        f"Neurons: {N_NEURONS} on {SHEET_W}×{SHEET_H} sheet",
        f"Mosaic types: {N_MOSAIC_TYPES} distinct morphologies",
        f"Mosaic size: {MOSAIC_SIZE}×{MOSAIC_SIZE} pixels",
        "",
        "COHERENCE MEASUREMENTS (Recovery Phase):",
        f"  Scalar (Kuramoto):  {np.mean(sc_arr[recovery]):.4f}",
        f"  Geometric (Moiré):  {np.mean(gc_arr[recovery]):.6f}",
    ]
    
    if not np.isnan(correlation):
        summary_lines.append(f"  Correlation:        r = {correlation:.4f}")
        summary_lines.append(f"  Divergence:         {divergence:.4f}")
    
    summary_lines += [
        "",
        "PER-TYPE RESONANCE (Recovery):",
    ]
    for t in range(N_MOSAIC_TYPES):
        tc_arr = np.array(type_coherence_history[t])
        summary_lines.append(f"  {type_names[t]:25s}: {np.mean(tc_arr[recovery]):.6f}")
    
    summary_lines += [
        f"  Cross-type spread: {type_spread:.6f}",
        "",
        "INTERPRETATION:",
        "Different geometries = different field response",
        "Same field, different Moiré = irreducible to phase",
    ]
    
    ax5b.text(0.05, 0.95, "\n".join(summary_lines), transform=ax5b.transAxes,
              fontsize=9, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig('/home/claude/deerskin_vector_field.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    
    print("Saved: deerskin_vector_field.png")
    
    # Also save the Moiré visualization: field * mosaic at one location
    fig2, axes2 = plt.subplots(N_MOSAIC_TYPES, 4, figsize=(16, 4 * N_MOSAIC_TYPES))
    fig2.suptitle("MOIRÉ COMPUTATION: Same Field × Different Geometries = Different Responses",
                  fontsize=14, fontweight='bold')
    
    # Use the final field state
    final_field = field.field.copy()
    
    for t in range(N_MOSAIC_TYPES):
        # Find a neuron of this type
        neuron = None
        for n in neurons:
            if n.mosaic_type == t:
                neuron = n
                break
        if neuron is None:
            continue
        
        # Get field patch at this neuron's location
        patch = field.get_local_patch(neuron)
        mosaic = neuron.base_mosaic
        
        # Moiré = element-wise product (interference)
        moire = patch * mosaic
        
        axes2[t, 0].imshow(patch, cmap='inferno', interpolation='bilinear')
        axes2[t, 0].set_title(f"Local Field", fontsize=9)
        axes2[t, 0].axis('off')
        
        axes2[t, 1].imshow(mosaic, cmap='RdBu_r', interpolation='nearest')
        axes2[t, 1].set_title(f"Type {t}: {type_names[t]}", fontsize=9)
        axes2[t, 1].axis('off')
        
        axes2[t, 2].imshow(moire, cmap='RdBu_r', interpolation='nearest')
        axes2[t, 2].set_title(f"Moiré (Field × Mosaic)", fontsize=9)
        axes2[t, 2].axis('off')
        
        res = neuron.resonance
        axes2[t, 3].bar([0], [res], color=colors_type[t])
        axes2[t, 3].set_xlim(-0.5, 0.5)
        axes2[t, 3].set_ylim(0, max(0.5, res * 1.3))
        axes2[t, 3].set_title(f"Resonance: {res:.4f}", fontsize=9)
        axes2[t, 3].set_xticks([])
    
    plt.tight_layout()
    plt.savefig('/home/claude/deerskin_moire_detail.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    
    print("Saved: deerskin_moire_detail.png")
    print("\nDone.")
    
    return {
        'correlation': float(correlation) if not np.isnan(correlation) else None,
        'divergence': float(divergence),
        'type_spread': float(type_spread),
        'scalar_recovery': float(np.mean(sc_arr[recovery])),
        'geometric_recovery': float(np.mean(gc_arr[recovery])),
    }

if __name__ == "__main__":
    results = run_simulation()
