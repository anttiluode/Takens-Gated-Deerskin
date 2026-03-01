import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import threading
import time

class DeerskinNode:
    """
    A pure geometric neuron. No weights, no backprop.
    Computes via Takens delay embedding and Moiré interference.
    """
    def __init__(self, target_freq=40.0, fs=1000.0, tau=4, initial_taps=3):
        self.target_freq = target_freq
        self.fs = fs
        self.tau = tau          # Delay spacing
        self.n_taps = initial_taps  # Dendrite length (number of dimensions)
        self.max_taps = 40
        self.frustration = 1.0  # Drives growth
        
    def get_mosaic(self):
        """The genetic 'shape' - a fixed geometric template"""
        k = np.arange(self.n_taps)
        return np.cos(2 * np.pi * self.target_freq * k * self.tau / self.fs)

    def forward(self, signal_buffer):
        """Translates temporal signal into geometry and computes resonance"""
        # 1. Takens Embedding (create the spatial object)
        if len(signal_buffer) < (self.n_taps * self.tau):
            return 0.0, np.zeros(3)
            
        # Get the latest state vector of length n_taps
        taps = np.array([signal_buffer[-(i * self.tau + 1)] for i in range(self.n_taps)])
        
        # Normalize the vector (contrast invariance)
        std = np.std(taps) if np.std(taps) > 0 else 1.0
        taps_norm = (taps - np.mean(taps)) / std
        
        # 2. Moiré Interference (dot product with mosaic)
        mosaic = self.get_mosaic()
        resonance = np.dot(taps_norm, mosaic)
        
        # 3. Power (phase invariant)
        power = (resonance ** 2) / self.n_taps
        
        # Return power and the first 3 dims for 3D visualization
        viz_3d = taps_norm[:3] if len(taps_norm) >= 3 else np.zeros(3)
        return power, viz_3d

    def adapt(self, current_resonance):
        """Homeostatic frustration-driven growth. No gradient descent."""
        # Update frustration (moving average of error)
        target_resonance = 5.0 # Arbitrary homeostatic target
        error = max(0, target_resonance - current_resonance)
        self.frustration = 0.95 * self.frustration + 0.05 * error
        
        # If frustrated, grow the dendrite!
        if self.frustration > 2.0 and self.n_taps < self.max_taps:
            self.n_taps += 1
            self.frustration = 0.0 # Reset frustration after growth to give it time to test
            return True # Indicates growth happened
        return False

class DeerskinDemoUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deerskin Architecture: Zero-Weight Geometric Computation")
        
        self.fs = 1000.0
        self.neuron = DeerskinNode(target_freq=40.0, fs=self.fs, tau=4, initial_taps=3)
        
        self.running = False
        self.auto_adapt = tk.BooleanVar(value=False)
        self.noise_level = tk.DoubleVar(value=0.5)
        
        self.time_step = 0
        self.signal_history = np.zeros(500)
        self.resonance_history = np.zeros(500)
        self.phase_history = np.zeros((100, 3))
        self.target_state_history = np.zeros(500) # 1 if 40Hz, 0 if 65Hz
        
        self.setup_ui()
        self.start_simulation()

    def setup_ui(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side="top", fill="x")
        
        ttk.Label(control_frame, text="Deerskin Node Demo", font=("Arial", 16, "bold")).pack(side="left", padx=10)
        
        self.btn_start = ttk.Button(control_frame, text="Pause", command=self.toggle_run)
        self.btn_start.pack(side="left", padx=10)
        
        ttk.Checkbutton(control_frame, text="Auto-Adapt Dendrite (Growth)", variable=self.auto_adapt).pack(side="left", padx=10)
        
        ttk.Label(control_frame, text="Noise Level:").pack(side="left")
        ttk.Scale(control_frame, from_=0.0, to=2.0, variable=self.noise_level, orient="horizontal").pack(side="left", padx=5)
        
        self.lbl_taps = ttk.Label(control_frame, text="Dendrite Length (Taps): 3", font=("Arial", 12, "bold"), foreground="blue")
        self.lbl_taps.pack(side="right", padx=20)

        # Matplotlib Figures
        self.fig = plt.Figure(figsize=(12, 8), facecolor='#f0f0f0')
        self.ax_sig = self.fig.add_subplot(221)
        self.ax_res = self.fig.add_subplot(223)
        self.ax_3d = self.fig.add_subplot(122, projection='3d')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def toggle_run(self):
        self.running = not self.running
        self.btn_start.config(text="Pause" if self.running else "Resume")

    def simulation_loop(self):
        while True:
            if not self.running:
                time.sleep(0.1)
                continue

            # 1. Generate Environment Signal (Switch between Target 40Hz and Distractor 65Hz every 2 seconds)
            t = self.time_step / self.fs
            is_target = (self.time_step % 4000) < 2000
            freq = 40.0 if is_target else 65.0
            
            clean_signal = np.sin(2 * np.pi * freq * t)
            noise = np.random.normal(0, self.noise_level.get())
            signal_val = clean_signal + noise
            
            # Roll buffers
            self.signal_history = np.roll(self.signal_history, -1)
            self.signal_history[-1] = signal_val
            
            self.target_state_history = np.roll(self.target_state_history, -1)
            self.target_state_history[-1] = 1.0 if is_target else 0.0

            # 2. Deerskin Computation
            resonance, viz_3d = self.neuron.forward(self.signal_history)
            
            self.resonance_history = np.roll(self.resonance_history, -1)
            self.resonance_history[-1] = resonance
            
            self.phase_history = np.roll(self.phase_history, -1, axis=0)
            self.phase_history[-1] = viz_3d
            
            # 3. Adaptation
            if self.auto_adapt.get() and is_target:
                # Only adapt if it's supposed to be firing but isn't
                if self.neuron.adapt(np.mean(self.resonance_history[-50:])):
                    # Update UI label safely
                    self.root.after(0, lambda: self.lbl_taps.config(text=f"Dendrite Length (Taps): {self.neuron.n_taps}"))

            self.time_step += 1
            
            # Throttle UI updates (update 20 times a second, process math faster)
            if self.time_step % 20 == 0:
                self.root.after(0, self.update_plots)
            
            time.sleep(0.001)

    def update_plots(self):
        # Update Input Signal Plot
        self.ax_sig.clear()
        self.ax_sig.plot(self.signal_history, color='gray', alpha=0.5, label="Raw Input (w/ Noise)")
        # Overlay a green background when Target is present
        self.ax_sig.fill_between(range(500), -3, 3, where=self.target_state_history>0.5, color='green', alpha=0.1, label="Target (40Hz) Present")
        self.ax_sig.set_title("Environmental Signal (1D Temporal)")
        self.ax_sig.set_ylim(-3, 3)
        self.ax_sig.legend(loc="upper right")

        # Update Resonance Plot
        self.ax_res.clear()
        self.ax_res.plot(self.resonance_history, color='blue', linewidth=2)
        self.ax_res.axhline(5.0, color='red', linestyle='--', label="Firing Threshold")
        self.ax_res.set_title("Deerskin Neuron Output (Resonance)")
        self.ax_res.set_ylim(0, 15)
        self.ax_res.legend(loc="upper left")
        
        # Update 3D Phase Space (Takens Embedding)
        self.ax_3d.clear()
        color = 'green' if self.target_state_history[-1] > 0.5 else 'red'
        self.ax_3d.plot(self.phase_history[:,0], self.phase_history[:,1], self.phase_history[:,2], color=color, alpha=0.6, linewidth=1)
        self.ax_3d.scatter(self.phase_history[-1,0], self.phase_history[-1,1], self.phase_history[-1,2], color='black', s=50) # Current point
        self.ax_3d.set_title(f"Takens Geometry (First 3 of {self.neuron.n_taps} dimensions)")
        self.ax_3d.set_xlim(-3, 3); self.ax_3d.set_ylim(-3, 3); self.ax_3d.set_zlim(-3, 3)
        
        self.canvas.draw()

    def start_simulation(self):
        self.running = True
        self.sim_thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.sim_thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    app = DeerskinDemoUI(root)
    root.mainloop()