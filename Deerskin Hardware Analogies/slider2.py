import numpy as np
import pyaudio
import time
from scipy.fft import fft, fftfreq
from scipy import signal
import matplotlib
matplotlib.use("TkAgg")  # Needed to integrate with Tkinter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import threading
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    filename='quantum_field_computer.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

class NoiseGenerator:
    """
    Advanced noise generation for signal injection and interference patterns.
    """
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.noise_types = {
            'white': self._generate_white_noise,
            'pink': self._generate_pink_noise,
            'brown': self._generate_brown_noise,
            'blue': self._generate_blue_noise,
            'violet': self._generate_violet_noise,
            'gaussian': self._generate_gaussian_noise,
            'uniform': self._generate_uniform_noise,
            'perlin': self._generate_perlin_noise,
            'quantum': self._generate_quantum_noise
        }
        
        # Noise state variables
        self.pink_filter_state = np.zeros(7)
        self.brown_state = 0.0
        
    def generate_noise(self, noise_type, duration_samples, amplitude=1.0, **kwargs):
        """Generate specified type of noise"""
        if noise_type not in self.noise_types:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        noise_func = self.noise_types[noise_type]
        noise = noise_func(duration_samples, **kwargs)
        return noise * amplitude
    
    def _generate_white_noise(self, duration_samples, **kwargs):
        """Pure white noise - equal power across all frequencies"""
        return np.random.normal(0, 1, duration_samples)
    
    def _generate_pink_noise(self, duration_samples, **kwargs):
        """Pink noise - 1/f power spectrum"""
        white = np.random.normal(0, 1, duration_samples)
        
        # Simple pink noise filter approximation
        # Using Voss-McCartney algorithm
        pink = np.zeros(duration_samples)
        for i in range(duration_samples):
            # Update filter states
            for j in range(7):
                if i % (2**j) == 0:
                    self.pink_filter_state[j] = np.random.normal(0, 1)
            pink[i] = np.sum(self.pink_filter_state) / 7
        
        return pink
    
    def _generate_brown_noise(self, duration_samples, **kwargs):
        """Brown noise - 1/f² power spectrum"""
        white = np.random.normal(0, 1, duration_samples)
        brown = np.zeros(duration_samples)
        
        for i in range(duration_samples):
            self.brown_state += white[i] * 0.1
            self.brown_state *= 0.999  # Prevent drift
            brown[i] = self.brown_state
            
        return brown
    
    def _generate_blue_noise(self, duration_samples, **kwargs):
        """Blue noise - f power spectrum"""
        white = self._generate_white_noise(duration_samples)
        # Simple high-pass filter for blue noise
        b, a = signal.butter(1, 0.1, 'high')
        return signal.filtfilt(b, a, white)
    
    def _generate_violet_noise(self, duration_samples, **kwargs):
        """Violet noise - f² power spectrum"""
        white = self._generate_white_noise(duration_samples)
        # High-pass filter with steeper rolloff
        b, a = signal.butter(2, 0.1, 'high')
        return signal.filtfilt(b, a, white)
    
    def _generate_gaussian_noise(self, duration_samples, mean=0, std=1, **kwargs):
        """Gaussian distributed noise"""
        return np.random.normal(mean, std, duration_samples)
    
    def _generate_uniform_noise(self, duration_samples, low=-1, high=1, **kwargs):
        """Uniformly distributed noise"""
        return np.random.uniform(low, high, duration_samples)
    
    def _generate_perlin_noise(self, duration_samples, frequency=1.0, **kwargs):
        """Perlin-like noise for smooth variations"""
        t = np.linspace(0, duration_samples / self.sample_rate, duration_samples)
        # Simple Perlin approximation using multiple sine waves
        noise = np.zeros(duration_samples)
        for i in range(5):
            freq = frequency * (2 ** i)
            amplitude = 1.0 / (2 ** i)
            phase = np.random.uniform(0, 2 * np.pi)
            noise += amplitude * np.sin(2 * np.pi * freq * t + phase)
        return noise
    
    def _generate_quantum_noise(self, duration_samples, **kwargs):
        """Quantum-inspired noise with interference patterns"""
        t = np.linspace(0, duration_samples / self.sample_rate, duration_samples)
        
        # Superposition of multiple frequencies with random phases
        frequencies = [1, 1.618, 2.718, 3.14159, 7.389]  # Golden ratio, e, pi, etc.
        noise = np.zeros(duration_samples)
        
        for freq in frequencies:
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.exponential(0.3)
            noise += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Add quantum tunneling effect (random spikes)
        tunnel_events = np.random.poisson(duration_samples * 0.001, 1)[0]
        for _ in range(tunnel_events):
            pos = np.random.randint(0, duration_samples)
            amplitude = np.random.exponential(2.0)
            noise[pos] += amplitude
        
        return noise
    
    def apply_frequency_filter(self, noise, freq_min, freq_max):
        """Apply bandpass filter to noise"""
        nyquist = self.sample_rate / 2
        low = freq_min / nyquist
        high = freq_max / nyquist
        
        if low <= 0:
            low = 0.001
        if high >= 1:
            high = 0.999
            
        try:
            b, a = signal.butter(4, [low, high], 'band')
            return signal.filtfilt(b, a, noise)
        except:
            return noise

class CrystalResonanceDetector:
    """
    Analyzes resonance patterns related to crystal oscillator frequencies within audio range.
    """
    def __init__(self, base_crystal_freq=1000, fft_size=4096):
        self.crystal_freq = base_crystal_freq
        self.fft_size = fft_size
        self.harmonics = []
        self.subharmonics = []
        self.resonance_history = []
        self.last_peak_frequencies = []
        self.interference_patterns = []

    def compute_harmonics(self, order=10):
        self.harmonics = [self.crystal_freq * n for n in range(1, order+1)]
        self.subharmonics = [self.crystal_freq / n for n in range(1, order+1)]

    def detect_resonance(self, signal_data, sample_rate):
        n = self.fft_size
        if len(signal_data) < n:
            logging.warning("Signal data shorter than FFT size. Zero-padding.")
            signal_data = np.pad(signal_data, (0, n - len(signal_data)), 'constant')
        windowed_data = signal_data[:n] * np.hanning(n)
        spectrum = np.abs(fft(windowed_data))
        freqs = fftfreq(n, 1/sample_rate)

        peaks = []
        for harm in self.harmonics + self.subharmonics:
            tolerance = harm * 0.01
            mask = (freqs >= harm - tolerance) & (freqs <= harm + tolerance)
            if np.any(mask):
                peak_idx = np.argmax(spectrum[mask])
                actual_freq = freqs[mask][peak_idx]
                magnitude = spectrum[mask][peak_idx]
                peaks.append({
                    'frequency': actual_freq,
                    'magnitude': magnitude,
                    'harmonic': harm
                })

        self.resonance_history.append(peaks)
        if len(self.resonance_history) > 1000:
            self.resonance_history.pop(0)

        return peaks

    def analyze_interference(self, current_freq):
        patterns = []
        for harm in self.harmonics + self.subharmonics:
            beat_freq = abs(harm - current_freq)
            wavelength_ratio = max(harm, current_freq) / min(harm, current_freq)
            patterns.append({
                'harmonic': harm,
                'beat_frequency': beat_freq,
                'wavelength_ratio': wavelength_ratio
            })

        self.interference_patterns = patterns
        return patterns

class PhysicalSpeakerNeuron:
    """
    Enhanced Audio Input/Output and Signal Generation with Noise Injection.
    """
    def __init__(self, sample_rate=44100, buffer_size=4096, input_device=None, output_device=None):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.base_freq = 1000  # Default base frequency
        self.phase = 0
        self.amplitude = 0.3
        self.audio = pyaudio.PyAudio()
        self.input_device = input_device
        self.output_device = output_device
        
        # Noise injection parameters
        self.noise_generator = NoiseGenerator(sample_rate)
        self.noise_enabled = False
        self.noise_type = 'white'
        self.noise_amplitude = 0.1
        self.noise_freq_min = 20
        self.noise_freq_max = 20000
        self.noise_mix_ratio = 0.5  # 0 = pure signal, 1 = pure noise
        
        self.setup_audio()

    def setup_audio(self):
        try:
            self.output_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device,
                frames_per_buffer=self.buffer_size
            )
            logging.info(f"Opened output device index {self.output_device}")
        except Exception as e:
            error_msg = f"Failed to open output device (Index {self.output_device}): {e}"
            logging.error(error_msg)
            messagebox.showerror("Audio Output Error", error_msg)
            raise e

        try:
            self.input_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device,
                frames_per_buffer=self.buffer_size
            )
            logging.info(f"Opened input device index {self.input_device}")
        except Exception as e:
            error_msg = f"Failed to open input device (Index {self.input_device}): {e}"
            logging.error(error_msg)
            messagebox.showerror("Audio Input Error", error_msg)
            raise e

    def generate_wave(self):
        t = np.linspace(0, self.buffer_size / self.sample_rate, self.buffer_size, endpoint=False)
        
        # Generate base wave
        wave = self.amplitude * np.sin(2 * np.pi * self.base_freq * t + self.phase)
        self.phase += 2 * np.pi * self.base_freq * self.buffer_size / self.sample_rate
        self.phase = self.phase % (2 * np.pi)
        
        # Add noise if enabled
        if self.noise_enabled:
            noise = self.noise_generator.generate_noise(
                self.noise_type, 
                self.buffer_size, 
                amplitude=self.noise_amplitude
            )
            
            # Apply frequency filtering to noise
            if self.noise_freq_min > 20 or self.noise_freq_max < 20000:
                noise = self.noise_generator.apply_frequency_filter(
                    noise, self.noise_freq_min, self.noise_freq_max
                )
            
            # Mix signal and noise
            mixed_wave = (1 - self.noise_mix_ratio) * wave + self.noise_mix_ratio * noise
            return mixed_wave
        
        return wave

    def set_noise_parameters(self, enabled, noise_type, amplitude, freq_min, freq_max, mix_ratio):
        """Update noise injection parameters"""
        self.noise_enabled = enabled
        self.noise_type = noise_type
        self.noise_amplitude = amplitude
        self.noise_freq_min = freq_min
        self.noise_freq_max = freq_max
        self.noise_mix_ratio = mix_ratio
        
        logging.info(f"Noise parameters updated: {enabled}, {noise_type}, {amplitude}, {freq_min}-{freq_max}Hz, mix:{mix_ratio}")

    def forward(self):
        try:
            signal = self.generate_wave()
            self.output_stream.write(signal.astype(np.float32).tobytes())
            recorded = np.frombuffer(
                self.input_stream.read(self.buffer_size, exception_on_overflow=False),
                dtype=np.float32
            )
            return recorded
        except Exception as e:
            logging.error(f"Forward pass error: {e}")
            return np.zeros(self.buffer_size, dtype=np.float32)

    def cleanup(self):
        try:
            if hasattr(self, 'output_stream'):
                self.output_stream.stop_stream()
                self.output_stream.close()
                logging.info("Closed output stream")
            if hasattr(self, 'input_stream'):
                self.input_stream.stop_stream()
                self.input_stream.close()
                logging.info("Closed input stream")
            self.audio.terminate()
            logging.info("Terminated PyAudio")
        except Exception as e:
            logging.error(f"Cleanup error: {e}")

class QuantumFieldUI:
    """
    Main Application UI with 3D visualization, enhanced frequency controls, and noise injection.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Quantum Field Computer with Noise Injection")
        self.setup_variables()
        self.setup_layout()
        self.setup_crystal_detector()

    def setup_variables(self):
        self.sample_rate = 44100
        self.buffer_size = 4096
        self.wave_history = []
        self.recorded_data = []
        self.running = False
        self.recording = False

        # Frequency range variables
        self.freq_min = tk.DoubleVar(value=20)
        self.freq_max = tk.DoubleVar(value=20000)
        self.current_freq = tk.DoubleVar(value=1000)

        # Noise injection variables
        self.noise_enabled = tk.BooleanVar(value=False)
        self.noise_type = tk.StringVar(value='white')
        self.noise_amplitude = tk.DoubleVar(value=0.1)
        self.noise_freq_min = tk.DoubleVar(value=20)
        self.noise_freq_max = tk.DoubleVar(value=20000)
        self.noise_mix_ratio = tk.DoubleVar(value=0.1)

        self.audio = pyaudio.PyAudio()
        self.input_devices = self.get_audio_devices(True)
        self.output_devices = self.get_audio_devices(False)

    def setup_crystal_detector(self):
        self.crystal_detector = CrystalResonanceDetector(
            base_crystal_freq=1000,
            fft_size=self.buffer_size
        )
        self.crystal_detector.compute_harmonics()
        self.resonance_detected = False
        self.last_resonance_time = time.time()
        self.resonance_count = 0

    def get_audio_devices(self, is_input):
        devices = {}
        for i in range(self.audio.get_device_count()):
            try:
                dev_info = self.audio.get_device_info_by_index(i)
                if is_input and dev_info['maxInputChannels'] > 0:
                    devices[dev_info['name']] = i
                elif not is_input and dev_info['maxOutputChannels'] > 0:
                    devices[dev_info['name']] = i
            except Exception as e:
                logging.error(f"Error accessing device index {i}: {e}")
        return devices

    def setup_layout(self):
        """
        Enhanced layout with frequency range controls, noise injection, and dual visualization.
        """
        # Root configuration
        self.root.rowconfigure(0, weight=0)  # Controls row
        self.root.rowconfigure(1, weight=0)  # Frequency controls row
        self.root.rowconfigure(2, weight=0)  # Noise controls row
        self.root.rowconfigure(3, weight=1)  # Visualization row
        self.root.columnconfigure(0, weight=1)

        # --- Row 0: Device Selection and Basic Controls ---
        top_frame = ttk.Frame(self.root, padding=5)
        top_frame.grid(row=0, column=0, sticky="ew")

        # Device Selection Frame
        device_frame = ttk.LabelFrame(top_frame, text="Device Selection", padding=5)
        device_frame.pack(side="left", padx=5)

        self.input_var = tk.StringVar()
        ttk.Label(device_frame, text="Input:").pack(side="left", padx=2)
        self.input_combobox = ttk.Combobox(
            device_frame,
            textvariable=self.input_var,
            values=list(self.input_devices.keys()),
            state="readonly",
            width=20
        )
        self.input_combobox.pack(side="left", padx=2)
        if self.input_devices:
            self.input_combobox.current(0)

        self.output_var = tk.StringVar()
        ttk.Label(device_frame, text="Output:").pack(side="left", padx=2)
        self.output_combobox = ttk.Combobox(
            device_frame,
            textvariable=self.output_var,
            values=list(self.output_devices.keys()),
            state="readonly",
            width=20
        )
        self.output_combobox.pack(side="left", padx=2)
        if self.output_devices:
            self.output_combobox.current(0)

        ttk.Button(device_frame, text="Initialize", command=self.initialize_audio).pack(side="left", padx=2)
        ttk.Button(device_frame, text="Refresh", command=self.refresh_audio_devices).pack(side="left", padx=2)

        # Controls Frame (Start/Stop, Record)
        control_frame = ttk.LabelFrame(top_frame, text="Controls", padding=5)
        control_frame.pack(side="left", padx=5)

        self.start_button = ttk.Button(control_frame, text="Start", command=self.toggle_running)
        self.start_button.pack(side="left", padx=2)

        self.record_button = ttk.Button(control_frame, text="Record", command=self.toggle_recording)
        self.record_button.pack(side="left", padx=2)

        # --- Row 1: Enhanced Frequency Controls ---
        freq_control_frame = ttk.LabelFrame(self.root, text="Frequency Control System", padding=10)
        freq_control_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # Range Settings Frame
        range_frame = ttk.Frame(freq_control_frame)
        range_frame.pack(fill="x", pady=5)

        # Bottom (Min) Frequency Controls
        min_frame = ttk.LabelFrame(range_frame, text="Base Frequency (Hz)", padding=5)
        min_frame.pack(side="left", padx=10, fill="both", expand=True)

        self.min_entry = ttk.Entry(min_frame, textvariable=self.freq_min, width=10)
        self.min_entry.pack(side="left", padx=2)
        self.min_entry.bind('<Return>', self.update_frequency_range)

        self.min_scale = ttk.Scale(
            min_frame,
            from_=1,
            to=10000,
            orient='horizontal',
            variable=self.freq_min,
            command=self.on_min_freq_changed,
            length=200
        )
        self.min_scale.pack(side="left", padx=5, fill="x", expand=True)

        # Top (Max) Frequency Controls
        max_frame = ttk.LabelFrame(range_frame, text="Top Frequency (Hz)", padding=5)
        max_frame.pack(side="right", padx=10, fill="both", expand=True)

        self.max_entry = ttk.Entry(max_frame, textvariable=self.freq_max, width=10)
        self.max_entry.pack(side="left", padx=2)
        self.max_entry.bind('<Return>', self.update_frequency_range)

        self.max_scale = ttk.Scale(
            max_frame,
            from_=100,
            to=20000,
            orient='horizontal',
            variable=self.freq_max,
            command=self.on_max_freq_changed,
            length=200
        )
        self.max_scale.pack(side="left", padx=5, fill="x", expand=True)

        # Current Frequency Slider (Main Control)
        current_frame = ttk.LabelFrame(freq_control_frame, text="Current Frequency", padding=5)
        current_frame.pack(fill="x", pady=5)

        ttk.Label(current_frame, text="Frequency (Hz):").pack(side="left", padx=5)

        self.current_freq_scale = ttk.Scale(
            current_frame,
            from_=20,
            to=20000,
            orient='horizontal',
            variable=self.current_freq,
            command=self.on_current_freq_changed,
            length=400
        )
        self.current_freq_scale.pack(side="left", padx=5, fill="x", expand=True)

        self.current_freq_label = ttk.Label(current_frame, text="1000 Hz", font=("Arial", 12, "bold"))
        self.current_freq_label.pack(side="right", padx=5)

        # --- Row 2: Noise Injection Controls ---
        noise_frame = ttk.LabelFrame(self.root, text="Noise Injection System", padding=10)
        noise_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        # Noise Enable and Type Selection
        noise_top_frame = ttk.Frame(noise_frame)
        noise_top_frame.pack(fill="x", pady=5)

        self.noise_checkbox = ttk.Checkbutton(
            noise_top_frame,
            text="Enable Noise Injection",
            variable=self.noise_enabled,
            command=self.on_noise_settings_changed
        )
        self.noise_checkbox.pack(side="left", padx=5)

        ttk.Label(noise_top_frame, text="Noise Type:").pack(side="left", padx=10)
        self.noise_type_combo = ttk.Combobox(
            noise_top_frame,
            textvariable=self.noise_type,
            values=['white', 'pink', 'brown', 'blue', 'violet', 'gaussian', 'uniform', 'perlin', 'quantum'],
            state="readonly",
            width=12
        )
        self.noise_type_combo.pack(side="left", padx=5)
        self.noise_type_combo.bind('<<ComboboxSelected>>', self.on_noise_settings_changed)

        # Noise Parameters Frame
        noise_params_frame = ttk.Frame(noise_frame)
        noise_params_frame.pack(fill="x", pady=5)

        # Noise Amplitude
        amp_frame = ttk.LabelFrame(noise_params_frame, text="Noise Amplitude", padding=5)
        amp_frame.pack(side="left", padx=5, fill="both", expand=True)

        self.noise_amp_scale = ttk.Scale(
            amp_frame,
            from_=0.0,
            to=1.0,
            orient='horizontal',
            variable=self.noise_amplitude,
            command=self.on_noise_settings_changed,
            length=150
        )
        self.noise_amp_scale.pack(side="left", fill="x", expand=True)

        self.noise_amp_label = ttk.Label(amp_frame, text="0.1")
        self.noise_amp_label.pack(side="right", padx=5)

        # Noise Mix Ratio
        mix_frame = ttk.LabelFrame(noise_params_frame, text="Signal/Noise Mix", padding=5)
        mix_frame.pack(side="right", padx=5, fill="both", expand=True)

        self.noise_mix_scale = ttk.Scale(
            mix_frame,
            from_=0.0,
            to=1.0,
            orient='horizontal',
            variable=self.noise_mix_ratio,
            command=self.on_noise_settings_changed,
            length=150
        )
        self.noise_mix_scale.pack(side="left", fill="x", expand=True)

        self.noise_mix_label = ttk.Label(mix_frame, text="0.1")
        self.noise_mix_label.pack(side="right", padx=5)

        # Noise Frequency Range
        noise_freq_frame = ttk.Frame(noise_frame)
        noise_freq_frame.pack(fill="x", pady=5)

        # Noise Min Frequency
        noise_min_frame = ttk.LabelFrame(noise_freq_frame, text="Noise Min Freq (Hz)", padding=5)
        noise_min_frame.pack(side="left", padx=5, fill="both", expand=True)

        self.noise_freq_min_scale = ttk.Scale(
            noise_min_frame,
            from_=20,
            to=10000,
            orient='horizontal',
            variable=self.noise_freq_min,
            command=self.on_noise_settings_changed,
            length=150
        )
        self.noise_freq_min_scale.pack(side="left", fill="x", expand=True)

        self.noise_freq_min_label = ttk.Label(noise_min_frame, text="20")
        self.noise_freq_min_label.pack(side="right", padx=5)

        # Noise Max Frequency
        noise_max_frame = ttk.LabelFrame(noise_freq_frame, text="Noise Max Freq (Hz)", padding=5)
        noise_max_frame.pack(side="right", padx=5, fill="both", expand=True)

        self.noise_freq_max_scale = ttk.Scale(
            noise_max_frame,
            from_=100,
            to=20000,
            orient='horizontal',
            variable=self.noise_freq_max,
            command=self.on_noise_settings_changed,
            length=150
        )
        self.noise_freq_max_scale.pack(side="left", fill="x", expand=True)

        self.noise_freq_max_label = ttk.Label(noise_max_frame, text="20000")
        self.noise_freq_max_label.pack(side="right", padx=5)

        # --- Row 3: 3D Visualization ---
        viz_frame = ttk.Frame(self.root)
        viz_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        viz_frame.rowconfigure(0, weight=1)
        viz_frame.columnconfigure(0, weight=1)

        # Create 3D plot
        self.fig = plt.Figure(figsize=(12, 8))
        
        # 3D Plot (Full area)
        self.ax_3d = self.fig.add_subplot(111, projection='3d')
        self.ax_3d.set_title("3D Phase Space with Noise Injection")
        self.ax_3d.set_xlabel('Wave(n-2)')
        self.ax_3d.set_ylabel('Wave(n-1)')
        self.ax_3d.set_zlabel('Wave(n)')

        # Place figure in canvas
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Update frequency range and noise settings
        self.update_frequency_range()
        self.update_noise_labels()

    def refresh_audio_devices(self):
        # Refresh input devices
        self.input_devices = self.get_audio_devices(True)
        self.input_combobox['values'] = list(self.input_devices.keys())
        if self.input_devices:
            self.input_combobox.current(0)
        else:
            self.input_combobox.set('')

        # Refresh output devices
        self.output_devices = self.get_audio_devices(False)
        self.output_combobox['values'] = list(self.output_devices.keys())
        if self.output_devices:
            self.output_combobox.current(0)
        else:
            self.output_combobox.set('')

    def on_min_freq_changed(self, val):
        """Callback when minimum frequency slider changes"""
        min_freq = float(val)
        max_freq = self.freq_max.get()
        
        # Ensure min < max
        if min_freq >= max_freq:
            self.freq_max.set(min_freq + 100)
        
        self.update_frequency_range()

    def on_max_freq_changed(self, val):
        """Callback when maximum frequency slider changes"""
        max_freq = float(val)
        min_freq = self.freq_min.get()
        
        # Ensure min < max
        if max_freq <= min_freq:
            self.freq_min.set(max_freq - 100)
        
        self.update_frequency_range()

    def on_current_freq_changed(self, val):
        """Callback when current frequency slider changes"""
        freq = float(val)
        self.set_frequency(freq)
        self.current_freq_label.config(text=f"{int(freq)} Hz")

    def on_noise_settings_changed(self, val=None):
        """Callback when noise settings change"""
        self.update_noise_labels()
        if hasattr(self, 'neuron'):
            self.neuron.set_noise_parameters(
                self.noise_enabled.get(),
                self.noise_type.get(),
                self.noise_amplitude.get(),
                self.noise_freq_min.get(),
                self.noise_freq_max.get(),
                self.noise_mix_ratio.get()
            )

    def update_noise_labels(self):
        """Update noise parameter labels"""
        self.noise_amp_label.config(text=f"{self.noise_amplitude.get():.2f}")
        self.noise_mix_label.config(text=f"{self.noise_mix_ratio.get():.2f}")
        self.noise_freq_min_label.config(text=f"{int(self.noise_freq_min.get())}")
        self.noise_freq_max_label.config(text=f"{int(self.noise_freq_max.get())}")

    def update_frequency_range(self, event=None):
        """Update the current frequency slider range based on min/max values"""
        min_freq = self.freq_min.get()
        max_freq = self.freq_max.get()
        
        # Update current frequency slider range
        self.current_freq_scale.config(from_=min_freq, to=max_freq)
        
        # Ensure current frequency is within range
        current = self.current_freq.get()
        if current < min_freq:
            self.current_freq.set(min_freq)
        elif current > max_freq:
            self.current_freq.set(max_freq)

    def update_visualization(self):
        """3D Phase Space visualization with noise effects"""
        if len(self.wave_history) < 3:
            return

        try:
            # Clear 3D plot
            self.ax_3d.clear()

            # 3D Plot: Phase space
            wave1 = self.wave_history[-3]
            wave2 = self.wave_history[-2]
            wave3 = self.wave_history[-1]

            # Color based on noise injection status
            color = 'red' if self.noise_enabled.get() else 'green'
            alpha = 0.3 if self.noise_enabled.get() else 0.6

            self.ax_3d.scatter(
                wave1, wave2, wave3,
                c=color, s=1, alpha=alpha
            )
            
            title = "3D Phase Space"
            if self.noise_enabled.get():
                noise_info = f" (Noise: {self.noise_type.get()}, {self.noise_amplitude.get():.2f})"
                title += noise_info
            
            self.ax_3d.set_title(title)
            self.ax_3d.set_xlabel('Wave(n-2)')
            self.ax_3d.set_ylabel('Wave(n-1)')
            self.ax_3d.set_zlabel('Wave(n)')

            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            logging.error(f"Visualization error: {e}")

    def processing_loop(self):
        while self.running:
            try:
                wave_data = self.neuron.forward()
                self.wave_history.append(wave_data)

                # Limit wave history if it gets large
                if len(self.wave_history) > 50:
                    self.wave_history.pop(0)

                # Crystal resonance analysis
                if len(self.wave_history) >= 2:
                    self.crystal_detector.analyze_interference(self.neuron.base_freq)
                    resonances = self.crystal_detector.detect_resonance(
                        self.wave_history[-1],
                        self.sample_rate
                    )
                    if resonances:
                        self.resonance_detected = True
                        self.resonance_count += 1
                        noise_status = "with noise" if self.noise_enabled.get() else "clean"
                        logging.info(f"Quantum field interaction detected ({noise_status})! Count: {self.resonance_count}")

                # Record if active
                if self.recording:
                    self.record_frame(wave_data)

                # Update 3D visualization
                self.update_visualization()

                time.sleep(0.01)

            except Exception as e:
                logging.error(f"Processing error: {e}")
                break

    def initialize_audio(self):
        input_name = self.input_var.get()
        output_name = self.output_var.get()

        if not input_name or not output_name:
            messagebox.showwarning("Device Selection", "Please select both input and output devices.")
            return

        try:
            self.neuron = PhysicalSpeakerNeuron(
                sample_rate=self.sample_rate,
                buffer_size=self.buffer_size,
                input_device=self.input_devices.get(input_name),
                output_device=self.output_devices.get(output_name)
            )
            # Set frequency from slider's current value
            freq = float(self.current_freq.get())
            self.set_frequency(freq)
            self.current_freq_label.config(text=f"{int(freq)} Hz")

            # Set initial noise parameters
            self.on_noise_settings_changed()

            messagebox.showinfo("Success", "Audio initialized successfully")
            logging.info("Audio initialized successfully with noise injection capabilities")
        except Exception as e:
            logging.error(f"Initialization failed: {e}")

    def set_frequency(self, freq):
        if hasattr(self, 'neuron'):
            self.neuron.base_freq = freq
            # Update crystal detector
            self.crystal_detector.crystal_freq = freq
            self.crystal_detector.compute_harmonics()
            logging.info(f"Set base frequency to {freq} Hz")
        else:
            logging.warning("Attempted to set frequency without initializing audio")

    def toggle_running(self):
        if not hasattr(self, 'neuron'):
            messagebox.showwarning("Not Initialized", "Please initialize audio first")
            logging.warning("Attempted to start processing without initializing audio")
            return

        self.running = not self.running
        self.start_button.config(text="Stop" if self.running else "Start")
        logging.info(f"Processing {'started' if self.running else 'stopped'}")

        if self.running:
            self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
            self.processing_thread.start()

    def toggle_recording(self):
        if not self.running:
            messagebox.showwarning("Not Running", "Please start the system first")
            logging.warning("Attempted to toggle recording while not running")
            return

        self.recording = not self.recording
        self.record_button.config(text="Stop Recording" if self.recording else "Record")
        logging.info(f"Recording {'started' if self.recording else 'stopped'}")

        if not self.recording and self.recorded_data:
            self.save_recorded_data()

    def record_frame(self, wave_data):
        frame_data = {
            'timestamp': datetime.now().isoformat(),
            'frequency': self.neuron.base_freq,
            'wave_data': wave_data.tolist(),
            'noise_settings': {
                'enabled': self.noise_enabled.get(),
                'type': self.noise_type.get(),
                'amplitude': self.noise_amplitude.get(),
                'freq_min': self.noise_freq_min.get(),
                'freq_max': self.noise_freq_max.get(),
                'mix_ratio': self.noise_mix_ratio.get()
            },
            'metadata': {
                'sample_rate': self.sample_rate,
                'buffer_size': self.buffer_size,
                'frequency_range': [self.freq_min.get(), self.freq_max.get()],
                'resonances': (self.crystal_detector.resonance_history[-1]
                               if self.crystal_detector.resonance_history else None)
            }
        }
        self.recorded_data.append(frame_data)
        logging.debug("Frame recorded with noise settings")

    def save_recorded_data(self):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json")]
            )
            if filename:
                with open(filename, 'w') as f:
                    json.dump(self.recorded_data, f, indent=2)
                self.recorded_data = []
                messagebox.showinfo("Success", "Data saved successfully with noise injection settings")
                logging.info(f"Recorded data saved successfully to {filename} with noise parameters")
        except Exception as e:
            logging.error(f"Save recorded data error: {e}")
            messagebox.showerror("Save Error", str(e))

def main():
    root = tk.Tk()
    root.geometry("1400x1000")  # Increased size for noise controls
    root.minsize(1000, 700)     # Reasonable minimum
    app = QuantumFieldUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: quit_application(root, app))
    root.mainloop()

def quit_application(root, app):
    if hasattr(app, 'neuron'):
        app.neuron.cleanup()
    root.quit()
    root.destroy()

if __name__ == "__main__":
    main()