"""
webcam_outer_deerskin_codec_v2.py - Live Webcam + Phase-Coherent Axon Fix
=============================================================
Combines the V2 "Cochlea/Axon" math fixes with a real-time 
webcam feed. 
"""

import os
import sys
import threading
import queue
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Use CPU for threading stability with Tkinter, or CUDA if preferred
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 1. ARCHITECTURE — Phase-Locked Transport
# ============================================================================

class FourierPositionEncoding(nn.Module):
    def __init__(self, n_freqs=64):
        super().__init__()
        self.n_freqs = n_freqs
        n_per_axis = int(math.sqrt(n_freqs))
        
        freq_x = torch.linspace(1.0, 20.0, n_per_axis)
        freq_y = torch.linspace(1.0, 20.0, n_per_axis)
        
        fx, fy = torch.meshgrid(freq_x, freq_y, indexing='ij')
        fx = fx.reshape(-1)[:n_freqs]
        fy = fy.reshape(-1)[:n_freqs]
        
        self.freq_x = nn.Parameter(fx)
        self.freq_y = nn.Parameter(fy)
        self.phase = nn.Parameter(torch.rand(n_freqs) * 2 * math.pi)
        self.out_dim = n_freqs * 2 + 2
    
    def forward(self, xy):
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        angles = (x * self.freq_x + y * self.freq_y) * math.pi + self.phase
        features = torch.cat([torch.sin(angles), torch.cos(angles), xy], dim=-1)
        return features

class ComplexLinear(nn.Module):
    """The Synapse: Pure spatial mixing of complex signals."""
    def __init__(self, in_features, out_features, init_scale=None):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=False)
        self.fc_i = nn.Linear(in_features, out_features, bias=False)
        self.bias_r = nn.Parameter(torch.zeros(out_features))
        self.bias_i = nn.Parameter(torch.zeros(out_features))
        
        bound = init_scale if init_scale else math.sqrt(6.0 / in_features) / 30.0
        nn.init.uniform_(self.fc_r.weight, -bound, bound)
        nn.init.uniform_(self.fc_i.weight, -bound, bound)
    
    def forward(self, z):
        r, i = z[..., 0], z[..., 1]
        out_r = self.fc_r(r) - self.fc_i(i) + self.bias_r
        out_i = self.fc_r(i) + self.fc_i(r) + self.bias_i
        return torch.stack([out_r, out_i], dim=-1)

class DeerskinTemporalLayer(nn.Module):
    """The Axon: Unitary phase rotation (Time Transport)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.omega = nn.Parameter(torch.linspace(0.5, 20.0, dim))
        self.bias_phase = nn.Parameter(torch.rand(dim) * 2 * math.pi)
    
    def forward(self, z, t):
        r, i = z[..., 0], z[..., 1]
        
        theta = self.omega * t + self.bias_phase
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        out_r = r * cos_t - i * sin_t
        out_i = r * sin_t + i * cos_t
        return torch.stack([out_r, out_i], dim=-1)

class InverseTransformOutput(nn.Module):
    """The Decoder: Reconstructing RGB via IFFT logic."""
    def __init__(self, latent_dim, n_output=3):
        super().__init__()
        self.w_real = nn.Linear(latent_dim, n_output)
        self.w_imag = nn.Linear(latent_dim, n_output)
        
        bound = math.sqrt(6.0 / latent_dim) / 30.0
        nn.init.uniform_(self.w_real.weight, -bound, bound)
        nn.init.uniform_(self.w_imag.weight, -bound, bound)
    
    def forward(self, z):
        real, imag = z[..., 0], z[..., 1]
        output = self.w_real(real) + self.w_imag(imag)
        return torch.sigmoid(output)

class DeerskinCodecV2(nn.Module):
    def __init__(self, n_freqs=64, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.pos_enc = FourierPositionEncoding(n_freqs=n_freqs)
        
        self.encode = ComplexLinear(self.pos_enc.out_dim, latent_dim)
        self.temporal1 = DeerskinTemporalLayer(latent_dim)
        self.temporal2 = DeerskinTemporalLayer(latent_dim)
        self.decode = InverseTransformOutput(latent_dim, n_output=3)
        
        self.register_buffer('prev_latent', torch.zeros(1, latent_dim, 2))
    
    def forward(self, xy, t, alpha=0.0):
        batch = xy.shape[0]
        
        # Analysis (Spatial -> Frequency)
        features = self.pos_enc(xy)
        z = torch.stack([features, torch.zeros_like(features)], dim=-1)
        
        # Synaptic Mixing
        z = self.encode(z)
        
        # Outer Loop Modulation
        if alpha > 0.0 and self.prev_latent.shape[0] == batch:
            z = z + alpha * self.prev_latent
        
        # Axonal Transport (Phase Rotation)
        z = self.temporal1(z, t)
        z = self.temporal2(z, t)
        
        self.prev_latent = z.detach()
        
        # Synthesis (Frequency -> Spatial RGB)
        return self.decode(z)

# ============================================================================
# 2. WEBCAM LOADER & TRAINER
# ============================================================================

class WebcamThread(threading.Thread):
    def __init__(self, img_size=64):
        super().__init__(daemon=True)
        self.img_size = img_size
        self.cap = cv2.VideoCapture(0)
        self.latest_frame = np.zeros((img_size, img_size, 3), dtype=np.float32)
        self.running = True
        
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                h, w = frame.shape[:2]
                min_dim = min(h, w)
                sy, sx = h//2 - min_dim//2, w//2 - min_dim//2
                cropped = frame[sy:sy+min_dim, sx:sx+min_dim]
                
                img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size))
                self.latest_frame = img.astype(np.float32) / 255.0
            time.sleep(0.03)
            
    def stop(self):
        self.running = False
        self.cap.release()

class TrainerThread(threading.Thread):
    def __init__(self, webcam, config_queue, update_queue, shared_model):
        super().__init__(daemon=True)
        self.webcam = webcam
        self.config_queue = config_queue
        self.update_queue = update_queue
        self.shared_model = shared_model
        self.optimizer = optim.Adam(self.shared_model.parameters(), lr=1e-3)
        self.running = True

    def run(self):
        step = 0
        loss_ema = 0.1
        start_time = time.time()
        
        while self.running:
            try:
                config = self.config_queue.get_nowait()
                if 'stop' in config: break
                alpha = config.get('alpha', 0.0)
            except queue.Empty: 
                alpha = getattr(self, 'current_alpha', 0.0)
            self.current_alpha = alpha

            # Continuous time loop (0.0 to 1.0 every 5 seconds)
            t_val = ((time.time() - start_time) % 5.0) / 5.0

            batch_size = 4096
            coords = torch.rand(batch_size, 2, device=device) * 2 - 1
            t_tensor = torch.full((batch_size, 1), t_val, device=device)
            
            frame = self.webcam.latest_frame
            sz = self.webcam.img_size - 1
            px = ((coords[:, 0].cpu().numpy() + 1) / 2 * sz).astype(int).clip(0, sz)
            py = ((coords[:, 1].cpu().numpy() + 1) / 2 * sz).astype(int).clip(0, sz)
            targets = torch.tensor(frame[py, px], device=device)

            self.optimizer.zero_grad()
            preds = self.shared_model(coords, t_tensor, alpha=alpha)
            loss = nn.MSELoss()(preds, targets)
            loss.backward()
            self.optimizer.step()

            loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            if step % 20 == 0:
                self.update_queue.put({
                    'loss_ema': loss_ema, 
                    'step': step, 
                    't_val': t_val,
                    'raw_frame': frame
                })
            step += 1

# ============================================================================
# 3. GUI (Dual Viewport)
# ============================================================================

class DeerskinCodecApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Webcam Deerskin V2")
        self.root.configure(bg="#0c0c18")
        
        self.webcam = WebcamThread(img_size=64)
        self.webcam.start()
        
        self.model = DeerskinCodecV2().to(device)
        self.config_q, self.update_q = queue.Queue(), queue.Queue()
        self.trainer = None
        
        self.current_t = 0.0
        
        self.setup_ui()
        self.update_loop()

    def setup_ui(self):
        tk.Label(self.root, text="LIVE WEBCAM: PHASE-LOCKED V2", bg="#0c0c18", fg="white", font=("Consolas", 14)).pack(pady=5)
        
        self.btn = tk.Button(self.root, text="START ENTRAINMENT", command=self.toggle, bg="#44ffaa")
        self.btn.pack(pady=5)
        
        self.alpha_scale = tk.Scale(self.root, from_=0, to=1.5, resolution=0.1, orient="horizontal", label="Outer Loop Alpha", length=400)
        self.alpha_scale.pack()
        
        self.status = tk.Label(self.root, text="Loss: --", bg="#0c0c18", fg="#ff4466")
        self.status.pack(pady=5)

        # Dual Canvas Frame
        frame = tk.Frame(self.root, bg="#0c0c18")
        frame.pack(pady=10)
        
        tk.Label(frame, text="Raw Reality", bg="#0c0c18", fg="#88aaff").grid(row=0, column=0)
        tk.Label(frame, text="Deerskin Reconstruction", bg="#0c0c18", fg="#88aaff").grid(row=0, column=1)
        
        self.lbl_raw = tk.Label(frame, bg="black")
        self.lbl_raw.grid(row=1, column=0, padx=10)
        
        self.lbl_recon = tk.Label(frame, bg="black")
        self.lbl_recon.grid(row=1, column=1, padx=10)

    def toggle(self):
        if not self.trainer:
            self.trainer = TrainerThread(self.webcam, self.config_q, self.update_q, self.model)
            self.trainer.start()
            self.btn.config(text="STOP ENTRAINMENT", bg="#ff4466")
        else:
            self.trainer.running = False
            self.trainer = None
            self.btn.config(text="START ENTRAINMENT", bg="#44ffaa")

    def update_loop(self):
        self.config_q.put({'alpha': self.alpha_scale.get()})
        try:
            latest = None
            while not self.update_q.empty():
                latest = self.update_q.get_nowait()
                
            if latest:
                self.status.config(text=f"Loss: {latest['loss_ema']:.5f} | Step: {latest['step']}")
                self.current_t = latest['t_val']
                
                # Render raw webcam
                raw_img = (latest['raw_frame'] * 255).astype(np.uint8)
                raw_tk = ImageTk.PhotoImage(Image.fromarray(cv2.resize(raw_img, (300, 300))))
                self.lbl_raw.config(image=raw_tk)
                self.lbl_raw.image = raw_tk
                
                # Render model output
                self.render(self.current_t)
        except: pass
        self.root.after(50, self.update_loop)

    def render(self, t):
        if not hasattr(self, 'model'): return
        with torch.no_grad():
            res = 128
            y, x = torch.meshgrid(torch.linspace(-1, 1, res), torch.linspace(-1, 1, res), indexing='ij')
            coords = torch.stack([x, y], dim=-1).reshape(-1, 2).to(device)
            t_tensor = torch.full((coords.shape[0], 1), t, device=device)
            
            self.model.prev_latent = torch.zeros(coords.shape[0], self.model.latent_dim, 2, device=device)
            
            pred = self.model(coords, t_tensor, alpha=self.alpha_scale.get())
            img = (pred.reshape(res, res, 3).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.resize(img, (300, 300))))
            self.lbl_recon.config(image=img_tk)
            self.lbl_recon.image = img_tk
            
    def __del__(self):
        self.webcam.stop()
        if self.trainer and self.trainer.is_alive():
            self.config_q.put({'stop': True})

if __name__ == "__main__":
    root = tk.Tk()
    app = DeerskinCodecApp(root)
    root.mainloop()
