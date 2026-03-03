"""
DEERSKIN TEST #3: Φ-DWELL × BETTI-1 UNIFIED CONSCIOUSNESS PROBE
=================================================================
Runs eigenmode vocabulary analysis (from the Φ-Dwell framework) on the
same RepOD Schizophrenia EEG dataset, then correlates eigenmode metrics
with the Betti-1 topological complexity and Theta PLV from Tests 1 & 2.

PREDICTIONS (from the Geometric Dysrhythmia / Fractal Surfer framework):

Schizophrenia (Hyper-Geometric) should show:
  1. HIGHER eigenmode vocabulary (more configurations visited)
  2. HIGHER top-5 concentration (hijacked gate forces dominant patterns)
  3. STEEPER Zipf α (more structured, not less — unlike Alzheimer's)
  4. CV DEVIATING from 1.0 toward SUPERCRITICAL (over-structured)
  5. Betti-1 POSITIVELY CORRELATES with vocabulary size across subjects
  6. Theta PLV CORRELATES with dwell rigidity (lower CV variance)

This bridges the two frameworks:
  - Betti-1 measures the TOPOLOGY of the phase-space attractor
  - Φ-Dwell measures the GRAMMAR of eigenmode configuration trajectories
  - Theta PLV measures the COHERENCE of the gating mechanism
  - Together they characterize consciousness as a fractal surf through
    a multi-scale eigenmode manifold

Dataset: RepOD "EEG in Schizophrenia" (Olejarczyk & Jernajczyk, 2017)
  14 Healthy Controls, 14 Schizophrenia patients
  19-channel 10-20 system, 250 Hz

Antti Luode (PerceptionLab | Independent Research, Finland)
Claude (Anthropic, Collaborative formalization)
March 2026
"""

import os
import requests
import numpy as np
import scipy.signal
import scipy.linalg
import scipy.stats as stats
from scipy.signal import hilbert
import mne
from ripser import ripser
from collections import Counter, defaultdict
from tqdm import tqdm
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

FRONTAL_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'F7', 'F8']

BANDS = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'gamma': (30, 50),
}
BAND_NAMES = list(BANDS.keys())

# 19-channel 10-20 electrode positions
ELECTRODE_POS_19 = {
    'Fp1': (-0.30, 0.90), 'Fp2': (0.30, 0.90),
    'F7':  (-0.70, 0.60), 'F3':  (-0.35, 0.60), 'Fz': (0.00, 0.60),
    'F4':  (0.35, 0.60),  'F8':  (0.70, 0.60),
    'T3':  (-0.90, 0.00), 'C3':  (-0.40, 0.00), 'Cz': (0.00, 0.00),
    'C4':  (0.40, 0.00),  'T4':  (0.90, 0.00),
    'T5':  (-0.70, -0.50), 'P3': (-0.35, -0.50), 'Pz': (0.00, -0.50),
    'P4':  (0.35, -0.50), 'T6':  (0.70, -0.50),
    'O1':  (-0.30, -0.85), 'O2': (0.30, -0.85),
}

CHANNEL_ALIASES = {
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
}

N_MODES = 6  # 19 channels → 6 usable eigenmodes
THETA_BAND = (4.0, 8.0)


# ═══════════════════════════════════════════════════════════════
# DATASET DOWNLOAD (same as schizophrenia.py)
# ═══════════════════════════════════════════════════════════════

def download_repod_dataset(download_dir="repod_sz_eeg"):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    print("Querying RepOD Dataverse API...")
    api_url = "https://repod.icm.edu.pl/api/datasets/:persistentId/?persistentId=doi:10.18150/repod.0107441"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        metadata = response.json()
        files = metadata['data']['latestVersion']['files']
    except Exception as e:
        print(f"Failed: {e}")
        return []

    edf_files = []
    for f in tqdm(files, desc="Downloading"):
        filename = f['dataFile']['filename']
        if not filename.endswith('.edf'):
            continue
        filepath = os.path.join(download_dir, filename)
        edf_files.append(filepath)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000000:
            continue
        file_id = f['dataFile']['id']
        url = f"https://repod.icm.edu.pl/api/access/datafile/{file_id}"
        data = requests.get(url, stream=True)
        with open(filepath, 'wb') as out:
            for chunk in data.iter_content(chunk_size=8192):
                out.write(chunk)
    return edf_files


# ═══════════════════════════════════════════════════════════════
# GRAPH LAPLACIAN & EIGENMODES (from Φ-Dwell)
# ═══════════════════════════════════════════════════════════════

def build_graph_laplacian(positions, sigma=0.5):
    names = sorted(positions.keys())
    N = len(names)
    coords = np.array([positions[n] for n in names])
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            A[i, j] = A[j, i] = np.exp(-d ** 2 / (2 * sigma ** 2))
    D = np.diag(A.sum(axis=1))
    L = D - A
    vals, vecs = scipy.linalg.eigh(L)
    return names, coords, vecs[:, 1:N_MODES + 1], vals[1:N_MODES + 1]


def map_channels(raw_names, graph_names):
    mapping = {}
    graph_lower = {n.lower(): n for n in graph_names}
    for ch in raw_names:
        clean = ch.strip().replace('EEG ', '').replace('-REF', '').replace('-Ref', '').strip()
        if clean in graph_names:
            mapping[ch] = clean
            continue
        if clean.lower() in graph_lower:
            mapping[ch] = graph_lower[clean.lower()]
            continue
        # Aliases
        alias = CHANNEL_ALIASES.get(clean, clean)
        if alias in graph_names:
            mapping[ch] = alias
            continue
        if alias.lower() in graph_lower:
            mapping[ch] = graph_lower[alias.lower()]
    return mapping


# ═══════════════════════════════════════════════════════════════
# BETTI-1 TOPOLOGY (from schizophrenia.py)
# ═══════════════════════════════════════════════════════════════

def takens_embedding(signal, delay_samples, dim=3, max_pts=400):
    N = len(signal)
    if N < (dim - 1) * delay_samples + 1:
        return np.array([])
    embedded = np.array([signal[i: i + (dim - 1) * delay_samples + 1: delay_samples]
                         for i in range(N - (dim - 1) * delay_samples)])
    if len(embedded) > max_pts:
        indices = np.linspace(0, len(embedded) - 1, max_pts, dtype=int)
        embedded = embedded[indices]
    embedded = embedded - np.mean(embedded, axis=0)
    std_dev = np.std(embedded, axis=0)
    std_dev[std_dev == 0] = 1.0
    return embedded / std_dev


def compute_betti1(embedded):
    if len(embedded) < 10:
        return 0.0
    try:
        diagrams = ripser(embedded, maxdim=1, thresh=2.0)['dgms']
        h1 = diagrams[1]
        if len(h1) == 0:
            return 0.0
        lifetimes = h1[:, 1] - h1[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes)]
        if len(lifetimes) == 0:
            return 0.0
        max_life = np.max(lifetimes)
        sig = lifetimes[lifetimes > 0.1 * max_life]
        return np.sum(sig) + 0.1 * len(sig)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════
# UNIFIED SUBJECT ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_subject_unified(filepath, graph_names, eigenvecs, max_duration_s=20):
    """
    Extracts ALL three measurement layers from a single EEG recording:
      Layer 1: Betti-1 topological complexity (phase-space geometry)
      Layer 2: Theta PLV (gate coherence)
      Layer 3: Φ-Dwell eigenmode vocabulary (configuration grammar)

    Returns dict with all metrics, or None on failure.
    """
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    except Exception as e:
        return None

    sfreq = raw.info['sfreq']
    raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
    data = raw.get_data()
    n_samp = min(data.shape[1], int(max_duration_s * sfreq))
    data = data[:, :n_samp]

    ch_map_upper = {ch.upper(): i for i, ch in enumerate(raw.ch_names)}

    # ─────────────────────────────────────────────────────────
    # LAYER 1: BETTI-1 (frontal topology)
    # ─────────────────────────────────────────────────────────
    valid_frontal = [ch_map_upper[c.upper()] for c in FRONTAL_CHANNELS if c.upper() in ch_map_upper]
    if len(valid_frontal) < 2:
        return None

    frontal_signal = np.mean(data[valid_frontal, :], axis=0)
    window_samples = int(2.0 * sfreq)
    delays_samples = [int(d * sfreq / 1000.0) for d in [10, 20, 40]]

    topo_scores = []
    for w in range(10):
        start = w * window_samples
        end = start + window_samples
        if end > len(frontal_signal):
            break
        window_sig = frontal_signal[start:end]
        delay_scores = [compute_betti1(takens_embedding(window_sig, d)) for d in delays_samples]
        topo_scores.append(np.mean(delay_scores))

    betti1 = np.mean(topo_scores) if topo_scores else None

    # ─────────────────────────────────────────────────────────
    # LAYER 2: THETA PLV (gate coherence)
    # ─────────────────────────────────────────────────────────
    frontal_data = data[valid_frontal, :]
    theta_phases = []
    for ch_idx in range(len(valid_frontal)):
        sig = frontal_data[ch_idx, :]
        filtered = mne.filter.filter_data(sig, sfreq, l_freq=THETA_BAND[0],
                                           h_freq=THETA_BAND[1], verbose=False)
        analytic = hilbert(filtered)
        theta_phases.append(np.angle(analytic))

    plv_scores = []
    for w in range(10):
        start = w * window_samples
        end = start + window_samples
        if end > frontal_data.shape[1]:
            break
        plvs = []
        for i in range(len(theta_phases)):
            for j in range(i + 1, len(theta_phases)):
                diff = theta_phases[i][start:end] - theta_phases[j][start:end]
                plvs.append(np.abs(np.mean(np.exp(1j * diff))))
        if plvs:
            plv_scores.append(np.mean(plvs))

    theta_plv = np.mean(plv_scores) if plv_scores else None

    # ─────────────────────────────────────────────────────────
    # LAYER 3: Φ-DWELL EIGENMODE VOCABULARY
    # ─────────────────────────────────────────────────────────
    graph_mapping = map_channels(raw.ch_names, graph_names)
    if len(graph_mapping) < 10:
        return None

    n_elec = len(graph_names)
    word_step_ms = 25
    step_samp = max(1, int(sfreq * word_step_ms / 1000))

    # Band-filter, extract phases, project onto eigenmodes
    tokens = np.zeros((5, n_samp), dtype=np.int32)

    for bi, band in enumerate(BAND_NAMES):
        lo, hi = BANDS[band]
        if hi >= sfreq / 2:
            hi = sfreq / 2 - 1
        if lo >= hi:
            continue

        sos = scipy.signal.butter(3, [lo, hi], btype='band', fs=sfreq, output='sos')

        # Phase per electrode
        phases = np.zeros((n_elec, n_samp), dtype=np.complex64)
        for raw_ch, graph_ch in graph_mapping.items():
            idx_g = graph_names.index(graph_ch)
            idx_r = raw.ch_names.index(raw_ch)
            sig = scipy.signal.sosfiltfilt(sos, data[idx_r, :n_samp])
            analytic = scipy.signal.hilbert(sig)
            phases[idx_g, :] = analytic / (np.abs(analytic) + 1e-9)

        # Project onto eigenmodes → dominant mode per timestep
        coeffs = np.abs(phases.T @ eigenvecs)  # (n_samp, N_MODES)
        tokens[bi, :] = np.argmax(coeffs, axis=1)

    # Downsample to word steps
    tokens_ds = tokens[:, ::step_samp]
    n_words = tokens_ds.shape[1]

    # Build words (5-tuples)
    words = [tuple(tokens_ds[:, t]) for t in range(n_words)]
    counts = Counter(words)

    # Vocabulary size
    vocab_size = len(counts)

    # Shannon entropy
    probs = np.array(list(counts.values())) / n_words
    entropy = -np.sum(probs * np.log2(probs + 1e-15))

    # Zipf exponent
    freqs = np.array(sorted(counts.values(), reverse=True))
    ranks = np.arange(1, len(freqs) + 1)
    zipf_alpha = 0.0
    zipf_r2 = 0.0
    if len(freqs) > 5:
        n_fit = min(50, len(freqs))
        slope, _, r, _, _ = stats.linregress(np.log(ranks[:n_fit]), np.log(freqs[:n_fit]))
        zipf_alpha = -slope
        zipf_r2 = r ** 2

    # Top-5 concentration
    top5_frac = sum(c for _, c in counts.most_common(5)) / n_words

    # Self-transition rate
    n_self = sum(1 for a, b in zip(words[:-1], words[1:]) if a == b)
    self_rate = n_self / (n_words - 1) if n_words > 1 else 0

    # Bigram perplexity
    bigram_counts = defaultdict(Counter)
    context_totals = defaultdict(int)
    for i in range(len(words) - 1):
        ctx = (words[i],)
        bigram_counts[ctx][words[i + 1]] += 1
        context_totals[ctx] += 1

    V = max(vocab_size, 1)
    k = 0.01
    total_log_prob = 0
    n_scored = 0
    for i in range(1, len(words)):
        ctx = (words[i - 1],)
        word = words[i]
        if ctx in bigram_counts:
            count = bigram_counts[ctx].get(word, 0)
            total = context_totals[ctx]
            prob = (count + k) / (total + k * V)
        else:
            prob = (counts.get(word, 0) + k) / (n_words + k * V)
        total_log_prob += np.log2(prob + 1e-15)
        n_scored += 1
    perplexity = 2.0 ** (-total_log_prob / n_scored) if n_scored > 0 else float('inf')

    # Per-band dwell times and CV (criticality)
    band_cv = {}
    band_mean_dwell = {}
    for bi, band in enumerate(BAND_NAMES):
        bt = tokens_ds[bi, :]
        dwells = []
        current_len = 1
        for t in range(1, len(bt)):
            if bt[t] == bt[t - 1]:
                current_len += 1
            else:
                dwells.append(current_len * word_step_ms)
                current_len = 1
        dwells.append(current_len * word_step_ms)
        dwells = np.array(dwells, dtype=float)
        if len(dwells) > 2:
            band_cv[band] = float(np.std(dwells) / (np.mean(dwells) + 1e-9))
            band_mean_dwell[band] = float(np.mean(dwells))
        else:
            band_cv[band] = 0.0
            band_mean_dwell[band] = 0.0

    mean_cv = np.mean(list(band_cv.values()))
    n_critical = sum(1 for v in band_cv.values() if v > 1.0)
    criticality_fraction = n_critical / 5

    # Cross-band coupling (correlation of dominant mode across bands)
    coupling_matrix = np.corrcoef(tokens_ds.astype(float))
    mean_coupling = np.mean(coupling_matrix[np.triu_indices(5, k=1)])
    delta_theta_coupling = coupling_matrix[0, 1] if coupling_matrix.shape[0] >= 2 else 0.0

    # Top words
    top_words = [''.join(str(x + 1) for x in w) for w, _ in counts.most_common(5)]

    return {
        # Layer 1: Topology
        'betti1': betti1,
        # Layer 2: Gate coherence
        'theta_plv': theta_plv,
        # Layer 3: Eigenmode vocabulary
        'vocab_size': vocab_size,
        'entropy': entropy,
        'perplexity': perplexity,
        'zipf_alpha': zipf_alpha,
        'zipf_r2': zipf_r2,
        'top5_concentration': top5_frac,
        'self_rate': self_rate,
        'mean_cv': mean_cv,
        'criticality_fraction': criticality_fraction,
        'mean_coupling': mean_coupling,
        'delta_theta_coupling': delta_theta_coupling,
        'band_cv': band_cv,
        'band_mean_dwell': band_mean_dwell,
        'top_words': top_words,
        'n_words': n_words,
        'n_channels': len(graph_mapping),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"{'=' * 80}")
    print("DEERSKIN TEST #3: Φ-DWELL × BETTI-1 UNIFIED CONSCIOUSNESS PROBE")
    print(f"{'=' * 80}")
    print()
    print("Three layers of measurement on the same brains:")
    print("  Layer 1: Betti-1 topological complexity (phase-space holes)")
    print("  Layer 2: Theta PLV (gate synchrony)")
    print("  Layer 3: Eigenmode vocabulary & grammar (Φ-Dwell)")
    print()
    print("PREDICTIONS:")
    print("  SZ → higher vocabulary, higher concentration, steeper Zipf")
    print("  SZ → CV deviates from 1.0 (supercritical / rigid)")
    print("  Betti-1 correlates with vocabulary size across all subjects")
    print("  Theta PLV correlates with CV rigidity")
    print()

    # Build eigenmode basis
    graph_names, coords, eigenvecs, eigenvals = build_graph_laplacian(ELECTRODE_POS_19)
    print(f"Eigenmode basis: {len(graph_names)} electrodes, {N_MODES} modes")
    print(f"Eigenvalues: {eigenvals.round(2)}")

    # Download data
    edf_files = download_repod_dataset()
    if not edf_files:
        return

    # Process all subjects
    hc_results = []
    sz_results = []

    print("\nAnalyzing all subjects (3 layers per subject)...")
    for filepath in tqdm(edf_files, desc="Processing"):
        filename = os.path.basename(filepath).lower()
        result = analyze_subject_unified(filepath, graph_names, eigenvecs)
        if result is None:
            continue
        if filename.startswith('h'):
            result['group'] = 'HC'
            hc_results.append(result)
        elif filename.startswith('s'):
            result['group'] = 'SZ'
            sz_results.append(result)

    if len(hc_results) < 3 or len(sz_results) < 3:
        print("Insufficient data.")
        return

    all_results = hc_results + sz_results

    # ═══════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'=' * 80}")
    print("LAYER 1 REPLICATION: BETTI-1 TOPOLOGICAL COMPLEXITY")
    print(f"{'=' * 80}")
    hc_b = [r['betti1'] for r in hc_results if r['betti1'] is not None]
    sz_b = [r['betti1'] for r in sz_results if r['betti1'] is not None]
    print(f"  HC (n={len(hc_b)}): Mean = {np.mean(hc_b):.3f}, SD = {np.std(hc_b):.3f}")
    print(f"  SZ (n={len(sz_b)}): Mean = {np.mean(sz_b):.3f}, SD = {np.std(sz_b):.3f}")
    t, p = stats.ttest_ind(hc_b, sz_b)
    print(f"  t = {t:.3f}, p = {p:.4f}")

    print(f"\n{'=' * 80}")
    print("LAYER 2 REPLICATION: THETA PLV (GATE COHERENCE)")
    print(f"{'=' * 80}")
    hc_t = [r['theta_plv'] for r in hc_results if r['theta_plv'] is not None]
    sz_t = [r['theta_plv'] for r in sz_results if r['theta_plv'] is not None]
    print(f"  HC (n={len(hc_t)}): Mean = {np.mean(hc_t):.4f}, SD = {np.std(hc_t):.4f}")
    print(f"  SZ (n={len(sz_t)}): Mean = {np.mean(sz_t):.4f}, SD = {np.std(sz_t):.4f}")
    t, p = stats.ttest_ind(hc_t, sz_t)
    print(f"  t = {t:.3f}, p = {p:.4f}")

    # ─────────────────────────────────────────────────────────
    # LAYER 3: Φ-DWELL EIGENMODE RESULTS
    # ─────────────────────────────────────────────────────────
    metrics_to_test = [
        ('vocab_size', 'Vocabulary Size'),
        ('entropy', 'Shannon Entropy'),
        ('perplexity', 'Bigram Perplexity'),
        ('top5_concentration', 'Top-5 Concentration'),
        ('zipf_alpha', 'Zipf α'),
        ('self_rate', 'Self-Transition Rate'),
        ('mean_cv', 'Mean CV (Criticality)'),
        ('criticality_fraction', 'Criticality Fraction'),
        ('mean_coupling', 'Mean Cross-Band Coupling'),
        ('delta_theta_coupling', 'δ-θ Coupling'),
    ]

    print(f"\n{'=' * 80}")
    print("LAYER 3: Φ-DWELL EIGENMODE VOCABULARY")
    print(f"{'=' * 80}")
    print(f"\n  {'Metric':<26} {'HC':>10} {'SZ':>10} {'t':>8} {'p':>10} {'Direction':>12}")
    print(f"  {'─' * 78}")

    for metric, title in metrics_to_test:
        hc_vals = [r[metric] for r in hc_results if r.get(metric) is not None and np.isfinite(r[metric])]
        sz_vals = [r[metric] for r in sz_results if r.get(metric) is not None and np.isfinite(r[metric])]
        if len(hc_vals) < 3 or len(sz_vals) < 3:
            continue
        t_stat, p_val = stats.ttest_ind(hc_vals, sz_vals)
        direction = "SZ > HC" if np.mean(sz_vals) > np.mean(hc_vals) else "HC > SZ"
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "†" if p_val < 0.1 else ""

        hc_m = np.mean(hc_vals)
        sz_m = np.mean(sz_vals)
        if hc_m > 100:
            print(f"  {title:<26} {hc_m:>9.0f} {sz_m:>9.0f} {t_stat:>+7.2f} {p_val:>9.4f} {sig:>2} {direction:>10}")
        else:
            print(f"  {title:<26} {hc_m:>9.3f} {sz_m:>9.3f} {t_stat:>+7.2f} {p_val:>9.4f} {sig:>2} {direction:>10}")

    # ─────────────────────────────────────────────────────────
    # PER-BAND DWELL TIMES AND CV
    # ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("PER-BAND DWELL ANALYSIS")
    print(f"{'=' * 80}")
    print(f"\n  {'Band':<8} {'HC Dwell':>10} {'SZ Dwell':>10} {'HC CV':>8} {'SZ CV':>8} {'Dwell-p':>10} {'CV-p':>10}")
    print(f"  {'─' * 68}")

    for band in BAND_NAMES:
        hc_dw = [r['band_mean_dwell'][band] for r in hc_results if r['band_mean_dwell'].get(band)]
        sz_dw = [r['band_mean_dwell'][band] for r in sz_results if r['band_mean_dwell'].get(band)]
        hc_cv = [r['band_cv'][band] for r in hc_results if r['band_cv'].get(band)]
        sz_cv = [r['band_cv'][band] for r in sz_results if r['band_cv'].get(band)]

        _, p_dw = stats.ttest_ind(hc_dw, sz_dw) if len(hc_dw) >= 3 and len(sz_dw) >= 3 else (0, 1)
        _, p_cv = stats.ttest_ind(hc_cv, sz_cv) if len(hc_cv) >= 3 and len(sz_cv) >= 3 else (0, 1)

        sig_dw = "*" if p_dw < 0.05 else "†" if p_dw < 0.1 else ""
        sig_cv = "*" if p_cv < 0.05 else "†" if p_cv < 0.1 else ""

        print(f"  {band:<8} {np.mean(hc_dw):>8.1f}ms {np.mean(sz_dw):>8.1f}ms "
              f"{np.mean(hc_cv):>7.3f} {np.mean(sz_cv):>7.3f} "
              f"{p_dw:>9.4f}{sig_dw:>1} {p_cv:>9.4f}{sig_cv:>1}")

    # ─────────────────────────────────────────────────────────
    # CROSS-LAYER CORRELATIONS (the key test)
    # ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("CROSS-LAYER CORRELATIONS (ALL 28 SUBJECTS)")
    print("This is the core test: do topology, gate coherence, and grammar connect?")
    print(f"{'=' * 80}")

    all_betti = [r['betti1'] for r in all_results if r['betti1'] is not None]
    all_plv = [r['theta_plv'] for r in all_results if r['theta_plv'] is not None]
    all_vocab = [r['vocab_size'] for r in all_results]
    all_entropy = [r['entropy'] for r in all_results]
    all_cv = [r['mean_cv'] for r in all_results]
    all_zipf = [r['zipf_alpha'] for r in all_results]
    all_top5 = [r['top5_concentration'] for r in all_results]
    all_perp = [r['perplexity'] for r in all_results]

    correlation_pairs = [
        ('Betti-1', 'Vocabulary', all_betti, all_vocab),
        ('Betti-1', 'Entropy', all_betti, all_entropy),
        ('Betti-1', 'Mean CV', all_betti, all_cv),
        ('Betti-1', 'Zipf α', all_betti, all_zipf),
        ('Betti-1', 'Top-5 Conc.', all_betti, all_top5),
        ('Betti-1', 'Perplexity', all_betti, all_perp),
        ('Betti-1', 'Theta PLV', all_betti, all_plv),
        ('Theta PLV', 'Mean CV', all_plv, all_cv),
        ('Theta PLV', 'Vocabulary', all_plv, all_vocab),
        ('Theta PLV', 'Zipf α', all_plv, all_zipf),
        ('Vocabulary', 'Mean CV', all_vocab, all_cv),
        ('Vocabulary', 'Entropy', all_vocab, all_entropy),
        ('Entropy', 'Perplexity', all_entropy, all_perp),
    ]

    print(f"\n  {'Pair':<30} {'Pearson r':>10} {'p':>10} {'Spearman ρ':>12} {'p':>10}")
    print(f"  {'─' * 74}")

    for name1, name2, v1, v2 in correlation_pairs:
        n = min(len(v1), len(v2))
        if n < 5:
            continue
        x, y = np.array(v1[:n]), np.array(v2[:n])
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 5:
            continue

        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)
        sig = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "†" if p_p < 0.1 else ""
        print(f"  {name1} × {name2:<18} {r_p:>+9.3f} {p_p:>9.4f}{sig:>2} {r_s:>+11.3f} {p_s:>9.4f}")

    # ─────────────────────────────────────────────────────────
    # WITHIN-GROUP CORRELATIONS
    # ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("WITHIN-GROUP: BETTI-1 × VOCABULARY")
    print(f"{'=' * 80}")

    for label, group_results in [("HC", hc_results), ("SZ", sz_results)]:
        b = [r['betti1'] for r in group_results if r['betti1'] is not None]
        v = [r['vocab_size'] for r in group_results if r['betti1'] is not None]
        if len(b) >= 5:
            r, p = stats.pearsonr(b, v)
            print(f"  {label} (n={len(b)}): Pearson r = {r:+.3f}, p = {p:.4f}")

    # ─────────────────────────────────────────────────────────
    # INDIVIDUAL SUBJECT TABLE
    # ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("INDIVIDUAL SUBJECT DATA")
    print(f"{'=' * 80}")
    print(f"  {'Subj':<8} {'Grp':<4} {'Betti-1':>8} {'θ PLV':>8} {'Vocab':>6} {'Entropy':>8} "
          f"{'CV':>6} {'Zipf α':>7} {'Top5%':>6} {'PPlex':>7} {'TopWord':>8}")
    print(f"  {'─' * 95}")

    for i, r in enumerate(hc_results):
        b = r['betti1'] if r['betti1'] is not None else 0
        t = r['theta_plv'] if r['theta_plv'] is not None else 0
        tw = r['top_words'][0] if r['top_words'] else '?'
        print(f"  HC_{i+1:<5} {'HC':<4} {b:>7.2f} {t:>7.4f} {r['vocab_size']:>5} {r['entropy']:>7.2f} "
              f"{r['mean_cv']:>5.3f} {r['zipf_alpha']:>6.3f} {r['top5_concentration']:>5.1%} "
              f"{r['perplexity']:>6.1f} {tw:>8}")

    for i, r in enumerate(sz_results):
        b = r['betti1'] if r['betti1'] is not None else 0
        t = r['theta_plv'] if r['theta_plv'] is not None else 0
        tw = r['top_words'][0] if r['top_words'] else '?'
        print(f"  SZ_{i+1:<5} {'SZ':<4} {b:>7.2f} {t:>7.4f} {r['vocab_size']:>5} {r['entropy']:>7.2f} "
              f"{r['mean_cv']:>5.3f} {r['zipf_alpha']:>6.3f} {r['top5_concentration']:>5.1%} "
              f"{r['perplexity']:>6.1f} {tw:>8}")

    # ─────────────────────────────────────────────────────────
    # INTERPRETATION
    # ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("DEERSKIN INTERPRETATION: THREE LAYERS OF CONSCIOUSNESS")
    print(f"{'=' * 80}")
    print("""
The three measurement layers capture different aspects of a single process:

  LAYER 1 — Betti-1 (Topology):
    How many holes in the phase-space attractor?
    → Measures the COMPLEXITY of the brain's geometric field.

  LAYER 2 — Theta PLV (Gate):
    How synchronously does the theta gate open/close across frontal cortex?
    → Measures the COHERENCE of the temporal segmentation mechanism.

  LAYER 3 — Φ-Dwell (Grammar):
    How does the brain move through its eigenmode configuration space?
    → Measures the VOCABULARY and SYNTAX of the macroscopic field trajectory.

CONSCIOUSNESS = Coherent field (PLV > threshold)
              + Critical dynamics (CV ≈ 1.0)
              + Structured grammar (intermediate perplexity)
              ...all holding simultaneously across the full band hierarchy.

SCHIZOPHRENIA disrupts this as a HIJACKED GATE:
  → Betti-1 ↑ (geometric overflow — more topology in the field)
  → Theta PLV ↑ and rigid (gate works, but serves internal predictions)
  → Vocabulary/grammar shifts (eigenmode trajectories restructured)

ALZHEIMER'S disrupts this as a DEGRADED MANIFOLD:
  → Vocabulary ↑ but structure ↓ (more words, less concentration, flatter Zipf)
  → CV drops from criticality (can't hold a pattern)
  → "The brain can no longer surf — it falls off the wave"

The fractal surfer is the coherent trajectory through multi-scale
eigenmode space. Health is staying on the wave. Disease is either
being hijacked by an internal wave (schizophrenia) or losing the
wave entirely (Alzheimer's).
""")


if __name__ == '__main__':
    main()
