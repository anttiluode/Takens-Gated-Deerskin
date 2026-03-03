"""
DEERSKIN TEST #2: THETA GATE INTEGRITY vs. TOPOLOGICAL COMPLEXITY
=================================================================
Prediction: If the Theta Phase Gate is what segments the geometric stream,
then broken theta coherence should correlate with elevated Betti-1.

Specifically:
  - Schizophrenia subjects should have LOWER frontal theta phase coherence
  - Within ALL subjects, theta coherence should NEGATIVELY correlate with Betti-1
    (worse gate = more geometric overflow)

This runs on the same RepOD dataset as schizophrenia.py — it will reuse
already-downloaded files if present.

Antti Luode (PerceptionLab | Independent Research, Finland)
Claude (Anthropic, Collaborative formalization)
March 2026
"""

import os
import requests
import numpy as np
import scipy.stats as stats
import mne
from scipy.signal import hilbert
from ripser import ripser
from tqdm import tqdm
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")

FRONTAL_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'F7', 'F8']
THETA_BAND = (4.0, 8.0)

def download_repod_schizophrenia_dataset(download_dir="repod_sz_eeg"):
    """Reuses the same download logic — skips if files already exist."""
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    print("Querying RepOD Dataverse API for Schizophrenia dataset...")
    api_url = "https://repod.icm.edu.pl/api/datasets/:persistentId/?persistentId=doi:10.18150/repod.0107441"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        metadata = response.json()
        files = metadata['data']['latestVersion']['files']
    except Exception as e:
        print(f"Failed to fetch dataset metadata: {e}")
        return []

    edf_files = []
    print(f"Found {len(files)} files. Downloading EDFs...")

    for f in tqdm(files, desc="Downloading"):
        filename = f['dataFile']['filename']
        if not filename.endswith('.edf'):
            continue
        filepath = os.path.join(download_dir, filename)
        edf_files.append(filepath)

        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000000:
            continue

        file_id = f['dataFile']['id']
        download_url = f"https://repod.icm.edu.pl/api/access/datafile/{file_id}"
        file_data = requests.get(download_url, stream=True)
        with open(filepath, 'wb') as out_file:
            for chunk in file_data.iter_content(chunk_size=8192):
                out_file.write(chunk)

    return edf_files


def takens_embedding(signal, delay_samples, dim=3, max_pts=400):
    """Same Takens embedding as the original script."""
    N = len(signal)
    if N < (dim - 1) * delay_samples + 1:
        return np.array([])

    embedded = np.array([signal[i : i + (dim - 1) * delay_samples + 1 : delay_samples]
                         for i in range(N - (dim - 1) * delay_samples)])

    if len(embedded) > max_pts:
        indices = np.linspace(0, len(embedded) - 1, max_pts, dtype=int)
        embedded = embedded[indices]

    embedded = embedded - np.mean(embedded, axis=0)
    std_dev = np.std(embedded, axis=0)
    std_dev[std_dev == 0] = 1.0
    return embedded / std_dev


def compute_topology_score(embedded):
    """Same Betti-1 persistence as the original script."""
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
        sig_lifetimes = lifetimes[lifetimes > 0.1 * max_life]
        return np.sum(sig_lifetimes) + (0.1 * len(sig_lifetimes))
    except Exception:
        return 0.0


def compute_theta_phase_coherence(signals, sfreq):
    """
    Computes mean pairwise Phase-Locking Value (PLV) in the theta band
    across frontal channels.
    
    PLV measures how consistently two signals maintain a fixed phase
    relationship. High PLV = theta oscillations are synchronized = 
    the gate is opening and closing in unison across frontal cortex.
    Low PLV = desynchronized = the gate is fragmented.
    """
    n_channels = len(signals)
    if n_channels < 2:
        return np.nan

    # Extract theta-band analytic signal for each channel
    theta_phases = []
    for sig in signals:
        # Bandpass filter to theta
        filtered = mne.filter.filter_data(sig, sfreq, l_freq=THETA_BAND[0],
                                          h_freq=THETA_BAND[1], verbose=False)
        analytic = hilbert(filtered)
        phase = np.angle(analytic)
        theta_phases.append(phase)

    # Compute pairwise PLV
    plv_values = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = theta_phases[i] - theta_phases[j]
            # PLV = magnitude of mean phase difference vector
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_values.append(plv)

    return np.mean(plv_values)


def process_subject(filepath):
    """
    Extracts BOTH the Betti-1 topological score AND theta phase coherence
    from a single subject's frontal EEG.
    
    Returns: (betti1_score, theta_plv) or (None, None)
    """
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
        data = raw.get_data()
        ch_map = {ch.upper(): i for i, ch in enumerate(raw.ch_names)}

        valid_chs = [ch_map[c.upper()] for c in FRONTAL_CHANNELS if c.upper() in ch_map]
        if len(valid_chs) < 2:
            return None, None

        # --- BETTI-1 SCORE (same as original) ---
        frontal_signal = np.mean(data[valid_chs, :], axis=0)
        window_samples = int(2.0 * sfreq)
        delays_samples = [int(d * sfreq / 1000.0) for d in [10, 20, 40]]

        topo_scores = []
        for w in range(10):
            start = w * window_samples
            end = start + window_samples
            if end > len(frontal_signal):
                break
            window_sig = frontal_signal[start:end]
            delay_scores = []
            for d in delays_samples:
                emb = takens_embedding(window_sig, d)
                delay_scores.append(compute_topology_score(emb))
            topo_scores.append(np.mean(delay_scores))

        betti1 = np.mean(topo_scores) if topo_scores else None

        # --- THETA PHASE COHERENCE (new measure) ---
        # Use the same 10 x 2-second windows, compute PLV per window, average
        frontal_channels_data = data[valid_chs, :]
        plv_scores = []
        for w in range(10):
            start = w * window_samples
            end = start + window_samples
            if end > frontal_channels_data.shape[1]:
                break
            window_channels = [frontal_channels_data[ch, start:end] for ch in range(len(valid_chs))]
            plv = compute_theta_phase_coherence(window_channels, sfreq)
            if not np.isnan(plv):
                plv_scores.append(plv)

        theta_plv = np.mean(plv_scores) if plv_scores else None

        return betti1, theta_plv

    except Exception as e:
        return None, None


def main():
    print(f"{'='*80}")
    print("DEERSKIN TEST #2: THETA GATE INTEGRITY vs. TOPOLOGICAL COMPLEXITY")
    print(f"{'='*80}")
    print()
    print("PREDICTIONS (from Deerskin Architecture):")
    print("  1. SZ subjects have LOWER frontal theta phase coherence (broken gate)")
    print("  2. Across ALL subjects, theta coherence NEGATIVELY correlates with Betti-1")
    print("     (worse gate synchrony → more geometric overflow)")
    print()

    edf_files = download_repod_schizophrenia_dataset()
    if not edf_files:
        return

    # Collect per-subject data
    results = {'HC': [], 'SZ': []}

    print("\nExtracting topology + theta coherence from each subject...")
    for filepath in tqdm(edf_files, desc="Processing"):
        filename = os.path.basename(filepath).lower()
        betti1, theta_plv = process_subject(filepath)

        if betti1 is None or theta_plv is None:
            continue

        if filename.startswith('h'):
            results['HC'].append((betti1, theta_plv))
        elif filename.startswith('s'):
            results['SZ'].append((betti1, theta_plv))

    hc = results['HC']
    sz = results['SZ']

    if len(hc) < 3 or len(sz) < 3:
        print("Insufficient data.")
        return

    hc_betti = [x[0] for x in hc]
    hc_theta = [x[1] for x in hc]
    sz_betti = [x[0] for x in sz]
    sz_theta = [x[1] for x in sz]

    all_betti = hc_betti + sz_betti
    all_theta = hc_theta + sz_theta

    # =====================================================================
    # TEST 1: GROUP DIFFERENCE IN THETA COHERENCE
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 1: THETA PHASE COHERENCE (Frontal PLV in 4-8 Hz)")
    print(f"{'='*80}")
    print(f"  Healthy Controls (n={len(hc)}):  Mean PLV = {np.mean(hc_theta):.4f}  (SD = {np.std(hc_theta):.4f})")
    print(f"  Schizophrenia    (n={len(sz)}):  Mean PLV = {np.mean(sz_theta):.4f}  (SD = {np.std(sz_theta):.4f})")

    t_theta, p_theta = stats.ttest_ind(hc_theta, sz_theta)
    print(f"\n  t-test: t = {t_theta:.3f}, p = {p_theta:.4f}")

    if p_theta < 0.1 and np.mean(sz_theta) < np.mean(hc_theta):
        print("  → PREDICTION 1 SUPPORTED: SZ shows reduced theta phase coherence.")
    elif p_theta < 0.1 and np.mean(sz_theta) > np.mean(hc_theta):
        print("  → PREDICTION 1 REVERSED: SZ shows elevated theta coherence (unexpected).")
    else:
        print("  → PREDICTION 1 NOT CONFIRMED at this sample size.")

    # =====================================================================
    # TEST 2: CORRELATION — THETA PLV vs. BETTI-1 (ALL SUBJECTS)
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 2: CORRELATION — THETA COHERENCE vs. TOPOLOGICAL COMPLEXITY")
    print(f"{'='*80}")

    r_all, p_all = stats.pearsonr(all_theta, all_betti)
    r_spearman, p_spearman = stats.spearmanr(all_theta, all_betti)

    print(f"  All subjects (n={len(all_betti)}):")
    print(f"    Pearson  r = {r_all:.3f},  p = {p_all:.4f}")
    print(f"    Spearman ρ = {r_spearman:.3f},  p = {p_spearman:.4f}")

    if p_all < 0.1 and r_all < 0:
        print("  → PREDICTION 2 SUPPORTED: Negative correlation — worse theta gate = more topology.")
    elif p_all < 0.1 and r_all > 0:
        print("  → PREDICTION 2 REVERSED: Positive correlation (unexpected).")
    else:
        print("  → PREDICTION 2 NOT CONFIRMED at this sample size.")

    # =====================================================================
    # TEST 3: WITHIN-GROUP CORRELATIONS
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 3: WITHIN-GROUP CORRELATIONS")
    print(f"{'='*80}")

    for label, b, t in [("HC", hc_betti, hc_theta), ("SZ", sz_betti, sz_theta)]:
        if len(b) >= 5:
            r, p = stats.pearsonr(t, b)
            print(f"  {label} (n={len(b)}): Pearson r = {r:.3f}, p = {p:.4f}")
        else:
            print(f"  {label}: Too few subjects for within-group correlation.")

    # =====================================================================
    # REPLICATION OF ORIGINAL BETTI-1 RESULT
    # =====================================================================
    print(f"\n{'='*80}")
    print("REPLICATION: BETTI-1 GROUP DIFFERENCE (same as schizophrenia.py)")
    print(f"{'='*80}")
    print(f"  HC Mean Betti-1: {np.mean(hc_betti):.3f}")
    print(f"  SZ Mean Betti-1: {np.mean(sz_betti):.3f}")
    t_b, p_b = stats.ttest_ind(hc_betti, sz_betti)
    print(f"  t = {t_b:.3f}, p = {p_b:.4f}")

    # =====================================================================
    # INDIVIDUAL SUBJECT DATA
    # =====================================================================
    print(f"\n{'='*80}")
    print("INDIVIDUAL SUBJECT DATA")
    print(f"{'='*80}")
    print(f"  {'Subject':<12} {'Group':<6} {'Betti-1':>10} {'Theta PLV':>12}")
    print(f"  {'-'*42}")

    for i, (b, t) in enumerate(hc):
        print(f"  HC_{i+1:<8} {'HC':<6} {b:>10.3f} {t:>12.4f}")
    for i, (b, t) in enumerate(sz):
        print(f"  SZ_{i+1:<8} {'SZ':<6} {b:>10.3f} {t:>12.4f}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*80}")
    print("DEERSKIN INTERPRETATION")
    print(f"{'='*80}")
    print("""
If both predictions hold, this demonstrates the MECHANISM, not just the symptom:
  - Test 1 (Betti-1 ↑ in SZ) shows the EFFECT: geometric overflow
  - Test 2 (Theta PLV ↓ in SZ) shows the CAUSE: broken phase gate
  - Correlation (PLV↓ ↔ Betti-1↑) shows they are LINKED within subjects

The Theta Phase Gate segments the continuous geometric stream into
processable frames. When it desynchronizes, the geometry piles up.
The hallucination is not a chemical event — it is an unlocked gate.
""")


if __name__ == '__main__':
    main()