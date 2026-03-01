#!/usr/bin/env python3
"""
ALPHA CONTROL — PREDICTION 5.4 BATCH TEST
==========================================
Identical pipeline to batch_runner.py but with ONE change:
  Before computing topology, we REMOVE the alpha band (8–13 Hz)
  using a 4th-order Butterworth bandstop (notch) filter.

If the inverse correlation (diversity ↔ lower persistence) survives
alpha removal, then the effect is NOT driven by occipital alpha rhythm.
If it disappears, alpha was the explanation all along.

This is the single most important control experiment for the result.

Usage:
    python alpha_control_batch.py <folder_with_edfs>
    python alpha_control_batch.py <folder> --sim
    python alpha_control_batch.py <folder> --max-files 6

Also runs a COMPARISON against the unfiltered results if
batch_prediction_5_4_results.json exists in the current directory.

Authors: Luode & Claude, March 1, 2026
"""

import numpy as np
import sys
import os
import glob
import json
import time
import argparse
import re
from collections import defaultdict
from scipy import signal as sig
from scipy.stats import spearmanr, wilcoxon, ttest_1samp

try:
    import mne
    mne.set_log_level('ERROR')
except ImportError:
    print("ERROR: pip install mne"); sys.exit(1)

try:
    from ripser import ripser
except ImportError:
    print("ERROR: pip install ripser"); sys.exit(1)


# ================================================================
# CHANNEL-TO-REGION MAPPING (identical to batch_runner.py)
# ================================================================

REGION_CHANNELS = {
    'Frontal': {
        'channels': ['FP1','FP2','FPZ','AF3','AF4','AF7','AF8',
                      'F1','F2','F3','F4','F5','F6','F7','F8','FZ',
                      'Fp1','Fp2','Fpz','Fz'],
        'diversity_rank': 5,
    },
    'Temporal': {
        'channels': ['T3','T4','T5','T6','T7','T8',
                      'TP7','TP8','TP9','TP10','FT7','FT8','FT9','FT10'],
        'diversity_rank': 4,
    },
    'Parietal': {
        'channels': ['P1','P2','P3','P4','P5','P6','P7','P8','PZ',
                      'CP1','CP2','CP3','CP4','CP5','CP6','CPZ',
                      'POZ','PO3','PO4','PO7','PO8','Pz'],
        'diversity_rank': 3,
    },
    'Central': {
        'channels': ['C1','C2','C3','C4','C5','C6','CZ',
                      'FC1','FC2','FC3','FC4','FC5','FC6','FCZ','Cz'],
        'diversity_rank': 2,
    },
    'Occipital': {
        'channels': ['O1','O2','OZ','Oz'],
        'diversity_rank': 1,
    },
}

REGION_ORDER = ['Occipital', 'Central', 'Parietal', 'Temporal', 'Frontal']


# ================================================================
# TOPOLOGY FUNCTIONS (identical to batch_runner.py)
# ================================================================

def takens_embed(x, tau=10, dim=3, max_points=400):
    n = len(x) - (dim - 1) * tau
    if n <= 0:
        return np.zeros((1, dim))
    embedded = np.zeros((n, dim))
    for d in range(dim):
        embedded[:, d] = x[d * tau: d * tau + n]
    if len(embedded) > max_points:
        idx = np.linspace(0, len(embedded)-1, max_points, dtype=int)
        embedded = embedded[idx]
    std = embedded.std(axis=0)
    std[std < 1e-12] = 1.0
    return (embedded - embedded.mean(axis=0)) / std


def compute_topology(signal_1d, fs, n_windows=6, window_sec=2.0,
                     taus_ms=[10, 20, 40], dim=3, max_points=400):
    window_samples = int(window_sec * fs)
    max_start = len(signal_1d) - window_samples
    if max_start <= 0:
        return {'score': 0.0, 'h1': 0.0, 'persistence': 0.0}

    taus = [max(1, int(t * fs / 1000)) for t in taus_ms]
    n_windows = min(n_windows, max(1, max_start // (window_samples // 2)))
    starts = np.linspace(0, max_start, n_windows, dtype=int)

    scores, h1s, persis = [], [], []

    for start in starts:
        segment = signal_1d[start:start + window_samples]
        w_h1, w_p = [], []
        for tau in taus:
            cloud = takens_embed(segment, tau=tau, dim=dim, max_points=max_points)
            if len(cloud) < 10:
                w_h1.append(0); w_p.append(0.0); continue
            try:
                result = ripser(cloud, maxdim=1, thresh=2.0)
            except:
                w_h1.append(0); w_p.append(0.0); continue
            dgm = result['dgms'][1]
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) > 0:
                lt = finite[:, 1] - finite[:, 0]
                thr = max(0.05, 0.1 * np.max(lt))
                sig_f = lt[lt > thr]
                w_h1.append(len(sig_f))
                w_p.append(float(np.sum(sig_f)))
            else:
                w_h1.append(0); w_p.append(0.0)

        scores.append(np.mean(w_p) + 0.1 * np.mean(w_h1))
        h1s.append(np.mean(w_h1))
        persis.append(np.mean(w_p))

    return {
        'score': float(np.mean(scores)),
        'h1': float(np.mean(h1s)),
        'persistence': float(np.mean(persis)),
        'score_std': float(np.std(scores)),
    }


# ================================================================
# THE CRITICAL ADDITION: ALPHA REMOVAL FILTER
# ================================================================

def remove_alpha(signal_1d, fs, low=8.0, high=13.0, order=4):
    """
    Bandstop (notch) filter to remove alpha rhythm (8-13 Hz).
    
    This is the control: if the inverse correlation between morphological
    diversity and topological persistence is driven by occipital alpha
    creating artificially persistent loops in the Takens embedding,
    then removing alpha should eliminate the effect.
    
    If the effect SURVIVES alpha removal, it is not an alpha artifact.
    """
    nyq = fs / 2.0
    if high >= nyq:
        high = nyq - 1.0
    if low >= high:
        return signal_1d
    
    b, a = sig.butter(order, [low / nyq, high / nyq], btype='bandstop')
    return sig.filtfilt(b, a, signal_1d)


# ================================================================
# PROCESS ONE FILE — WITH ALPHA REMOVAL
# ================================================================

def process_edf(filepath, max_duration=0, filter_alpha=True, filter_range=(8.0, 13.0)):
    """
    Same as batch_runner.py process_edf, but with alpha bandstop filter
    applied AFTER the broadband filter and BEFORE topology computation.
    """
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return None

    raw.rename_channels(lambda name: name.strip().replace('.', '').upper())
    fs = raw.info['sfreq']

    if max_duration > 0:
        raw.crop(tmax=min(max_duration, raw.n_times / fs))

    all_ch_upper = [ch.upper() for ch in raw.ch_names]
    region_results = {}

    for rname in REGION_ORDER:
        info = REGION_CHANNELS[rname]
        target_upper = [ch.upper() for ch in info['channels']]
        matched = [raw.ch_names[i] for i, ch in enumerate(all_ch_upper) if ch in target_upper]

        if not matched:
            continue

        try:
            raw_r = raw.copy().pick(matched)
            data = raw_r.get_data()
            avg = np.mean(data, axis=0)
            nyq = fs / 2

            # Step 1: Same broadband filter as original (1-45 Hz)
            if nyq > 45:
                b, a = sig.butter(4, [1.0/nyq, 45.0/nyq], btype='band')
                avg = sig.filtfilt(b, a, avg)

            # Step 2: THE CONTROL — remove alpha band
            if filter_alpha:
                avg = remove_alpha(avg, fs, low=filter_range[0], high=filter_range[1])

            topo = compute_topology(avg, fs)
            topo['n_channels'] = len(matched)
            topo['diversity_rank'] = info['diversity_rank']
            region_results[rname] = topo
        except Exception as e:
            print(f"  WARNING: {rname} failed: {e}")

    if len(region_results) < 3:
        return None

    return region_results


# ================================================================
# DEERSKIN SIMULATION (identical to batch_runner.py)
# ================================================================

class DeerskinNeuron:
    def __init__(self, target_freq, n_taps=16, tau=4, fs=1000.0, theta_freq=6.0):
        self.n_taps, self.tau, self.fs = n_taps, tau, fs
        self.theta_phase = np.random.uniform(0, 2*np.pi)
        freq = target_freq + np.random.randn() * 2.0
        k = np.arange(n_taps)
        self.mosaic = np.cos(2*np.pi * freq * k * tau / fs)
        self.theta_freq = theta_freq

    def process(self, x):
        n = len(x)
        out = np.zeros(n)
        for t in range(self.n_taps * self.tau, n):
            v = np.array([x[t - k*self.tau] for k in range(self.n_taps)])
            R = (np.dot(v, self.mosaic))**2
            G = max(0.0, np.sin(2*np.pi*self.theta_freq * t / self.fs + self.theta_phase))
            out[t] = R * G
        return out

MORPH_TYPES = {
    'stellate_L4':   {'target_freq':40.0, 'n_taps':8,  'tau':3, 'theta_freq':7.0},
    'pyramidal_L5':  {'target_freq':20.0, 'n_taps':32, 'tau':6, 'theta_freq':5.0},
    'pyramidal_L23': {'target_freq':30.0, 'n_taps':16, 'tau':4, 'theta_freq':6.0},
    'basket':        {'target_freq':55.0, 'n_taps':6,  'tau':2, 'theta_freq':8.0},
    'martinotti':    {'target_freq':12.0, 'n_taps':24, 'tau':5, 'theta_freq':4.5},
}
TYPE_NAMES = list(MORPH_TYPES.keys())

def run_simulation(duration_sec=10.0, n_neurons=30, fs_sim=1000.0):
    n_samples = int(duration_sec * fs_sim)
    t = np.arange(n_samples) / fs_sim
    thalamic = np.zeros(n_samples)
    for freq in [4, 8, 12, 20, 30, 40, 55]:
        thalamic += (1.0/freq) * np.sin(2*np.pi*freq*t + np.random.uniform(0, 2*np.pi))
    thalamic += 0.3 * np.random.randn(n_samples)
    thalamic /= np.max(np.abs(thalamic))

    sim_results = {}
    for n_types in range(1, 6):
        types_used = TYPE_NAMES[:n_types]
        outputs = np.zeros((n_neurons, n_samples))
        for i in range(n_neurons):
            mt = types_used[i % n_types]
            p = MORPH_TYPES[mt]
            neuron = DeerskinNeuron(p['target_freq'], p['n_taps'], p['tau'], fs_sim, p['theta_freq'])
            outputs[i] = neuron.process(thalamic)
        eeg = np.mean(outputs, axis=0)
        b, a = sig.butter(4, [1.0/(fs_sim/2), 45.0/(fs_sim/2)], btype='band')
        eeg = sig.filtfilt(b, a, eeg)
        topo = compute_topology(eeg, fs_sim)
        sim_results[n_types] = topo['score']

    return sim_results


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ALPHA CONTROL — Prediction 5.4 Batch Test with Alpha Removal')
    parser.add_argument('directory', help='Directory containing .edf files')
    parser.add_argument('--pattern', default='*.edf', help='Glob pattern (default: *.edf)')
    parser.add_argument('--sim', action='store_true', help='Run Deerskin simulation comparison')
    parser.add_argument('--max-files', type=int, default=0, help='Max files to process (0=all)')
    parser.add_argument('--duration', type=float, default=0, help='Max seconds per file')
    parser.add_argument('--alpha-low', type=float, default=8.0, help='Alpha band lower edge (Hz)')
    parser.add_argument('--alpha-high', type=float, default=13.0, help='Alpha band upper edge (Hz)')
    parser.add_argument('--no-alpha-filter', action='store_true',
                        help='Disable alpha filter (run identical to batch_runner for comparison)')
    args = parser.parse_args()

    filter_alpha = not args.no_alpha_filter
    filter_label = f"ALPHA REMOVED ({args.alpha_low}-{args.alpha_high} Hz)" if filter_alpha else "NO ALPHA FILTER (control run)"

    print("=" * 80)
    print("PREDICTION 5.4 — ALPHA CONTROL BATCH TEST")
    print(f"Filter: {filter_label}")
    print("=" * 80)
    print()
    if filter_alpha:
        print("PURPOSE: Test whether the inverse correlation between morphological")
        print("diversity and EEG topological persistence SURVIVES removal of the")
        print("alpha rhythm (8-13 Hz).")
        print()
        print("If the effect survives → NOT an alpha artifact → supports Deerskin")
        print("If the effect disappears → alpha was driving it → confound confirmed")
    print()

    # Find files
    search_path = os.path.join(args.directory, '**', args.pattern)
    files = sorted(glob.glob(search_path, recursive=True))

    if not files:
        search_path = os.path.join(args.directory, args.pattern)
        files = sorted(glob.glob(search_path))

    if not files:
        print(f"No .edf files found in {args.directory}")
        sys.exit(1)

    if args.max_files > 0:
        files = files[:args.max_files]

    print(f"Found {len(files)} EDF files\n")

    # Process all files
    all_file_results = []
    subject_data = defaultdict(list)

    for i, filepath in enumerate(files):
        fname = os.path.basename(filepath)
        match = re.search(r'S(\d+)', fname)
        subject_id = match.group(0) if match else fname

        print(f"[{i+1}/{len(files)}] {fname} ({subject_id})...", end=" ", flush=True)
        t0 = time.time()

        result = process_edf(filepath, max_duration=args.duration,
                            filter_alpha=filter_alpha,
                            filter_range=(args.alpha_low, args.alpha_high))

        if result is None:
            print("SKIPPED")
            continue

        regions_present = [r for r in REGION_ORDER if r in result]
        ranks = [result[r]['diversity_rank'] for r in regions_present]
        scores = [result[r]['score'] for r in regions_present]

        if len(regions_present) >= 3:
            rho, p = spearmanr(ranks, scores)
        else:
            rho, p = 0.0, 1.0

        file_result = {
            'file': fname,
            'subject': subject_id,
            'regions': result,
            'rho': float(rho),
            'p': float(p),
            'n_regions': len(regions_present),
        }
        all_file_results.append(file_result)
        subject_data[subject_id].append(file_result)

        scores_str = "  ".join([f"{r[:3]}={result[r]['score']:.1f}" for r in regions_present])
        print(f"rho={rho:+.2f}  {scores_str}  ({time.time()-t0:.1f}s)")

    if not all_file_results:
        print("\nNo files processed successfully.")
        sys.exit(1)

    # ================================================================
    # AGGREGATE
    # ================================================================

    all_rhos = [r['rho'] for r in all_file_results]

    print("\n" + "=" * 80)
    print("PER-FILE SUMMARY")
    print("=" * 80)
    print(f"\n{'File':20s} | {'Subject':8s} | {'rho':7s}")
    print("-" * 42)
    for r in all_file_results:
        print(f"{r['file']:20s} | {r['subject']:8s} | {r['rho']:+7.3f}")

    print(f"\nAll rho: mean={np.mean(all_rhos):+.3f}, std={np.std(all_rhos):.3f}")
    print(f"  Negative: {sum(1 for r in all_rhos if r < 0)}/{len(all_rhos)}")
    print(f"  Positive: {sum(1 for r in all_rhos if r > 0)}/{len(all_rhos)}")
    print(f"  Zero:     {sum(1 for r in all_rhos if r == 0)}/{len(all_rhos)}")

    # ================================================================
    # PER-SUBJECT
    # ================================================================

    print("\n" + "=" * 80)
    print("PER-SUBJECT AVERAGES")
    print("=" * 80)

    subject_rhos = {}
    for sid in sorted(subject_data.keys()):
        runs = subject_data[sid]
        avg_rho = np.mean([r['rho'] for r in runs])
        subject_rhos[sid] = avg_rho
        print(f"  {sid}: rho = {avg_rho:+.3f} ({len(runs)} runs)")

    # ================================================================
    # GRAND REGION AVERAGES
    # ================================================================

    print("\n" + "=" * 80)
    print("GRAND AVERAGE BY REGION")
    print("=" * 80)

    region_scores_all = defaultdict(list)
    region_h1_all = defaultdict(list)

    for fr in all_file_results:
        for rname, rdata in fr['regions'].items():
            region_scores_all[rname].append(rdata['score'])
            region_h1_all[rname].append(rdata['h1'])

    print(f"\n{'Region':12s} | {'Rank':4s} | {'Mean Score':12s} | {'Std':8s} | {'Mean H1':8s} | {'N':5s}")
    print("-" * 60)

    grand_ranks = []
    grand_scores = []
    for rname in REGION_ORDER:
        if rname in region_scores_all:
            rank = REGION_CHANNELS[rname]['diversity_rank']
            scores = region_scores_all[rname]
            h1s = region_h1_all[rname]
            print(f"{rname:12s} | {rank:4d} | {np.mean(scores):12.4f} | {np.std(scores):8.4f} | {np.mean(h1s):8.1f} | {len(scores):5d}")
            grand_ranks.append(rank)
            grand_scores.append(np.mean(scores))

    grand_rho, grand_p = spearmanr(grand_ranks, grand_scores)
    print(f"\nGrand Spearman rho = {grand_rho:+.3f} (p = {grand_p:.4f})")

    # ================================================================
    # POPULATION STATISTICS
    # ================================================================

    print("\n" + "=" * 80)
    print("POPULATION-LEVEL STATISTICAL TESTS")
    print("=" * 80)

    subject_rho_values = list(subject_rhos.values())
    n_subjects = len(subject_rho_values)

    if n_subjects >= 5:
        t_stat, p_ttest = ttest_1samp(subject_rho_values, 0.0)
        print(f"\nOne-sample t-test (H0: mean rho = 0):")
        print(f"  N subjects = {n_subjects}")
        print(f"  Mean rho = {np.mean(subject_rho_values):+.3f}")
        print(f"  t = {t_stat:.3f}, p = {p_ttest:.6f}")

        try:
            w_stat, p_wilcox = wilcoxon(subject_rho_values)
            print(f"\nWilcoxon signed-rank (H0: median rho = 0):")
            print(f"  W = {w_stat:.1f}, p = {p_wilcox:.6f}")
        except:
            p_wilcox = 1.0
            print("\n  Wilcoxon: could not compute")

        p_one_sided = p_ttest / 2 if np.mean(subject_rho_values) < 0 else 1 - p_ttest / 2
        print(f"\nOne-sided test (H0: rho >= 0, HA: rho < 0):")
        print(f"  p = {p_one_sided:.6f}")
    else:
        p_ttest = 1.0
        p_one_sided = 1.0

    # ================================================================
    # VERDICT
    # ================================================================

    mean_rho = np.mean(all_rhos)
    frac_negative = sum(1 for r in all_rhos if r < 0) / len(all_rhos)

    print("\n" + "=" * 80)
    print(f"VERDICT — {filter_label}")
    print("=" * 80)

    print(f"""
  Files tested:       {len(all_file_results)}
  Subjects:           {n_subjects}
  Mean rho:           {mean_rho:+.3f}
  Fraction negative:  {frac_negative:.1%}
  Grand rho:          {grand_rho:+.3f} (p={grand_p:.4f})
""")

    if mean_rho < -0.3 and frac_negative > 0.7:
        print("  RESULT: INVERSE RELATIONSHIP SURVIVES ALPHA REMOVAL")
        print()
        if filter_alpha:
            print("  *** THIS IS THE KEY FINDING ***")
            print("  The effect is NOT driven by occipital alpha rhythm.")
            print("  Removing 8-13 Hz does not eliminate the inverse correlation")
            print("  between morphological diversity and topological persistence.")
            print("  This supports the Deerskin interpretation: the effect reflects")
            print("  genuine destructive Moiré interference, not alpha dominance.")
    elif mean_rho > 0.3 and frac_negative < 0.3:
        print("  RESULT: POSITIVE RELATIONSHIP (alpha removal reversed the effect!)")
        if filter_alpha:
            print("  The original inverse correlation WAS driven by alpha.")
            print("  Without alpha, more diversity → more complexity.")
    elif abs(mean_rho) < 0.15:
        print("  RESULT: EFFECT DISAPPEARS WITH ALPHA REMOVAL")
        if filter_alpha:
            print("  The original inverse correlation was LARGELY driven by alpha.")
            print("  This is the confound scenario — occipital alpha was inflating")
            print("  persistence in low-diversity regions, creating an artifact.")
    elif abs(mean_rho) < 0.3:
        print("  RESULT: EFFECT WEAKENED BUT PRESENT")
        if filter_alpha:
            print("  Alpha contributes to the effect but does not fully explain it.")
            print("  Partial support for Deerskin interpretation.")
    else:
        print("  RESULT: MIXED / INCONCLUSIVE")

    # ================================================================
    # COMPARISON WITH ORIGINAL (UNFILTERED) RESULTS
    # ================================================================

    original_file = 'batch_prediction_5_4_results.json'
    if os.path.exists(original_file) and filter_alpha:
        print("\n" + "=" * 80)
        print("COMPARISON: ALPHA-FILTERED vs ORIGINAL (UNFILTERED)")
        print("=" * 80)

        with open(original_file) as f:
            orig = json.load(f)

        orig_mean_rho = orig.get('mean_rho', None)
        orig_frac_neg = orig.get('fraction_negative', None)
        orig_grand_rho = orig.get('grand_rho', None)
        orig_regions = orig.get('grand_region_means', {})

        print(f"\n{'Metric':30s} | {'Original':12s} | {'Alpha-Removed':14s} | {'Change':10s}")
        print("-" * 72)

        if orig_mean_rho is not None:
            delta = mean_rho - orig_mean_rho
            print(f"{'Mean rho':30s} | {orig_mean_rho:+12.3f} | {mean_rho:+14.3f} | {delta:+10.3f}")

        if orig_frac_neg is not None:
            delta = frac_negative - orig_frac_neg
            print(f"{'Fraction negative':30s} | {orig_frac_neg:12.1%} | {frac_negative:14.1%} | {delta:+10.1%}")

        if orig_grand_rho is not None:
            delta = grand_rho - orig_grand_rho
            print(f"{'Grand rho':30s} | {orig_grand_rho:+12.3f} | {grand_rho:+14.3f} | {delta:+10.3f}")

        # Per-region comparison
        print(f"\n{'Region':12s} | {'Original':12s} | {'Alpha-Removed':14s} | {'Change':8s} | {'% Change':10s}")
        print("-" * 65)
        for rname in REGION_ORDER:
            if rname in orig_regions and rname in region_scores_all:
                orig_s = orig_regions[rname]
                new_s = np.mean(region_scores_all[rname])
                delta = new_s - orig_s
                pct = (delta / orig_s * 100) if orig_s != 0 else 0
                print(f"{rname:12s} | {orig_s:12.4f} | {new_s:14.4f} | {delta:+8.4f} | {pct:+9.1f}%")

        # Per-subject comparison
        orig_subject_rhos = orig.get('per_subject_rho', {})
        if orig_subject_rhos:
            print(f"\n{'Subject':8s} | {'Original':10s} | {'Alpha-Removed':14s} | {'Change':10s}")
            print("-" * 48)
            for sid in sorted(subject_rhos.keys()):
                if sid in orig_subject_rhos:
                    o = orig_subject_rhos[sid]
                    n = subject_rhos[sid]
                    print(f"{sid:8s} | {o:+10.3f} | {n:+14.3f} | {n-o:+10.3f}")

        # The key interpretation
        print("\n" + "-" * 72)
        if orig_mean_rho is not None:
            rho_ratio = mean_rho / orig_mean_rho if orig_mean_rho != 0 else 0
            if abs(mean_rho) < 0.15:
                print("INTERPRETATION: Alpha removal ELIMINATES the effect.")
                print("The original result was primarily an alpha artifact.")
                print("The Deerskin prediction is NOT supported by this control.")
            elif rho_ratio > 0.5 and mean_rho < -0.2:
                print(f"INTERPRETATION: Effect SURVIVES alpha removal.")
                print(f"  Original mean rho: {orig_mean_rho:+.3f}")
                print(f"  After alpha removal: {mean_rho:+.3f}")
                print(f"  Retention: {rho_ratio:.0%} of original effect size")
                print()
                print("  The inverse correlation is NOT primarily driven by alpha.")
                print("  This is evidence AGAINST the alpha confound hypothesis")
                print("  and evidence FOR the Deerskin framework's prediction.")
                if rho_ratio < 0.8:
                    print()
                    print("  NOTE: Some reduction in effect size suggests alpha contributes")
                    print("  partially, but does not account for the full effect.")
            elif mean_rho > 0:
                print("INTERPRETATION: Alpha removal REVERSES the effect direction.")
                print("The original inverse correlation was entirely alpha-driven.")
            else:
                print(f"INTERPRETATION: Effect is weakened (ratio: {rho_ratio:.0%})")

    # ================================================================
    # SIMULATION COMPARISON
    # ================================================================

    if args.sim:
        print("\n" + "=" * 80)
        print("DEERSKIN SIMULATION COMPARISON")
        print("=" * 80)
        print("\nRunning simulation...")
        sim = run_simulation()
        sim_types = sorted(sim.keys())
        sim_scores = [sim[k] for k in sim_types]
        sim_rho, sim_p = spearmanr(sim_types, sim_scores)

        print(f"  Types:  {sim_types}")
        print(f"  Scores: {[f'{s:.2f}' for s in sim_scores]}")
        print(f"  Simulation rho: {sim_rho:+.3f}")
        print(f"  Real EEG rho (alpha-filtered): {mean_rho:+.3f}")

        if sim_rho < 0 and mean_rho < 0:
            print("\n  BOTH simulation and alpha-filtered EEG show inverse relationship.")
        elif sim_rho * mean_rho > 0:
            print("\n  Simulation and alpha-filtered EEG agree in direction.")
        else:
            print("\n  Simulation and alpha-filtered EEG DISAGREE.")

    # ================================================================
    # SAVE
    # ================================================================

    output = {
        'test': 'Prediction 5.4 — Alpha Control',
        'filter': filter_label,
        'alpha_band_removed': [args.alpha_low, args.alpha_high] if filter_alpha else None,
        'n_files': len(all_file_results),
        'n_subjects': n_subjects,
        'mean_rho': float(mean_rho),
        'std_rho': float(np.std(all_rhos)),
        'fraction_negative': float(frac_negative),
        'grand_rho': float(grand_rho),
        'grand_p': float(grand_p),
        'per_subject_rho': {k: float(v) for k, v in subject_rhos.items()},
        'grand_region_means': {
            rname: float(np.mean(region_scores_all[rname]))
            for rname in REGION_ORDER if rname in region_scores_all
        },
        'per_file': [
            {'file': r['file'], 'subject': r['subject'], 'rho': r['rho']}
            for r in all_file_results
        ],
    }

    out_file = 'alpha_control_results.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_file}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()