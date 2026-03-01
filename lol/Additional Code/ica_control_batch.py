import os
import sys
import glob
import json
import numpy as np
import scipy.stats as stats
import mne
from ripser import ripser

# Suppress MNE warnings for clean console output
mne.set_log_level('ERROR')

# Region definitions (Standard 10-10)
REGIONS = {
    'Occipital': ['O1', 'O2', 'Oz'],
    'Central':['FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz', 
                'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'Cz'],
    'Parietal':['CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 
                 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'Pz', 
                 'PO3', 'PO4', 'PO7', 'PO8', 'POz'],
    'Temporal':['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8'],
    'Frontal':['Fp1', 'Fp2', 'Fpz', 'AF3', 'AF4', 'AF7', 'AF8', 
                'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Fz']
}

# Morphological Diversity Ranks (1 = lowest, 5 = highest)
RANKS = {'Occipital': 1, 'Central': 2, 'Parietal': 3, 'Temporal': 4, 'Frontal': 5}

def takens_embedding(signal, delay_samples, dim=3, max_pts=400):
    """Constructs Takens delay embedding."""
    N = len(signal)
    if N < (dim - 1) * delay_samples + 1:
        return np.array([])
    
    embedded = np.array([signal[i : i + (dim - 1) * delay_samples + 1 : delay_samples] 
                         for i in range(N - (dim - 1) * delay_samples)])
    
    # Subsample to max_pts to keep Ripser fast
    if len(embedded) > max_pts:
        indices = np.linspace(0, len(embedded) - 1, max_pts, dtype=int)
        embedded = embedded[indices]
        
    # Normalize phase space to unit variance and zero mean
    embedded = embedded - np.mean(embedded, axis=0)
    std_dev = np.std(embedded, axis=0)
    std_dev[std_dev == 0] = 1.0
    embedded = embedded / std_dev
    return embedded

def compute_topology_score(embedded):
    """Computes persistent homology and returns complexity score."""
    if len(embedded) < 10:
        return 0.0
    try:
        # maxdim=1 for loops, threshold=2.0 to ignore massive outliers
        diagrams = ripser(embedded, maxdim=1, thresh=2.0)['dgms']
        h1 = diagrams[1]
        if len(h1) == 0:
            return 0.0
            
        lifetimes = h1[:, 1] - h1[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes)]
        
        if len(lifetimes) == 0:
            return 0.0
            
        max_life = np.max(lifetimes)
        # Keep features exceeding 10% of max lifetime
        sig_lifetimes = lifetimes[lifetimes > 0.1 * max_life]
        
        # Score: Sum of significant lifetimes + 0.1 * count
        return np.sum(sig_lifetimes) + 0.1 * len(sig_lifetimes)
    except Exception:
        return 0.0

def process_file_with_ica(filepath):
    """Loads EDF, applies ICA to remove EOG, and computes region topologies."""
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True)
        # Standardize channel names for PhysioNet
        mne.datasets.eegbci.standardize(raw)
        
        sfreq = raw.info['sfreq']
        
        # Filter 1-45Hz (ICA needs 1Hz highpass to avoid drift)
        raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=45.0)
        
        # Identify EOG proxy channels (Frontal poles)
        ch_names_upper =[ch.upper() for ch in raw.ch_names]
        eog_proxies =[ch for ch, ch_u in zip(raw.ch_names, ch_names_upper) if ch_u in['FP1', 'FP2', 'FPZ']]
        
        if len(eog_proxies) > 0:
            # Fit ICA
            ica = mne.preprocessing.ICA(n_components=15, random_state=42, max_iter=800)
            ica.fit(raw_filtered)
            
            # Find and exclude EOG artifact components
            bad_idx =[]
            for eog_ch in eog_proxies:
                eog_indices, _ = ica.find_bads_eog(raw_filtered, ch_name=eog_ch)
                bad_idx.extend(eog_indices)
            
            ica.exclude = list(set(bad_idx))
            clean_raw = ica.apply(raw_filtered)
        else:
            # Fallback if no frontal channels found (rare for PhysioNet)
            clean_raw = raw_filtered
            
        data = clean_raw.get_data()
        ch_names = clean_raw.ch_names
        ch_map = {ch.upper(): i for i, ch in enumerate(ch_names)}
        
        region_scores = {}
        
        # Analysis parameters
        delays_ms = [10, 20, 40]
        delays_samples =[max(1, int(d * sfreq / 1000.0)) for d in delays_ms]
        window_sec = 2.0
        window_samples = int(window_sec * sfreq)
        
        for region, channels in REGIONS.items():
            valid_chs =[ch_map[c.upper()] for c in channels if c.upper() in ch_map]
            if not valid_chs:
                continue
                
            # Average channels in region
            region_signal = np.mean(data[valid_chs, :], axis=0)
            
            # 6 non-overlapping windows
            scores =[]
            for w in range(6):
                start = w * window_samples
                end = start + window_samples
                if end > len(region_signal):
                    break
                    
                window_sig = region_signal[start:end]
                
                # Average across delays
                delay_scores =[]
                for d in delays_samples:
                    emb = takens_embedding(window_sig, d)
                    delay_scores.append(compute_topology_score(emb))
                scores.append(np.mean(delay_scores))
                
            region_scores[region] = np.mean(scores)
            
        return region_scores
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python ica_control_batch.py <path_to_eegmmidb> [--sim]")
        sys.exit(1)
        
    data_dir = sys.argv[1]
    files = glob.glob(os.path.join(data_dir, '**', '*.edf'), recursive=True)
    
    # Filter for target runs (R04, R08, R12)
    target_files =[f for f in files if any(run in f for run in['R04.edf', 'R08.edf', 'R12.edf'])]
    
    if len(target_files) == 0:
        print("No target files found.")
        sys.exit(1)
        
    print(f"{'='*80}")
    print(f"PREDICTION 5.4 — ICA ARTIFACT REJECTION CONTROL")
    print(f"Filter: 1-45 Hz + FastICA EOG/Muscle Projection")
    print(f"Processing {len(target_files)} files...")
    print(f"{'='*80}")
    
    results =[]
    
    for i, filepath in enumerate(target_files):
        print(f"[{i+1}/{len(target_files)}] Processing {os.path.basename(filepath)}...", end='\r')
        scores = process_file_with_ica(filepath)
        
        if scores and len(scores) == 5:
            # Sort by rank
            sorted_regions = sorted(scores.keys(), key=lambda r: RANKS[r])
            x_ranks =[RANKS[r] for r in sorted_regions]
            y_scores = [scores[r] for r in sorted_regions]
            
            rho, pval = stats.spearmanr(x_ranks, y_scores)
            
            results.append({
                'file': os.path.basename(filepath),
                'subject': os.path.basename(os.path.dirname(filepath)),
                'scores': scores,
                'rho': rho
            })
            
    print("\n\n" + "="*80)
    print("ICA CONTROL RESULTS")
    print("="*80)
    
    valid_rhos = [r['rho'] for r in results if not np.isnan(r['rho'])]
    mean_rho = np.mean(valid_rhos)
    fraction_negative = sum(1 for r in valid_rhos if r < 0) / len(valid_rhos)
    
    t_stat, p_val_t = stats.ttest_1samp(valid_rhos, 0)
    p_val_t_one_sided = p_val_t / 2 if t_stat < 0 else 1.0
    
    print(f"Total valid recordings: {len(valid_rhos)}")
    print(f"Fraction negative ρ:    {fraction_negative:.1%} ({sum(1 for r in valid_rhos if r < 0)}/{len(valid_rhos)})")
    print(f"Mean Spearman ρ:        {mean_rho:.3f}")
    print(f"One-sample t-test:      t = {t_stat:.3f}, p = {p_val_t_one_sided:.6f}\n")
    
    # Region averages
    print("GRAND AVERAGE TOPOLOGY BY REGION (ICA CLEANED):")
    for r in sorted(REGIONS.keys(), key=lambda x: RANKS[x]):
        avg = np.mean([res['scores'][r] for res in results])
        print(f"  {r.ljust(10)} (Rank {RANKS[r]}): {avg:.2f}")
        
    # Save to JSON for side-by-side
    with open('ica_control_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()