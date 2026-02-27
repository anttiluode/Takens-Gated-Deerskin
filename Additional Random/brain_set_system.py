import numpy as np
import torch
import torch.nn as nn
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mne
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy import stats
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import cognitive set theory components
try:
    from cognitive_set_theory import (
        CognitiveSignature, convert_to_cognitive_signature, 
        analyze_cognitive_sets, cognitive_similarity
    )
    SET_THEORY_AVAILABLE = True
except ImportError:
    SET_THEORY_AVAILABLE = False
    print("Note: cognitive_set_theory module not found, using fallback")

# Core classes
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, latent_dim))
    def forward(self, x): return self.net(x)

class Predictor(nn.Module):
    def __init__(self, latent_dim=32, depth=3):
        super().__init__()
        layers = [nn.Linear(latent_dim, 128), nn.ReLU()]
        for _ in range(depth - 1): layers.extend([nn.Linear(128, 128), nn.ReLU()])
        layers.append(nn.Linear(128, latent_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z)

class WorldModel(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.predictor = Predictor(latent_dim)

EEG_REGIONS = {"All": [], "Occipital": ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4', 'PO7', 'PO8'], 
               "Temporal": ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8'], 
               "Parietal": ['P1', 'P2', 'P3', 'P4', 'PZ', 'CP1', 'CP2'], 
               "Frontal": ['FP1', 'FP2', 'FZ', 'F1', 'F2', 'F3', 'F4'], 
               "Central": ['C1', 'C2', 'C3', 'C4', 'CZ', 'FC1', 'FC2']}

def create_eeg_features(edf_file, region="All"):
    frequency_bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    raw.rename_channels(lambda name: name.strip().replace('.', '').upper())
    
    if region != "All":
        region_channels = EEG_REGIONS[region]
        available_channels = [ch for ch in region_channels if ch in raw.ch_names]
        if not available_channels: return np.array([]), 0, []
        raw.pick_channels(available_channels)
    
    fs = 100.0
    raw.resample(fs, verbose=False)
    
    band_filtered_data = {band: raw.copy().filter(l_freq=low, h_freq=high, fir_design='firwin', verbose=False).get_data() 
                         for band, (low, high) in frequency_bands.items()}
    samples_per_epoch = int(0.5 * fs)
    all_epochs_features = []
    
    for i in range(0, raw.n_times - samples_per_epoch, samples_per_epoch):
        epoch_band_powers = [np.log1p(np.mean(band_filtered_data[band][:, i:i+samples_per_epoch]**2, axis=1)) 
                           for band in frequency_bands.keys()]
        all_epochs_features.append(np.stack(epoch_band_powers, axis=1))
        
    all_epochs_features = np.array(all_epochs_features)
    feature_names = [f"{ch}-{band}" for ch in raw.ch_names for band in frequency_bands.keys()]
    
    n_epochs, n_channels, n_bands = all_epochs_features.shape
    flattened_features = all_epochs_features.reshape(n_epochs, n_channels * n_bands)
    
    mean = np.mean(flattened_features, axis=0, keepdims=True)
    std = np.std(flattened_features, axis=0, keepdims=True)
    std[std == 0] = 1
    normalized_features = (flattened_features - mean) / std
    
    return normalized_features, feature_names

def create_correlation_fingerprint(feature_data, feature_names):
    """Visualizes the correlation between all input features."""
    if feature_data.shape[0] < 2: return go.Figure()
    
    correlation_matrix = np.corrcoef(feature_data.T)
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=feature_names,
        y=feature_names,
        colorscale='RdBu_r',
        zmid=0
    ))
    fig.update_layout(
        title="Functional Connectivity Fingerprint",
        template='plotly_dark',
        height=600,
        yaxis=dict(autorange='reversed')
    )
    return fig

def analyze_state_dynamics(latent_trajectory, n_states=20):
    """Comprehensive state-space analysis with entropy, modularity, and pattern detection."""
    if len(latent_trajectory) < n_states: 
        return None, None, {}
    
    # Cluster to find stable states
    kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    state_labels = kmeans.fit_predict(latent_trajectory)
    state_centers = kmeans.cluster_centers_
    
    # Build transition matrix
    transitions = np.zeros((n_states, n_states))
    for i in range(len(state_labels) - 1):
        transitions[state_labels[i], state_labels[i+1]] += 1
    
    # Normalize rows to get transition probabilities
    row_sums = transitions.sum(axis=1)
    transition_probs = transitions / row_sums[:, np.newaxis]
    transition_probs[np.isnan(transition_probs)] = 0
    
    # Calculate metrics
    metrics = {}
    
    # 1. State entropy
    state_counts = np.bincount(state_labels)
    state_probs = state_counts / len(state_labels)
    state_entropy = stats.entropy(state_probs[state_probs > 0])
    metrics['state_entropy'] = state_entropy
    
    # 2. Transition entropy
    flat_transitions = transition_probs.flatten()
    transition_entropy = stats.entropy(flat_transitions[flat_transitions > 0])
    metrics['transition_entropy'] = transition_entropy
    
    # 3. Modularity analysis
    try:
        G = nx.from_numpy_array(transitions, create_using=nx.DiGraph)
        communities = list(nx.community.greedy_modularity_communities(G.to_undirected()))
        metrics['n_communities'] = len(communities)
        metrics['modularity'] = nx.community.modularity(G.to_undirected(), communities)
    except:
        metrics['n_communities'] = 1
        metrics['modularity'] = 0
    
    # 4. Identify cognitive patterns
    patterns = {
        'loops': 0,
        'hubs': [],
        'linear_chains': 0
    }
    
    for i in range(n_states):
        if transition_probs[i, i] > 0.3:
            patterns['loops'] += 1
    
    connectivity = (transitions > 0).sum(axis=0) + (transitions > 0).sum(axis=1)
    hub_threshold = np.percentile(connectivity, 75)
    patterns['hubs'] = [i for i, c in enumerate(connectivity) if c > hub_threshold]
    
    for i in range(n_states):
        if (transition_probs[i] > 0.7).sum() == 1:
            patterns['linear_chains'] += 1
    
    metrics['patterns'] = patterns
    
    return state_labels, transition_probs, metrics

def create_enhanced_state_flow(latent_trajectory, metrics, method='umap'):
    """Creates an enhanced visualization with pattern annotations."""
    if len(latent_trajectory) < 20:
        return go.Figure()
    
    # Dimensionality reduction
    if method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    
    embedded = reducer.fit_transform(latent_trajectory)
    
    fig = go.Figure()
    
    # Add trajectory as scatter plot with color gradient
    fig.add_trace(go.Scatter(
        x=embedded[:, 0],
        y=embedded[:, 1],
        mode='markers+lines',
        marker=dict(
            size=6,
            color=np.arange(len(embedded)),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Time")
        ),
        line=dict(
            width=1,
            color='rgba(255, 255, 255, 0.3)'
        ),
        name='Thought Trajectory',
        text=[f'Time: {i}' for i in range(len(embedded))],
        hovertemplate='%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}'
    ))
    
    title = f"Enhanced State-Space Map ({method.upper()})<br>"
    title += f"<sub>Entropy: {metrics.get('state_entropy', 0):.2f} | "
    title += f"Communities: {metrics.get('n_communities', 0)} | "
    title += f"Loops: {metrics.get('patterns', {}).get('loops', 0)}</sub>"
    
    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=600,
        xaxis_title=f'{method.upper()} 1',
        yaxis_title=f'{method.upper()} 2'
    )
    
    return fig

def create_state_flow_diagram(latent_trajectory, n_states=20):
    """Creates an enhanced Sankey flow diagram showing thought transitions."""
    if len(latent_trajectory) < n_states: 
        return go.Figure()

    # Find stable states using K-Means clustering
    kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    state_labels = kmeans.fit_predict(latent_trajectory)
    
    # Count transitions between states
    transitions = np.zeros((n_states, n_states))
    for i in range(len(state_labels) - 1):
        transitions[state_labels[i], state_labels[i+1]] += 1
    
    # Analyze state properties
    state_properties = []
    for i in range(n_states):
        state_mask = state_labels == i
        duration = np.sum(state_mask)
        
        incoming = np.sum(transitions[:, i])
        outgoing = np.sum(transitions[i, :])
        connectivity = incoming + outgoing
        self_loop = transitions[i, i]
        
        properties = {
            'duration': duration,
            'connectivity': connectivity,
            'self_loop': self_loop,
            'incoming': incoming,
            'outgoing': outgoing
        }
        state_properties.append(properties)
    
    # Create meaningful labels
    labels = []
    for i, props in enumerate(state_properties):
        if props['self_loop'] > np.mean([p['self_loop'] for p in state_properties]) * 2:
            label = f"Loop {i}"
        elif props['connectivity'] > np.mean([p['connectivity'] for p in state_properties]) * 1.5:
            label = f"Hub {i}"
        elif props['outgoing'] > props['incoming'] * 2:
            label = f"Source {i}"
        elif props['incoming'] > props['outgoing'] * 2:
            label = f"Sink {i}"
        else:
            label = f"State {i}"
        
        label += f"\n({props['duration']} steps)"
        labels.append(label)
    
    # Filter transitions for clarity
    threshold = np.max(transitions) * 0.05
    
    # Create node colors
    node_colors = []
    for props in state_properties:
        if props['self_loop'] > np.mean([p['self_loop'] for p in state_properties]) * 2:
            node_colors.append('rgba(255, 100, 100, 0.8)')  # Red for loops
        elif props['connectivity'] > np.mean([p['connectivity'] for p in state_properties]) * 1.5:
            node_colors.append('rgba(100, 255, 100, 0.8)')  # Green for hubs
        else:
            node_colors.append('rgba(100, 150, 255, 0.8)')  # Blue for regular
    
    # Build Sankey diagram
    source_nodes, target_nodes, values, link_colors = [], [], [], []
    
    for i in range(n_states):
        for j in range(n_states):
            if transitions[i, j] > threshold:
                source_nodes.append(i)
                target_nodes.append(j)
                values.append(transitions[i, j])
                
                if i == j:
                    link_colors.append('rgba(255, 150, 150, 0.4)')
                else:
                    link_colors.append('rgba(150, 150, 255, 0.4)')
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=1),
            label=labels,
            color=node_colors,
            hovertemplate='%{label}<br>Connections: %{value}<extra></extra>'
        ),
        link=dict(
            source=source_nodes,
            target=target_nodes,
            value=values,
            color=link_colors,
            hovertemplate='%{source.label} → %{target.label}<br>Transitions: %{value}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title={
            'text': "Map of Thoughts: Cognitive State Flow<br><sub>Red: Loop states | Green: Hub states | Blue: Regular states</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        template='plotly_dark',
        height=700,
        font=dict(size=12)
    )
    
    return fig

def compare_multiple_brains(all_data):
    """Creates comparative visualizations across multiple subjects."""
    if len(all_data) < 2:
        return go.Figure(), go.Figure()
    
    all_metrics = []
    for name, (features, trajectory, metrics) in all_data.items():
        metrics['name'] = name
        all_metrics.append(metrics)
    
    # Metric comparison radar chart
    categories = ['State Entropy', 'Transition Entropy', 'Modularity', 'Loop Patterns', 'Hub States']
    
    fig_radar = go.Figure()
    
    for m in all_metrics:
        values = [
            m.get('state_entropy', 0),
            m.get('transition_entropy', 0),
            m.get('modularity', 0),
            m.get('patterns', {}).get('loops', 0) / 5,
            len(m.get('patterns', {}).get('hubs', [])) / 5
        ]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=m['name']
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max([v for m in all_metrics for v in [
                    m.get('state_entropy', 0),
                    m.get('transition_entropy', 0),
                    m.get('modularity', 0),
                    m.get('patterns', {}).get('loops', 0) / 5,
                    len(m.get('patterns', {}).get('hubs', [])) / 5
                ]])]
            )),
        showlegend=True,
        template='plotly_dark',
        title="Cognitive Signatures Comparison"
    )
    
    # State-space similarity matrix
    n_subjects = len(all_data)
    similarity_matrix = np.zeros((n_subjects, n_subjects))
    names = list(all_data.keys())
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                traj1 = all_data[name1][1]
                traj2 = all_data[name2][1]
                
                if traj1.shape[1] == traj2.shape[1]:
                    min_len = min(len(traj1), len(traj2))
                    if min_len > 10:
                        correlations = []
                        for dim in range(traj1.shape[1]):
                            corr = np.corrcoef(traj1[:min_len, dim], traj2[:min_len, dim])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)
                        
                        if correlations:
                            similarity_matrix[i, j] = (np.mean(correlations) + 1) / 2
                        else:
                            similarity_matrix[i, j] = 0.5
                else:
                    m1, m2 = all_metrics[i], all_metrics[j]
                    metric_vec1 = np.array([m1.get('state_entropy', 0), m1.get('transition_entropy', 0), m1.get('modularity', 0)])
                    metric_vec2 = np.array([m2.get('state_entropy', 0), m2.get('transition_entropy', 0), m2.get('modularity', 0)])
                    dist = np.linalg.norm(metric_vec1 - metric_vec2)
                    similarity_matrix[i, j] = 1 / (1 + dist)
    
    fig_similarity = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=names,
        y=names,
        colorscale='Viridis',
        text=np.round(similarity_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig_similarity.update_layout(
        title="Thought Pattern Similarity Matrix",
        template='plotly_dark',
        height=500,
        yaxis=dict(autorange='reversed')
    )
    
    return fig_radar, fig_similarity

# Global storage
person_data = {}

def run_analysis(edf_file, region, latent_dim, reduction_method, person_name, progress=gr.Progress()):
    """Analyzes a single person's EEG data."""
    if edf_file is None: 
        raise gr.Error("Please upload an EEG file.")
    
    if not person_name:
        person_name = f"Person_{len(person_data) + 1}"
    
    progress(0.1, desc="Extracting features...")
    feature_data, feature_names = create_eeg_features(edf_file.name, region)
    if len(feature_data) == 0: 
        raise gr.Error("Could not extract features for the selected region.")
    
    progress(0.3, desc="Generating latent trajectory...")
    latent_dim = int(latent_dim)
    latent_dim = min(latent_dim, min(feature_data.shape[0], feature_data.shape[1]) - 1)
    
    pca = PCA(n_components=latent_dim)
    latent_trajectory = pca.fit_transform(feature_data)
    
    progress(0.5, desc="Analyzing state dynamics...")
    state_labels, transition_probs, metrics = analyze_state_dynamics(latent_trajectory)
    
    person_data[person_name] = (feature_data, latent_trajectory, metrics)
    
    progress(0.7, desc="Creating visualizations...")
    fig_corr = create_correlation_fingerprint(feature_data, feature_names)
    fig_flow = create_enhanced_state_flow(latent_trajectory, metrics, reduction_method)
    fig_thought_flow = create_state_flow_diagram(latent_trajectory)
    
    summary = f"**Analysis Complete for {person_name}**\n\n"
    summary += f"- **State Entropy:** {metrics['state_entropy']:.3f} (cognitive exploration)\n"
    summary += f"- **Transition Entropy:** {metrics['transition_entropy']:.3f} (predictability)\n"
    summary += f"- **Modularity:** {metrics['modularity']:.3f} (community structure)\n"
    summary += f"- **Loop Patterns:** {metrics['patterns']['loops']} (habitual states)\n"
    summary += f"- **Hub States:** {len(metrics['patterns']['hubs'])} (attractor points)\n"
    summary += f"- **Linear Chains:** {metrics['patterns']['linear_chains']} (focused sequences)\n"
    
    return fig_corr, fig_flow, fig_thought_flow, summary, gr.update(choices=list(person_data.keys()))

def run_comparison(progress=gr.Progress()):
    """Compares all loaded persons."""
    if len(person_data) < 2:
        raise gr.Error("Please analyze at least 2 people before comparison.")
    
    progress(0.5, desc="Comparing cognitive signatures...")
    fig_radar, fig_similarity = compare_multiple_brains(person_data)
    
    summary = f"**Comparison of {len(person_data)} Individuals**\n\n"
    summary += "Detected cognitive patterns across subjects. "
    summary += "The radar chart shows individual cognitive signatures, "
    summary += "while the similarity matrix reveals shared thought patterns."
    
    return fig_radar, fig_similarity, summary

def run_set_analysis(progress=gr.Progress()):
    """Run formal set-theoretic analysis."""
    if len(person_data) < 2:
        raise gr.Error("Please analyze at least 2 people before set analysis.")
    
    progress(0.5, desc="Performing set-theoretic analysis...")
    
    if SET_THEORY_AVAILABLE:
        fig_sets, fig_algebra, analysis_text = analyze_cognitive_sets(person_data)
    else:
        # Fallback visualization
        fig_sets = go.Figure()
        fig_algebra = go.Figure()
        
        fig_sets.add_annotation(
            text="Set Theory Module Not Available<br>Save cognitive_set_theory.py to enable",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        
        analysis_text = "## Set-Theoretic Analysis (Preview)\n\n"
        analysis_text += "To enable full set-theoretic analysis:\n"
        analysis_text += "1. Save the cognitive_set_theory.py module\n"
        analysis_text += "2. Restart the application\n\n"
        analysis_text += "This will enable:\n"
        analysis_text += "- Formal cognitive signatures: P_i = (C_i, L_i, H_i^T, H_i^S, M_i)\n"
        analysis_text += "- Set operations on mental states\n"
        analysis_text += "- Category theory functors between minds\n"
        analysis_text += "- Algebraic structure of consciousness"
    
    return fig_sets, fig_algebra, analysis_text

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧠 Advanced Brain Architecture Analyzer")
    gr.Markdown("Reveal the functional architecture and 'grammar of thought' from EEG data")
    
    with gr.Tab("Individual Analysis"):
        with gr.Row():
            with gr.Column(scale=1):
                edf_input = gr.File(label="Upload EEG File (.edf)")
                person_name = gr.Textbox(label="Person Name (optional)", placeholder="Person_1")
                region_selector = gr.Dropdown(choices=list(EEG_REGIONS.keys()), value="Occipital", label="Brain Region")
                latent_dim = gr.Slider(2, 64, value=8, step=1, label="Latent Dimensions")
                reduction_method = gr.Radio(["pca", "tsne", "umap"], value="umap", label="Visualization Method")
                run_button = gr.Button("Analyze Brain Architecture", variant="primary")
            
            with gr.Column(scale=2):
                analysis_summary = gr.Markdown("Upload an EEG file to begin analysis...")
        
        with gr.Row():
            with gr.Column():
                corr_plot = gr.Plot(label="Functional Connectivity")
            with gr.Column():
                flow_plot = gr.Plot(label="Enhanced State-Space Map")
        
        with gr.Row():
            thought_flow = gr.Plot(label="Thought Flow Diagram")
    
    with gr.Tab("Multi-Person Comparison"):
        gr.Markdown("### Compare Cognitive Signatures Across Multiple Individuals")
        
        with gr.Row():
            person_list = gr.Dropdown(choices=[], label="Analyzed Persons", multiselect=True, interactive=False)
            compare_button = gr.Button("Compare All Persons", variant="primary")
        
        comparison_summary = gr.Markdown("Analyze at least 2 people to enable comparison...")
        
        with gr.Row():
            radar_plot = gr.Plot(label="Cognitive Signatures")
            similarity_plot = gr.Plot(label="Thought Pattern Similarity")
    
    with gr.Tab("Set-Theoretic Analysis"):
        gr.Markdown("### Formal Mathematical Analysis of Cognitive Structures")
        gr.Markdown("Formalizing consciousness using set theory, category theory, and algebraic structures")
        
        set_analysis_button = gr.Button("Run Set-Theoretic Analysis", variant="primary")
        set_analysis_summary = gr.Markdown("Analyze at least 2 people to enable set analysis...")
        
        with gr.Row():
            set_operations_plot = gr.Plot(label="Set Operations on Cognitive Spaces")
            cognitive_algebra_plot = gr.Plot(label="Cognitive Algebra")
    
    # Event handlers
    run_button.click(
        fn=run_analysis,
        inputs=[edf_input, region_selector, latent_dim, reduction_method, person_name],
        outputs=[corr_plot, flow_plot, thought_flow, analysis_summary, person_list]
    )
    
    compare_button.click(
        fn=run_comparison,
        inputs=[],
        outputs=[radar_plot, similarity_plot, comparison_summary]
    )
    
    set_analysis_button.click(
        fn=run_set_analysis,
        inputs=[],
        outputs=[set_operations_plot, cognitive_algebra_plot, set_analysis_summary]
    )

if __name__ == "__main__":
    app.launch(debug=True)