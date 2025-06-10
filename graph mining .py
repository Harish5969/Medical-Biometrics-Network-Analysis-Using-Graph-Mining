import pandas as pd
import networkx as nx
import panel as pn
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from networkx.algorithms import community
from sklearn.preprocessing import StandardScaler
import numpy as np
import io
import traceback
import requests
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import holoviews as hv
from holoviews import opts

pn.extension(sizing_mode="stretch_width")
hv.opts.defaults(opts.Graph(tools=['hover', 'tap'], node_size=10, edge_alpha=0.5, width=800, height=600))

def get_community_colors(n_communities):
    import colorsys
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if n_communities <= len(colors):
        return colors[:n_communities]
    hsv_tuples = [(x * 1.0 / n_communities, 0.8, 0.9) for x in range(n_communities)]
    rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
    return ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in rgb_tuples]

pn.extension('plotly', 'tabulator', sizing_mode="stretch_width", throttled=True)

def get_distinct_dark_colors(n):
    import colorsys
    base_colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#17becf', '#ff7f0e', '#4a1486', '#006d2c', '#8c6d31']
    if n > len(base_colors):
        colors = base_colors.copy()
        golden_ratio = 0.618033988749895
        saturation = 0.8
        value = 0.7
        curr_hue = 0.1
        while len(colors) < n:
            curr_hue = (curr_hue + golden_ratio) % 1.0
            rgb = colorsys.hsv_to_rgb(curr_hue, saturation, value)
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            if all(color_distance(hex_color, existing) > 0.2 for existing in colors):
                colors.append(hex_color)
        return colors
    return base_colors[:n]

def color_distance(hex1, hex2):
    rgb1 = tuple(int(hex1[i:i+2], 16)/255 for i in (1, 3, 5))
    rgb2 = tuple(int(hex2[i:i+2], 16)/255 for i in (1, 3, 5))
    return sum((a-b)**2 for a, b in zip(rgb1, rgb2))**0.5

def update_progress(value):
    progress_bar.value = value
    progress_text.object = f"{value}%"

data = None
scaler = StandardScaler()
feature_columns = []
disease_column = None
file_data_store = None

file_input = pn.widgets.FileInput(accept='.csv', name='Upload Dataset', sizing_mode='stretch_width')
url_input = pn.widgets.TextInput(name="Enter Dataset URL (Optional)", placeholder="Enter URL or use file upload above", sizing_mode='stretch_width', value=None)
similarity_slider = pn.widgets.IntSlider(name="Similarity Threshold", start=5, end=95, step=5, value=30, tooltips=True)
centrality_choice = pn.widgets.Select(name="Centrality Measure", options=["Degree", "Betweenness"])
community_algo = pn.widgets.Select(name="Community Detection Algorithm", options=["Greedy Modularity", "Louvain"])
layout_choice = pn.widgets.Select(name="Graph Layout", options=["Spring Layout", "Circular Layout", "Kamada-Kawai Layout"])

start_button = pn.widgets.Button(name='Start Analysis', button_type='success')
export_button = pn.widgets.Button(name='Export Results', button_type='primary')
reset_button = pn.widgets.Button(name='Reset Dataset', button_type='danger')

graph_pane = pn.pane.Matplotlib(height=600, sizing_mode='stretch_width')
table_pane = pn.pane.DataFrame(sizing_mode='stretch_width')
community_table_pane = pn.pane.DataFrame(escape=False, sizing_mode='stretch_width')
status_text = pn.pane.Markdown("No dataset loaded")

progress_bar = pn.widgets.Progress(value=0, max=100, sizing_mode='stretch_width', name='Progress', bar_color='success')
progress_text = pn.pane.Markdown("0%", align='center')

def fetch_data_from_api(url):
    try:
        if not url or not url.strip():
            status_text.object = "Error: The URL is empty. Please provide a valid URL."
            return None
        if not url.startswith(('http://', 'https://')):
            status_text.object = "Error: The URL is invalid. Please ensure it starts with 'http://' or 'https://'."
            return None
        response = requests.get(url)
        if response.status_code == 200:
            data = pd.read_csv(io.BytesIO(response.content))
            data = preprocess_data(data)
            if 'ID' not in data.columns:
                data['ID'] = range(1, len(data) + 1)
            for col in data.columns:
                if col != 'ID' and col.lower() not in ['disease', 'condition', 'diagnosis']:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    except:
                        continue
            return data
        else:
            raise Exception(f"Failed to fetch data: HTTP {response.status_code}")
    except Exception as e:
        status_text.object = f"Error fetching dataset: {str(e)}"
        traceback.print_exc()
        return None

def load_data_from_url(event):
    if event.new and event.new.strip():
        file_input.disabled = True
    global data, feature_columns, disease_column, file_data_store
    try:
        url = event.new
        data = fetch_data_from_api(url)
        if data is None:
            status_text.object = "Failed to load dataset. Please check the URL."
            return
        if 'ID' not in data.columns:
            data['ID'] = [f'P{i:03d}' for i in range(1, len(data) + 1)]
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'ID' in numeric_columns:
            numeric_columns.remove('ID')
        potential_disease_cols = [col for col in data.columns if col.lower() in ['disease', 'condition', 'diagnosis', 'target', 'label', 'class']]
        if potential_disease_cols:
            disease_column = potential_disease_cols[0]
            feature_columns = [col for col in numeric_columns if col != disease_column]
        else:
            disease_column = 'condition'
            data[disease_column] = 'Group A'
            feature_columns = numeric_columns
            if len(numeric_columns) >= 2:
                try:
                    from sklearn.cluster import KMeans
                    features = data[numeric_columns].fillna(0)
                    kmeans = KMeans(n_clusters=min(5, len(data)), random_state=42)
                    clusters = kmeans.fit_predict(features)
                    data[disease_column] = [f'Group {chr(65+c)}' for c in clusters]
                except Exception:
                    pass
        if len(feature_columns) < 1:
            data['feature1'] = np.random.normal(size=len(data))
            data['feature2'] = np.random.normal(size=len(data))
            feature_columns = ['feature1', 'feature2']
            status_text.object = "Created synthetic features for visualization."
        else:
            status_text.object = f"Dataset loaded successfully with {len(feature_columns)} features."
        url_input.disabled = True
    except Exception as e:
        status_text.object = f"Error loading dataset: {str(e)}"
        traceback.print_exc()
        data = None
        feature_columns = []
        disease_column = None
        file_data_store = None
        url_input.disabled = False
        url_input.value = ''

def reset_upload(event):
    global data, feature_columns, disease_column, file_data_store
    try:
        data = None
        feature_columns = []
        disease_column = None
        file_data_store = None
        try:
            url_input.value = ""
            url_input.disabled = False
        except Exception:
            pass
        try:
            file_input.value = None
            file_input.disabled = False
        except Exception:
            pass
        try:
            graph_pane.object = None
        except Exception:
            pass
        try:
            table_pane.object = pd.DataFrame()
        except Exception:
            pass
        try:
            community_table_pane.object = pd.DataFrame()
        except Exception:
            pass
        try:
            similarity_slider.value = 30
            centrality_choice.value = "Degree"
            community_algo.value = "Greedy Modularity"
            layout_choice.value = "Spring Layout"
        except Exception:
            pass
        try:
            status_text.object = "Dataset cleared. Ready for new analysis."
            update_progress(0)
        except Exception:
            pass
    except Exception as e:
        try:
            status_text.object = f"Error during reset: {str(e)}"
        except Exception:
            print(f"Critical error during reset: {str(e)}")
        traceback.print_exc()
        data = None
        feature_columns = []
        disease_column = None
        file_data_store = None

def process_chunk(chunk_data, full_data, similarity_threshold):
    distances = euclidean_distances(chunk_data, full_data)
    threshold = np.percentile(distances, similarity_threshold)
    edges = []
    for idx, row in enumerate(distances):
        similar_indices = np.where(row < threshold)[0]
        for j in similar_indices:
            if idx < j:
                similarity = 1 / (1 + row[j])
                edges.append((idx, j, similarity))
    return edges

@lru_cache(maxsize=32)
def calculate_distances_parallel(feature_matrix, chunk_size=1000):
    n_samples = len(feature_matrix)
    distances = np.zeros((n_samples, n_samples))
    def process_chunk(start_idx):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk = feature_matrix[start_idx:end_idx]
        return start_idx, euclidean_distances(chunk, feature_matrix)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, i) for i in range(0, n_samples, chunk_size)]
        for future in futures:
            start_idx, chunk_distances = future.result()
            end_idx = min(start_idx + chunk_size, n_samples)
            distances[start_idx:end_idx] = chunk_distances
    return distances

def get_optimal_threshold(distances, similarity_threshold):
    try:
        base_percentile = max(5, 100 - similarity_threshold)
        thresholds = []
        for modifier in range(0, 5):
            percentile = max(1, min(99, base_percentile - modifier * 10))
            threshold = np.percentile(distances[distances > 0], percentile)
            thresholds.append(threshold)
        return thresholds[0]
    except Exception as e:
        return np.median(distances[distances > 0])

def build_graph(similarity_threshold):
    G = nx.Graph()
    try:
        if data is None or data.empty:
            update_progress(100)
            return G
        update_progress(10)
        cols_to_drop = ['ID']
        if disease_column:
            cols_to_drop.append(disease_column)
        biometric_features = data.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=[np.number])
        biometric_features = biometric_features.fillna(biometric_features.mean())
        biometric_features = biometric_features.fillna(0)
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(biometric_features)
        features_normalized = np.nan_to_num(features_normalized)
        update_progress(30)
        distances = euclidean_distances(features_normalized)
        update_progress(50)
        for i, row in data.iterrows():
            node_attributes = {col: row[col] for col in biometric_features.columns}
            if disease_column:
                node_attributes['disease'] = row[disease_column]
            G.add_node(row['ID'], **node_attributes)
        update_progress(70)
        percentile_value = 100 - similarity_threshold
        flat_distances = distances.flatten()
        non_zero_distances = flat_distances[flat_distances > 0]
        if len(non_zero_distances) > 0:
            adaptive_threshold = np.percentile(non_zero_distances, percentile_value)
        else:
            adaptive_threshold = 1.0
        edges = []
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                if distances[i][j] < adaptive_threshold:
                    similarity = 1 / (1 + distances[i][j])
                    edges.append((data.iloc[i]['ID'], data.iloc[j]['ID'], similarity))
        G.add_weighted_edges_from(edges)
        update_progress(45)
        components = list(nx.connected_components(G))
        if len(components) > 1:
            status_text.object = f"Connecting {len(components)} separate components to ensure a fully connected graph..."
            sorted_components = sorted(components, key=len, reverse=True)
            largest_component = sorted_components[0]
            for component in sorted_components[1:]:
                min_distance = float('inf')
                closest_pair = (None, None)
                for node1 in largest_component:
                    for node2 in component:
                        idx1 = data[data['ID'] == node1].index[0]
                        idx2 = data[data['ID'] == node2].index[0]
                        if distances[idx1][idx2] < min_distance:
                            min_distance = distances[idx1][idx2]
                            closest_pair = (node1, node2)
                if closest_pair[0] is not None and closest_pair[1] is not None:
                    similarity = 1 / (1 + min_distance)
                    G.add_edge(closest_pair[0], closest_pair[1], weight=similarity)
        update_progress(60)
        if not nx.is_connected(G):
            status_text.object = "Warning: Graph still disconnected. Adding spanning tree edges..."
            components = list(nx.connected_components(G))
            if len(components) > 1:
                for i in range(len(components)-1):
                    node1 = next(iter(components[i]))
                    node2 = next(iter(components[i+1]))
                    G.add_edge(node1, node2, weight=0.01)
        update_progress(90)
        return G
    except Exception as e:
        status_text.object = f"Error building graph: {str(e)}"
        traceback.print_exc()
        update_progress(100)
        return G

def calculate_community_statistics(G, communities):
    try:
        community_stats = []
        for idx, community in enumerate(communities):
            subgraph = G.subgraph(community)
            size = len(community)
            internal_edges = subgraph.number_of_edges()
            external_edges = sum(1 for u in community for v in G.neighbors(u) if v not in community)
            density = nx.density(subgraph) if size > 1 else 0
            avg_clustering = nx.average_clustering(subgraph) if size > 1 else 0
            avg_degree = sum(dict(subgraph.degree()).values()) / size if size > 0 else 0
            if size > 0:
                central_node = max(subgraph.degree(), key=lambda x: x[1])[0]
            else:
                central_node = "N/A"
            community_stats.append({
                'Community ID': idx + 1,
                'Size': size,
                'Internal Edges': internal_edges,
                'External Edges': external_edges,
                'Density': f"{density:.3f}",
                'Avg Clustering': f"{avg_clustering:.3f}",
                'Avg Degree': f"{avg_degree:.2f}",
                'Central Node': central_node
            })
        return pd.DataFrame(community_stats)
    except Exception as e:
        status_text.object = f"Error calculating community statistics: {str(e)}"
        return pd.DataFrame([{'Error': 'Failed to calculate community statistics'}])

def analyze_graph(G, centrality_measure, community_algorithm):
    try:
        if G.number_of_nodes() > 10000:
            status_text.object = "Warning: Large network detected. Using fast community detection..."
            communities = list(nx.community.label_propagation_communities(G))
            return centrality, communities, clustering_coefficient, density, modularity
        if G.number_of_edges() == 0:
            status_text.object = "Warning: Graph has no edges. Try reducing similarity threshold."
            communities = [set(G.nodes())]
            centrality = {node: 0 for node in G.nodes()}
            return centrality, communities, 0, 0, 0
        update_progress(75)
        try:
            if centrality_measure == "Degree":
                centrality = nx.degree_centrality(G)
            else:
                centrality = nx.betweenness_centrality(G, normalized=True)
        except Exception as e:
            status_text.object = f"Warning: Centrality calculation failed: {str(e)}. Using degree centrality."
            centrality = {node: len(list(G.neighbors(node)))/G.number_of_nodes() for node in G.nodes()}
        update_progress(80)
        try:
            if community_algorithm == "Greedy Modularity":
                communities = list(community.greedy_modularity_communities(G, resolution=0.7))
            elif community_algorithm == "Louvain":
                try:
                    from community import community_louvain
                    partition = community_louvain.best_partition(G, resolution=0.7)
                    community_dict = {}
                    for node, comm_id in partition.items():
                        if comm_id not in community_dict:
                            community_dict[comm_id] = set()
                        community_dict[comm_id].add(node)
                    communities = list(community_dict.values())
                except ImportError:
                    status_text.object = "Warning: Louvain algorithm not available. Using greedy modularity."
                    communities = list(community.greedy_modularity_communities(G))
                except Exception as e:
                    status_text.object = f"Warning: Louvain algorithm failed: {str(e)}. Using connected components."
                    communities = [set(c) for c in nx.connected_components(G)]
        except Exception as e:
            status_text.object = f"Warning: Community detection failed: {str(e)}. Using connected components."
            communities = [set(c) for c in nx.connected_components(G)]
        update_progress(85)
        communities.sort(key=len, reverse=True)
        if not communities:
            communities = [set(G.nodes())]
        try:
            clustering_coefficient = nx.average_clustering(G)
        except Exception as e:
            status_text.object = f"Warning: Clustering coefficient calculation failed: {str(e)}"
            clustering_coefficient = 0
        try:
            density = nx.density(G)
        except Exception as e:
            status_text.object = f"Warning: Density calculation failed: {str(e)}"
            density = 0
        try:
            modularity = community.modularity(G, communities) if G.number_of_edges() > 0 else 0
        except Exception as e:
            status_text.object = f"Warning: Modularity calculation failed: {str(e)}"
            modularity = 0
        update_progress(90)
        community_info = f"Detected {len(communities)} communities\n"
        for i, comm in enumerate(communities):
            community_info += f"Community {i+1} size: {len(comm)}\n"
        status_text.object = community_info
        return centrality, communities, clustering_coefficient, density, modularity
    except Exception as e:
        status_text.object = f"Error in graph analysis: {str(e)}"
        traceback.print_exc()
        return {}, [set(G.nodes())], 0, 0, 0

def node_statistics_table(G, communities):
    try:
        rows = []
        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i + 1
        for node in G.nodes():
            try:
                node_data = G.nodes[node]
                connected_nodes = list(G.neighbors(node))
                community_id = community_map.get(node, "N/A")
                row = {
                    "Node ID": str(node),
                    "Community": f"Community {community_id}",
                    "Connected Nodes": len(connected_nodes),
                    "Connections": ", ".join(map(str, connected_nodes))
                }
                for attr, value in node_data.items():
                    if isinstance(value, (int, float)):
                        row[attr] = f"{value:.2f}" if isinstance(value, float) else str(value)
                    else:
                        row[attr] = str(value)
                rows.append(row)
            except Exception as node_error:
                status_text.object = f"Warning: Issue processing node {node}: {str(node_error)}"
                continue
        if not rows:
            status_text.object = "No node statistics available"
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        required_cols = ["Node ID", "Community", "Connected Nodes", "Connections"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = "N/A"
        other_cols = [col for col in df.columns if col not in required_cols]
        df = df[required_cols + other_cols]
        status_text.object = f"Generated statistics for {len(rows)} nodes"
        return df
    except Exception as e:
        status_text.object = f"Status: Network analysis complete. {G.number_of_nodes()} nodes in {len(communities)} communities."
        return pd.DataFrame([{
            "Node ID": node,
            "Community": community_map.get(node, "N/A"),
            "Connected Nodes": len(list(G.neighbors(node))),
            "Connections": "Error loading details"
        } for node in G.nodes()])

def get_layout(G, layout_type):
    try:
        if layout_type == "Circular Layout":
            return nx.circular_layout(G)
        elif layout_type == "Spring Layout":
            if G.number_of_nodes() > 500:
                return nx.spring_layout(G, k=1, iterations=20)
            else:
                return nx.spring_layout(G, k=0.5, iterations=50)
        else:
            return nx.kamada_kawai_layout(G)
    except Exception as e:
        status_text.object = f"Warning: Layout calculation failed: {str(e)}. Using spring layout."
        return nx.spring_layout(G)

def visualize_network(similarity_threshold, centrality_measure, community_algorithm, layout_type):
    """Create network visualization with community colors"""
    try:
        status_text.object = "Starting graph construction..."
        G = build_graph(similarity_threshold)
        
        if G is None or G.number_of_nodes() == 0:
            status_text.object = "No valid graph could be created."
            return None, pd.DataFrame(), pd.DataFrame()
        
        # Clear all existing figures
        plt.close('all')
        
        # Get communities and centrality
        centrality, communities, _, _, _ = analyze_graph(G, centrality_measure, community_algorithm)
        
        # Get layout
        pos = get_layout(G, layout_type)
        
        # Generate colors for communities
        community_colors = get_community_colors(len(communities))
        
        # Create community mapping
        node_community = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                node_community[node] = idx

        # Create new figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw network
        nx.draw(G, pos,
               node_color=[community_colors[node_community.get(node, 0)] for node in G.nodes()],
               node_size=[3000 * centrality.get(node, 0.1) for node in G.nodes()],
               with_labels=True,
               font_size=8,
               font_weight='bold',
               edge_color='gray',
               alpha=0.7,
               ax=ax)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color,
                                    label=f'Community {i+1}',
                                    markersize=10)
                         for i, color in enumerate(community_colors)]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.title("Network Visualization\nNode size indicates centrality measure")
        plt.tight_layout()
        
        # Calculate statistics
        node_stats = node_statistics_table(G, communities)
        community_stats = calculate_community_statistics(G, communities)
        
        return fig, node_stats, community_stats
            
    except Exception as e:
        plt.close('all')  # Ensure cleanup on error
        status_text.object = f"Visualization error: {str(e)}"
        traceback.print_exc()
        return None, pd.DataFrame(), pd.DataFrame()

# Add at the top of the file with other imports
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 50  # Increase warning threshold if needed

pn.extension(sizing_mode="stretch_width")

def start_analysis(event):
    global data
    try:
        if data is None or data.empty:
            status_text.object = "Please upload a dataset first"
            return
        status_text.object = "Building graph..."
        update_progress(0)
        viz_result, node_table, community_table = visualize_network(
            similarity_slider.value,
            centrality_choice.value,
            community_algo.value,
            layout_choice.value
        )
        if viz_result is not None:
            graph_pane.object = viz_result
            table_pane.object = node_table
            community_table_pane.object = community_table
            status_text.object = "Analysis complete!"
        update_progress(100)
    except Exception as e:
        status_text.object = f"Error during analysis: {str(e)}"
        traceback.print_exc()
        update_progress(0)

def export_results(event):
    try:
        viz_result = graph_pane.object
        if viz_result is not None:
            download_container = pn.Column(pn.pane.Markdown("### Download Files"), sizing_mode='stretch_width')
            if hasattr(table_pane, 'object') and not table_pane.object.empty:
                import tempfile
                import os
                node_stats_file = os.path.join(tempfile.gettempdir(), 'node_statistics.csv')
                table_pane.object.to_csv(node_stats_file, index=False)
                node_download = pn.widgets.FileDownload(file=node_stats_file, filename="node_statistics.csv", button_type="primary", label="Download Node Statistics")
                download_container.append(node_download)
            if hasattr(community_table_pane, 'object') and not community_table_pane.object.empty:
                comm_stats_file = os.path.join(tempfile.gettempdir(), 'community_statistics.csv')
                community_table_pane.object.to_csv(comm_stats_file, index=False)
                comm_download = pn.widgets.FileDownload(file=comm_stats_file, filename="community_statistics.csv", button_type="primary", label="Download Community Statistics")
                download_container.append(comm_download)
            if viz_result:
                temp_png = os.path.join(tempfile.gettempdir(), 'network_viz_temp.png')
                viz_result.savefig(temp_png, format='png', bbox_inches='tight', dpi=300)
                png_download = pn.widgets.FileDownload(file=temp_png, filename="network_visualization.png", button_type="primary", label="Download PNG Image")
                download_container.append(png_download)
            if 'Downloads' in [tab[0] for tab in tabs]:
                tabs.pop('Downloads')
            tabs.append(('Downloads', download_container))
            tabs.active = len(tabs) - 1
            status_text.object = "Export ready! Click the download buttons to save files."
        else:
            status_text.object = "No visualization results available. Please run the analysis first."
    except Exception as e:
        status_text.object = f"Error during export: {str(e)}"
        traceback.print_exc()

url_input.param.watch(load_data_from_url, 'value')
reset_button.on_click(reset_upload)
start_button.on_click(start_analysis)
export_button.on_click(export_results)

def optimize_dataframe(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def preprocess_data(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        medians = df[numeric_cols].median().fillna(0)
        df[numeric_cols] = df[numeric_cols].fillna(medians)
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        if df[col].isna().any():
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("Unknown")
    if df.isna().any().any():
        df = df.fillna(0)
    return df

def validate_dataset(df):
    required_numeric_cols = 2
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < required_numeric_cols:
        raise ValueError(f"Dataset must have at least {required_numeric_cols} numeric columns")
    return True

def handle_file_upload(event):
    if event.new:
        url_input.disabled = True
    global data, feature_columns, disease_column, file_data_store
    try:
        if event.new:
            chunks = []
            for chunk in pd.read_csv(io.BytesIO(event.new), chunksize=10000):
                chunk = optimize_dataframe(chunk)
                chunk = preprocess_data(chunk)
                chunks.append(chunk)
            data = pd.concat(chunks)
            if 'ID' not in data.columns:
                data['ID'] = [f'P{i:03d}' for i in range(1, len(data) + 1)]
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if 'ID' in numeric_columns:
                numeric_columns.remove('ID')
            potential_disease_cols = [col for col in data.columns if col.lower() in ['disease', 'condition', 'diagnosis', 'target', 'label', 'class']]
            if potential_disease_cols:
                disease_column = potential_disease_cols[0]
                feature_columns = [col for col in numeric_columns if col != disease_column]
            else:
                disease_column = 'condition'
                data[disease_column] = 'Group A'
                feature_columns = numeric_columns
                if len(numeric_columns) >= 2:
                    try:
                        from sklearn.cluster import KMeans
                        features = data[numeric_columns].fillna(0)
                        kmeans = KMeans(n_clusters=min(5, len(data)), random_state=42)
                        clusters = kmeans.fit_predict(features)
                        data[disease_column] = [f'Group {chr(65+c)}' for c in clusters]
                    except Exception:
                        pass
            if len(feature_columns) < 1:
                num_rows = len(data)
                data['feature1'] = np.random.normal(size=num_rows)
                data['feature2'] = np.random.normal(size=num_rows)
                feature_columns = ['feature1', 'feature2']
                status_text.object = "Created synthetic features for visualization."
            else:
                status_text.object = f"Dataset loaded successfully with {len(feature_columns)} features."
            url_input.disabled = True
        else:
            reset_upload(None)
    except Exception as e:
        status_text.object = f"Error loading file: {str(e)}"
        traceback.print_exc()
        reset_upload(None)

file_input.param.watch(handle_file_upload, 'value')

title = pn.Column(pn.pane.Markdown("# Network Analysis Dashboard"), styles={'text-align': 'center'})
tabs = pn.Tabs(("Graph Visualization", pn.Column(graph_pane, sizing_mode='stretch_width', height=650)), ("Node Statistics", table_pane), ("Community Statistics", community_table_pane))
data_input_section = pn.Column(pn.pane.Markdown("## Data Input", styles={'text-align': 'center', 'margin-bottom': '5px'}), pn.pane.Markdown("### Upload Dataset", styles={'text-align': 'center', 'margin-bottom': '2px'}), file_input, pn.Spacer(height=2), pn.pane.Markdown("### OR", styles={'text-align': 'center', 'font-weight': 'bold', 'font-size': '16px', 'color': '#666', 'margin': '0px'}), pn.Spacer(height=2), pn.pane.Markdown("### Enter Dataset URL", styles={'text-align': 'center', 'margin-bottom': '2px'}), url_input, pn.Spacer(height=8), styles={'background': '#f8f9fa', 'padding': '8px', 'border-radius': '5px'})
sidebar = pn.Column(data_input_section, pn.pane.Markdown("## Analysis Settings", styles={'text-align': 'center'}), similarity_slider, centrality_choice, community_algo, layout_choice, pn.Spacer(height=20), pn.Row(start_button, pn.Spacer(width=50), export_button), pn.Spacer(height=10), pn.Row(reset_button), pn.Spacer(height=20), pn.pane.Markdown("## Progress", styles={'text-align': 'center'}), pn.Column(status_text, progress_bar, progress_text), width=350, styles={'background': '#f5f5f5', 'padding': '20px'})
app = pn.template.MaterialTemplate(title="Medical Biometrics Network Analysis Using Graph Mining", sidebar=sidebar, main=[title, tabs], header_background="#1f77b4")
if __name__ == '__main__':
    app.show(port=5006)
else:
    app.servable()