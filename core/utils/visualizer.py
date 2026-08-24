import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import os

def generate_pictorial_graph(dataset_path="data/pyg_dataset_norm.pt", output_rel_path="img/graph_example.png"):
    """Generates a high-quality visualization of a single malware graph."""
    # Define output absolute path
    static_dir = os.path.join("web", "static")
    output_path = os.path.join(static_dir, output_rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load dataset
    try:
        dataset = torch.load(dataset_path, weights_only=False)
        g1 = dataset[0]
    except Exception:
        return None
    
    # Define node mapping
    NODE_LABELS = {0: "Header", 1: "Entropy", 2: "API", 3: "Network"}
    NODE_COLORS = {0: "#A855F7", 1: "#1E90FF", 2: "#2ED573", 3: "#FFA502"}
    
    # Convert PyG graph to NetworkX
    G = to_networkx(g1, to_undirected=False)
    pos = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    
    # Create figure
    plt.ioff()
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0B0E14")
    ax.set_facecolor("#0B0E14")
    
    colors = [NODE_COLORS[i] for i in range(4)]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=6000, alpha=0.95, edgecolors="white", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#7A8B99", width=2.5, arrowsize=25, connectionstyle="arc3,rad=0.2", alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=NODE_LABELS, font_size=13, font_color="white", font_weight="bold", ax=ax)
    
    plt.title("Neural Topology Audit (G_1 Sample)", color="white", fontsize=16, fontweight="bold", pad=20)
    plt.axis('off')
    
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor="#0B0E14")
    plt.close(fig)
    
    return output_rel_path

if __name__ == "__main__":
    path = generate_pictorial_graph()
    print(f"Generated: {path}")
