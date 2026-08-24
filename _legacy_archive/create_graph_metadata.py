"""
Quick converter to create graph_metadata.json from existing data
This enables GNN training to work with the demo graphs
"""

import json
import os
import pandas as pd

# Load the CSV data
csv_file = "data/merge_csv_20240704.csv"
df = pd.read_csv(csv_file)

# Limit to first 500 samples for quick testing
df_subset = df.head(500)

# Create metadata for each sample
metadata = {}
for idx, row in df_subset.iterrows():
    graph_id = f"graph_{idx:06d}"
    metadata[graph_id] = {
        "filename": str(row.get('filename', f'sample_{idx}')),
        "label": str(row.get('label', 'unknown')),
        "source_index": int(idx)
    }

# Save metadata
os.makedirs("graph_output/graphs", exist_ok=True)

with open("graph_output/graph_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Created graph_metadata.json with {len(metadata)} samples")
print(f"   File: graph_output/graph_metadata.json")

# Create dummy graph files for each
print("Creating dummy graph files...")
for graph_id in list(metadata.keys()):  # Create all graphs
    graph_data = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {"id": f"node_0", "type": "SAMPLE"},
            {"id": f"node_1", "type": "API_CALL"},
            {"id": f"node_2", "type": "STRING"}
        ],
        "links": [
            {"source": "node_0", "target": "node_1", "type": "HAS_IMPORT"},
            {"source": "node_0", "target": "node_2", "type": "HAS_STRING"}
        ]
    }
    
    graph_file = f"graph_output/graphs/{graph_id}.json"
    with open(graph_file, "w") as f:
        json.dump(graph_data, f)

print(f"✅ Created {len(metadata)} graph files in graph_output/graphs/")
print("\n📦 Data ready for GNN training!")
print("   Run: python agentic_pacx/gnn_training.py")
