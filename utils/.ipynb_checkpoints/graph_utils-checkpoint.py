import os
import pickle
import numpy as np
import random
from dwave.system import DWaveSampler, EmbeddingComposite

def load_or_create_graph(graph_path, n_visible, n_hidden, token=None):
    if os.path.exists(graph_path):
        with open(graph_path, 'rb') as f:
            graph_info = pickle.load(f)
        print("✅ Loaded saved D-Wave graph.")
    else:
        print("⚠️ No saved graph found. Creating new graph from D-Wave QPU.")
        sampler = EmbeddingComposite(DWaveSampler(solver={'name__contains': 'Advantage2'}, token=token))
        adjacency = sampler.child.adjacency
        all_qubits = list(adjacency.keys())
        random.shuffle(all_qubits)
        visible = all_qubits[:n_visible]
        hidden = all_qubits[n_visible:n_visible + n_hidden]
        mask = np.zeros((n_visible, n_hidden))
        for i, v in enumerate(visible):
            for j, h in enumerate(hidden):
                if h in adjacency[v]:
                    mask[i, j] = 1.0
        graph_info = {'adjacency': adjacency, 'visible': visible, 'hidden': hidden, 'mask': mask}
        with open(graph_path, 'wb') as f:
            pickle.dump(graph_info, f)
        print("✅ Saved new D-Wave graph.")
    return graph_info
