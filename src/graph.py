import json
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx


def main():
    file_names = [
        'data/office/s01e01_frames.txt',
        'data/office/s01e02_frames.txt',
        'data/office/s01e03_frames.txt',
        'data/office/s01e05_frames.txt',
        'data/office/s01e06_frames.txt',
    ]
    face_frequencies = defaultdict(int)
    edges = defaultdict(int)
    for file_name in file_names:
        with open(file_name) as f:
            content = f.readlines()
            for line in content:
                [frame_number, face_names] = json.loads(line)
                for i, face_name in enumerate(face_names):
                    face_frequencies[face_name] += 1
                    for face_name_j in face_names[i + 1:]:
                        key = tuple(sorted([face_name, face_name_j]))
                        edges[key] += 1

    G = nx.Graph()
    for face_name, face_count in face_frequencies.items():
        if face_count == 1:
            continue
        if face_name in ('Unknown', 'Bob Vance'):
            continue
        G.add_node(face_name, size=face_count)
    for pair, edge_count in edges.items():
        if edge_count == 1:
            continue
        if pair[0] in ('Unknown', 'Bob Vance') or pair[1] in ('Unknown', 'Bob Vance'):
            continue
        G.add_edge(pair[0], pair[1], weight=1./edge_count)

    nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold')
    plt.show()


main()
