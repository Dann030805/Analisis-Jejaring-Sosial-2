import matplotlib.pyplot as plt
import community as community_louvain  # Package: python-louvain
import networkx as nx
import pandas as pd

# ==========================================
# 1. IMPORT DATA KE NETWORKX
# ==========================================
file_path = "facebook_combined.txt"

# Membaca file edgelist SNAP
df = pd.read_csv(file_path, sep=" ", names=["Source", "Target"])

# Membuat objek graph undirect (tak berarah, tidak berbobot)
G = nx.from_pandas_edgelist(df, source="Source", target="Target")

print("--- Data Berhasil Dimuat ---")
print(f"Total Node dalam jaringan: {G.number_of_nodes()}")
print(f"Total Edge dalam jaringan: {G.number_of_edges()}")

# ==========================================
# 2. HITUNG & ANALISIS CENTRALITY (SOAL NO. 2)
# ==========================================
print("\n[MENGHITUNG METRIK CENTRALITY...]")

# • Degree Centrality
degree_cent = nx.degree_centrality(G)

# • Betweenness Centrality (Menggunakan sampel k=100 agar proses cepat)
betweenness_cent = nx.betweenness_centrality(G, k=100, seed=42)

# • Closeness Centrality
closeness_cent = nx.closeness_centrality(G)

# • Eigenvector Centrality
eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)


# Fungsi pembantu untuk mencari node dengan nilai tertinggi
def get_top_node(centrality_dict):
    top_node = max(centrality_dict, key=centrality_dict.get)
    return top_node, centrality_dict[top_node]


print("\n--- HASIL ANALISIS CENTRALITY ---")
node, val = get_top_node(degree_cent)
print(f"Top Degree Centrality      -> Node: {node} (Nilai: {val:.4f})")

node, val = get_top_node(betweenness_cent)
print(f"Top Betweenness Centrality -> Node: {node} (Nilai: {val:.4f})")

node, val = get_top_node(closeness_cent)
print(f"Top Closeness Centrality   -> Node: {node} (Nilai: {val:.4f})")

node, val = get_top_node(eigenvector_cent)
print(f"Top Eigenvector Centrality -> Node: {node} (Nilai: {val:.4f})")


# ==========================================
# 3. METRIK GLOBAL & DETEKSI LOUVAIN (SOAL NO. 3)
# ==========================================
print("\n--- HASIL METRIK GLOBAL ---")

# • Density
density = nx.density(G)
print(f"Density Jaringan         : {density:.5f}")

# • Average Clustering Coefficient
avg_clustering = nx.average_clustering(G)
print(f"Clustering Coefficient   : {avg_clustering:.5f}")

# Cek keterhubungan komponen untuk Diameter & Average Path Length
if nx.is_connected(G):
    diameter = nx.diameter(G)
    avg_path_length = nx.average_shortest_path_length(G)
    print(f"Diameter Jaringan        : {diameter}")
    print(f"Average Path Length      : {avg_path_length:.4f}")
else:
    print(
        "Graf tidak terhubung sepenuhnya. Menghitung komponen terbesar (LCC)..."
    )
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)

    diameter = nx.diameter(subgraph)
    avg_path_length = nx.average_shortest_path_length(subgraph)
    print(f"Diameter (LCC)           : {diameter}")
    print(f"Average Path Length (LCC): {avg_path_length:.4f}")

# • Deteksi Komunitas Louvain
partition = community_louvain.best_partition(G)
num_communities = len(set(partition.values()))
print(f"Jumlah Komunitas Louvain : {num_communities}")


# ==========================================
# 5. VISUALISASI JEJARING (SOAL NO. 5)
# ==========================================
print("\n[MEMPROSES VISUALISASI... MOHON TUNGGU SEBENTAR]")

plt.figure(figsize=(12, 12))

# Layout spring
pos = nx.spring_layout(G, seed=42)

# Ukuran node proporsional terhadap Degree Centrality agar terlihat mana yang 'Hub/Influencer'
node_sizes = [degree_cent[node] * 500 for node in G.nodes()]

# Warna node dikelompokkan berdasarkan ID sirkel/komunitas Louvain-nya
node_colors = [partition[node] for node in G.nodes()]

# Menggambar elemen graf
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap=plt.cm.jet,
    alpha=0.7,
)
nx.draw_networkx_edges(G, pos, alpha=0.05)  # Alpha tipis agar graf padat tidak semrawut

plt.title(
    "Visualisasi Jejaring Sosial Facebook\nUkuran Node: Degree Centrality | Warna Node: Komunitas Louvain",
    fontsize=14,
)
plt.axis("off")
plt.show()