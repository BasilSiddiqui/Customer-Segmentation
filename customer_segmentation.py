# 📦 Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# 🧾 Load Data
df = pd.read_excel(r"C:\Users\basil\OneDrive\Desktop\Base\Work\Personal projects\Mall_customers\Mall Customers.xlsx")

# 🧹 Fix Column Names
df.rename(columns={'Education ': 'Education'}, inplace=True)

# ✅ EDA - Distribution Plots

# --------------------------------------------
# 📊 Distribution of Annual Income (Histogram)
# --------------------------------------------

sns.set_style("whitegrid", {'grid.linestyle': '--', 'axes.edgecolor': '0.3'})
plt.figure(figsize=(10, 6))
ax = sns.histplot(
    df['Annual Income (k$)'], 
    kde=True,                          # Overlay KDE curve
    color='royalblue', 
    bins=20, 
    alpha=0.7,                         # Slight transparency
    linewidth=0.5,
    edgecolor='white'                 # White edges between bars
)
plt.title('Distribution of Annual Income', fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Annual Income (k$)', fontsize=12, labelpad=10)
plt.ylabel('Count', fontsize=12, labelpad=10)
sns.despine(left=True)               # Remove left & top spines
plt.grid(axis='y', alpha=0.3)
plt.show()


# --------------------------------------------
# 🧓 Age Distribution (Histogram with KDE)
# --------------------------------------------

sns.set_style("dark", {'axes.facecolor': '#f5f5f5', 'text.color': '0.3'})
plt.figure(figsize=(10, 6))

ax = sns.histplot(
    df['Age'], 
    kde=True,                         # Keep KDE line
    color='crimson', 
    bins=15, 
    alpha=0.8,
    edgecolor='white', 
    linewidth=0.5
)

plt.title('Distribution of Age', fontsize=16, pad=20, fontweight='bold', color='0.3')
plt.xlabel('Age', fontsize=12, labelpad=10, color='0.3')
plt.ylabel('Count', fontsize=12, labelpad=10, color='0.3')
ax.grid(alpha=0.2)
sns.despine()
plt.show()


# --------------------------------------------
# 🟣 Gender Distribution (Pie Chart)
# --------------------------------------------

labels = ['Female', 'Male']
size = df['Gender'].value_counts().values
colors = ['#ff007f', '#007fff']       # Hot pink and hot blue
explode = [0, 0.05]                   # Slight separation for one slice

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    size, 
    colors=colors, 
    explode=explode, 
    labels=labels, 
    autopct='%.1f%%', 
    startangle=90,
    shadow=False,
    textprops={'fontsize': 28, 'color': '0.4'},
    wedgeprops={'edgecolor': 'black', 'linewidth': 4}
)
plt.title('Gender Distribution', fontsize=16, pad=20, fontweight='bold', color='0.3')
plt.legend(wedges, labels, loc='upper right', bbox_to_anchor=(1.1, 1))
plt.setp(autotexts, size=12, weight='bold', color='white')  # Style % labels
plt.axis('equal')
plt.show()


# --------------------------------------------
# 🔁 Pairplot Colored by Gender
# --------------------------------------------

sns.set_theme(
    style="ticks",
    rc={"axes.spines.right": False, "axes.spines.top": False}
)
g = sns.pairplot(
    df, 
    hue='Gender', 
    palette={'F': '#ff007f', 'M': '#007fff'},     # Use mapped values
    plot_kws={'alpha': 0.7, 'edgecolor': 'w', 'linewidth': 0.5},
    diag_kws={'alpha': 0.8, 'edgecolor': 'w'}
)
g.fig.suptitle('Pairwise Relationships by Gender', y=1.02, fontsize=16, fontweight='bold')
plt.show()


# --------------------------------------------
# 🔁 Pairplot (No Hue - General Overview)
# --------------------------------------------

sns.set_theme(
    style="ticks",
    rc={"axes.spines.right": False, "axes.spines.top": False}
)
g = sns.pairplot(df)
g.fig.suptitle('Pairwise Relationships by Gender', y=1.02, fontsize=16, fontweight='bold')
plt.show()


# 🧠 Data Preprocessing
# 🎯 Ordinal Encode Education
education_map = {
    'Uneducated': 0, 
    'High School': 1, 
    'College': 2,
    'Graduate': 3, 
    'Post-Graduate': 4, 
    'Doctorate': 5, 
    'Unknown': 0
    }

df['Education_enc'] = df['Education'].map(education_map)

# ❌ Drop unused columns
df_encoded = df.drop(columns=['CustomerID', 'Education'])

# 🧠 One-Hot Encode Gender and Marital Status
df_encoded = pd.get_dummies(df_encoded, columns=['Gender', 'Marital Status'], drop_first=True)

# 📏 Scale features
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df_encoded)

# 📉 Dimensionality Reduction for Visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_df)

# 🚀 Clustering Algorithms
algorithms = {
    "KMeans": KMeans(n_clusters=4, random_state=0),
    "Agglomerative": AgglomerativeClustering(n_clusters=4),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "GMM": GaussianMixture(n_components=4, random_state=0)
}

# 📊 Visualization & Metric Evaluation
plt.figure(figsize=(16, 12))
for i, (name, algo) in enumerate(algorithms.items(), 1):
    if name == "GMM":
        labels = algo.fit_predict(scaled_df)
    else:
        labels = algo.fit(scaled_df).labels_

    # 📈 Plot
    plt.subplot(2, 2, i)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='tab10', s=50)
    plt.title(name)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    # 🧪 Print evaluation metrics
    print(f"\n{name} Evaluation Metrics:")
    try:
        print(f"Silhouette Score: {silhouette_score(scaled_df, labels):.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin_score(scaled_df, labels):.4f}")
        print(f"Calinski-Harabasz Score: {calinski_harabasz_score(scaled_df, labels):.4f}")
    except Exception as e:
        print(f"Evaluation Error: {e} (likely due to noise or single cluster)")

plt.tight_layout()
plt.show()