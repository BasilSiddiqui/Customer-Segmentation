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
sns.set(style='whitegrid')
sns.histplot(df['Annual Income (k$)'], kde=True)
plt.title('Distribution of Annual Income')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Count')
plt.show()

sns.set(style='dark')
sns.histplot(df['Age'], kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# ✅ EDA - Gender Pie Chart
labels = ['Female', 'Male']
size = df['Gender'].value_counts().values
colors = ['pink', 'skyblue']
explode = [0, 0.1]

plt.figure(figsize=(6, 6))
plt.pie(size, colors=colors, explode=explode, labels=labels, autopct='%.2f%%', shadow=True)
plt.title('Gender Distribution')
plt.axis('equal')
plt.legend()
plt.show()

# ✅ EDA - Pair Plots
sns.pairplot(df)
plt.show()

sns.pairplot(df, hue='Gender')
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
df = df.drop(columns=['CustomerID', 'Education'])

# 🧠 One-Hot Encode Gender and Marital Status
df = pd.get_dummies(df, columns=['Gender', 'Marital Status'], drop_first=True)

# 📏 Scale features
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

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
