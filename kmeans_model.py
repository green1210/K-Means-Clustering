import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")

df = pd.read_csv("iris.csv")  

X = df.select_dtypes(include=['number'])
if X.shape[1] < 2:
    raise SystemExit("Need at least two numeric features to plot 2D scatter. Add numeric columns or remove target.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method')
plt.show()

k = 3  
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette='tab10', s=60)
plt.title(f'K-Means Clustering (k={k}) — first 2 features (scaled)')
plt.xlabel(f'{X.columns[0]} (scaled)')
plt.ylabel(f'{X.columns[1]} (scaled)')
plt.legend(title='Cluster')
plt.show()

print("\nCluster counts:")
print(df['Cluster'].value_counts().sort_index())

print("\nCluster centers (in scaled feature space):")
print(kmeans.cluster_centers_)
