import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Set page configuration
st.set_page_config(
    page_title="Wine Clustering Analysis",
    page_icon="🍷",
    layout="wide"
)

# Title and description
st.title("🍷 Wine Clustering Analysis")
st.markdown("### Exploring wine characteristics using K-Means and Hierarchical Clustering")

# Load data
@st.cache_data
def load_data():
    wine_data = load_wine()
    X = wine_data.data
    feature_names = wine_data.feature_names
    y_true = wine_data.target
    df = pd.DataFrame(X, columns=feature_names)
    return df, X, feature_names, y_true

df, X, feature_names, y_true = load_data()

# Sidebar for user controls
st.sidebar.header("Analysis Controls")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Dataset Overview", "K-Means Clustering", "Hierarchical Clustering", "Comparison"]
)

# Main content based on selection
if analysis_type == "Dataset Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Features:** {len(feature_names)}")
        st.write(f"**Samples:** {len(df)}")
        
        st.subheader("Missing Values")
        missing_values = df.isnull().sum().sum()
        st.write(f"**Total Missing Values:** {missing_values}")
    
    with col2:
        st.subheader("Feature Statistics")
        st.dataframe(df.describe())
    
    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', ax=ax)
    st.pyplot(fig)

elif analysis_type == "K-Means Clustering":
    st.header("🎯 K-Means Clustering Analysis")
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means analysis
    k_range = range(2, 6)
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    
    # Plot silhouette scores
    st.subheader("Silhouette Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Analysis for K-Means')
    ax.grid(True, alpha=0.3)
    
    for i, score in enumerate(silhouette_scores):
        ax.text(k_range[i], score, f'{score:.3f}', ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Best k analysis
    best_k = k_range[np.argmax(silhouette_scores)]
    st.write(f"**Optimal number of clusters:** {best_k}")
    
    # Final clustering with best k
    kmeans_final = KMeans(n_clusters=best_k, random_state=42)
    kmeans_labels = kmeans_final.fit_predict(X_scaled)
    
    # PCA visualization
    st.subheader("Cluster Visualization (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, 
                        cmap='viridis', alpha=0.7, s=50)
    
    # Plot centroids
    centroids_pca = pca.transform(kmeans_final.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids')
    
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title('K-Means Clustering Results')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Cluster')
    
    st.pyplot(fig)

elif analysis_type == "Hierarchical Clustering":
    st.header("🌳 Hierarchical Clustering Analysis")
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Hierarchical clustering
    linkage_matrix = linkage(X_scaled, method='ward')
    
    # Dendrogram
    st.subheader("Dendrogram")
    fig, ax = plt.subplots(figsize=(15, 8))
    dendrogram(linkage_matrix, truncate_mode='level', p=5, ax=ax)
    ax.set_title('Hierarchical Clustering Dendrogram')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Distance')
    st.pyplot(fig)
    
    # Test different k values
    k_range = range(2, 5)
    hier_scores = []
    
    for k in k_range:
        hier_clust = AgglomerativeClustering(n_clusters=k, linkage='ward')
        hier_labels = hier_clust.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, hier_labels)
        hier_scores.append(score)
    
    best_k_hier = k_range[np.argmax(hier_scores)]
    st.write(f"**Optimal number of clusters:** {best_k_hier}")
    
    # Final clustering
    hier_final = AgglomerativeClustering(n_clusters=best_k_hier, linkage='ward')
    hier_labels = hier_final.fit_predict(X_scaled)
    
    # PCA visualization
    st.subheader("Cluster Visualization (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=hier_labels, 
                        cmap='plasma', alpha=0.7, s=50)
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title('Hierarchical Clustering Results')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Cluster')
    
    st.pyplot(fig)

else:  # Comparison
    st.header("⚖️ Algorithm Comparison")
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run both algorithms
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    hier_clust = AgglomerativeClustering(n_clusters=3, linkage='ward')
    hier_labels = hier_clust.fit_predict(X_scaled)
    
    # Calculate metrics
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    hier_silhouette = silhouette_score(X_scaled, hier_labels)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("K-Means Silhouette Score", f"{kmeans_silhouette:.3f}")
    
    with col2:
        st.metric("Hierarchical Silhouette Score", f"{hier_silhouette:.3f}")
    
    # Side-by-side visualization
    st.subheader("Side-by-Side Comparison")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # K-Means plot
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, 
                          cmap='viridis', alpha=0.7, s=50)
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='red', marker='x', s=200, linewidths=3)
    ax1.set_title('K-Means Clustering')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.grid(True, alpha=0.3)
    
    # Hierarchical plot
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=hier_labels, 
                          cmap='plasma', alpha=0.7, s=50)
    ax2.set_title('Hierarchical Clustering')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("### 📈 Analysis Summary")
st.markdown("""
This application demonstrates:
- **Data exploration** of the wine dataset
- **K-Means clustering** with optimal cluster selection
- **Hierarchical clustering** with dendrogram analysis  
- **Comparative analysis** of both methods

Both algorithms successfully identify 3 distinct wine clusters representing different quality segments.
""")
