import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    def __init__(self):
        self.customers_df = None 
        self.transactions_df = None
        self.features = None
        self.clusters = None
        self.db_index = None
        self.silhouette_avg = None

    def load_data(self):
        """Load and validate data"""
        try:
            self.customers_df = pd.read_csv('data/raw/Customers.csv')
            self.transactions_df = pd.read_csv('data/raw/Transactions.csv')
            self.customers_df['SignupDate'] = pd.to_datetime(self.customers_df['SignupDate'])
            self.transactions_df['TransactionDate'] = pd.to_datetime(self.transactions_df['TransactionDate'])
            print("Data loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def create_features(self):
        """Engineer features for clustering"""
        transaction_metrics = self.transactions_df.groupby('CustomerID').agg({
            'TransactionID': 'count',
            'TotalValue': ['sum', 'mean', 'std'],
            'Quantity': ['sum', 'mean'],
            'TransactionDate': lambda x: (x.max() - x.min()).days
        }).reset_index()

        transaction_metrics.columns = [
            'CustomerID', 'num_transactions', 'total_spend', 
            'avg_transaction', 'std_transaction',
            'total_quantity', 'avg_quantity', 'customer_lifetime'
        ]

        transaction_metrics['purchase_frequency'] = (
            transaction_metrics['num_transactions'] / 
            transaction_metrics['customer_lifetime'].clip(lower=1)
        )
        transaction_metrics['avg_basket_size'] = (
            transaction_metrics['total_quantity'] / 
            transaction_metrics['num_transactions']
        )

        region_dummies = pd.get_dummies(self.customers_df['Region'], prefix='region')

        self.features = pd.merge(
            transaction_metrics,
            pd.concat([self.customers_df[['CustomerID']], region_dummies], axis=1),
            on='CustomerID'
        )

        numeric_cols = self.features.select_dtypes(include=[np.number]).columns
        self.features[numeric_cols] = self.features[numeric_cols].fillna(0)

        return self.features

    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using DB Index and Silhouette Score"""
        feature_cols = [col for col in self.features.columns if col not in ['CustomerID']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features[feature_cols])

        db_scores = []
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features_scaled)
            db_score = davies_bouldin_score(features_scaled, labels)
            silhouette_avg = silhouette_score(features_scaled, labels)
            db_scores.append(db_score)
            silhouette_scores.append(silhouette_avg)
            print(f"k={k}, DB Index={db_score:.4f}, Silhouette Score={silhouette_avg:.4f}")

        optimal_k = np.argmin(db_scores) + 2
        self.db_index = min(db_scores)
        self.silhouette_avg = max(silhouette_scores)

        return optimal_k, features_scaled

    def perform_clustering(self, n_clusters):
        """Perform clustering with optimal k"""
        feature_cols = [col for col in self.features.columns if col not in ['CustomerID']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features[feature_cols])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(features_scaled)

        return features_scaled

    def visualize_clusters(self, features_scaled):
        """Create cluster visualizations"""
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=self.clusters, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('Customer Segments')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.show()

        self.features['Cluster'] = self.clusters
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        sns.boxplot(data=self.features, x='Cluster', y='total_spend', ax=axes[0,0])
        axes[0,0].set_title('Total Spend by Cluster')

        sns.boxplot(data=self.features, x='Cluster', y='purchase_frequency', ax=axes[0,1])
        axes[0,1].set_title('Purchase Frequency by Cluster')

        sns.boxplot(data=self.features, x='Cluster', y='avg_basket_size', ax=axes[1,0])
        axes[1,0].set_title('Basket Size by Cluster')

        sns.boxplot(data=self.features, x='Cluster', y='customer_lifetime', ax=axes[1,1])
        axes[1,1].set_title('Customer Lifetime by Cluster')

        plt.tight_layout()
        plt.show()

    def generate_report(self):
        """Generate clustering report"""
        numeric_cols = self.features.select_dtypes(include=[np.number]).columns
        report = {
            'n_clusters': len(np.unique(self.clusters)),
            'db_index': self.db_index,
            'silhouette_avg': self.silhouette_avg,
            'cluster_sizes': pd.Series(self.clusters).value_counts().to_dict(),
            'cluster_profiles': self.features.groupby('Cluster')[numeric_cols].mean().round(2).to_dict()
        }
        return report

def main():
    segmentation = CustomerSegmentation()

    if not segmentation.load_data():
        return

    print("\nCreating customer features...")
    segmentation.create_features()

    print("\nFinding optimal number of clusters...")
    n_clusters, features_scaled = segmentation.find_optimal_clusters()

    print(f"\nPerforming clustering with k={n_clusters}...")
    features_scaled = segmentation.perform_clustering(n_clusters)

    print("\nCreating visualizations...")
    segmentation.visualize_clusters(features_scaled)

    report = segmentation.generate_report()

    print("\nClustering Results:")
    print("-" * 50)
    print(f"Number of clusters: {report['n_clusters']}")
    print(f"Davies-Bouldin Index: {report['db_index']:.4f}")
    print(f"Silhouette Score: {report['silhouette_avg']:.4f}")
    print("\nCluster sizes:")
    for cluster, size in report['cluster_sizes'].items():
        print(f"Cluster {cluster}: {size} customers")

if __name__ == "__main__":
    main()