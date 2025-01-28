import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import warnings
warnings.filterwarnings('ignore')

# Create required directories
os.makedirs('reports', exist_ok=True)

class LookalikeModel:
    def __init__(self):
        self.customers_df = None
        self.transactions_df = None
        self.customer_features = None
        self.similarity_matrix = None
        
    def load_data(self):
        """Load and clean data"""
        try:
            self.customers_df = pd.read_csv('data/raw/Customers.csv')
            self.transactions_df = pd.read_csv('data/raw/Transactions.csv')
            
            # Convert dates
            self.customers_df['SignupDate'] = pd.to_datetime(self.customers_df['SignupDate'])
            self.transactions_df['TransactionDate'] = pd.to_datetime(self.transactions_df['TransactionDate'])
            
            print("Data loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_features(self):
        """Create customer features"""
        # Calculate customer metrics
        customer_metrics = self.transactions_df.groupby('CustomerID').agg({
            'TransactionID': 'count',
            'TotalValue': ['sum', 'mean'],
            'Quantity': 'sum',
            'TransactionDate': lambda x: (x.max() - x.min()).days
        }).reset_index()
        
        customer_metrics.columns = [
            'CustomerID', 'num_transactions', 'total_spend', 
            'avg_spend', 'total_quantity', 'active_days'
        ]
        
        # Calculate derived features
        customer_metrics['frequency'] = customer_metrics['num_transactions'] / \
                                      customer_metrics['active_days'].clip(lower=1)
        customer_metrics['avg_basket'] = customer_metrics['total_spend'] / \
                                       customer_metrics['num_transactions']
        
        # Encode regions
        region_dummies = pd.get_dummies(self.customers_df['Region'], prefix='region')
        
        # Combine features
        self.customer_features = pd.merge(
            customer_metrics,
            pd.concat([self.customers_df[['CustomerID']], region_dummies], axis=1),
            on='CustomerID'
        ).fillna(0)
        
        return self.customer_features
    
    def calculate_similarity(self):
        """Calculate similarity matrix"""
        feature_cols = [col for col in self.customer_features.columns 
                       if col != 'CustomerID']
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.customer_features[feature_cols])
        
        # Calculate similarity
        self.similarity_matrix = cosine_similarity(features_scaled)
        return self.similarity_matrix
    
    def find_lookalikes(self, customer_id, n_similar=3):
        """Find similar customers"""
        try:
            idx = self.customer_features[
                self.customer_features['CustomerID'] == customer_id
            ].index[0]
            
            similarities = self.similarity_matrix[idx]
            similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
            
            return [{
                'similar_customer': self.customer_features.iloc[i]['CustomerID'],
                'similarity_score': similarities[i]
            } for i in similar_indices]
            
        except Exception as e:
            print(f"Error finding lookalikes for {customer_id}: {e}")
            return []
    
    def generate_report(self):
        """Generate recommendations report"""
        results = []
        
        for i in range(1, 21):
            customer_id = f'C{i:04d}'
            lookalikes = self.find_lookalikes(customer_id)
            
            if lookalikes:
                results.append({
                    'customer_id': customer_id,
                    'lookalikes': [
                        f"{rec['similar_customer']}:{rec['similarity_score']:.3f}"
                        for rec in lookalikes
                    ]
                })
        
        report_df = pd.DataFrame(results)
        report_df.to_csv('reports/Lookalike.csv', index=False)
        return report_df

def main():
    model = LookalikeModel()
    
    if not model.load_data():
        return
    
    print("Creating features...")
    model.create_features()
    
    print("Calculating similarity...")
    model.calculate_similarity()
    
    print("Generating report...")
    report = model.generate_report()
    
    print("\nAnalysis complete!")
    print("Results saved to: reports/Lookalike.csv")
    print("\nSample results:")
    print(report.head())

if __name__ == "__main__":
    main()