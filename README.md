# ECommerce Analysis Project

## Overview
This project performs comprehensive analysis of eCommerce transaction data to derive actionable business insights. Through exploratory data analysis (EDA), predictive modeling, and customer segmentation, we aim to enhance business strategies using customer, product, and transaction data.

## Project Structure
```
ecommerce-analysis
├── data
│   └── raw
│       ├── Customers.csv
│       ├── Products.csv
│       └── Transactions.csv
├── notebooks
│   ├── EDA.ipynb
│   ├── Lookalike.ipynb
│   └── Clustering.ipynb
├── src
│   ├── data_processing
│   │   └── __init__.py
│   ├── models
│   │   └── __init__.py
│   └── visualization
│       └── __init__.py
├── reports
│   ├── Business Insight Report.pdf
│   ├── Customer Segmentation Report.pdf
│   └── Lookalike.csv
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/vladelets-vselennoy/Ecommerce-Analysis
   ```

2. Navigate to the project directory:
   ```bash
   cd ecommerce-analysis
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Analysis Components

### 1. Exploratory Data Analysis (EDA.ipynb)
Comprehensive analysis of the dataset focusing on:
- Customer demographics and behavior patterns
- Product category performance metrics
- Transaction patterns and trends
- Seasonal sales analysis
- Customer Lifetime Value (CLV) calculation
- RFM (Recency, Frequency, Monetary) Analysis

### 2. Lookalike Modeling (Lookalike.py)
Customer similarity analysis including:
- Data preprocessing and feature engineering
- Similarity computation using cosine similarity
- Generation of lookalike customer recommendations
- Model validation and performance metrics

### 3. Customer Segmentation (Clustering.py)
Advanced clustering analysis featuring:
- Data preparation and feature selection
- K-Means clustering implementation
- Cluster evaluation using Davies-Bouldin Index
- Silhouette Score analysis
- Cluster visualization and interpretation

## Deliverables

### Notebooks
- `EDA.ipynb`: Contains all exploratory data analysis code and visualizations
- `Lookalike.ipynb`: Implementation of the lookalike modeling system
- `Clustering.ipynb`: Customer segmentation analysis and results

### Reports
- `Business Insight Report.pdf`: Comprehensive business insights derived from the exploratory analysis
- `Customer Segmentation Report.pdf`: Detailed clustering analysis results including:
  - Optimal cluster count determination
  - Davies-Bouldin Index evaluation
  - Cluster visualization and interpretation
- `Lookalike.csv`: Customer similarity recommendations and matching scores

## Usage Guidelines
1. Begin with the EDA notebook to understand data patterns and distributions
2. Use the Clustering notebook to identify distinct customer segments
3. Leverage the Lookalike notebook to find similar customers for targeted marketing

## License
This project is licensed under the MIT License 


## Contact
For any queries regarding this project, please open an issue in the repository.# Ecommerce-Analysis
