# Customer Revenue Optimizer

A comprehensive data science project that analyzes retail transaction data to identify at-risk customers and simulate the ROI of targeted retention campaigns. This project demonstrates proficiency in data engineering, customer segmentation using RFM analysis, and business intelligence for marketing optimization.

## 🎯 Project Overview

The Customer Revenue Optimizer project enables businesses to:
- **Identify Customer Segments**: Classify customers into actionable segments (Champions, Loyal, At Risk, Lost, etc.) using RFM analysis
- **Predict Churn Risk**: Identify customers at risk of leaving based on purchasing behavior patterns
- **Simulate Campaign ROI**: Calculate the potential return on investment for targeted discount campaigns
- **Optimize Marketing Spend**: Make data-driven decisions about customer retention strategies

## 📊 Business Impact

This analysis helps businesses:
- **Reduce Customer Churn**: Proactively identify and engage at-risk customers before they leave
- **Maximize ROI**: Target marketing campaigns to customer segments with the highest potential return
- **Optimize Budget Allocation**: Understand the cost and benefit of retention campaigns before execution
- **Increase Customer Lifetime Value**: Retain valuable customers through strategic interventions

### Projected ROI Example
For the "At Risk" segment, a targeted 15% discount campaign with a 30% conversion rate can yield significant returns by retaining customers who might otherwise churn.

## 🛠️ Tech Stack

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **DuckDB**: High-performance SQL analytics on DataFrames
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization

## 📁 Project Structure

```
revenue-optimizer/
├── analysis.py              # Main analysis module with RevenueOptimizer class
├── data.csv                 # Transaction data (CSV format)
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── segment_distribution.png # Customer segment distribution visualization
└── revenue_comparison.png  # Revenue comparison chart for At Risk segment
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**:
   - Ensure your transaction data is saved as `data.csv` in the project root
   - Required columns: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`
   - If your file has a different name, update the `data_path` parameter in `analysis.py`

### Running the Analysis

Execute the complete pipeline:

```bash
python analysis.py
```

This will:
1. Load and clean the transaction data
2. Calculate RFM metrics (Recency, Frequency, Monetary) using DuckDB
3. Segment customers based on RFM scores
4. Generate visualizations:
   - Customer segment distribution
   - Revenue comparison for "At Risk" segment
5. Display ROI simulation results

## 📈 RFM Analysis Explained

### RFM Metrics

- **Recency (R)**: Days since last purchase (lower is better)
  - Score 5: Most recent customers
  - Score 1: Least recent customers

- **Frequency (F)**: Total number of transactions
  - Score 5: Most frequent purchasers
  - Score 1: Least frequent purchasers

- **Monetary (M)**: Total revenue from customer
  - Score 5: Highest value customers
  - Score 1: Lowest value customers

### Customer Segments

- **Champions**: High R, F, M scores - Best customers, keep them engaged
- **Loyal**: High F and M, moderate R - Regular customers, maintain loyalty
- **At Risk**: Low R, moderate F and M - Need immediate attention to prevent churn
- **Lost**: Low R, F, M scores - Customers who have churned
- **New Customers**: High R, low F - Recent customers with potential
- **Promising**: Moderate R, low F - Growing customers
- **Need Attention**: Mixed scores - Require personalized approach

## 💡 Usage Examples

### Basic Usage

```python
from analysis import RevenueOptimizer

# Initialize optimizer
optimizer = RevenueOptimizer(data_path='data.csv')

# Run complete pipeline
optimizer.load_data()
optimizer.clean_data()
optimizer.calculate_rfm_metrics()
optimizer.create_segments()

# Generate visualizations
optimizer.plot_segment_distribution('segment_distribution.png')
optimizer.plot_revenue_comparison('At Risk', discount_pct=15.0, 
                                 conversion_rate=0.3,
                                 save_path='revenue_comparison.png')

# Close connection
optimizer.close()
```

### ROI Simulation

```python
# Simulate ROI for a retention campaign
roi_results = optimizer.simulate_discount_roi(
    segment_name='At Risk',
    discount_pct=15.0,      # 15% discount
    conversion_rate=0.3      # 30% expected conversion
)

print(f"Net ROI: ${roi_results['net_roi']:,.2f}")
print(f"ROI Percentage: {roi_results['roi_percentage']:.2f}%")
```

## 📊 Key Features

### Data Engineering
- Robust data cleaning (handles missing values, negative quantities/prices)
- Efficient SQL-based RFM calculation using DuckDB
- Date handling and transaction aggregation

### Analytics & Modeling
- Quintile-based RFM scoring (1-5 scale)
- Multi-dimensional customer segmentation
- ROI simulation with configurable parameters

### Visualization
- High-quality, publication-ready charts
- Segment distribution analysis
- Revenue comparison visualizations
- Professional styling with matplotlib/seaborn

### Code Quality
- Object-oriented design with comprehensive class structure
- Google-style docstrings for all methods
- Error handling and validation
- Modular, maintainable code

## 🔧 Customization

### Adjusting Segment Definitions

Modify the `assign_segment` function in the `create_segments` method to customize segment thresholds based on your business needs.

### Changing Campaign Parameters

Update the discount percentage and conversion rate in the `simulate_discount_roi` calls to test different campaign scenarios.

### Reference Date for Recency

Specify a custom reference date when calling `calculate_rfm_metrics()`:
```python
optimizer.calculate_rfm_metrics(reference_date='2011-12-09')
```

## 📝 Output Files

- **segment_distribution.png**: Bar chart showing the distribution of customers across segments
- **revenue_comparison.png**: Comparison of current vs. potential revenue for the "At Risk" segment

## 🤝 Contributing

This is a portfolio project. For questions or suggestions, please open an issue.

## 📄 License

This project is created for educational and portfolio purposes.

## 👤 Author

Developed as a demonstration of data science and MLOps engineering skills for internship applications.

---

**Note**: This project uses synthetic or publicly available retail transaction data. Ensure you have appropriate permissions and comply with data privacy regulations when working with customer data.
