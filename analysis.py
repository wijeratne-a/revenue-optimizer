"""
Customer Revenue Optimizer - Data Analysis and Modeling Module

This module provides a comprehensive solution for analyzing retail transaction data,
calculating RFM metrics, segmenting customers, and simulating retention campaign ROI.
"""

import pandas as pd
import duckdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Tuple, Optional, List

# Try to import rich library, fallback to standard output if not available
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Create dummy classes for fallback
    class Console:
        def print(self, *args, **kwargs): print(*args, **kwargs)
        def log(self, *args, **kwargs): print(*args, **kwargs)
    class Progress:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def add_task(self, *args, **kwargs): return 0
        def update(self, *args, **kwargs): pass
    class Table:
        def __init__(self, *args, **kwargs): pass
        def add_column(self, *args, **kwargs): pass
        def add_row(self, *args, **kwargs): pass
    class Panel:
        def __init__(self, *args, **kwargs): self.renderable = args[0] if args else ""
        def __str__(self): return str(self.renderable)


class RevenueOptimizer:
    """
    A class to optimize customer revenue through RFM analysis and retention campaigns.

    This class handles the complete pipeline from data loading and cleaning to
    customer segmentation and ROI simulation for targeted marketing campaigns.

    Attributes:
        data (pd.DataFrame): The cleaned transaction dataset.
        rfm_data (pd.DataFrame): Customer-level RFM metrics and segments.
        connection (duckdb.DuckDBPyConnection): DuckDB database connection.
        scenario_results (pd.DataFrame): Results from scenario analysis.
        console (Console): Rich console instance for formatted output.
    """

    def __init__(self, data_path: str = 'data.csv'):
        """
        Initialize the RevenueOptimizer with data file path.

        Args:
            data_path (str): Path to the CSV file containing transaction data.
                            Default is 'data.csv'.
        """
        self.data_path = data_path
        self.data = None
        self.rfm_data = None
        self.scenario_results = None
        self.connection = None
        self.console = Console() if RICH_AVAILABLE else Console()

        # Initialize DuckDB connection with error handling
        try:
            self.connection = duckdb.connect()
        except Exception as e:
            raise ConnectionError(f"Failed to initialize DuckDB connection: {str(e)}")

    def load_data(self, encoding: str = 'ISO-8859-1') -> pd.DataFrame:
        """
        Load transaction data from CSV file using Pandas.

        Args:
            encoding (str): Character encoding of the CSV file. Default is 'ISO-8859-1'
                           which handles most non-UTF-8 CSV files. Common options:
                           'ISO-8859-1', 'latin-1', 'cp1252', 'utf-8'.

        Returns:
            pd.DataFrame: Raw transaction data loaded from CSV.

        Raises:
            FileNotFoundError: If the deata file does not exist.
            UnicodeDecodeError: If the specified encoding fails.
            IOError: If file reading fails.
        """
        try:
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console
                ) as progress:
                    task = progress.add_task(
                        f"[cyan]Loading data from {self.data_path}...", total=None
                    )
                    self.data = pd.read_csv(self.data_path, encoding=encoding)
                    progress.update(task, completed=True)
            else:
                print(f"Loading data from {self.data_path}...")
                self.data = pd.read_csv(self.data_path, encoding=encoding)

            self.console.log(
                f"[green]Data loaded successfully:[/green] "
                f"{len(self.data)} rows, {len(self.data.columns)} columns"
            )
            return self.data

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Data file '{self.data_path}' not found. Please ensure the file exists."
            )
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                f"Failed to decode file with encoding '{encoding}'. Try a different encoding."
            )
        except IOError as e:
            raise IOError(f"Error reading file '{self.data_path}': {str(e)}")

    def validate_data(self) -> Dict[str, int]:
        """
        Validate and flag dirty data in the transaction dataset.

        Identifies and removes:
        - Rows with missing CustomerID
        - Rows with Quantity <= 0
        - Rows with UnitPrice <= 0

        Returns:
            dict: Dictionary containing cleaning statistics:
                - 'initial_rows': Number of rows before cleaning
                - 'missing_customer_id': Number of rows with missing CustomerID
                - 'invalid_quantity': Number of rows with Quantity <= 0
                - 'invalid_price': Number of rows with UnitPrice <= 0
                - 'final_rows': Number of rows after cleaning
                - 'total_removed': Total number of rows removed
        """
        if self.data is None:
            raise ValueError("Data must be loaded before validation. Call load_data() first.")

        initial_rows = len(self.data)
        stats = {
            'initial_rows': initial_rows,
            'missing_customer_id': 0,
            'invalid_quantity': 0,
            'invalid_price': 0,
            'final_rows': 0,
            'total_removed': 0
        }

        # Flag and count missing CustomerIDs
        missing_customer_id = self.data['CustomerID'].isna()
        stats['missing_customer_id'] = missing_customer_id.sum()

        # Flag and count invalid quantities
        invalid_quantity = self.data['Quantity'] <= 0
        stats['invalid_quantity'] = invalid_quantity.sum()

        # Flag and count invalid prices
        invalid_price = self.data['UnitPrice'] <= 0
        stats['invalid_price'] = invalid_price.sum()

        # Remove dirty data
        dirty_mask = missing_customer_id | invalid_quantity | invalid_price
        self.data = self.data[~dirty_mask].copy()

        # Convert CustomerID to integer (handle any string representations)
        if len(self.data) > 0:
            self.data['CustomerID'] = self.data['CustomerID'].astype(str).str.replace(
                '.0', '', regex=False
            )
            self.data = self.data[self.data['CustomerID'].str.isdigit()].copy()
            self.data['CustomerID'] = self.data['CustomerID'].astype(int)

        stats['final_rows'] = len(self.data)
        stats['total_removed'] = stats['initial_rows'] - stats['final_rows']

        # Report cleaning statistics
        if RICH_AVAILABLE:
            table = Table(title="Data Validation Results", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Count", style="green", justify="right")

            table.add_row("Initial Rows", f"{stats['initial_rows']:,}")
            table.add_row("Missing CustomerID", f"{stats['missing_customer_id']:,}", style="red")
            table.add_row("Invalid Quantity (<= 0)", f"{stats['invalid_quantity']:,}", style="red")
            table.add_row("Invalid Price (<= 0)", f"{stats['invalid_price']:,}", style="red")
            table.add_row("Total Removed", f"{stats['total_removed']:,}", style="yellow")
            table.add_row("Final Rows", f"{stats['final_rows']:,}", style="green")

            self.console.print(table)
        else:
            print("\n" + "=" * 60)
            print("DATA VALIDATION RESULTS")
            print("=" * 60)
            print(f"Initial Rows: {stats['initial_rows']:,}")
            print(f"Missing CustomerID: {stats['missing_customer_id']:,}")
            print(f"Invalid Quantity (<= 0): {stats['invalid_quantity']:,}")
            print(f"Invalid Price (<= 0): {stats['invalid_price']:,}")
            print(f"Total Removed: {stats['total_removed']:,}")
            print(f"Final Rows: {stats['final_rows']:,}")
            print("=" * 60 + "\n")

        return stats

    def clean_data(self) -> pd.DataFrame:
        """
        Clean the transaction data by performing additional operations.

        Performs the following cleaning operations:
        - Convert InvoiceDate to datetime format
        - Calculate TotalPrice for each transaction

        Note: Data validation (removing dirty data) should be done via validate_data().

        Returns:
            pd.DataFrame: Cleaned transaction data.
        """
        if self.data is None:
            raise ValueError("Data must be loaded before cleaning. Call load_data() first.")

        # Convert InvoiceDate to datetime
        self.data['InvoiceDate'] = pd.to_datetime(self.data['InvoiceDate'], errors='coerce')
        self.data = self.data.dropna(subset=['InvoiceDate'])

        # Calculate TotalPrice for each transaction
        self.data['TotalPrice'] = self.data['Quantity'] * self.data['UnitPrice']

        self.console.log(
            f"[green]Data cleaning completed:[/green] "
            f"{len(self.data)} rows remaining after date validation"
        )

        return self.data

    def calculate_rfm_metrics(self, reference_date: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics using DuckDB SQL.

        RFM Metrics:
        - Recency: Days since last purchase (lower is better)
        - Frequency: Total number of transactions
        - Monetary: Total revenue from customer

        Args:
            reference_date (str, optional): Reference date for recency calculation
                                           in 'YYYY-MM-DD' format. If None, uses
                                           the maximum date in the dataset.

        Returns:
            pd.DataFrame: DataFrame with CustomerID, Recency, Frequency, and Monetary columns.

        Raises:
            ValueError: If data is not loaded or cleaned.
            ConnectionError: If DuckDB connection fails.
        """
        if self.data is None:
            raise ValueError("Data must be loaded and cleaned before RFM calculation.")

        if self.connection is None:
            raise ConnectionError("DuckDB connection is not available.")

        try:
            # Determine reference date
            if reference_date is None:
                max_date = self.data['InvoiceDate'].max()
                reference_date = max_date
            else:
                max_date = pd.to_datetime(reference_date)

            # Create a temporary view in DuckDB with date as string
            data_with_dates = self.data.copy()
            data_with_dates['InvoiceDate_str'] = data_with_dates['InvoiceDate'].dt.strftime(
                '%Y-%m-%d'
            )

            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console
                ) as progress:
                    task = progress.add_task(
                        "[cyan]Calculating RFM metrics with DuckDB...", total=None
                    )
                    self.connection.register('transactions', data_with_dates)

                    # SQL query to calculate RFM metrics
                    rfm_query = """
                    SELECT
                        CustomerID,
                        COUNT(DISTINCT InvoiceNo) AS Frequency,
                        SUM(TotalPrice) AS Monetary,
                        MAX(InvoiceDate_str) AS LastPurchaseDate
                    FROM transactions
                    GROUP BY CustomerID
                    ORDER BY CustomerID
                    """

                    self.rfm_data = self.connection.execute(rfm_query).df()
                    progress.update(task, completed=True)
            else:
                print("Calculating RFM metrics with DuckDB...")
                self.connection.register('transactions', data_with_dates)

                rfm_query = """
                SELECT
                    CustomerID,
                    COUNT(DISTINCT InvoiceNo) AS Frequency,
                    SUM(TotalPrice) AS Monetary,
                    MAX(InvoiceDate_str) AS LastPurchaseDate
                FROM transactions
                GROUP BY CustomerID
                ORDER BY CustomerID
                """

                self.rfm_data = self.connection.execute(rfm_query).df()

            # Calculate Recency in Python (more reliable)
            self.rfm_data['LastPurchaseDate'] = pd.to_datetime(
                self.rfm_data['LastPurchaseDate']
            )
            self.rfm_data['Recency'] = (
                max_date - self.rfm_data['LastPurchaseDate']
            ).dt.days
            self.rfm_data = self.rfm_data.drop('LastPurchaseDate', axis=1)

            # Reorder columns
            self.rfm_data = self.rfm_data[
                ['CustomerID', 'Recency', 'Frequency', 'Monetary']
            ]

            self.console.log(
                f"[green]RFM metrics calculated for {len(self.rfm_data)} customers[/green]"
            )
            return self.rfm_data

        except Exception as e:
            raise ConnectionError(f"Error during RFM calculation: {str(e)}")

    def create_segments(self) -> pd.DataFrame:
        """
        Create customer segments based on RFM scores using pandas qcut.

        Scores R, F, M are created using quintiles (1-5) where:
        - R: 5 = most recent, 1 = least recent
        - F: 5 = most frequent, 1 = least frequent
        - M: 5 = highest monetary, 1 = lowest monetary

        Segments are assigned based on RFM score combinations:
        - Champions: R=5, F=5, M=5
        - Loyal: High F and M scores
        - At Risk: Low R, moderate F and M
        - Lost: Low R, F, M scores

        Returns:
            pd.DataFrame: RFM data with R, F, M scores and Segment column added.
        """
        if self.rfm_data is None:
            raise ValueError("RFM metrics must be calculated before segmentation.")

        # Create R, F, M scores using qcut (quintiles)
        # For Recency: lower is better, so we reverse it
        self.rfm_data['R'] = pd.qcut(
            self.rfm_data['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop'
        )
        self.rfm_data['F'] = pd.qcut(
            self.rfm_data['Frequency'].rank(method='first'),
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        )
        self.rfm_data['M'] = pd.qcut(
            self.rfm_data['Monetary'].rank(method='first'),
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        )

        # Convert to numeric
        self.rfm_data['R'] = pd.to_numeric(self.rfm_data['R'])
        self.rfm_data['F'] = pd.to_numeric(self.rfm_data['F'])
        self.rfm_data['M'] = pd.to_numeric(self.rfm_data['M'])

        # Calculate RFM score (average)
        self.rfm_data['RFM_Score'] = (
            self.rfm_data['R'] + self.rfm_data['F'] + self.rfm_data['M']
        ) / 3

        # Assign segments based on RFM scores
        def assign_segment(row):
            r, f, m = row['R'], row['F'], row['M']

            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 4 and m >= 4:
                return 'Loyal'
            elif r >= 4 and f <= 2 and m <= 3:
                return 'New Customers'
            elif r <= 2 and f >= 3 and m >= 3:
                return 'At Risk'
            elif r <= 2 and f <= 2 and m <= 2:
                return 'Lost'
            elif r >= 3 and f <= 2:
                return 'Promising'
            else:
                return 'Need Attention'

        self.rfm_data['Segment'] = self.rfm_data.apply(assign_segment, axis=1)

        # Display segment distribution with Rich table
        segment_counts = self.rfm_data['Segment'].value_counts()

        if RICH_AVAILABLE:
            table = Table(
                title="Customer Segment Distribution", show_header=True, header_style="bold magenta"
            )
            table.add_column("Segment", style="cyan", no_wrap=True)
            table.add_column("Count", style="green", justify="right")
            table.add_column("Percentage", style="yellow", justify="right")

            total_customers = len(self.rfm_data)
            for segment, count in segment_counts.items():
                percentage = (count / total_customers) * 100
                table.add_row(segment, f"{count:,}", f"{percentage:.2f}%")

            self.console.print(table)
        else:
            print("\nSegmentation completed. Segment distribution:")
            print(segment_counts)

        return self.rfm_data

    def simulate_discount_roi(
        self, segment_name: str, discount_pct: float, conversion_rate: float
    ) -> Dict[str, float]:
        """
        Simulate the ROI of a discount campaign for a specific customer segment.

        Calculates:
        - Current Revenue: Total revenue from segment
        - Campaign Cost: Discount cost assuming all customers convert
        - Revenue Recovered: Additional revenue from retained customers
        - Net ROI: Revenue recovered minus campaign cost
        - ROI Percentage: (Net ROI / Campaign Cost) * 100

        Args:
            segment_name (str): Name of the customer segment to target.
            discount_pct (float): Discount percentage (0-100) to offer.
            conversion_rate (float): Expected conversion rate (0-1) of customers
                                   who will use the discount.

        Returns:
            dict: Dictionary containing:
                - 'current_revenue': Total current revenue from segment
                - 'campaign_cost': Total discount cost
                - 'revenue_recovered': Additional revenue from retained customers
                - 'net_roi': Net return on investment
                - 'roi_percentage': ROI as a percentage
                - 'num_customers': Number of customers in segment
        """
        if self.rfm_data is None:
            raise ValueError("Segments must be created before ROI simulation.")

        # Filter segment
        segment_data = self.rfm_data[self.rfm_data['Segment'] == segment_name]

        if len(segment_data) == 0:
            raise ValueError(f"Segment '{segment_name}' not found in data.")

        # Current revenue from segment
        current_revenue = segment_data['Monetary'].sum()
        num_customers = len(segment_data)

        # Calculate average transaction value
        avg_monetary = segment_data['Monetary'].mean()

        # Campaign cost: assuming customers make purchases with discount
        # We assume retained customers make one additional purchase at discounted rate
        expected_additional_purchases = num_customers * conversion_rate
        discount_per_purchase = avg_monetary * (discount_pct / 100)
        campaign_cost = expected_additional_purchases * discount_per_purchase

        # Revenue recovered: additional revenue from retained customers
        # Assuming retained customers continue purchasing at normal rates
        revenue_recovered = expected_additional_purchases * avg_monetary * (
            1 - discount_pct / 100
        )

        # Net ROI
        net_roi = revenue_recovered - campaign_cost

        # ROI percentage
        roi_percentage = (net_roi / campaign_cost * 100) if campaign_cost > 0 else 0

        results = {
            'current_revenue': current_revenue,
            'campaign_cost': campaign_cost,
            'revenue_recovered': revenue_recovered,
            'net_roi': net_roi,
            'roi_percentage': roi_percentage,
            'num_customers': num_customers
        }

        return results

    def run_scenario_analysis(
        self,
        segment_name: str = 'At Risk',
        discount_rates: List[float] = [0.05, 0.10, 0.20],
        conversion_rate: float = 0.15
    ) -> pd.DataFrame:
        """
        Run scenario analysis for multiple discount rates on a customer segment.

        For each discount rate, calculates:
        - Current Revenue: Total revenue currently held by the segment
        - Recovered Revenue: Projected revenue recovered (assuming conversion rate)
        - Net ROI: Recovered Revenue minus campaign cost

        Args:
            segment_name (str): Name of the customer segment to analyze.
                               Default is 'At Risk'.
            discount_rates (List[float]): List of discount percentages (0-1).
                                         Default is [0.05, 0.10, 0.20] (5%, 10%, 20%).
            conversion_rate (float): Expected conversion rate (0-1). Default is 0.15 (15%).

        Returns:
            pd.DataFrame: DataFrame containing scenario analysis results with columns:
                - Discount_Rate_Pct: Discount rate as percentage
                - Current_Revenue: Current revenue from segment
                - Recovered_Revenue: Projected recovered revenue
                - Campaign_Cost: Cost of the discount campaign
                - Net_ROI: Net return on investment
                - ROI_Percentage: ROI as percentage
                - Num_Customers: Number of customers in segment
        """
        if self.rfm_data is None:
            raise ValueError("Segments must be created before scenario analysis.")

        # Filter segment
        segment_data = self.rfm_data[self.rfm_data['Segment'] == segment_name]

        if len(segment_data) == 0:
            raise ValueError(f"Segment '{segment_name}' not found in data.")

        current_revenue = segment_data['Monetary'].sum()
        num_customers = len(segment_data)
        avg_monetary = segment_data['Monetary'].mean()

        scenario_results = []

        for discount_rate in discount_rates:
            discount_pct = discount_rate * 100  # Convert to percentage

            # Calculate metrics
            expected_additional_purchases = num_customers * conversion_rate
            discount_per_purchase = avg_monetary * discount_rate
            campaign_cost = expected_additional_purchases * discount_per_purchase
            recovered_revenue = expected_additional_purchases * avg_monetary * (
                1 - discount_rate
            )
            net_roi = recovered_revenue - campaign_cost
            roi_percentage = (net_roi / campaign_cost * 100) if campaign_cost > 0 else 0

            scenario_results.append({
                'Discount_Rate_Pct': discount_pct,
                'Current_Revenue': current_revenue,
                'Recovered_Revenue': recovered_revenue,
                'Campaign_Cost': campaign_cost,
                'Net_ROI': net_roi,
                'ROI_Percentage': roi_percentage,
                'Num_Customers': num_customers
            })

        self.scenario_results = pd.DataFrame(scenario_results)

        return self.scenario_results

    def plot_segment_distribution(self, save_path: str = 'segment_distribution.png') -> None:
        """
        Create a high-quality visualization of customer segment distribution.

        Args:
            save_path (str): Path to save the plot. Default is 'segment_distribution.png'.

        Raises:
            ValueError: If segments are not created.
            IOError: If file saving fails.
        """
        if self.rfm_data is None:
            raise ValueError("Segments must be created before plotting.")

        try:
            plt.figure(figsize=(12, 6))

            segment_counts = self.rfm_data['Segment'].value_counts()

            # Create bar plot
            ax = sns.barplot(
                x=segment_counts.index,
                y=segment_counts.values,
                hue=segment_counts.index,
                palette='viridis',
                legend=False
            )

            plt.title('Customer Segment Distribution', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Customer Segment', fontsize=12)
            plt.ylabel('Number of Customers', fontsize=12)
            plt.xticks(rotation=45, ha='right')

            # Add value labels on bars
            for i, v in enumerate(segment_counts.values):
                ax.text(
                    i,
                    v + max(segment_counts.values) * 0.01,
                    str(v),
                    ha='center',
                    va='bottom',
                    fontweight='bold'
                )

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.console.log(f"[green]Segment distribution plot saved to {save_path}[/green]")

        except Exception as e:
            raise IOError(f"Error saving plot to {save_path}: {str(e)}")

    def plot_revenue_comparison(
        self,
        segment_name: str = 'At Risk',
        discount_pct: float = 15.0,
        conversion_rate: float = 0.3,
        save_path: str = 'revenue_comparison.png'
    ) -> None:
        """
        Create a bar chart comparing current revenue vs potential revenue for a segment.

        Args:
            segment_name (str): Segment to analyze. Default is 'At Risk'.
            discount_pct (float): Discount percentage for simulation. Default is 15.0.
            conversion_rate (float): Conversion rate for simulation. Default is 0.3.
            save_path (str): Path to save the plot. Default is 'revenue_comparison.png'.

        Raises:
            ValueError: If segments are not created.
            IOError: If file saving fails.
        """
        if self.rfm_data is None:
            raise ValueError("Segments must be created before plotting.")

        try:
            # Get ROI simulation results
            roi_results = self.simulate_discount_roi(segment_name, discount_pct, conversion_rate)

            # Prepare data for plotting
            categories = ['Current Revenue', 'Potential Revenue\n(After Campaign)']
            values = [
                roi_results['current_revenue'],
                roi_results['current_revenue'] + roi_results['net_roi']
            ]

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            bars = ax.bar(
                categories,
                values,
                color=['#3498db', '#2ecc71'],
                alpha=0.8,
                edgecolor='black',
                linewidth=1.5
            )

            # Customize plot
            ax.set_title(
                f'Revenue Comparison: {segment_name} Segment\n'
                f'(Discount: {discount_pct}%, Conversion: {conversion_rate*100}%)',
                fontsize=14,
                fontweight='bold',
                pad=20
            )
            ax.set_ylabel('Revenue ($)', fontsize=12)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'${height:,.0f}',
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    fontsize=11
                )

            # Add ROI annotation
            roi_text = f"Net ROI: ${roi_results['net_roi']:,.0f} ({roi_results['roi_percentage']:.1f}%)"
            ax.text(
                0.5,
                0.95,
                roi_text,
                transform=ax.transAxes,
                ha='center',
                va='top',
                fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.console.log(f"[green]Revenue comparison plot saved to {save_path}[/green]")

        except Exception as e:
            raise IOError(f"Error saving plot to {save_path}: {str(e)}")

    def save_scenario_analysis(self, output_path: str = 'roi_report.csv') -> None:
        """
        Save scenario analysis results to CSV file.

        Args:
            output_path (str): Path to save the CSV file. Default is 'roi_report.csv'.

        Raises:
            ValueError: If scenario analysis has not been run.
            IOError: If file writing fails.
        """
        if self.scenario_results is None:
            raise ValueError(
                "Scenario analysis must be run before saving. Call run_scenario_analysis() first."
            )

        try:
            self.scenario_results.to_csv(output_path, index=False)
            self.console.log(f"[green]Scenario analysis saved to {output_path}[/green]")
        except Exception as e:
            raise IOError(f"Error saving scenario analysis to {output_path}: {str(e)}")

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self.connection:
            try:
                self.connection.close()
                self.console.log("[green]DuckDB connection closed[/green]")
            except Exception as e:
                self.console.log(f"[yellow]Warning: Error closing connection: {str(e)}[/yellow]")


if __name__ == "__main__":
    """
    Main execution block to run the complete revenue optimization pipeline.
    """
    optimizer = RevenueOptimizer(data_path='data.csv')

    try:
        console = Console() if RICH_AVAILABLE else Console()

        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold cyan]Customer Revenue Optimizer[/bold cyan]\n"
                "[yellow]Production-Grade Analysis Pipeline[/yellow]",
                border_style="green"
            ))

        # Step 1: Load data
        console.print("\n[bold cyan]STEP 1: Loading Data[/bold cyan]")
        optimizer.load_data(encoding='ISO-8859-1')

        # Step 2: Validate data
        console.print("\n[bold cyan]STEP 2: Validating Data[/bold cyan]")
        optimizer.validate_data()

        # Step 3: Clean data
        console.print("\n[bold cyan]STEP 3: Cleaning Data[/bold cyan]")
        optimizer.clean_data()

        # Step 4: Calculate RFM metrics
        console.print("\n[bold cyan]STEP 4: Calculating RFM Metrics[/bold cyan]")
        optimizer.calculate_rfm_metrics()

        # Step 5: Create segments
        console.print("\n[bold cyan]STEP 5: Creating Customer Segments[/bold cyan]")
        optimizer.create_segments()

        # Step 6: Run scenario analysis
        console.print("\n[bold cyan]STEP 6: Running Scenario Analysis[/bold cyan]")
        scenario_results = optimizer.run_scenario_analysis(
            segment_name='At Risk',
            discount_rates=[0.05, 0.10, 0.20],
            conversion_rate=0.15
        )

        # Display scenario results
        if RICH_AVAILABLE:
            table = Table(
                title="Scenario Analysis Results - At Risk Segment",
                show_header=True,
                header_style="bold magenta"
            )
            table.add_column("Discount %", style="cyan", justify="right")
            table.add_column("Current Revenue", style="green", justify="right")
            table.add_column("Recovered Revenue", style="green", justify="right")
            table.add_column("Campaign Cost", style="yellow", justify="right")
            table.add_column("Net ROI", style="green", justify="right")
            table.add_column("ROI %", style="green", justify="right")

            for _, row in scenario_results.iterrows():
                table.add_row(
                    f"{row['Discount_Rate_Pct']:.1f}%",
                    f"${row['Current_Revenue']:,.0f}",
                    f"${row['Recovered_Revenue']:,.0f}",
                    f"${row['Campaign_Cost']:,.0f}",
                    f"${row['Net_ROI']:,.0f}",
                    f"{row['ROI_Percentage']:.1f}%"
                )

            console.print(table)

            # Executive recommendation
            best_scenario = scenario_results.loc[scenario_results['Net_ROI'].idxmax()]
            recommendation = (
                f"[bold green]Recommended Discount: {best_scenario['Discount_Rate_Pct']:.1f}%[/bold green]\n\n"
                f"Expected Net ROI: [green]${best_scenario['Net_ROI']:,.0f}[/green]\n"
                f"ROI Percentage: [green]{best_scenario['ROI_Percentage']:.1f}%[/green]\n"
                f"Campaign Cost: [yellow]${best_scenario['Campaign_Cost']:,.0f}[/yellow]\n"
                f"Recovered Revenue: [green]${best_scenario['Recovered_Revenue']:,.0f}[/green]\n\n"
                f"Targeting [cyan]{int(best_scenario['Num_Customers']):,} customers[/cyan] "
                f"in the 'At Risk' segment with a [cyan]15% conversion rate[/cyan]."
            )

            console.print(Panel(
                recommendation,
                title="[bold]Executive Recommendation[/bold]",
                border_style="green"
            ))
        else:
            print("\n" + "=" * 60)
            print("SCENARIO ANALYSIS RESULTS - At Risk Segment")
            print("=" * 60)
            print(scenario_results.to_string(index=False))
            print("=" * 60)

            best_scenario = scenario_results.loc[scenario_results['Net_ROI'].idxmax()]
            print(f"\nRecommended Discount: {best_scenario['Discount_Rate_Pct']:.1f}%")
            print(f"Expected Net ROI: ${best_scenario['Net_ROI']:,.0f}")
            print(f"ROI Percentage: {best_scenario['ROI_Percentage']:.1f}%")

        # Step 7: Save scenario analysis
        console.print("\n[bold cyan]STEP 7: Saving Scenario Analysis[/bold cyan]")
        optimizer.save_scenario_analysis('roi_report.csv')

        # Step 8: Generate visualizations
        console.print("\n[bold cyan]STEP 8: Generating Visualizations[/bold cyan]")
        optimizer.plot_segment_distribution('segment_distribution.png')
        optimizer.plot_revenue_comparison(
            'At Risk', discount_pct=15.0, conversion_rate=0.3, save_path='revenue_comparison.png'
        )

        # Completion message
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold green]Pipeline completed successfully![/bold green]\n\n"
                "[cyan]Output files:[/cyan]\n"
                "  • roi_report.csv\n"
                "  • segment_distribution.png\n"
                "  • revenue_comparison.png",
                border_style="green"
            ))
        else:
            console.print("\n" + "=" * 60)
            console.print("Pipeline completed successfully!")
            console.print("=" * 60)
            console.print("\nOutput files:")
            console.print("  • roi_report.csv")
            console.print("  • segment_distribution.png")
            console.print("  • revenue_comparison.png")

    except Exception as e:
        error_msg = f"Error occurred: {str(e)}"
        if RICH_AVAILABLE:
            console.print(Panel.fit(f"[bold red]{error_msg}[/bold red]", border_style="red"))
        else:
            console.print(f"\n{error_msg}")
        raise
    finally:
        optimizer.close()
