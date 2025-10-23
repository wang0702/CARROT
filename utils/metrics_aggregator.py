import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from .logger import get_carrot_logger

logger = get_carrot_logger(__name__)


class MetricsAggregator:
    """Utility for aggregating and analyzing evaluation metrics."""
    
    def __init__(self):
        self.metrics_data = []
        self.aggregated_results = {}
    
    def add_result(self, result: Dict[str, Any]):
        """Add a single evaluation result."""
        self.metrics_data.append(result.copy())
        logger.debug(f"Added result with keys: {list(result.keys())}")
    
    def add_batch_results(self, results: List[Dict[str, Any]]):
        """Add multiple evaluation results."""
        for result in results:
            self.add_result(result)
        logger.info(f"Added {len(results)} results to metrics aggregator")
    
    def compute_statistics(self, metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Compute descriptive statistics for specified metrics."""
        if not self.metrics_data:
            logger.warning("No metrics data available for statistics")
            return {}
        
        df = pd.DataFrame(self.metrics_data)
        
        if metrics is None:
            # Auto-detect numeric metrics
            metrics = df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats = {}
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    stats[metric] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median()),
                        'count': int(len(values))
                    }
                else:
                    logger.warning(f"No valid values found for metric: {metric}")
        
        self.aggregated_results['statistics'] = stats
        return stats
    
    def compute_method_comparison(self, method_column: str = 'method',
                                score_columns: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Compare performance across different methods."""
        if not self.metrics_data:
            logger.warning("No metrics data available for method comparison")
            return {}
        
        df = pd.DataFrame(self.metrics_data)
        
        if method_column not in df.columns:
            logger.error(f"Method column '{method_column}' not found in data")
            return {}
        
        if score_columns is None:
            score_columns = ['f1', 'rouge_1', 'rouge_2', 'rouge_l', 'bleu_1']
            score_columns = [col for col in score_columns if col in df.columns]
        
        comparison = {}
        for method in df[method_column].unique():
            method_data = df[df[method_column] == method]
            comparison[method] = {}
            
            for score_col in score_columns:
                if score_col in method_data.columns:
                    values = method_data[score_col].dropna()
                    if len(values) > 0:
                        comparison[method][score_col] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'count': int(len(values))
                        }
        
        self.aggregated_results['method_comparison'] = comparison
        return comparison
    
    def compute_correlation_analysis(self, metrics: List[str] = None) -> Dict[str, float]:
        """Compute correlations between metrics."""
        if not self.metrics_data:
            logger.warning("No metrics data available for correlation analysis")
            return {}
        
        df = pd.DataFrame(self.metrics_data)
        
        if metrics is None:
            metrics = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter metrics that exist in the data
        available_metrics = [m for m in metrics if m in df.columns]
        
        if len(available_metrics) < 2:
            logger.warning("Need at least 2 numeric metrics for correlation analysis")
            return {}
        
        correlation_matrix = df[available_metrics].corr()
        
        # Convert to dictionary format
        correlations = {}
        for i, metric1 in enumerate(available_metrics):
            for j, metric2 in enumerate(available_metrics):
                if i < j:  # Only upper triangle to avoid duplicates
                    corr_key = f"{metric1}_vs_{metric2}"
                    correlations[corr_key] = float(correlation_matrix.loc[metric1, metric2])
        
        self.aggregated_results['correlations'] = correlations
        return correlations
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        if not self.metrics_data:
            return "No metrics data available for summary report."
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("CARROT EVALUATION METRICS SUMMARY REPORT")
        report_lines.append("=" * 60)
        
        # Basic statistics
        stats = self.compute_statistics()
        if stats:
            report_lines.append("\n📊 PERFORMANCE STATISTICS:")
            report_lines.append("-" * 30)
            for metric, values in stats.items():
                report_lines.append(f"{metric.upper():15} | Mean: {values['mean']:.4f} ± {values['std']:.4f}")
                report_lines.append(f"               | Range: [{values['min']:.4f}, {values['max']:.4f}] | Count: {values['count']}")
        
        # Method comparison (if available)
        if 'method' in pd.DataFrame(self.metrics_data).columns:
            comparison = self.compute_method_comparison()
            if comparison:
                report_lines.append("\n🔍 METHOD COMPARISON:")
                report_lines.append("-" * 30)
                for method, metrics in comparison.items():
                    report_lines.append(f"\n{method.upper()}:")
                    for metric, values in metrics.items():
                        report_lines.append(f"  {metric:12} | {values['mean']:.4f} ± {values['std']:.4f}")
        
        # Correlations
        correlations = self.compute_correlation_analysis()
        if correlations:
            report_lines.append("\n📈 METRIC CORRELATIONS:")
            report_lines.append("-" * 30)
            sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            for corr_pair, value in sorted_corrs[:5]:  # Top 5 correlations
                report_lines.append(f"{corr_pair:25} | {value:6.3f}")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)
    
    def save_results(self, filepath: str, format: str = 'json'):
        """Save aggregated results to file."""
        try:
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump({
                        'raw_data': self.metrics_data,
                        'aggregated_results': self.aggregated_results
                    }, f, indent=2)
            elif format.lower() == 'csv':
                df = pd.DataFrame(self.metrics_data)
                df.to_csv(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def create_visualization(self, metrics: List[str], 
                           save_path: Optional[str] = None) -> None:
        """Create visualizations for specified metrics."""
        if not self.metrics_data:
            logger.warning("No data available for visualization")
            return
        
        try:
            df = pd.DataFrame(self.metrics_data)
            available_metrics = [m for m in metrics if m in df.columns]
            
            if not available_metrics:
                logger.warning("No valid metrics found for visualization")
                return
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('CARROT Evaluation Metrics', fontsize=16)
            
            # Distribution plots
            for i, metric in enumerate(available_metrics[:4]):
                row, col = i // 2, i % 2
                ax = axes[row, col]
                
                values = df[metric].dropna()
                if len(values) > 0:
                    ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
                    ax.set_title(f'{metric.title()} Distribution')
                    ax.set_xlabel(metric)
                    ax.set_ylabel('Frequency')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")


def create_metrics_aggregator(results: List[Dict[str, Any]] = None) -> MetricsAggregator:
    """Convenience function to create and populate metrics aggregator."""
    aggregator = MetricsAggregator()
    if results:
        aggregator.add_batch_results(results)
    return aggregator