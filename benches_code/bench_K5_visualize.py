import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up professional styling
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

class ResultsVisualizer:
    def __init__(self, normal_csv, dynamic_csv):
        self.normal_csv = normal_csv
        self.dynamic_csv = dynamic_csv
        self.df_normal = None
        self.df_dynamic = None
        self.df_combined = None
        
    def load_data(self):
        """Load the evaluation results"""
        print("📊 Loading evaluation results...")
        
        self.df_normal = pd.read_csv(self.normal_csv)
        self.df_dynamic = pd.read_csv(self.dynamic_csv)
        
        # Add model identifiers
        self.df_normal['model'] = 'Gemma-2-2b-it (Normal)'
        self.df_dynamic['model'] = 'Gemma-2-2b-it (Dynamic KV)'
        
        # Combine datasets
        self.df_combined = pd.concat([self.df_normal, self.df_dynamic], ignore_index=True)
        
        print(f"✅ Normal model samples: {len(self.df_normal)}")
        print(f"✅ Dynamic model samples: {len(self.df_dynamic)}")
        
        return self.df_combined
    
    def create_performance_dashboard(self):
        """Create a comprehensive performance dashboard"""
        print("🎨 Creating performance dashboard...")
        
        # Create a 2x3 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Model Performance Comparison Dashboard\ Normal vs Dynamic KV Architecture', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Color scheme
        colors = ['#2E86AB', '#A23B72']  # Blue for normal, Purple for dynamic
        model_names = ['Gemma 2 2B IT (Normal)', 'Gemma 2 2B IT (Dynamic KV)']
        
        # 1. Time Per Output Token (TPOT) - Box plot
        self._create_boxplot(axes[0,0], 'time_per_output_token_ms', 
                           'Time Per Output Token (TPOT)', 'TPOT (ms)', colors)
        
        # 2. Tokens per Second - Box plot
        self._create_boxplot(axes[0,1], 'tokens_per_second', 
                           'Tokens per Second', 'Tokens/sec', colors)
        
        # 3. GPU Memory Usage - Box plot
        self._create_boxplot(axes[0,2], 'memory_allocated_gb', 
                           'GPU Memory Usage', 'Memory (GB)', colors)
        
        # 4. Accuracy Comparison - Bar plot
        self._create_bar_plot(axes[1,0], 'accuracy_pass1', 
                            'Accuracy (Pass@1)', 'Accuracy Score', colors)
        
        # 5. MR-Score Comparison - Bar plot
        self._create_bar_plot(axes[1,1], 'mr_score', 
                            'MR-Score', 'MR-Score', colors)
        
        # 6. First Token Time - Box plot
        self._create_boxplot(axes[1,2], 'first_token_time_ms', 
                           'First Token Time', 'Time (ms)', colors)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
    def _create_boxplot(self, ax, column, title, ylabel, colors):
        """Helper function to create box plots"""
        data = [self.df_normal[column], self.df_dynamic[column]]
        box_plot = ax.boxplot(data, labels=['Normal', 'Dynamic KV'], 
                            patch_artist=True, widths=0.6)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add mean values
        for i, (model_data, color) in enumerate(zip(data, colors)):
            mean_val = np.mean(model_data)
            ax.text(i + 1, mean_val, f'μ={mean_val:.1f}', 
                   ha='center', va='bottom', fontweight='bold', color=color)
    
    def _create_bar_plot(self, ax, column, title, ylabel, colors):
        """Helper function to create bar plots"""
        means = [self.df_normal[column].mean(), self.df_dynamic[column].mean()]
        stds = [self.df_normal[column].std(), self.df_dynamic[column].std()]
        
        bars = ax.bar(['Normal', 'Dynamic KV'], means, yerr=stds, 
                     capsize=10, alpha=0.8, color=colors)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def create_improvement_radar_chart(self):
        """Create a radar chart showing improvements"""
        print("📈 Creating improvement radar chart...")
        
        # Calculate improvement percentages
        normal_means = self.df_normal.mean()
        dynamic_means = self.df_dynamic.mean()
        
        improvements = {
            'Speed (TPOT)': ((normal_means['time_per_output_token_ms'] - dynamic_means['time_per_output_token_ms']) 
                           / normal_means['time_per_output_token_ms'] * 100),
            'Memory Usage': ((normal_means['memory_allocated_gb'] - dynamic_means['memory_allocated_gb']) 
                           / normal_means['memory_allocated_gb'] * 100),
            'Tokens/sec': ((dynamic_means['tokens_per_second'] - normal_means['tokens_per_second']) 
                         / normal_means['tokens_per_second'] * 100),
            'Accuracy': (dynamic_means['accuracy_pass1'] - normal_means['accuracy_pass1']) * 100,  # Percentage points
            'MR-Score': (dynamic_means['mr_score'] - normal_means['mr_score']) * 100  # Percentage points
        }
        
        # Prepare data for radar chart
        categories = list(improvements.keys())
        values = list(improvements.values())
        
        # Normalize values for radar chart (scale to 0-100)
        max_abs = max(abs(min(values)), abs(max(values)))
        normalized_values = [v / max_abs * 100 for v in values]
        
        # Complete the circle
        categories += [categories[0]]
        normalized_values += [normalized_values[0]]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True).tolist()
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot the data
        ax.plot(angles, normalized_values, 'o-', linewidth=2, label='Improvement', color='#A23B72')
        ax.fill(angles, normalized_values, alpha=0.25, color='#A23B72')
        
        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1], fontsize=12)
        
        # Add value annotations
        for angle, value, orig_value in zip(angles[:-1], normalized_values[:-1], values):
            ax.annotate(f'{orig_value:+.1f}%', 
                       xy=(angle, value), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontweight='bold',
                       fontsize=10)
        
        ax.set_ylim(0, 120)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=10)
        ax.grid(True)
        ax.set_title('Dynamic KV Model Improvements Over Normal Model\n(Percentage Improvement)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('improvement_radar_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return improvements
    
    def create_time_series_analysis(self):
        """Create time series analysis of performance metrics"""
        print("⏱️ Creating time series analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Create sample indices for x-axis
        x_normal = range(len(self.df_normal))
        x_dynamic = range(len(self.df_dynamic))
        
        # 1. TPOT over samples
        axes[0,0].plot(x_normal, self.df_normal['time_per_output_token_ms'], 
                      alpha=0.7, label='Normal', color='#2E86AB', linewidth=2)
        axes[0,0].plot(x_dynamic, self.df_dynamic['time_per_output_token_ms'], 
                      alpha=0.7, label='Dynamic KV', color='#A23B72', linewidth=2)
        axes[0,0].set_title('TPOT Over Evaluation Samples', fontweight='bold')
        axes[0,0].set_ylabel('TPOT (ms)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Memory usage over samples
        axes[0,1].plot(x_normal, self.df_normal['memory_allocated_gb'], 
                      alpha=0.7, label='Normal', color='#2E86AB', linewidth=2)
        axes[0,1].plot(x_dynamic, self.df_dynamic['memory_allocated_gb'], 
                      alpha=0.7, label='Dynamic KV', color='#A23B72', linewidth=2)
        axes[0,1].set_title('GPU Memory Usage Over Evaluation Samples', fontweight='bold')
        axes[0,1].set_ylabel('Memory (GB)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Accuracy over samples
        axes[1,0].plot(x_normal, self.df_normal['accuracy_pass1'], 
                      alpha=0.7, label='Normal', color='#2E86AB', linewidth=2)
        axes[1,0].plot(x_dynamic, self.df_dynamic['accuracy_pass1'], 
                      alpha=0.7, label='Dynamic KV', color='#A23B72', linewidth=2)
        axes[1,0].set_title('Accuracy Over Evaluation Samples', fontweight='bold')
        axes[1,0].set_ylabel('Accuracy Score')
        axes[1,0].set_xlabel('Sample Index')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Tokens per second over samples
        axes[1,1].plot(x_normal, self.df_normal['tokens_per_second'], 
                      alpha=0.7, label='Normal', color='#2E86AB', linewidth=2)
        axes[1,1].plot(x_dynamic, self.df_dynamic['tokens_per_second'], 
                      alpha=0.7, label='Dynamic KV', color='#A23B72', linewidth=2)
        axes[1,1].set_title('Tokens per Second Over Evaluation Samples', fontweight='bold')
        axes[1,1].set_ylabel('Tokens/sec')
        axes[1,1].set_xlabel('Sample Index')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_statistical_significance_plot(self):
        """Create plot showing statistical significance"""
        print("🔬 Creating statistical significance plot...")
        
        # Perform t-tests
        metrics_to_test = {
            'TPOT': ('time_per_output_token_ms', 'Lower is better'),
            'Tokens/sec': ('tokens_per_second', 'Higher is better'),
            'Memory': ('memory_allocated_gb', 'Lower is better'),
            'Accuracy': ('accuracy_pass1', 'Higher is better'),
            'MR-Score': ('mr_score', 'Higher is better')
        }
        
        results = []
        for metric_name, (column, direction) in metrics_to_test.items():
            t_stat, p_value = stats.ttest_ind(self.df_normal[column], self.df_dynamic[column])
            normal_mean = self.df_normal[column].mean()
            dynamic_mean = self.df_dynamic[column].mean()
            
            improvement = ((dynamic_mean - normal_mean) / normal_mean * 100 
                          if 'Higher' in direction else 
                          (normal_mean - dynamic_mean) / normal_mean * 100)
            
            results.append({
                'Metric': metric_name,
                'P_Value': p_value,
                'Normal_Mean': normal_mean,
                'Dynamic_Mean': dynamic_mean,
                'Improvement_Pct': improvement,
                'Significant': p_value < 0.05,
                'Direction': direction
            })
        
        results_df = pd.DataFrame(results)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot with significance coding
        colors = ['green' if sig else 'red' for sig in results_df['Significant']]
        sizes = [200 if sig else 100 for sig in results_df['Significant']]
        
        scatter = ax.scatter(results_df['Improvement_Pct'], 
                           -np.log10(results_df['P_Value']),
                           c=colors, s=sizes, alpha=0.7)
        
        # Add labels and annotations
        for i, row in results_df.iterrows():
            ax.annotate(row['Metric'], 
                       (row['Improvement_Pct'], -np.log10(row['P_Value'])),
                       xytext=(5, 5), textcoords='offset points',
                       fontweight='bold' if row['Significant'] else 'normal')
        
        # Add significance threshold line
        ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, 
                  label='Significance Threshold (p=0.05)')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Improvement (%)', fontweight='bold')
        ax.set_ylabel('-log10(P-Value)', fontweight='bold')
        ax.set_title('Statistical Significance of Improvements\nDynamic KV vs Normal Model', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add quadrant labels
        ax.text(0.05, 0.95, 'Significant\nImprovement', transform=ax.transAxes, 
               fontweight='bold', color='green', va='top')
        ax.text(0.65, 0.95, 'Non-significant\nImprovement', transform=ax.transAxes, 
               fontweight='bold', color='red', va='top')
        
        plt.tight_layout()
        plt.savefig('statistical_significance.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return results_df
    
    def create_summary_infographic(self):
        """Create a summary infographic with key metrics"""
        print("📋 Creating summary infographic...")
        
        # Calculate key metrics
        improvements = self.calculate_improvements()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall Performance Score
        performance_score = (
            improvements['tpot_improvement_pct'] * 0.3 +  # Speed weight
            improvements['memory_improvement_pct'] * 0.3 +  # Memory weight  
            improvements['accuracy_improvement_pct'] * 0.2 +  # Accuracy weight
            improvements['mr_score_improvement_pct'] * 0.2  # MR-Score weight
        )
        
        ax1.barh(['Overall Score'], [performance_score], color='skyblue', alpha=0.7)
        ax1.set_xlim(-100, 100)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_title('Overall Performance Score', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Key Improvements
        metrics = ['Speed', 'Memory', 'Accuracy', 'MR-Score']
        values = [
            improvements['tpot_improvement_pct'],
            improvements['memory_improvement_pct'], 
            improvements['accuracy_improvement_pct'],
            improvements['mr_score_improvement_pct']
        ]
        
        bars = ax2.bar(metrics, values, color=['green' if v > 0 else 'red' for v in values], alpha=0.7)
        ax2.set_title('Key Performance Improvements', fontweight='bold')
        ax2.set_ylabel('Improvement (%)')
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if value > 0 else -3),
                    f'{value:+.1f}%', ha='center', va='bottom' if value > 0 else 'top',
                    fontweight='bold')
        
        # 3. Memory vs Speed scatter
        ax3.scatter(self.df_normal['memory_allocated_gb'], self.df_normal['tokens_per_second'],
                   alpha=0.6, label='Normal', color='#2E86AB', s=50)
        ax3.scatter(self.df_dynamic['memory_allocated_gb'], self.df_dynamic['tokens_per_second'],
                   alpha=0.6, label='Dynamic KV', color='#A23B72', s=50)
        ax3.set_xlabel('GPU Memory (GB)')
        ax3.set_ylabel('Tokens per Second')
        ax3.set_title('Memory vs Speed Trade-off', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Quality vs Speed scatter
        ax4.scatter(self.df_normal['tokens_per_second'], self.df_normal['accuracy_pass1'],
                   alpha=0.6, label='Normal', color='#2E86AB', s=50)
        ax4.scatter(self.df_dynamic['tokens_per_second'], self.df_dynamic['accuracy_pass1'],
                   alpha=0.6, label='Dynamic KV', color='#A23B72', s=50)
        ax4.set_xlabel('Tokens per Second')
        ax4.set_ylabel('Accuracy Score')
        ax4.set_title('Speed vs Quality Trade-off', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('summary_infographic.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def calculate_improvements(self):
        """Calculate improvement percentages"""
        normal_means = self.df_normal.mean()
        dynamic_means = self.df_dynamic.mean()
        
        improvements = {
            'tpot_improvement_pct': ((normal_means['time_per_output_token_ms'] - dynamic_means['time_per_output_token_ms']) 
                                   / normal_means['time_per_output_token_ms'] * 100),
            'memory_improvement_pct': ((normal_means['memory_allocated_gb'] - dynamic_means['memory_allocated_gb']) 
                                     / normal_means['memory_allocated_gb'] * 100),
            'tokens_per_second_improvement_pct': ((dynamic_means['tokens_per_second'] - normal_means['tokens_per_second']) 
                                                / normal_means['tokens_per_second'] * 100),
            'accuracy_improvement_pct': (dynamic_means['accuracy_pass1'] - normal_means['accuracy_pass1']) * 100,
            'mr_score_improvement_pct': (dynamic_means['mr_score'] - normal_means['mr_score']) * 100
        }
        
        return improvements
    
    def generate_all_graphs(self):
        """Generate all graphs in sequence"""
        print("🚀 Starting graph generation...")
        
        # Load data first
        self.load_data()
        
        # Generate all visualizations
        self.create_performance_dashboard()
        self.create_improvement_radar_chart()
        self.create_time_series_analysis()
        self.create_statistical_significance_plot()
        self.create_summary_infographic()
        
        print("\n🎉 All graphs generated successfully!")
        print("📁 Generated files:")
        print("   - performance_dashboard.png")
        print("   - improvement_radar_chart.png") 
        print("   - time_series_analysis.png")
        print("   - statistical_significance.png")
        print("   - summary_infographic.png")

# Main execution
if __name__ == "__main__":
    # File paths - update these if your CSV files have different names/locations
    normal_csv = "Gemma-2-2B-IT-GMSK8.csv"
    dynamic_csv = "Hyde-IKV-Gemma-2-2B-IT-GMSK8.csv"
    
    # Initialize visualizer
    visualizer = ResultsVisualizer(normal_csv, dynamic_csv)
    
    # Generate all graphs
    visualizer.generate_all_graphs()
    
    # Print quick summary
    improvements = visualizer.calculate_improvements()
    print("\n" + "="*60)
    print("📊 QUICK SUMMARY")
    print("="*60)
    print(f"⚡ TPOT Improvement: {improvements['tpot_improvement_pct']:+.1f}%")
    print(f"💾 Memory Improvement: {improvements['memory_improvement_pct']:+.1f}%")
    print(f"🚀 Tokens/sec Improvement: {improvements['tokens_per_second_improvement_pct']:+.1f}%")
    print(f"🎯 Accuracy Improvement: {improvements['accuracy_improvement_pct']:+.2f}% points")
    print(f"📊 MR-Score Improvement: {improvements['mr_score_improvement_pct']:+.2f}% points")