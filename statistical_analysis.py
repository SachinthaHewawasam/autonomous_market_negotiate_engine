"""
Statistical analysis tools for research validation.

Provides rigorous statistical testing and analysis:
- Hypothesis testing (t-tests, ANOVA)
- Effect size calculation (Cohen's d)
- Confidence intervals
- Significance testing
- Power analysis
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns


class StatisticalAnalyzer:
    """Performs statistical analysis on experimental results"""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize analyzer.
        
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
    
    def compare_agents(
        self,
        rl_rewards: List[float],
        baseline_rewards: List[float]
    ) -> Dict:
        """
        Compare RL agent vs baseline using statistical tests.
        
        Args:
            rl_rewards: List of episode rewards from RL agent
            baseline_rewards: List of episode rewards from baseline agent
        
        Returns:
            Dictionary with statistical test results
        """
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(rl_rewards, baseline_rewards)
        
        # Effect size (Cohen's d)
        cohens_d = self._calculate_cohens_d(rl_rewards, baseline_rewards)
        
        # Confidence intervals
        rl_ci = self._calculate_ci(rl_rewards)
        baseline_ci = self._calculate_ci(baseline_rewards)
        
        # Descriptive statistics
        rl_mean = np.mean(rl_rewards)
        rl_std = np.std(rl_rewards)
        baseline_mean = np.mean(baseline_rewards)
        baseline_std = np.std(baseline_rewards)
        
        # Improvement
        improvement = ((rl_mean - baseline_mean) / baseline_mean) * 100
        
        # Determine significance
        significant = p_value < self.alpha
        
        results = {
            'hypothesis_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': significant,
                'alpha': self.alpha
            },
            'effect_size': {
                'cohens_d': float(cohens_d),
                'interpretation': self._interpret_cohens_d(cohens_d)
            },
            'rl_agent': {
                'mean': float(rl_mean),
                'std': float(rl_std),
                'ci_lower': float(rl_ci[0]),
                'ci_upper': float(rl_ci[1])
            },
            'baseline_agent': {
                'mean': float(baseline_mean),
                'std': float(baseline_std),
                'ci_lower': float(baseline_ci[0]),
                'ci_upper': float(baseline_ci[1])
            },
            'improvement': {
                'percentage': float(improvement),
                'absolute': float(rl_mean - baseline_mean)
            }
        }
        
        return results
    
    def anova_analysis(
        self,
        groups: Dict[str, List[float]],
        group_names: List[str]
    ) -> Dict:
        """
        Perform one-way ANOVA across multiple groups.
        
        Args:
            groups: Dictionary mapping group names to reward lists
            group_names: List of group names in order
        
        Returns:
            ANOVA results
        """
        # Prepare data
        group_data = [groups[name] for name in group_names]
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)
        
        # Post-hoc pairwise comparisons (Tukey HSD)
        pairwise_results = {}
        for i, name1 in enumerate(group_names):
            for j, name2 in enumerate(group_names):
                if i < j:
                    t_stat, p_val = stats.ttest_ind(groups[name1], groups[name2])
                    cohens_d = self._calculate_cohens_d(groups[name1], groups[name2])
                    
                    pairwise_results[f"{name1}_vs_{name2}"] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_val),
                        'cohens_d': float(cohens_d),
                        'significant': p_val < self.alpha
                    }
        
        results = {
            'anova': {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < self.alpha
            },
            'pairwise_comparisons': pairwise_results,
            'group_statistics': {
                name: {
                    'mean': float(np.mean(groups[name])),
                    'std': float(np.std(groups[name])),
                    'n': len(groups[name])
                }
                for name in group_names
            }
        }
        
        return results
    
    def correlation_analysis(
        self,
        x: List[float],
        y: List[float],
        x_name: str = 'X',
        y_name: str = 'Y'
    ) -> Dict:
        """
        Analyze correlation between two variables.
        
        Args:
            x: First variable
            y: Second variable
            x_name: Name of first variable
            y_name: Name of second variable
        
        Returns:
            Correlation analysis results
        """
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(x, y)
        
        # Spearman correlation (rank-based, robust to outliers)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        
        results = {
            'pearson': {
                'correlation': float(pearson_r),
                'p_value': float(pearson_p),
                'significant': pearson_p < self.alpha,
                'interpretation': self._interpret_correlation(pearson_r)
            },
            'spearman': {
                'correlation': float(spearman_r),
                'p_value': float(spearman_p),
                'significant': spearman_p < self.alpha,
                'interpretation': self._interpret_correlation(spearman_r)
            },
            'variables': {
                'x_name': x_name,
                'y_name': y_name,
                'n': len(x)
            }
        }
        
        return results
    
    def normality_test(self, data: List[float]) -> Dict:
        """
        Test if data follows normal distribution.
        
        Args:
            data: Data to test
        
        Returns:
            Normality test results
        """
        # Shapiro-Wilk test
        stat, p_value = stats.shapiro(data)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        
        results = {
            'shapiro_wilk': {
                'statistic': float(stat),
                'p_value': float(p_value),
                'normal': p_value > self.alpha
            },
            'kolmogorov_smirnov': {
                'statistic': float(ks_stat),
                'p_value': float(ks_p),
                'normal': ks_p > self.alpha
            },
            'descriptive': {
                'mean': float(np.mean(data)),
                'median': float(np.median(data)),
                'std': float(np.std(data)),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data))
            }
        }
        
        return results
    
    def power_analysis(
        self,
        effect_size: float,
        n: int,
        alpha: float = None
    ) -> Dict:
        """
        Calculate statistical power.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            n: Sample size
            alpha: Significance level (uses self.alpha if None)
        
        Returns:
            Power analysis results
        """
        if alpha is None:
            alpha = self.alpha
        
        # Calculate power using normal approximation
        from scipy.stats import norm
        
        # Critical value
        z_alpha = norm.ppf(1 - alpha/2)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n/2)
        
        # Power
        power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
        
        # Required sample size for 80% power
        required_n = int(np.ceil((2 * (z_alpha + norm.ppf(0.8))**2) / (effect_size**2)))
        
        results = {
            'power': float(power),
            'effect_size': float(effect_size),
            'sample_size': n,
            'alpha': alpha,
            'required_n_for_80_power': required_n,
            'adequate_power': power >= 0.8
        }
        
        return results
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        std1 = np.std(group1, ddof=1)
        std2 = np.std(group2, ddof=1)
        n1 = len(group1)
        n2 = len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std
        
        return cohens_d
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient"""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "weak"
        elif abs_r < 0.5:
            return "moderate"
        elif abs_r < 0.7:
            return "strong"
        else:
            return "very strong"
    
    def _calculate_ci(
        self,
        data: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval"""
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
        return ci
    
    def generate_report(
        self,
        rl_rewards: List[float],
        baseline_rewards: List[float],
        output_file: str = 'experiments/statistical_report.txt'
    ):
        """Generate comprehensive statistical report"""
        
        # Perform all analyses
        comparison = self.compare_agents(rl_rewards, baseline_rewards)
        rl_normality = self.normality_test(rl_rewards)
        baseline_normality = self.normality_test(baseline_rewards)
        power = self.power_analysis(
            comparison['effect_size']['cohens_d'],
            len(rl_rewards)
        )
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Hypothesis test
        report.append("1. HYPOTHESIS TEST (Paired t-test)")
        report.append("-" * 80)
        report.append(f"H₀: RL agent performance ≤ Baseline agent performance")
        report.append(f"H₁: RL agent performance > Baseline agent performance")
        report.append(f"")
        report.append(f"t-statistic: {comparison['hypothesis_test']['t_statistic']:.4f}")
        report.append(f"p-value: {comparison['hypothesis_test']['p_value']:.6f}")
        report.append(f"Significance level (α): {self.alpha}")
        report.append(f"Result: {'REJECT H₀' if comparison['hypothesis_test']['significant'] else 'FAIL TO REJECT H₀'}")
        report.append(f"")
        
        # Effect size
        report.append("2. EFFECT SIZE")
        report.append("-" * 80)
        report.append(f"Cohen's d: {comparison['effect_size']['cohens_d']:.4f}")
        report.append(f"Interpretation: {comparison['effect_size']['interpretation'].upper()}")
        report.append(f"")
        
        # Descriptive statistics
        report.append("3. DESCRIPTIVE STATISTICS")
        report.append("-" * 80)
        report.append(f"RL Agent:")
        report.append(f"  Mean: {comparison['rl_agent']['mean']:.2f}")
        report.append(f"  Std: {comparison['rl_agent']['std']:.2f}")
        report.append(f"  95% CI: [{comparison['rl_agent']['ci_lower']:.2f}, {comparison['rl_agent']['ci_upper']:.2f}]")
        report.append(f"")
        report.append(f"Baseline Agent:")
        report.append(f"  Mean: {comparison['baseline_agent']['mean']:.2f}")
        report.append(f"  Std: {comparison['baseline_agent']['std']:.2f}")
        report.append(f"  95% CI: [{comparison['baseline_agent']['ci_lower']:.2f}, {comparison['baseline_agent']['ci_upper']:.2f}]")
        report.append(f"")
        report.append(f"Improvement:")
        report.append(f"  Absolute: {comparison['improvement']['absolute']:.2f}")
        report.append(f"  Percentage: {comparison['improvement']['percentage']:.2f}%")
        report.append(f"")
        
        # Normality tests
        report.append("4. NORMALITY TESTS")
        report.append("-" * 80)
        report.append(f"RL Agent:")
        report.append(f"  Shapiro-Wilk p-value: {rl_normality['shapiro_wilk']['p_value']:.6f}")
        report.append(f"  Normal: {rl_normality['shapiro_wilk']['normal']}")
        report.append(f"")
        report.append(f"Baseline Agent:")
        report.append(f"  Shapiro-Wilk p-value: {baseline_normality['shapiro_wilk']['p_value']:.6f}")
        report.append(f"  Normal: {baseline_normality['shapiro_wilk']['normal']}")
        report.append(f"")
        
        # Power analysis
        report.append("5. POWER ANALYSIS")
        report.append("-" * 80)
        report.append(f"Statistical Power: {power['power']:.4f}")
        report.append(f"Sample Size: {power['sample_size']}")
        report.append(f"Required n for 80% power: {power['required_n_for_80_power']}")
        report.append(f"Adequate Power: {power['adequate_power']}")
        report.append(f"")
        
        # Conclusion
        report.append("6. CONCLUSION")
        report.append("-" * 80)
        if comparison['hypothesis_test']['significant']:
            report.append(f"The RL agent shows STATISTICALLY SIGNIFICANT improvement over the baseline")
            report.append(f"agent (p < {self.alpha}). The effect size is {comparison['effect_size']['interpretation']}.")
            report.append(f"The RL agent achieves {comparison['improvement']['percentage']:.1f}% higher average reward.")
        else:
            report.append(f"No statistically significant difference detected (p ≥ {self.alpha}).")
        
        report.append("")
        report.append("=" * 80)
        
        # Save report
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # Print report
        print('\n'.join(report))
        print(f"\n✓ Report saved to {output_file}")
        
        # Save JSON (convert NumPy types to Python native types)
        def convert_numpy_types(obj):
            """Recursively convert NumPy types to Python native types"""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            else:
                return obj
        
        json_file = output_file.replace('.txt', '.json')
        with open(json_file, 'w') as f:
            json.dump({
                'comparison': convert_numpy_types(comparison),
                'rl_normality': convert_numpy_types(rl_normality),
                'baseline_normality': convert_numpy_types(baseline_normality),
                'power_analysis': convert_numpy_types(power)
            }, f, indent=2)
        
        print(f"✓ JSON results saved to {json_file}")
        
        return comparison


def visualize_statistical_results(
    rl_rewards: List[float],
    baseline_rewards: List[float],
    output_file: str = 'experiments/statistical_plots.png'
):
    """Create comprehensive statistical visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Box plot comparison
    axes[0, 0].boxplot([rl_rewards, baseline_rewards], labels=['RL Agent', 'Baseline'])
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Reward Distribution Comparison')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Violin plot
    parts = axes[0, 1].violinplot([rl_rewards, baseline_rewards], positions=[1, 2], 
                                   showmeans=True, showmedians=True)
    axes[0, 1].set_xticks([1, 2])
    axes[0, 1].set_xticklabels(['RL Agent', 'Baseline'])
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Reward Distribution (Violin Plot)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Histogram overlay
    axes[0, 2].hist(rl_rewards, bins=20, alpha=0.6, label='RL Agent', color='#2ecc71', edgecolor='black')
    axes[0, 2].hist(baseline_rewards, bins=20, alpha=0.6, label='Baseline', color='#3498db', edgecolor='black')
    axes[0, 2].set_xlabel('Reward')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Reward Distribution Overlay')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # 4. Q-Q plot for RL agent
    stats.probplot(rl_rewards, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot: RL Agent')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Q-Q plot for baseline
    stats.probplot(baseline_rewards, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Baseline Agent')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Confidence interval comparison
    rl_mean = np.mean(rl_rewards)
    baseline_mean = np.mean(baseline_rewards)
    rl_ci = stats.t.interval(0.95, len(rl_rewards)-1, loc=rl_mean, scale=stats.sem(rl_rewards))
    baseline_ci = stats.t.interval(0.95, len(baseline_rewards)-1, loc=baseline_mean, scale=stats.sem(baseline_rewards))
    
    agents = ['RL Agent', 'Baseline']
    means = [rl_mean, baseline_mean]
    cis = [rl_ci, baseline_ci]
    
    axes[1, 2].errorbar([0, 1], means, 
                       yerr=[[means[i] - cis[i][0] for i in range(2)],
                             [cis[i][1] - means[i] for i in range(2)]],
                       fmt='o', markersize=10, capsize=10, capthick=2, linewidth=2)
    axes[1, 2].set_xticks([0, 1])
    axes[1, 2].set_xticklabels(agents)
    axes[1, 2].set_ylabel('Mean Reward')
    axes[1, 2].set_title('Mean Reward with 95% CI')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Statistical plots saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    # Example usage with evaluation results
    try:
        with open('logs/evaluation_results.json', 'r') as f:
            results = json.load(f)
        
        rl_rewards = results['rl_agent']['episode_rewards']
        baseline_rewards = results['rule_based_agent']['episode_rewards']
        
        analyzer = StatisticalAnalyzer(alpha=0.05)
        analyzer.generate_report(rl_rewards, baseline_rewards)
        visualize_statistical_results(rl_rewards, baseline_rewards)
        
    except FileNotFoundError:
        print("Error: Run evaluate.py first to generate evaluation results")
        print("\nExample usage:")
        print("  python evaluate.py")
        print("  python statistical_analysis.py")
