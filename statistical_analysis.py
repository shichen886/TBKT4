import numpy as np
from scipy import stats

# Test AUC data from 5 seeds
nopos_auc = np.array([0.8006, 0.7949, 0.8017, 0.8006, 0.8019])
rope_qk_auc = np.array([0.8021, 0.8020, 0.8026, 0.8011, 0.8007])

print("=" * 80)
print("Paired t-test: NoPos vs RoPE-QK")
print("=" * 80)
print()

# Print raw data
print("Raw Data (Test AUC):")
print("-" * 80)
print(f"Seed\tNoPos\t\tRoPE-QK\t\tDifference")
print("-" * 80)
for i in range(5):
    diff = rope_qk_auc[i] - nopos_auc[i]
    print(f"{i+1}\t{nopos_auc[i]:.4f}\t\t{rope_qk_auc[i]:.4f}\t\t{diff:+.4f}")
print()

# Calculate differences
differences = rope_qk_auc - nopos_auc

# Paired t-test
t_statistic, p_value = stats.ttest_rel(rope_qk_auc, nopos_auc)

# Cohen's d for paired samples
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)
cohens_d = mean_diff / std_diff

# 95% Confidence Interval for the mean difference
n = len(differences)
sem = std_diff / np.sqrt(n)  # Standard Error of the Mean
ci_lower = mean_diff - stats.t.ppf(0.975, n-1) * sem
ci_upper = mean_diff + stats.t.ppf(0.975, n-1) * sem

# Print results
print("Statistical Analysis Results:")
print("-" * 80)
print(f"Number of paired samples (seeds): {n}")
print()
print(f"Mean difference (RoPE-QK - NoPos): {mean_diff:.6f}")
print(f"Standard deviation of differences: {std_diff:.6f}")
print(f"Standard error of the mean: {sem:.6f}")
print()
print(f"t-statistic: {t_statistic:.6f}")
print(f"Degrees of freedom: {n-1}")
print(f"p-value: {p_value:.6f}")
print()
print(f"Cohen's d: {cohens_d:.6f}")
print(f"95% CI for mean difference: [{ci_lower:.6f}, {ci_upper:.6f}]")
print()

# Interpretation
print("=" * 80)
print("Interpretation:")
print("=" * 80)
print()

# Cohen's d interpretation
if abs(cohens_d) < 0.2:
    effect_size = "negligible"
elif abs(cohens_d) < 0.5:
    effect_size = "small"
elif abs(cohens_d) < 0.8:
    effect_size = "medium"
else:
    effect_size = "large"

print(f"Effect size (Cohen's d): {effect_size}")
print()

# p-value interpretation
alpha = 0.05
if p_value < alpha:
    print(f"Result: Statistically significant (p < {alpha})")
    print(f"The difference between NoPos and RoPE-QK is statistically significant.")
else:
    print(f"Result: Not statistically significant (p >= {alpha})")
    print(f"The difference between NoPos and RoPE-QK is not statistically significant.")
print()

# Practical significance
print("=" * 80)
print("Practical Significance:")
print("=" * 80)
print()
print(f"Mean improvement of RoPE-QK over NoPos: {mean_diff*100:.4f}%")
print(f"Relative improvement: {(mean_diff/np.mean(nopos_auc))*100:.4f}%")
print()

# Additional statistics
print("=" * 80)
print("Additional Statistics:")
print("=" * 80)
print()
print(f"NoPos - Mean: {np.mean(nopos_auc):.6f}, Std: {np.std(nopos_auc, ddof=1):.6f}")
print(f"RoPE-QK - Mean: {np.mean(rope_qk_auc):.6f}, Std: {np.std(rope_qk_auc, ddof=1):.6f}")
print()
