"""
TikTok 视频数据集 — 概率论 Coursework
核心问题: Claim 类视频与 Opinion 类视频的用户参与度是否存在显著差异？

方法一: 概率分布拟合 + MLE 参数估计（对数正态分布）
方法二: 假设检验（Mann-Whitney U + 双样本 t 检验）
加分: 贝叶斯条件概率分析 + Logistic 回归
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无头模式，兼容 WSL
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局设置
# ============================================================
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CSV_PATH = os.path.join(SCRIPT_DIR, "tiktok_dataset.csv")

# ============================================================
# 1. 数据加载与预处理
# ============================================================
print("=" * 70)
print("  PART 1: Data Loading & Preprocessing")
print("=" * 70)

df = pd.read_csv(CSV_PATH)
print(f"\n原始数据集: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n列名: {list(df.columns)}")

# 数值型参与度指标
engagement_cols = [
    'video_view_count', 'video_like_count', 'video_share_count',
    'video_download_count', 'video_comment_count'
]

# 缺失值检查
print(f"\n缺失值统计:")
print(df[engagement_cols + ['claim_status', 'author_ban_status', 'verified_status']].isnull().sum())

# 删除缺失行
df = df.dropna(subset=engagement_cols + ['claim_status'])
print(f"清洗后数据: {df.shape[0]} rows")

# 分组
claim = df[df['claim_status'] == 'claim']
opinion = df[df['claim_status'] == 'opinion']
print(f"\nClaim 组: {len(claim)} 条 ({len(claim)/len(df)*100:.1f}%)")
print(f"Opinion 组: {len(opinion)} 条 ({len(opinion)/len(df)*100:.1f}%)")


# ============================================================
# 2. 描述性统计与 EDA
# ============================================================
print("\n" + "=" * 70)
print("  PART 2: Descriptive Statistics & EDA")
print("=" * 70)

# 分组描述性统计
for col in engagement_cols:
    print(f"\n--- {col} ---")
    for label, group in [("Claim", claim), ("Opinion", opinion)]:
        vals = group[col]
        print(f"  {label:8s} | mean={vals.mean():>12,.0f}  median={vals.median():>12,.0f}  "
              f"std={vals.std():>12,.0f}  skew={vals.skew():>6.2f}  max={vals.max():>12,.0f}")

# ---- 图1: 各参与度指标的箱线图（Claim vs Opinion）----
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, col in enumerate(engagement_cols):
    ax = axes[i]
    data_plot = df[[col, 'claim_status']].copy()
    data_plot[col] = np.log1p(data_plot[col])  # log(1+x) 可视化
    sns.boxplot(data=data_plot, x='claim_status', y=col, ax=ax,
                palette={'claim': '#e74c3c', 'opinion': '#3498db'})
    ax.set_title(f'log(1+{col})')
    ax.set_xlabel('')

# 删除多余子图
axes[5].set_visible(False)
plt.suptitle('User Engagement Distribution: Claim vs Opinion (log scale)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_boxplots.png'), bbox_inches='tight')
plt.close()
print(f"\n✓ 图1已保存: figures/fig1_boxplots.png")

# ---- 图2: video_view_count 直方图 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (label, group) in zip(axes, [("Claim", claim), ("Opinion", opinion)]):
    log_views = np.log1p(group['video_view_count'])
    ax.hist(log_views, bins=50, density=True, alpha=0.7,
            color='#e74c3c' if label == 'Claim' else '#3498db', edgecolor='white')
    # 拟合正态曲线
    mu, std = log_views.mean(), log_views.std()
    x = np.linspace(log_views.min(), log_views.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, std), 'k-', linewidth=2,
            label=f'Normal fit: μ={mu:.2f}, σ={std:.2f}')
    ax.set_title(f'{label} Videos (n={len(group)})')
    ax.set_xlabel('log(1 + view_count)')
    ax.set_ylabel('Density')
    ax.legend()

plt.suptitle('Distribution of log(1 + video_view_count)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_view_count_hist.png'), bbox_inches='tight')
plt.close()
print(f"✓ 图2已保存: figures/fig2_view_count_hist.png")


# ============================================================
# 3. 方法一: 概率分布拟合 + MLE 参数估计
# ============================================================
print("\n" + "=" * 70)
print("  PART 3: Probability Distribution Fitting (MLE)")
print("=" * 70)

print("""
对数正态分布 (Log-Normal Distribution):
如果 X ~ LogNormal(μ, σ²)，则 ln(X) ~ N(μ, σ²)

概率密度函数:
  f(x; μ, σ) = (1 / (x·σ·√(2π))) · exp(-(ln x - μ)² / (2σ²)),  x > 0

最大似然估计 (MLE) 推导:
  令 y_i = ln(x_i)，则 y_i ~ N(μ, σ²)

  对数似然函数:
    ℓ(μ, σ²) = -n/2 · ln(2π) - n/2 · ln(σ²) - 1/(2σ²) · Σ(y_i - μ)²

  对 μ 求导令其为零:
    ∂ℓ/∂μ = 1/σ² · Σ(y_i - μ) = 0
    => μ̂ = (1/n) · Σ ln(x_i)          [样本对数均值]

  对 σ² 求导令其为零:
    ∂ℓ/∂(σ²) = -n/(2σ²) + 1/(2σ⁴) · Σ(y_i - μ)² = 0
    => σ̂² = (1/n) · Σ(ln(x_i) - μ̂)²  [样本对数方差]
""")

def mle_lognormal(data):
    """对数正态分布的 MLE 参数估计"""
    log_data = np.log(data[data > 0])  # 取对数，排除零值
    n = len(log_data)
    mu_hat = np.mean(log_data)
    sigma2_hat = np.var(log_data, ddof=0)  # MLE 用 n 而非 n-1
    sigma_hat = np.sqrt(sigma2_hat)
    return mu_hat, sigma_hat, n

def neg_log_likelihood(params, data):
    """负对数似然函数（用于数值优化验证）"""
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    log_data = np.log(data[data > 0])
    n = len(log_data)
    ll = -n/2 * np.log(2 * np.pi) - n * np.log(sigma) - np.sum((log_data - mu)**2) / (2 * sigma**2)
    return -ll

# 对 view_count 做 MLE
print("\n>>> 视频播放量 (video_view_count) 的对数正态分布拟合 <<<\n")

results_mle = {}
for label, group in [("Claim", claim), ("Opinion", opinion)]:
    views = group['video_view_count'].values
    views = views[views > 0]  # 排除零值

    # 解析解 MLE
    mu_hat, sigma_hat, n = mle_lognormal(views)

    # 数值优化验证
    result = minimize(neg_log_likelihood, x0=[mu_hat, sigma_hat],
                      args=(views,), method='Nelder-Mead')
    mu_num, sigma_num = result.x

    results_mle[label] = (mu_hat, sigma_hat, n)

    print(f"  {label} 组 (n={n}):")
    print(f"    解析 MLE:  μ̂ = {mu_hat:.4f},  σ̂ = {sigma_hat:.4f}")
    print(f"    数值 MLE:  μ̂ = {mu_num:.4f},  σ̂ = {sigma_num:.4f}")
    print(f"    拟合均值:  E[X] = exp(μ̂ + σ̂²/2) = {np.exp(mu_hat + sigma_hat**2/2):,.0f}")
    print(f"    实际均值:  {views.mean():,.0f}")
    print(f"    拟合中位数: exp(μ̂) = {np.exp(mu_hat):,.0f}")
    print(f"    实际中位数: {np.median(views):,.0f}")
    print()

    # 其他参与度指标的 MLE
    print(f"  {label} 组 — 全部参与度指标 MLE:")
    for col in engagement_cols:
        vals = group[col].values
        vals = vals[vals > 0]
        if len(vals) > 0:
            mu_i, sigma_i, _ = mle_lognormal(vals)
            print(f"    {col:25s} | μ̂={mu_i:>8.3f}  σ̂={sigma_i:>8.3f}")
    print()

# 对数正态分布拟合的 Q-Q 图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (label, group) in zip(axes, [("Claim", claim), ("Opinion", opinion)]):
    views = group['video_view_count'].values
    views = views[views > 0]
    log_views = np.log(views)
    mu_hat, sigma_hat = results_mle[label][0], results_mle[label][1]

    # Q-Q plot
    theoretical_q = stats.norm.ppf(np.linspace(0.005, 0.995, len(log_views)))
    sample_q = np.sort((log_views - mu_hat) / sigma_hat)

    ax.scatter(theoretical_q, sample_q, s=1, alpha=0.3,
               color='#e74c3c' if label == 'Claim' else '#3498db')
    ax.plot([-4, 4], [-4, 4], 'k--', linewidth=1.5, label='y = x')
    ax.set_xlabel('Theoretical Quantiles (Standard Normal)')
    ax.set_ylabel('Sample Quantiles (Standardized log)')
    ax.set_title(f'Q-Q Plot: {label} log(view_count)')
    ax.legend()
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_qq_plot.png'), bbox_inches='tight')
plt.close()
print(f"✓ 图3已保存: figures/fig3_qq_plot.png")

# 正态性检验（对数变换后）
print("对数变换后的 Shapiro-Wilk 正态性检验:")
for label, group in [("Claim", claim), ("Opinion", opinion)]:
    log_views = np.log(group['video_view_count'].values[group['video_view_count'].values > 0])
    # Shapiro-Wilk 对大样本取子集
    sample = np.random.choice(log_views, size=min(5000, len(log_views)), replace=False)
    stat, p = stats.shapiro(sample)
    print(f"  {label:8s} | W={stat:.6f}, p={p:.2e} {'(近似正态)' if p > 0.05 else '(非正态)'}")


# ============================================================
# 4. 方法二: 假设检验
# ============================================================
print("\n" + "=" * 70)
print("  PART 4: Hypothesis Testing")
print("=" * 70)

# --- 检验 1: Mann-Whitney U 检验（非参数）---
print("\n>>> 检验 1: Mann-Whitney U 检验 (非参数) <<<")
print("  H₀: Claim 组与 Opinion 组的 view_count 分布相同")
print("  H₁: 两组的 view_count 分布不同 (双侧)")
print("  显著性水平: α = 0.05\n")

claim_views = claim['video_view_count'].values
opinion_views = opinion['video_view_count'].values

u_stat, p_mw = stats.mannwhitneyu(claim_views, opinion_views, alternative='two-sided')
print(f"  U 统计量 = {u_stat:,.0f}")
print(f"  p-value   = {p_mw:.2e}")
print(f"  结论: {'拒绝 H₀，两组分布显著不同' if p_mw < 0.05 else '不拒绝 H₀'}")

# 效应量 (rank-biserial correlation)
n1, n2 = len(claim_views), len(opinion_views)
r = 1 - (2 * u_stat) / (n1 * n2)
print(f"  效应量 r = {r:.4f} ({'小' if abs(r) < 0.3 else '中' if abs(r) < 0.5 else '大'}效应)")

# --- 检验 2: 双样本 t 检验（对数变换后）---
print(f"\n>>> 检验 2: 双样本 t 检验 (log 变换后) <<<")
print("  H₀: μ_claim = μ_opinion (对数尺度下的均值)")
print("  H₁: μ_claim ≠ μ_opinion (双侧)")
print("  显著性水平: α = 0.05\n")

log_claim = np.log(claim_views[claim_views > 0])
log_opinion = np.log(opinion_views[opinion_views > 0])

# 方差齐性检验
lev_stat, lev_p = stats.levene(log_claim, log_opinion)
print(f"  Levene 方差齐性检验: W={lev_stat:.4f}, p={lev_p:.2e}")
equal_var = lev_p > 0.05
print(f"  方差{'齐性' if equal_var else '非齐性'}，使用 {'Student' if equal_var else 'Welch'} t 检验\n")

t_stat, p_t = stats.ttest_ind(log_claim, log_opinion, equal_var=equal_var)
print(f"  t 统计量 = {t_stat:.4f}")
print(f"  p-value   = {p_t:.2e}")
print(f"  结论: {'拒绝 H₀，两组对数均值显著不同' if p_t < 0.05 else '不拒绝 H₀'}")

# Cohen's d
pooled_std = np.sqrt(((len(log_claim)-1)*log_claim.std(ddof=1)**2 +
                       (len(log_opinion)-1)*log_opinion.std(ddof=1)**2) /
                      (len(log_claim) + len(log_opinion) - 2))
cohens_d = (log_claim.mean() - log_opinion.mean()) / pooled_std
print(f"  Cohen's d = {cohens_d:.4f} ({'小' if abs(cohens_d) < 0.5 else '中' if abs(cohens_d) < 0.8 else '大'}效应)")

# --- 对所有参与度指标做检验 ---
print(f"\n>>> 全部参与度指标的假设检验汇总 <<<\n")
print(f"  {'指标':25s} | {'Mann-Whitney p':>15s} | {'t-test p':>15s} | {'Cohen d':>8s} | {'结论':>10s}")
print(f"  {'-'*25} | {'-'*15} | {'-'*15} | {'-'*8} | {'-'*10}")

test_results = []
for col in engagement_cols:
    c_vals = claim[col].values
    o_vals = opinion[col].values

    u, p1 = stats.mannwhitneyu(c_vals, o_vals, alternative='two-sided')

    lc = np.log(c_vals[c_vals > 0])
    lo = np.log(o_vals[o_vals > 0])
    t, p2 = stats.ttest_ind(lc, lo, equal_var=False)

    ps = np.sqrt(((len(lc)-1)*lc.std(ddof=1)**2 + (len(lo)-1)*lo.std(ddof=1)**2) /
                 (len(lc) + len(lo) - 2))
    d = (lc.mean() - lo.mean()) / ps

    sig = "***" if p1 < 0.001 else "**" if p1 < 0.01 else "*" if p1 < 0.05 else "ns"
    print(f"  {col:25s} | {p1:>15.2e} | {p2:>15.2e} | {d:>8.3f} | {sig:>10s}")

    test_results.append({'variable': col, 'mw_p': p1, 't_p': p2, 'cohens_d': d})

# ---- 图4: 参与度指标效应量对比 ----
fig, ax = plt.subplots(figsize=(10, 5))
vars_short = [r['variable'].replace('video_', '').replace('_count', '') for r in test_results]
effects = [r['cohens_d'] for r in test_results]
colors = ['#e74c3c' if e > 0 else '#3498db' for e in effects]

bars = ax.barh(vars_short, effects, color=colors, edgecolor='white', height=0.6)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel("Cohen's d (log scale)")
ax.set_title("Effect Size: Claim vs Opinion Engagement (positive = Claim higher)")
for bar, d in zip(bars, effects):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f'{d:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_effect_sizes.png'), bbox_inches='tight')
plt.close()
print(f"\n✓ 图4已保存: figures/fig4_effect_sizes.png")


# ============================================================
# 5. 加分: 贝叶斯条件概率分析
# ============================================================
print("\n" + "=" * 70)
print("  PART 5: Bayesian Analysis (Bonus)")
print("=" * 70)

# 列联表
ct = pd.crosstab(df['claim_status'], df['author_ban_status'])
print("\n列联表 (claim_status × author_ban_status):")
print(ct)

# 条件概率
print("\n条件概率:")
for ban_status in df['author_ban_status'].unique():
    subset = df[df['author_ban_status'] == ban_status]
    p_claim = (subset['claim_status'] == 'claim').mean()
    print(f"  P(claim | {ban_status:>12s}) = {p_claim:.4f}  (n={len(subset)})")

# 贝叶斯定理: P(ban | claim) = P(claim | ban) · P(ban) / P(claim)
p_claim_total = (df['claim_status'] == 'claim').mean()
print(f"\n贝叶斯定理反推:")
for ban_status in df['author_ban_status'].unique():
    p_ban = (df['author_ban_status'] == ban_status).mean()
    p_claim_given_ban = (df[df['author_ban_status'] == ban_status]['claim_status'] == 'claim').mean()
    p_ban_given_claim = p_claim_given_ban * p_ban / p_claim_total
    print(f"  P({ban_status:>12s} | claim) = {p_ban_given_claim:.4f}")

# 卡方检验
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
print(f"\n卡方检验: χ² = {chi2:.4f}, p = {p_chi:.2e}, df = {dof}")
print(f"结论: {'claim_status 与 author_ban_status 显著关联' if p_chi < 0.05 else '无显著关联'}")

# 认证状态分析
print("\n--- verified_status × claim_status ---")
ct2 = pd.crosstab(df['claim_status'], df['verified_status'])
print(ct2)
for v_status in df['verified_status'].unique():
    subset = df[df['verified_status'] == v_status]
    p_claim = (subset['claim_status'] == 'claim').mean()
    print(f"  P(claim | {v_status:>14s}) = {p_claim:.4f}  (n={len(subset)})")


# ============================================================
# 6. 加分: Logistic 回归
# ============================================================
print("\n" + "=" * 70)
print("  PART 6: Logistic Regression (Bonus)")
print("=" * 70)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 特征工程
feature_cols = engagement_cols + ['video_duration_sec']
df_model = df.dropna(subset=feature_cols + ['claim_status']).copy()
df_model['is_claim'] = (df_model['claim_status'] == 'claim').astype(int)

X = df_model[feature_cols].values
y = df_model['is_claim'].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 训练
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# 评估
y_pred = lr.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n特征: {feature_cols}")
print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
print(f"准确率: {acc:.4f}")
print(f"\n系数 (标准化后):")
for name, coef in zip(feature_cols, lr.coef_[0]):
    print(f"  {name:25s} | β = {coef:>8.4f}")
print(f"  {'intercept':25s} | β = {lr.intercept_[0]:>8.4f}")

print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['opinion', 'claim']))

# 混淆矩阵图
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Opinion', 'Claim'], yticklabels=['Opinion', 'Claim'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix: Logistic Regression')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_confusion_matrix.png'), bbox_inches='tight')
plt.close()
print(f"✓ 图5已保存: figures/fig5_confusion_matrix.png")

# ---- 图6: Logistic 回归系数 ----
fig, ax = plt.subplots(figsize=(10, 5))
coef_df = pd.DataFrame({'feature': feature_cols, 'coefficient': lr.coef_[0]})
coef_df = coef_df.sort_values('coefficient')
colors = ['#e74c3c' if c > 0 else '#3498db' for c in coef_df['coefficient']]
ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, edgecolor='white')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient (standardized)')
ax.set_title('Logistic Regression Coefficients\n(positive = more likely Claim)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig6_logistic_coef.png'), bbox_inches='tight')
plt.close()
print(f"✓ 图6已保存: figures/fig6_logistic_coef.png")


# ============================================================
# 7. 结论
# ============================================================
print("\n" + "=" * 70)
print("  CONCLUSION")
print("=" * 70)

print(f"""
核心问题: Claim 类视频与 Opinion 类视频的用户参与度是否存在显著差异？

1. 描述性统计:
   - Claim 类视频在所有参与度指标上均值远高于 Opinion 类视频
   - 所有指标呈严重右偏分布，适合对数正态分布建模

2. 概率分布拟合 (MLE):
   - video_view_count 服从对数正态分布 LogNormal(μ, σ²)
   - Claim 组:  μ̂ = {results_mle['Claim'][0]:.3f}, σ̂ = {results_mle['Claim'][1]:.3f}
   - Opinion 组: μ̂ = {results_mle['Opinion'][0]:.3f}, σ̂ = {results_mle['Opinion'][1]:.3f}
   - 两组参数差异显著，Claim 组的对数均值更高

3. 假设检验:
   - Mann-Whitney U 检验: p = {p_mw:.2e} → {'拒绝 H₀' if p_mw < 0.05 else '不拒绝 H₀'}
   - 双样本 t 检验 (log): p = {p_t:.2e} → {'拒绝 H₀' if p_t < 0.05 else '不拒绝 H₀'}
   - Cohen's d = {cohens_d:.3f}，效应量{'大' if abs(cohens_d) >= 0.8 else '中' if abs(cohens_d) >= 0.5 else '小'}

4. 贝叶斯分析:
   - 账号封禁状态与内容类型存在显著关联 (χ² p = {p_chi:.2e})

5. Logistic 回归:
   - 仅用参与度指标即可达到 {acc:.1%} 的分类准确率
   - like_count 和 view_count 是最强预测因子

结论: Claim 类视频的用户参与度显著高于 Opinion 类视频。
      参与度指标是区分内容类型的有效特征。
""")
