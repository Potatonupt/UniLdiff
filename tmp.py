# -*- coding: utf-8 -*-
"""
t-SNE 可视化消融：w/o DAFF（混在一起） vs w/ DAFF（五个单团，但形状/密度/朝向各异）
依赖：numpy, scikit-learn, matplotlib
保存：tsne_daff_ablation.png / .svg
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# --------------------------
# 全局参数
# --------------------------
TASKS = ["dehaze", "derain", "denoise", "deblur", "low-light"]
NUM_CLASSES = len(TASKS)
SAMPLES_PER_CLASS = 200      # 左图每类样本数（右图会有轻微浮动，但仍单团）
FEAT_DIM = 64
RANDOM_STATE_LEFT  = 42
RANDOM_STATE_RIGHT = 77

TSNE_KW = dict(
    n_components=2,
    perplexity=35,
    learning_rate=240,
    n_iter=1200,
    init="pca",
    random_state=42,
    metric="euclidean",
    verbose=0,
)

# --------------------------
# 1) w/o DAFF：五类来自同一分布（难以区分）
# --------------------------
def synth_no_daff(total_samples: int, feat_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(total_samples, feat_dim)).astype(np.float32)
    return X

# --------------------------
# 2) w/ DAFF：每类一个“单团”但各向异性/朝向/密度不同（更自然）
#    关键：不再用多分量 GMM，避免分裂成多团；仅对单峰施加轻微非线性，不至于撕裂
# --------------------------
def synth_with_daff_single_cluster(num_classes: int, base_n: int, feat_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    X_list, y = [], []

    # 把类中心放在一个半径有抖动的环上（前3维），拉开间隔
    R_base = 9.0
    angles = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    for cls, ang in enumerate(angles):
        # 每类样本数有轻微不均衡，但保持单团（不拆分）
        n_cls = base_n + int(rng.integers(-30, 31))  # ±30 浮动

        # 类中心（前两维圆环，第3维小抖动）
        r = R_base + rng.normal(0, 0.7)
        center = np.zeros(feat_dim, dtype=np.float32)
        center[:3] = [r*np.cos(ang), r*np.sin(ang), rng.normal(0, 0.5)]

        # 为该类构造一个“单峰”的各向异性高斯：
        # 1) 在低维子空间（subdim）采样，再映射回高维，得到旋转后的椭球形状
        subdim = 6
        U, _ = np.linalg.qr(rng.normal(size=(feat_dim, subdim)))  # 随机正交基
        # 2) 轴向方差谱：主轴更长，且每类不同，形状和朝向不一
        axis_var = np.linspace(1.0, 0.25, subdim) * rng.uniform(0.7, 1.6)
        axis_var[0] *= rng.uniform(2.0, 3.0)   # 拉长主轴，椭圆更明显
        axis_var[1] *= rng.uniform(1.3, 2.0)
        Sigma_sub = U @ np.diag(axis_var) @ U.T

        # 采样单个高斯分布（单团）
        feats = rng.multivariate_normal(mean=center, cov=Sigma_sub, size=n_cls).astype(np.float32)

        # 轻微“局部非线性”扰动（幅度很小，保持连通性，防止裂变成多团）
        # 仅在前两维上做小幅弯曲+剪切，使轮廓不规则但不破碎
        a = 0.06 + 0.02 * (cls % 3)              # 弯曲强度
        b = 0.9  + 0.15 * rng.random()           # 周期尺度
        c = 0.03 + 0.02 * rng.random()           # 轻微剪切
        x0 = feats[:, 0].copy()
        x1 = feats[:, 1].copy()
        feats[:, 1] = x1 + a * np.sin(x0 / b)    # 细微弯曲
        feats[:, 0] = x0 + c * x1                # 轻微剪切

        # 加一点点全局噪声（不会破坏单团）
        feats += rng.normal(0, 0.03, size=feats.shape).astype(np.float32)

        X_list.append(feats)
        y.extend([cls] * n_cls)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y, dtype=int)
    return X, y

# --------------------------
# 3) t-SNE
# --------------------------
def run_tsne(X: np.ndarray, **tsne_kw):
    tsne = TSNE(**tsne_kw)
    return tsne.fit_transform(X)

# --------------------------
# 4) 作图
# --------------------------
def plot_side_by_side(X_left_2d, y_left, X_right_2d, y_right, tasks, out_png="tsne_daff_ablation.png"):
    plt.figure(figsize=(12, 5), dpi=160)
    colors = plt.get_cmap("tab10")

    # 左：混在一起
    ax1 = plt.subplot(1, 2, 1)
    for cls in range(len(tasks)):
        idx = (y_left == cls)
        ax1.scatter(X_left_2d[idx, 0], X_left_2d[idx, 1], s=8, alpha=0.85, label=tasks[cls], c=[colors(cls)])
    ax1.set_title("w/o DAFF：五类混在一起（不可分）", fontsize=11)
    ax1.set_xlabel("t-SNE 1"); ax1.set_ylabel("t-SNE 2")
    ax1.legend(markerscale=2, fontsize=8, frameon=False)

    # 右：五个单团（每类一个团），但形状/密度/朝向各异
    ax2 = plt.subplot(1, 2, 2)
    for cls in range(len(tasks)):
        idx = (y_right == cls)
        ax2.scatter(X_right_2d[idx, 0], X_right_2d[idx, 1], s=8, alpha=0.85, label=tasks[cls], c=[colors(cls)])
    ax2.set_title("w/ DAFF：五个单团（形状/密度/朝向不同）", fontsize=11)
    ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")
    ax2.legend(markerscale=2, fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_png.replace(".png", ".svg"), bbox_inches="tight")
    print(f"已保存：{out_png} 与 {out_png.replace('.png', '.svg')}")
    plt.show()

# --------------------------
# 5) 主流程
# --------------------------
def main():
    # 左图：不可分（五类同分布）
    total = NUM_CLASSES * SAMPLES_PER_CLASS
    X_no  = synth_no_daff(total, FEAT_DIM, seed=RANDOM_STATE_LEFT)
    y_left = np.repeat(np.arange(NUM_CLASSES), SAMPLES_PER_CLASS)

    # 右图：五个单团（每类一个团），但各异
    X_yes, y_right = synth_with_daff_single_cluster(NUM_CLASSES, SAMPLES_PER_CLASS, FEAT_DIM, seed=RANDOM_STATE_RIGHT)

    # t-SNE（分开实例化）
    X_left_2d  = run_tsne(X_no, **TSNE_KW)
    X_right_2d = run_tsne(X_yes, **TSNE_KW)

    # 可分性参考指标
    sil_left  = silhouette_score(X_left_2d,  y_left,  metric="euclidean")
    sil_right = silhouette_score(X_right_2d, y_right, metric="euclidean")
    print(f"Silhouette（w/o DAFF）: {sil_left:.3f}")
    print(f"Silhouette（w/  DAFF）: {sil_right:.3f}")

    # 作图
    plot_side_by_side(X_left_2d, y_left, X_right_2d, y_right, TASKS, out_png="tsne_daff_ablation.png")

if __name__ == "__main__":
    main()

"""
—— 调参小贴士 ——
1) 如果右图某类出现内裂：
   - 减小非线性幅度 a/c，或把 R_base 再增大到 9.5~10.5 增强类间隔。
   - 降低 axis_var[0] 的放大倍数，避免过长“尾巴”。
2) 想让五团不完全对称：
   - 对每类单独调 subdim、axis_var 的范围或加入不同的剪切系数 c。
3) 替换为真实特征：
   - 把 X_no / X_yes 换成你的特征矩阵（[N, D]），y_left / y_right 替换为任务标签（0~4）。
"""
