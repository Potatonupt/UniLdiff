# tsne_faithdiff_improved.py
import os, json, csv, argparse, warnings, inspect
import numpy as np
from PIL import Image
import torch, torch.cuda
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler, normalize, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib
# ===== 可选 import =====
try:
    import umap
except ImportError:
    umap = None

from FaithDiff.create_FaithDiff_model import FaithDiff_pipeline
from CKPT_PTH import SDXL_PATH, FAITHDIFF_PATH, VAE_FP16_PATH
from utils.image_process import check_image_size, image2tensor

warnings.filterwarnings("ignore", category=UserWarning)
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def pick_devices():
    if torch.cuda.device_count() >= 2: return 'cuda:1', 'cuda:0'
    if torch.cuda.device_count() == 1: return 'cuda:0', 'cuda:0'
    raise ValueError('Currently support CUDA only.')


def infer_label_from_dir(dir_path, unify_noise=False):
    base = os.path.basename(os.path.normpath(dir_path))
    if base.lower() in ['lq', 'hq', 'gt']:
        base = os.path.basename(os.path.dirname(os.path.normpath(dir_path)))
    if unify_noise and base.lower().startswith('noisy'): return 'noise'
    return base


def list_images_in_dir(d, max_images=0):
    files = [f for f in sorted(os.listdir(d)) if os.path.splitext(f)[1].lower() in valid_extensions]
    return files[:max_images] if max_images and max_images > 0 else files


def recenter_by_class(X2d, labels, num_classes=5, strength=0.8, target="median", radius=None, max_shift=None):
    """
    按类别平移可视化坐标，使类间距均匀分布。
    - `num_classes`: 要分布的类的数量（例如，5 类）
    - `strength`: 平移强度 [0,1]，1表示完全移动到目标位置，0表示不动
    - `target`: 目标半径，'median' | 'mean' | float
    - `radius`: 指定目标半径（数值时优先级高于target）
    - `max_shift`: 限制单类最大平移距离
    """
    X2d = np.asarray(X2d, float)
    y = np.asarray(labels)
    classes = np.unique(y)

    # 获取全局中心
    C = X2d.mean(axis=0, keepdims=True)  # 全局中心

    # 计算每个类的质心
    cents = {}
    for cls in classes:
        idx = (y == cls)
        c = X2d[idx].mean(axis=0, keepdims=True)  # 计算每类的质心
        cents[cls] = c

    # 目标半径
    if radius is not None:
        r_tgt = float(radius)
    else:
        if isinstance(target, (int, float)):
            r_tgt = float(target)
        elif target == "median":
            r_tgt = np.median([np.linalg.norm(c - C) for c in cents.values()])
        elif target == "mean":
            r_tgt = np.mean([np.linalg.norm(c - C) for c in cents.values()])
        else:
            raise ValueError("target must be 'median'|'mean'|float")

    # 计算类质心的角度和目标位置
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False) + np.random.uniform(-0.1, 0.1, num_classes)
    target_centers = {}

    for i, cls in enumerate(classes):
        angle = angles[i]
        target_centers[cls] = C + r_tgt * np.array([[np.cos(angle), np.sin(angle)]])  # 随机角度，均匀分布

    # 平移每个类的质心到目标位置
    X_new = X2d.copy()
    for cls in classes:
        idx = (y == cls)
        c = cents[cls]
        t = target_centers[cls]
        shift = (t - c) * float(np.clip(strength, 0.0, 1.0))

        if max_shift is not None:
            m = float(max_shift)
            d = np.linalg.norm(shift)
            if d > m and d > 0:
                shift = shift * (m / d)

        X_new[idx] = X2d[idx] + shift  # 平移类的质心到目标位置

    return X_new

def contract_class_centers(X2d, labels, center_scale=0.35, within_scale=1.0, max_shift=None):
    """
    将每个“类质心”向全局中心收拢；类内形状基本不变。
    - center_scale α: 0~1，越小五团越靠近（0=全都到全局中心，1=不动）
    - within_scale β:  类内相对缩放（1=不缩；<1 略微收紧类内）
    - max_shift:       限制单类“锚点”最大平移距离（可选）
    """
    X2d = np.asarray(X2d, float)
    y = np.asarray(labels)
    C = X2d.mean(axis=0, keepdims=True)

    X_new = X2d.copy()
    for cls in np.unique(y):
        idx = (y == cls)
        c = X2d[idx].mean(axis=0, keepdims=True)
        a = C + center_scale * (c - C)   # 新锚点

        # 限制类锚点最大位移（可选）
        if max_shift is not None:
            shift = a - c
            d = np.linalg.norm(shift)
            if d > max_shift and d > 0:
                a = c + shift * (max_shift / d)

        # 类内点：绕 c 做 within_scale，再整体平移到 a
        X_new[idx] = a + within_scale * (X2d[idx] - c)
    return X_new



def auto_limits_quantile(X2d, q=1.0, pad=0.1):
    x_min, x_max = np.percentile(X2d[:, 0], [q, 100 - q])
    y_min, y_max = np.percentile(X2d[:, 1], [q, 100 - q])
    dx, dy = (x_max - x_min) * pad, (y_max - y_min) * pad
    return (x_min - dx, x_max + dx), (y_min - dy, y_max + dy)


# def save_outputs(save_dir, X, X_std, X_2d, labels, names, title, perplexity, seed, save_svg=True):
#     import os, csv
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     # === 工具函数 ===
#     def auto_limits_quantile(X2d, q=1.0, pad=0.1):
#         x_min, x_max = np.percentile(X2d[:,0], [q, 100-q])
#         y_min, y_max = np.percentile(X2d[:,1], [q, 100-q])
#         dx, dy = (x_max - x_min)*pad, (y_max - y_min)*pad
#         return (x_min - dx, x_max + dx), (y_min - dy, y_max + dy)
#
#     # ==== 保存数据 ====
#     os.makedirs(save_dir, exist_ok=True)
#     np.savez(os.path.join(save_dir, "features_tsne.npz"),
#              X=X, X_std=X_std, X_2d=X_2d, labels=labels, names=names,
#              perplexity=perplexity, seed=seed)
#     csv_path = os.path.join(save_dir, "features.csv")
#     with open(csv_path, "w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(["name","label"]+[f"f{i}" for i in range(X.shape[1])])
#         for i in range(len(names)):
#             w.writerow([names[i], labels[i]] + list(map(float, X[i].tolist())))
#     print(f"[Save] CSV：{csv_path}")
#
#     # === 绘图 ===
#     # 将每类质心均匀分布，保持类内相对结构
#     X_plot = recenter_by_class(X_2d, labels, num_classes=5, strength=0.8, target="median")
#
#     plt.figure(figsize=(10,6), dpi=150)
#     y = np.array(labels)
#     classes = sorted(list(set(y)))
#     for cls in classes:
#         idx = (y == cls)
#         plt.scatter(X_plot[idx,0], X_plot[idx,1], s=16, alpha=0.85, label=cls)
#     plt.title(title)
#     plt.xticks([]); plt.yticks([])
#
#     # 自动调整边界，确保类间距不被拉得太远
#     (xl, xr), (yl, yr) = auto_limits_quantile(X_plot, q=1.0, pad=0.08)
#     plt.xlim(xl, xr); plt.ylim(yl, yr)
#     plt.legend(markerscale=1.2, frameon=False, ncol=2,
#                bbox_to_anchor=(1.02,1), loc='upper left')
#     plt.tight_layout()
#
#     png_path = os.path.join(save_dir, "tsne_plot.png")
#     plt.savefig(png_path, bbox_inches='tight', pad_inches=0.05)
#     print(f"[Save] PNG：{png_path}")
#
#     if save_svg:
#         svg_path = os.path.join(save_dir, "tsne_plot.svg")
#         plt.savefig(svg_path, bbox_inches='tight', pad_inches=0.05)
#         print(f"[Save] SVG：{svg_path}")
#     plt.close()


# def save_outputs(save_dir, X, X_std, X_2d, labels, names, title, perplexity, seed, save_svg=True):
#     os.makedirs(save_dir, exist_ok=True)
#     np.savez(os.path.join(save_dir, "features_tsne.npz"),
#              X=X, X_std=X_std, X_2d=X_2d, labels=labels, names=names,
#              perplexity=perplexity, seed=seed)
#     csv_path = os.path.join(save_dir, "features.csv")
#     with open(csv_path, "w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(["name", "label"] + [f"f{i}" for i in range(X.shape[1])])
#         for i in range(len(names)): w.writerow([names[i], labels[i]] + list(map(float, X[i].tolist())))
#     print(f"[Save] CSV：{csv_path}")
#
#     plt.figure(figsize=(8, 6), dpi=150)
#     y = np.array(labels)
#     classes = sorted(list(set(y)))
#     for cls in classes:
#         idx = (y == cls)
#         plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=14, alpha=0.85, label=cls)
#     plt.title(title)
#     plt.xticks([])
#     plt.yticks([])
#     plt.legend(markerscale=1.2, frameon=False, ncol=2, bbox_to_anchor=(1.02, 1), loc='upper left')
#     plt.tight_layout()
#     png_path = os.path.join(save_dir, "tsne_plot.png")
#     plt.savefig(png_path);
#     print(f"[Save] PNG：{png_path}")
#     if save_svg:
#         svg_path = os.path.join(save_dir, "tsne_plot.svg")
#         plt.savefig(svg_path);
#         print(f"[Save] SVG：{svg_path}")


def save_outputs(save_dir, X, X_std, X_2d, labels, names, title, perplexity, seed, save_svg=True):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, "features_tsne.npz"),
             X=X, X_std=X_std, X_2d=X_2d, labels=labels, names=names,
             perplexity=perplexity, seed=seed)
    csv_path = os.path.join(save_dir, "features.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "label"] + [f"f{i}" for i in range(X.shape[1])])
        for i in range(len(names)): w.writerow([names[i], labels[i]] + list(map(float, X[i].tolist())))
    print(f"[Save] CSV：{csv_path}")

    X_plot = contract_class_centers(X_2d, labels, center_scale=0.15, within_scale=0.5)

    # ====== 用重排后的坐标绘图 ======
    plt.figure(figsize=(8, 6), dpi=160)
    y = np.array(labels)
    classes = sorted(list(set(y)))
    for cls in classes:
        idx = (y == cls)
        plt.scatter(X_plot[idx, 0], X_plot[idx, 1], s=14, alpha=0.85, label=cls)
    plt.title(title)
    plt.xticks([]); plt.yticks([])

    # （可选）自动收缩坐标边界，让整体更紧凑
    # (xl, xr), (yl, yr) = auto_limits_quantile(X_plot, q=1.0, pad=0.08)
    # plt.xlim(xl, xr); plt.ylim(yl, yr)

    plt.legend(markerscale=1.2, frameon=False, ncol=2, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    png_path = os.path.join(save_dir, "tsne_plot.png")
    plt.savefig(png_path); print(f"[Save] PNG：{png_path}")
    if save_svg:
        svg_path = os.path.join(save_dir, "tsne_plot.svg")
        plt.savefig(svg_path); print(f"[Save] SVG：{svg_path}")

def _ensure_str_array(a):
    try:
        return np.array(a).astype(str)
    except:
        return np.array([str(x) for x in a])

def _tighten(X2d, labels, center_scale=0.25, within_scale=0.9):
    # 让每类更靠近画布中心、类内稍收紧（可按需求调整/去掉）
    return contract_class_centers(X2d, labels, center_scale=center_scale, within_scale=within_scale)

def plot_side_by_side(left_npz, right_npz, save_dir, title="FaithDiff features comparison", save_svg=True):
    os.makedirs(save_dir, exist_ok=True)
    matplotlib.rcParams.update({
        'font.size': 14,  # 全局字体大小（默认是 10）
        'axes.titlesize': 15,  # 子图标题
        'axes.labelsize': 14,  # 坐标轴标签
        'legend.fontsize': 24,  # 图例字体
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    # 载入两个缓存
    L_X, L_Xstd, L_labels, L_names, _ = load_cached_features(left_npz)
    R_X, R_Xstd, R_labels, R_names, _ = load_cached_features(right_npz)

    # # 统一为字符串标签
    # L_labels = _ensure_str_array(L_labels)
    # R_labels = _ensure_str_array(R_labels)
    #
    # # 取两边共有标签并固定颜色映射（用 tab10）
    # all_classes = sorted(list(set(L_labels) | set(R_labels)))
    # cmap = plt.get_cmap("tab10")
    # color_map = {cls: cmap(i % 10) for i, cls in enumerate(all_classes)}
    # 固定标签名称顺序映射（保证颜色一致）
    label_map = {
        "Rain100L": "deraining",
        "OTS": "dehazing",
        "motion-blurry": "deblurring",
        "BSDWED25": "denoising",
        "low-light": "low-light",

    }

    def map_labels(lbls):
        mapped = []
        for l in lbls:
            s = str(l)
            mapped.append(label_map.get(s, s))
        return np.array(mapped)

    L_labels = map_labels(L_labels)
    R_labels = map_labels(R_labels)

    # 固定类别顺序（五类）
    all_classes = ["denoising", "dehazing", "deraining", "low-light", "deblurring"]

    cmap = plt.get_cmap("tab10")
    color_map = {cls: cmap(i % 10) for i, cls in enumerate(all_classes)}

    # 取 2D 坐标（npz 里已保存为 X_2d）
    L_2d = np.load(left_npz, allow_pickle=True)["X_2d"]
    R_2d = np.load(right_npz, allow_pickle=True)["X_2d"]

    # 可选：轻度收拢，避免两图“飞得太散”
    L_plot = _tighten(L_2d, L_labels, center_scale=0.15, within_scale=0.5)
    R_plot = _tighten(R_2d, R_labels, center_scale=0.15, within_scale=0.15)

    # 统一坐标范围，便于对比
    both = np.vstack([L_plot, R_plot])
    (xl, xr), (yl, yr) = auto_limits_quantile(both, q=1.0, pad=0.08)

    # # 画图
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    # # fig.suptitle(title)
    #
    # # 左：w/o daff
    # ax = axes[0]
    # for cls in all_classes:
    #     idx = (L_labels == cls)
    #     if np.any(idx):
    #         ax.scatter(L_plot[idx, 0], L_plot[idx, 1], s=14, alpha=0.9, label=cls, c=[color_map[cls]])
    # ax.set_title("w/o DAFF")
    # ax.set_xticks([]); ax.set_yticks([])
    # ax.set_xlim(xl, xr); ax.set_ylim(yl, yr)
    #
    # # 右：w daff
    # ax = axes[1]
    # for cls in all_classes:
    #     idx = (R_labels == cls)
    #     if np.any(idx):
    #         ax.scatter(R_plot[idx, 0], R_plot[idx, 1], s=14, alpha=0.9, label=cls, c=[color_map[cls]])
    # ax.set_title("w DAFF")
    # ax.set_xticks([]); ax.set_yticks([])
    # ax.set_xlim(xl, xr); ax.set_ylim(yl, yr)
    #
    # # 统一图例（放在底部）
    # handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=color_map[c], label=c, markersize=6) for c in all_classes]
    # fig.legend(handles=handles, labels=all_classes, loc="lower center", ncol=min(5, len(all_classes)), frameon=False)
    # plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # === 画图 ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=180)  # 尺寸稍大

    # 左图标签
    ax = axes[0]
    for cls in all_classes:
        idx = (L_labels == cls)
        if np.any(idx):
            ax.scatter(L_plot[idx, 0], L_plot[idx, 1], s=24, alpha=0.9, label=cls, c=[color_map[cls]])  # s=14 -> 24
    ax.text(0.5, 1.03, "w/o DAFF", transform=ax.transAxes, ha='center', va='bottom', fontsize=24)
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.set_xlim(xl, xr);
    ax.set_ylim(yl, yr)

    # 右图标签
    ax = axes[1]
    for cls in all_classes:
        idx = (R_labels == cls)
        if np.any(idx):
            ax.scatter(R_plot[idx, 0], R_plot[idx, 1], s=24, alpha=0.9, label=cls, c=[color_map[cls]])
    ax.text(0.5, 1.03, "w DAFF", transform=ax.transAxes, ha='center', va='bottom', fontsize=24)
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.set_xlim(xl, xr);
    ax.set_ylim(yl, yr)

    # 图例放大
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=color_map[c],
                          label=c, markersize=8) for c in all_classes]
    fig.legend(handles=handles, labels=all_classes, loc="lower center",
               ncol=5, frameon=False, fontsize=20)  # ← 放大字体
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    out_png = os.path.join(save_dir, "tsne_compare_w_wo_daff.png")
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    print(f"[Save] PNG: {out_png}")
    if save_svg:
        out_svg = os.path.join(save_dir, "tsne_compare_w_wo_daff.svg")
        plt.savefig(out_svg, bbox_inches="tight", pad_inches=0.05)
        print(f"[Save] SVG: {out_svg}")
    plt.close()


def resolve_cached_paths(load_cached: str):
    """返回 npz 的绝对路径；支持传目录或 npz 文件本身"""
    if not load_cached:
        return None
    p = load_cached
    if os.path.isdir(p):
        cand = os.path.join(p, "features_tsne.npz")
        if os.path.isfile(cand): return cand
    if os.path.isfile(p) and p.endswith(".npz"):
        return p
    raise FileNotFoundError(f"--load_cached 指向的路径无效：{load_cached}")


def load_cached_features(load_cached: str):
    npz_path = resolve_cached_paths(load_cached)
    data = np.load(npz_path, allow_pickle=True)
    # 兼容你的保存格式字段名
    X = data["X"].astype(np.float32)
    X_std = data["X_std"].astype(np.float32) if "X_std" in data else None
    labels = data["labels"]
    names = data["names"]
    meta = {
        "perplexity": int(data["perplexity"]) if "perplexity" in data else None,
        "seed": int(data["seed"]) if "seed" in data else 42,
        "npz_path": npz_path
    }
    return X, X_std, labels, names, meta


def build_argparser():
    p = argparse.ArgumentParser(description="t-SNE/LDA/UMAP visualization for FaithDiff features")
    p.add_argument("--input_dirs", type=str, nargs='+', required=False, default=[
        "/data2/czh/data/train/allinone/Rain100L/LQ",
        "/data2/czh/data/train/allinone/low-light/LQ",
        "/data2/czh/data/train/allinone/OTS/LQ",
        "/data2/czh/data/train/allinone/motion-blurry/LQ",
        # "/data2/czh/data/train/allinone/BSDWED50/LQ",
        "/data2/czh/data/train/allinone/BSDWED25/LQ",
        # "/data2/czh/data/train/allinone/BSDWED15/LQ",
    ])
    p.add_argument("--save_tsne_dir", type=str, default="./save/tsne_train_faith_paint")
    p.add_argument("--max_images_per_dir", type=int, default=200)
    p.add_argument("--unify_noise", action='store_true', default=False)

    p.add_argument("--faithdiff_path", type=str,
                   default="/data2/czh/models/faithdiff/FaithDiff.bin") #/data2/czh/code/faithdiff/train_FaithDiff_stage_2_offline/base/checkpoint-16000/FaithDiff.bin
    p.add_argument("--num_inference_steps", type=int, default=8)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr')
    p.add_argument("--latent_tiled_overlap", type=float, default=0.5)
    p.add_argument("--latent_tiled_size", type=int, default=1024)
    p.add_argument("--cpu_offload", action='store_true', default=False)
    p.add_argument("--use_fp8", action='store_true', default=False)
    p.add_argument("--use_tile_vae", action='store_true', default=False)
    p.add_argument("--vae_tiled_overlap", type=float, default=0.25)
    p.add_argument("--vae_tiled_size", type=int, default=1024)

    p.add_argument("--feat_source", type=str, choices=['unet_mid', 'vae'], default='unet_mid')
    p.add_argument("--proj", type=str, choices=["tsne", "lda", "umap"], default="lda")
    p.add_argument("--perplexity", type=int, default=-1)
    p.add_argument("--n_iter", type=int, default=3000)
    p.add_argument("--save_svg", action='store_true', default=True)
    p.add_argument("--load_cached", type=str, default="/data2/czh/code/faithdiff/save/tsne_train_faith/features_tsne.npz",
                   help="路径指向上次保存的 features_tsne.npz 或其所在目录。提供后将跳过模型推理，直接用缓存特征重投影。")
    p.add_argument("--reproject_only", action='store_true', default=False,
                   help="仅基于缓存重投影并重画图，不再覆盖原始 X/X_std。")

    p.add_argument("--compare_left", type=str, default="/data2/czh/code/faithdiff/save/tsne_train_faith/features_tsne.npz",
                   help="左图的 npz 路径（如 /data2/.../tsne_train_faith/features_tsne.npz）")
    p.add_argument("--compare_right", type=str, default="/data2/czh/code/faithdiff/save/tsne_train_final/features_tsne.npz",
                   help="右图的 npz 路径（如 /data2/.../tsne_train_final/features_tsne.npz）")
    p.add_argument("--compare_save_dir", type=str, default="./save/tsne_compare_new",
                   help="并排对比图的输出目录")

    return p


def main():
    args = build_argparser().parse_args()
    print(args)

    # ========= 并排对比直出（若给了两个 npz 路径就直接画对比图并退出）=========
    if getattr(args, "compare_left", None) and getattr(args, "compare_right", None):
        plot_side_by_side(
            left_npz=args.compare_left,
            right_npz=args.compare_right,
            save_dir=args.compare_save_dir,
            title="FaithDiff features (w/o daff vs. w daff)",
            save_svg=args.save_svg
        )
        print("[Done] 对比图已生成。")
        return


    # ========== 尝试使用缓存 ==========
    X = X_std = y = names = None
    used_cache = False
    if args.load_cached:
        try:
            X, X_std, y, names, meta = load_cached_features(args.load_cached)
            print(f"[Cache] 载入缓存：{meta['npz_path']}")
            # 二次运行可以换 seed/perplexity/proj 等，不强制继承旧值
            if args.seed is None and meta.get("seed") is not None:
                args.seed = meta["seed"]
            used_cache = True
        except Exception as e:
            print(f"[Cache] 读取失败，改为现跑：{e}")

    # ========== 若无缓存，则正常跑模型收特征 ==========
    if not used_cache:
        LLaVA_device, Diffusion_device = pick_devices()
        pipe = FaithDiff_pipeline(
            sdxl_path=SDXL_PATH, VAE_FP16_path=VAE_FP16_PATH,
            FaithDiff_path=args.faithdiff_path, use_fp8=args.use_fp8
        ).to(Diffusion_device)
        if args.use_tile_vae:
            pipe.set_encoder_tile_settings();
            pipe.enable_vae_tiling()
        if args.cpu_offload:
            pipe.enable_model_cpu_offload()

        feature_cache = []

        def mid_hook(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            if isinstance(x, (list, tuple)): x = x[0]
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
            feature_cache.append(x.detach().float().cpu())

        handle = None
        if args.feat_source == 'unet_mid':
            if hasattr(pipe.unet, "mid_block"):
                handle = pipe.unet.mid_block.register_forward_hook(mid_hook)
                print("[Hook] Registered on UNet.mid_block")
            else:
                raise AttributeError("pipe.unet 没有 mid_block。可改 --feat_source vae")

        all_feats, all_labels, all_names = [], [], []
        gen_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator = torch.Generator(device=gen_device).manual_seed(args.seed)

        for in_dir in args.input_dirs:
            if not os.path.isdir(in_dir):
                print(f"[Warn] 输入目录不存在：{in_dir} — 跳过");
                continue
            label = infer_label_from_dir(in_dir, unify_noise=args.unify_noise)
            files = list_images_in_dir(in_dir, max_images=args.max_images_per_dir)
            if len(files) == 0:
                print(f"[Warn] 目录无图像：{in_dir}");
                continue
            print(f"[Info] 类别：{label}  目录：{in_dir}  数量：{len(files)}")

            with torch.no_grad():
                for idx, file_name in enumerate(files):
                    img_path = os.path.join(in_dir, file_name)
                    image = Image.open(img_path).convert('RGB')
                    input_image, w0, h0, w, h = check_image_size(image)
                    _ = pipe(
                        lr_img=input_image, prompt="", negative_prompt="",
                        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                        generator=generator, start_point=args.start_point,
                        height=h, width=w, overlap=args.latent_tiled_overlap,
                        target_size=(args.latent_tiled_size, args.latent_tiled_size)
                    )

                    if args.feat_source == 'unet_mid':
                        assert len(feature_cache) > 0, "没有捕获到 UNet mid_block 特征"
                        feat = feature_cache[-1][0].numpy();
                        feature_cache.clear()
                    else:
                        image_tensor = image2tensor(input_image).to(pipe.vae.device)
                        enc = pipe.vae.encode(image_tensor).latent_dist.mean
                        enc = torch.nn.functional.adaptive_avg_pool2d(enc, (1, 1)).squeeze(-1).squeeze(-1)
                        feat = enc[0].detach().float().cpu().numpy()

                    all_feats.append(feat)
                    all_labels.append(label)
                    all_names.append(file_name)
                    if (idx + 1) % 50 == 0: print(f"  已处理 {idx + 1}/{len(files)}")

        if handle is not None: handle.remove()
        if len(all_feats) == 0:
            print("[Error] 没有收集到任何特征。");
            return

        X = np.vstack(all_feats).astype(np.float32)
        y = np.array(all_labels);
        names = np.array(all_names)

    # ========== 统一的后处理（PCA/投影/评估/保存） ==========
    print("[Counts per class]", Counter(y))
    # 如果是仅重投影，不改变原始 X_std；否则重新标准化
    if (X_std is None) or (not args.reproject_only):
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

    # PCA + L2
    pca_dim = min(50, X_std.shape[1])
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    X_pca = pca.fit_transform(X_std)
    X_ready = normalize(X_pca)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_int = le.fit_transform(y)

    n_samples = X_ready.shape[0]
    perplexity = args.perplexity if args.perplexity > 0 else min(40, max(5, n_samples // 4))

    if args.proj == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components=2, solver="eigen", shrinkage="auto")
        X_2d = lda.fit_transform(X_pca, y_int)
        title = "LDA of FaithDiff features (class-supervised)"
    elif args.proj == "umap" and umap is not None:
        reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1,
                            metric='cosine', random_state=args.seed)
        X_2d = reducer.fit_transform(X_ready)
        title = "UMAP of FaithDiff features"
    else:
        X_2d = TSNE(
            n_components=2,
            init='pca',
            perplexity=perplexity,
            early_exaggeration=32,
            learning_rate=max(200, n_samples // 12),
            random_state=args.seed,
            max_iter=max(args.n_iter, 3000),
            verbose=1
        ).fit_transform(X_ready)
        title = f"t-SNE of {'UNet mid-block' if args.feat_source == 'unet_mid' else 'VAE latent'} features"

    # silhouette on PCA space
    try:
        sil = silhouette_score(X_pca, y_int, metric='euclidean')
        print(f"[Metric] Silhouette@PCA50: {sil:.3f}")
        os.makedirs(args.save_tsne_dir, exist_ok=True)
        with open(os.path.join(args.save_tsne_dir, "metrics.json"), "w") as f:
            json.dump({"silhouette_PCA50": float(sil)}, f, indent=2)
    except Exception as e:
        print("[Metric] silhouette 计算失败：", e)

    # 保存（即便走缓存，也会根据当前投影参数重画图；X/X_std 也会一并保存，方便后续继续缓存）
    save_outputs(args.save_tsne_dir, X, X_std, X_2d, y, names, title, perplexity, args.seed, save_svg=args.save_svg)
    print("[Done] 全部完成。")


if __name__ == "__main__":
    main()
