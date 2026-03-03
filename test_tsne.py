# tsne_faithdiff.py
import os, json, csv, argparse, warnings, inspect
import numpy as np
from PIL import Image
import torch, torch.cuda
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

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
    if base.lower() in ['lq','hq','gt']:
        base = os.path.basename(os.path.dirname(os.path.normpath(dir_path)))
    if unify_noise and base.lower().startswith('noisy'): return 'noise'
    return base

def list_images_in_dir(d, max_images=0):
    files = [f for f in sorted(os.listdir(d)) if os.path.splitext(f)[1].lower() in valid_extensions]
    return files[:max_images] if max_images and max_images > 0 else files

def tsne_fit_transform(X_std, seed, perplexity, n_iter, verbose=1):
    # 兼容 sklearn：1.6+ 用 max_iter；旧版用 n_iter。learning_rate 用数值更稳
    sig = inspect.signature(TSNE.__init__)
    tsne_kwargs = dict(
        n_components=2, init='pca', perplexity=perplexity,
        random_state=seed, verbose=verbose, learning_rate=200.0
    )
    if 'max_iter' in sig.parameters:
        tsne_kwargs['max_iter'] = n_iter
    else:
        tsne_kwargs['n_iter'] = n_iter
    tsne = TSNE(**tsne_kwargs)
    return tsne.fit_transform(X_std)

def save_outputs(save_dir, X, X_std, X_2d, labels, names, title, perplexity, seed, save_svg=True):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, "features_tsne.npz"),
             X=X, X_std=X_std, X_2d=X_2d, labels=labels, names=names,
             perplexity=perplexity, seed=seed)
    csv_path = os.path.join(save_dir, "features.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name","label"]+[f"f{i}" for i in range(X.shape[1])])
        for i in range(len(names)): w.writerow([names[i], labels[i]] + list(map(float, X[i].tolist())))
    print(f"[Save] CSV：{csv_path}")

    plt.figure(figsize=(8,6), dpi=150)
    y = np.array(labels)
    classes = sorted(list(set(y)))
    for cls in classes:
        idx = (y == cls)
        plt.scatter(X_2d[idx,0], X_2d[idx,1], s=14, alpha=0.85, label=cls)
    plt.title(title); plt.xticks([]); plt.yticks([])
    plt.legend(markerscale=1.2, frameon=False, ncol=2)
    plt.tight_layout()
    png_path = os.path.join(save_dir, "tsne_plot.png")
    plt.savefig(png_path); print(f"[Save] PNG：{png_path}")
    if save_svg:
        svg_path = os.path.join(save_dir, "tsne_plot.svg")
        plt.savefig(svg_path); print(f"[Save] SVG：{svg_path}")

    try:
        sil = silhouette_score(X_std, y, metric='euclidean')
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump({"silhouette_euclidean": float(sil)}, f, indent=2)
        print(f"[Metric] Silhouette (high-dim): {sil:.3f}")
    except Exception as e:
        print("[Metric] Silhouette 计算失败：", e)

def build_argparser():
    p = argparse.ArgumentParser(description="t-SNE visualization for FaithDiff features")
    p.add_argument("--input_dirs", type=str, nargs='+', required=False, default=[
        "/data2/czh/data/train/allinone/Rain100L/LQ",
        "/data2/czh/data/train/allinone/low-light/LQ",
        "/data2/czh/data/train/allinone/OTS/LQ",
        "/data2/czh/data/train/allinone/motion-blurry/LQ",
        "/data2/czh/data/train/allinone/BSDWED50/LQ",
        "/data2/czh/data/train/allinone/BSDWED25/LQ",
        "/data2/czh/data/train/allinone/BSDWED15/LQ",
    ])
    p.add_argument("--save_tsne_dir", type=str, default="./save/tsne_train")
    p.add_argument("--max_images_per_dir", type=int, default=200)
    p.add_argument("--unify_noise", action='store_true', default=False)

    p.add_argument("--faithdiff_path", type=str, default="/data2/czh/code/faithdiff/train_FaithDiff_stage_2_offline/tmp/checkpoint-14000/FaithDiff.bin")
    p.add_argument("--num_inference_steps", type=int, default=8)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start_point", type=str, choices=['lr','noise'], default='lr')
    p.add_argument("--latent_tiled_overlap", type=float, default=0.5)
    p.add_argument("--latent_tiled_size", type=int, default=1024)
    p.add_argument("--cpu_offload", action='store_true', default=False)
    p.add_argument("--use_fp8", action='store_true', default=False)
    p.add_argument("--use_tile_vae", action='store_true', default=False)
    p.add_argument("--vae_tiled_overlap", type=float, default=0.25)
    p.add_argument("--vae_tiled_size", type=int, default=1024)

    p.add_argument("--feat_source", type=str, choices=['unet_mid','vae'], default='unet_mid')
    p.add_argument("--perplexity", type=int, default=-1)
    p.add_argument("--n_iter", type=int, default=1500)
    p.add_argument("--save_svg", action='store_true', default=True)

    # DRY-RUN（不跑模型，秒测）
    p.add_argument("--dry_run", action='store_true', default=False)
    p.add_argument("--dry_classes", type=str, nargs="+",
                   default=["rainy1","low-light","SOTS","motion-blurry","noisy50","noisy25","noisy15"])
    p.add_argument("--dry_points_per_class", type=int, default=60)
    p.add_argument("--dry_dim", type=int, default=1280)
    p.add_argument("--dry_separation", type=float, default=3.0)
    return p

def main():
    args = build_argparser().parse_args()
    print(args)

    # -------- DRY RUN：不跑模型，几秒自检 --------
    if args.dry_run:
        rng = np.random.default_rng(args.seed)
        C, D, N = len(args.dry_classes), args.dry_dim, args.dry_points_per_class
        feats, labels, names = [], [], []

        centers = rng.normal(loc=0.0, scale=1.0, size=(C, D))
        for i in range(C): centers[i, i % D] += args.dry_separation * (i + 1)

        for ci, cls in enumerate(args.dry_classes):
            mean, cov = centers[ci], np.eye(D)
            Xc = rng.multivariate_normal(mean=mean, cov=cov, size=N).astype(np.float32)
            feats.append(Xc)
            labels += [cls] * N
            # 修复：使用 range(N) 生成名字（原先用了未定义的 k）
            names += [f"{cls}_{i:04d}.png" for i in range(N)]

        X = np.vstack(feats); y = np.array(labels); names = np.array(names)
        X_std = StandardScaler().fit_transform(X)
        n_samples = X.shape[0]
        perplexity = args.perplexity if args.perplexity > 0 else max(5, min(30, n_samples // 3))
        if perplexity >= n_samples: perplexity = max(5, n_samples // 2)
        print(f"[DRY] n_samples={n_samples}, dim={D}, classes={C}, perplexity={perplexity}")
        X_2d = tsne_fit_transform(X_std, seed=args.seed, perplexity=perplexity, n_iter=args.n_iter, verbose=1)
        title = f"t-SNE (DRY RUN) of synthetic features (C={C}, D={D})"
        save_outputs(args.save_tsne_dir, X, X_std, X_2d, y, names, title, perplexity, args.seed, save_svg=args.save_svg)
        print("[Done] DRY RUN 完成。"); return

    # -------- 真实运行：抽模型特征 --------
    LLaVA_device, Diffusion_device = pick_devices()
    pipe = FaithDiff_pipeline(
        sdxl_path=SDXL_PATH, VAE_FP16_path=VAE_FP16_PATH,
        FaithDiff_path=args.faithdiff_path, use_fp8=args.use_fp8
    ).to(Diffusion_device)
    if args.use_tile_vae:
        pipe.set_encoder_tile_settings(); pipe.enable_vae_tiling()
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()

    feature_cache = []
    def mid_hook(module, inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        if isinstance(x, (list, tuple)): x = x[0]
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1)).squeeze(-1).squeeze(-1)
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
            print(f"[Warn] 输入目录不存在：{in_dir} — 跳过"); continue
        label = infer_label_from_dir(in_dir, unify_noise=args.unify_noise)
        files = list_images_in_dir(in_dir, max_images=args.max_images_per_dir)
        if len(files) == 0:
            print(f"[Warn] 目录无图像：{in_dir}"); continue
        print(f"[Info] 类别：{label}  目录：{in_dir}  数量：{len(files)}")

        with torch.no_grad():
            for idx, file_name in enumerate(files):
                img_path = os.path.join(in_dir, file_name)
                image = Image.open(img_path).convert('RGB')
                input_image, w0, h0, w, h = check_image_size(image)
                prompt_init, negative_prompt_init = "", ""

                _ = pipe(
                    lr_img=input_image, prompt=prompt_init, negative_prompt=negative_prompt_init,
                    num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                    generator=generator, start_point=args.start_point,
                    height=h, width=w, overlap=args.latent_tiled_overlap,
                    target_size=(args.latent_tiled_size, args.latent_tiled_size)
                )

                if args.feat_source == 'unet_mid':
                    assert len(feature_cache) > 0, "没有捕获到 UNet mid_block 特征"
                    feat = feature_cache[-1][0].numpy(); feature_cache.clear()
                else:
                    image_tensor = image2tensor(input_image).to(Diffusion_device)
                    enc = pipe.vae.encode(image_tensor).latent_dist.mean
                    enc = torch.nn.functional.adaptive_avg_pool2d(enc, (1,1)).squeeze(-1).squeeze(-1)
                    feat = enc[0].detach().float().cpu().numpy()

                all_feats.append(feat); all_labels.append(label); all_names.append(file_name)
                if (idx + 1) % 50 == 0: print(f"  已处理 {idx + 1}/{len(files)}")

    if handle is not None: handle.remove()
    if len(all_feats) == 0:
        print("[Error] 没有收集到任何特征。"); return

    X = np.vstack(all_feats).astype(np.float32)
    y = np.array(all_labels); names = np.array(all_names)
    X_std = StandardScaler().fit_transform(X)

    n_samples = X_std.shape[0]
    if n_samples < 5:
        print(f"[Error] 样本太少（{n_samples}）。"); return
    perplexity = args.perplexity if args.perplexity > 0 else max(5, min(30, n_samples // 3))
    if perplexity >= n_samples:
        perplexity = max(5, n_samples // 2); print(f"[Warn] perplexity 过大，自动调整为 {perplexity}")
    print(f"[t-SNE] n_samples={n_samples}, perplexity={perplexity}, n_iter={args.n_iter}")

    X_2d = tsne_fit_transform(X_std, seed=args.seed, perplexity=perplexity, n_iter=args.n_iter, verbose=1)
    title = f"t-SNE of {'UNet mid-block' if args.feat_source=='unet_mid' else 'VAE latent'} features"
    save_outputs(args.save_tsne_dir, X, X_std, X_2d, y, names, title, perplexity, args.seed, save_svg=args.save_svg)
    print("[Done] 全部完成。")

if __name__ == "__main__":
    main()
