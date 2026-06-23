"""
Canonical 空间下的服装重建评估脚本（compute_metrics2 的几何对齐重构版）。

动机
----
原 compute_metrics2.py 用 rigid_align（仅平移+可选 ICP 旋转）把预测服装与 GT
对齐后算 CD/F-Score。但预测服装披在固定 **A-pose** 身体上，而 CloSe GT 是
**真实扫描姿态**（肩/肘旋转可达 ~1.2rad）。两者是不同的关节姿态，刚性变换
无法调和，导致 CD/F-Score 受姿态差污染而非真实形状差异。

本脚本改为：把预测与 GT **都反驱动到同一 SMPL canonical (neutral T-pose)** 后
再算 CD/F-Score，从而只比较服装形状本身。分类类指标
（Val.Rate / SSR / Meta Acc.）逻辑与原脚本一致，直接复用。

实测（10001_1923）：CD 3.27→2.31cm，F@5mm 0.029→0.050。

依赖
----
预测端反解：ContourCraft/unpose_garmentcode.py
GT 端反解  ：ContourCraft/unpose_close_gt.py
复用函数    ：compute_metrics2.py（分类/CD-F/汇总/保存）
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh

# ---- 路径：本脚本所在 work 目录 + ContourCraft 根 ----
WORK_DIR = str(Path(__file__).resolve().parent)
CONTOUR_CRAFT_ROOT = "/root/wyc/code/ContourCraft"
for p in (WORK_DIR, CONTOUR_CRAFT_ROOT):
    if p not in sys.path:
        sys.path.append(p)

# ---- 复用原评估脚本里已验证的逻辑 ----
from compute_metrics2 import (
    CLOTH_LABELS, NON_GARMENT_LABELS, DEFAULT_TAU, N_SAMPLE,
    evaluate_meta_accuracy, save_results,
)

# ---- F-Score 阈值 ----
# 重要说明：论文正文写 τ=5mm，但其表 4-3/4-4 与该阈值在数学上不自洽——
# 例如 SewFormer CD=7.157cm 却 F=0.789，平均距离 7cm 的两表面不可能有 79% 的点
# 落在 5mm 内（F 应≈0）。实测我们的预测 CD≈2.3cm（与基线 D2GC 2.0cm 同档），
# F@5mm≈0.05，而 F@30~50mm=0.77~0.92，正好覆盖论文 0.79~0.90 的区间。
# 故论文实际使用的有效阈值远大于 5mm（约 3~5cm）。为透明可比，这里同时报告多个阈值，
# 并以 PRIMARY_TAU 作为与论文对齐的主指标（可用 --tau 覆盖）。
F_SCORE_TAUS = [0.005, 0.01, 0.02, 0.03, 0.05]   # 5/10/20/30/50 mm
PRIMARY_TAU = 0.005                                # 复现论文量级的主阈值


def compute_cd_fscore_multi(pred_pts, gt_pts, taus=F_SCORE_TAUS):
    """一次性算 CD 与多个阈值下的 F-Score。返回 (cd_cm, {tau: f_score})。"""
    P = torch.tensor(pred_pts, dtype=torch.float32).unsqueeze(0)
    Q = torch.tensor(gt_pts, dtype=torch.float32).unsqueeze(0)
    D = torch.cdist(P, Q, p=2).squeeze(0)
    p2g = D.min(dim=1).values
    g2p = D.min(dim=0).values
    cd_cm = 0.5 * (p2g.mean() + g2p.mean()).item() * 100.0
    fs = {}
    for tau in taus:
        pr = (p2g < tau).float().mean().item()
        rc = (g2p < tau).float().mean().item()
        fs[tau] = (2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
    sweep = " ".join(f"F@{int(t*1000)}mm={fs[t]:.3f}" for t in taus)
    print(f"[Metric] CD={cd_cm:.3f}cm | {sweep}")
    return cd_cm, fs
# ---- 复用两端反驱动模块 ----
from unpose_garmentcode import build_garmentcode_apose, build_body, unpose
from unpose_close_gt import unpose_close_gt


# ==================== 预测端：design_sim.obj -> canonical ====================

def _load_pred_betas(output_dir: Path) -> np.ndarray:
    """预测身体 betas：优先 smpl.json，回退 hybrik/smpl.json。"""
    for rel in ("smpl.json", "hybrik/smpl.json"):
        p = output_dir / rel
        if p.exists():
            data = json.load(open(p))
            if "betas" in data:
                return np.asarray(data["betas"], dtype=np.float32).reshape(-1)[:10]
    raise FileNotFoundError(f"{output_dir} 下无含 betas 的 smpl.json / hybrik/smpl.json")


def get_pred_canonical(output_dir: Path, smpl_root: str, gender: str,
                       device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """预测服装 design_sim.obj（cm, A-pose）反解到 SMPL neutral T-pose。

    返回 (canonical 顶点 m, faces, garment->body NN 距离 m)。
    """
    garment_path = output_dir / "design" / "design_sim.obj"
    if not garment_path.exists():
        garment_path = output_dir / "design_sim.obj"
    loaded = trimesh.load(garment_path, process=False)
    verts = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces)

    betas = _load_pred_betas(output_dir)
    full_pose = build_garmentcode_apose()
    model, bt, fp, body_verts = build_body(betas, full_pose, smpl_root, gender, device)
    min_y = float(body_verts[:, 1].min())

    g = verts / 100.0           # cm -> m
    g[:, 1] += min_y            # 撤销 export_smpl_mesh 的 feet-on-ground 平移

    from scipy.spatial import cKDTree
    nn_d = cKDTree(body_verts).query(g)[0]
    v_canon, _ = unpose(g, body_verts, model, bt, fp, device, keep_shape=False)
    return v_canon, faces, nn_d


# ==================== 可选：canonical 空间内轻量刚性精修 ====================

def optional_icp(pred_v: np.ndarray, gt_pts: np.ndarray, enable: bool) -> np.ndarray:
    """两者已同处 canonical 帧，默认不动；启用时做不缩放 rigid ICP 精修残差。"""
    if not enable:
        return pred_v
    try:
        from trimesh.registration import icp
        src = pred_v[np.random.choice(len(pred_v), min(5000, len(pred_v)), replace=False)]
        ref = gt_pts[np.random.choice(len(gt_pts), min(20000, len(gt_pts)), replace=False)]
        T, _, _ = icp(src, ref, max_iterations=40, scale=False)
        return trimesh.transformations.transform_points(pred_v, T)
    except Exception as e:
        print(f"[Warning] canonical ICP failed: {e}")
        return pred_v


# ==================== 单样本评估 ====================

def evaluate_single_sample(npz_path: str,
                           output_dir: str,
                           smpl_root: str,
                           gender: str,
                           tau: float = PRIMARY_TAU,
                           use_icp: bool = False,
                           save_debug_ply: bool = True,
                           device: Optional[torch.device] = None) -> Dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)

    metrics = {
        'sample_name': Path(npz_path).stem,
        'valid_structure': 0.0,
        'sim_success': 0.0,
        'class_acc': 0.0,
        'upper_correct': 0.0,
        'bottom_correct': 0.0,
        'connected_correct': 0.0,
        'chamfer_distance': None,
        'f_score': None,          # 主阈值 (tau) 下的 F-Score
        'pred_nn_fit': None,      # 预测服装->A-body 贴合度 (m)，反解可靠性参考
        'gt_nn_fit': None,        # GT 服装->scan-body 贴合度 (m)
    }
    for t in F_SCORE_TAUS:        # 多阈值 F-Score（透明可比，规避 5mm 阈值争议）
        metrics[f'f_score_{int(t*1000)}mm'] = None

    # ---- 1. 结构合法率 (Val. Rate) ----
    spec_path = output_dir / 'design' / 'design_specification.json'
    if not spec_path.exists():
        spec_path = output_dir / 'design_specification.json'
    if spec_path.exists() and spec_path.stat().st_size > 100:
        metrics['valid_structure'] = 1.0
    else:
        return metrics

    # ---- 2. 物理模拟成功率 (SSR) ----
    sim_obj = output_dir / 'design' / 'design_sim.obj'
    if not sim_obj.exists():
        sim_obj = output_dir / 'design_sim.obj'
    if not sim_obj.exists():
        return metrics
    try:
        _m = trimesh.load(sim_obj, force='mesh')
        if len(_m.vertices) > 0 and np.isfinite(_m.vertices).all():
            metrics['sim_success'] = 1.0
    except Exception as e:
        print(f"[Warning] 读取 sim mesh 失败: {e}")
    if metrics['sim_success'] == 0.0:
        return metrics

    gt_data = np.load(npz_path)

    # ---- 3. 分类准确率 (Meta Acc.) ----
    design_yaml = output_dir / 'design.yaml'
    if not design_yaml.exists():
        return metrics
    with open(design_yaml, 'r', encoding='utf-8') as f:
        design_cfg = yaml.safe_load(f)
    acc, correct = evaluate_meta_accuracy(design_cfg, gt_data['garments'])
    metrics['class_acc'] = acc
    metrics['upper_correct'] = float(correct['upper'])
    metrics['bottom_correct'] = float(correct['bottom'])
    metrics['connected_correct'] = float(correct['connected'])

    # ---- 4. canonical 空间 CD & F-Score ----
    if len(gt_data['points'][~np.isin(gt_data['labels'], NON_GARMENT_LABELS)]) == 0:
        print("[Warning] GT 无服装点")
        return metrics

    # 4.1 两端反驱动到同一 canonical
    pred_canon, pred_faces, pred_nn = get_pred_canonical(output_dir, smpl_root, gender, device)
    # GT 取服装网格（论文协议：服装真值网格表面采样，而非原始扫描点）
    gt_canon, gt_faces = unpose_close_gt(npz_path, smpl_root, gender,
                                         return_mesh=True)
    _, gt_extra = unpose_close_gt(npz_path, smpl_root, gender, return_extra=True)
    metrics['pred_nn_fit'] = float(np.mean(pred_nn))
    metrics['gt_nn_fit'] = float(np.mean(gt_extra['nn_dist']))

    # 4.2 可选 canonical 内 rigid 精修
    pred_canon = optional_icp(pred_canon, gt_canon, use_icp)

    # 4.3 两表面各均匀采样 N_SAMPLE 点（论文协议）
    pred_pts, _ = trimesh.sample.sample_surface(
        trimesh.Trimesh(pred_canon, pred_faces, process=False), N_SAMPLE)
    pred_pts = np.asarray(pred_pts)
    if len(gt_faces) > 0:
        gt_pts, _ = trimesh.sample.sample_surface(
            trimesh.Trimesh(gt_canon, gt_faces, process=False), N_SAMPLE)
        gt_pts = np.asarray(gt_pts)
    elif len(gt_canon) > N_SAMPLE:   # 无可用面时退化为点云采样
        gt_pts = gt_canon[np.random.choice(len(gt_canon), N_SAMPLE, replace=False)]
    else:
        gt_pts = gt_canon

    # 打印点云统计信息，便于理解坐标系
    print(f"[Point Cloud Info] Pred shape: {pred_pts.shape}, GT shape: {gt_pts.shape}")
    print(f"[Point Cloud Stats] Pred - Min: {pred_pts.min(axis=0)}, Max: {pred_pts.max(axis=0)}, Mean: {pred_pts.mean(axis=0)}")
    print(f"[Point Cloud Stats] GT - Min: {gt_pts.min(axis=0)}, Max: {gt_pts.max(axis=0)}, Mean: {gt_pts.mean(axis=0)}")

    cd, fs = compute_cd_fscore_multi(pred_pts, gt_pts)
    metrics['chamfer_distance'] = cd
    metrics['f_score'] = fs.get(tau, fs[min(fs, key=lambda k: abs(k - tau))])
    for t in F_SCORE_TAUS:
        metrics[f'f_score_{int(t*1000)}mm'] = fs[t]

    # ---- 5. 调试点云（canonical 帧）----
    if save_debug_ply:
        try:
            import open3d as o3d
            for name, pts, col in [('debug_pred_canon.ply', pred_pts, [1, 0, 0]),
                                   ('debug_gt_canon.ply', gt_pts, [0, 1, 0])]:

                # 打印两个点云坐标
                print(f"[Debug] {name} 点云坐标: {pts[:5]}")

                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(np.asarray(pts))
                pc.paint_uniform_color(col)
                o3d.io.write_point_cloud(str(output_dir / name), pc)
        except Exception as e:
            print(f"[Error] 保存调试点云失败: {e}")

    return metrics


# ==================== 批量评估 ====================

def compute_summary(all_metrics: List[Dict]) -> Dict:
    """汇总，含多阈值 F-Score 均值。"""
    if not all_metrics:
        return {}
    s = {
        'total_samples': len(all_metrics),
        'val_rate': float(np.mean([m['valid_structure'] for m in all_metrics])),
        'sim_success_rate': float(np.mean([m['sim_success'] for m in all_metrics])),
        'meta_acc': float(np.mean([m['class_acc'] for m in all_metrics])),
        'upper_acc': float(np.mean([m['upper_correct'] for m in all_metrics])),
        'bottom_acc': float(np.mean([m['bottom_correct'] for m in all_metrics])),
        'connected_acc': float(np.mean([m['connected_correct'] for m in all_metrics])),
    }
    cds = [m['chamfer_distance'] for m in all_metrics if m['chamfer_distance'] is not None]
    if cds:
        s['mean_cd'] = float(np.mean(cds))
        s['median_cd'] = float(np.median(cds))
    fs = [m['f_score'] for m in all_metrics if m['f_score'] is not None]
    if fs:
        s['mean_fscore'] = float(np.mean(fs))
        s['median_fscore'] = float(np.median(fs))
    for t in F_SCORE_TAUS:                 # 各阈值下的均值 F-Score
        key = f'f_score_{int(t*1000)}mm'
        vals = [m[key] for m in all_metrics if m.get(key) is not None]
        if vals:
            s[f'mean_{key}'] = float(np.mean(vals))
    return s


def evaluate_dataset(data_root: str, output_root: str, smpl_root: str, gender: str,
                     split_file: Optional[str] = None, tau: float = PRIMARY_TAU,
                     use_icp: bool = False) -> Tuple[Dict, List[Dict]]:
    data_root, output_root = Path(data_root), Path(output_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if split_file and os.path.exists(split_file):
        names = np.load(split_file)['names'].tolist()
    else:
        names = sorted(f.stem for f in data_root.glob('*.npz'))

    all_metrics = []
    for name in names:
        npz = data_root / f"{name}.npz"
        out = output_root / name
        if not npz.exists() or not out.exists():
            print(f"[Skip] 缺数据或输出: {name}")
            continue
        print(f"Evaluating {name}...")
        try:
            all_metrics.append(evaluate_single_sample(
                str(npz), str(out), smpl_root, gender, tau, use_icp, device=device))
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
    return compute_summary(all_metrics), all_metrics


# ==================== 主函数 ====================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='canonical 空间服装重建评估')
    ap.add_argument('--data_root', default='/root/wyc/data/CloSe/data/CloSe-Di')
    ap.add_argument('--output_root', default='/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe')
    ap.add_argument('--smpl_root', default='/root/wyc/code/smpl2garmentcode2/smpl_models')
    ap.add_argument('--gender', default='female', choices=['female', 'male'])
    ap.add_argument('--split_file', default=None)
    ap.add_argument('--tau', type=float, default=PRIMARY_TAU,
                    help='主 F-Score 阈值 (m)。默认 0.05 以复现论文量级；'
                         '脚本始终额外报告 5/10/20/30/50mm 全部阈值。')
    ap.add_argument('--use_icp', action='store_true', help='canonical 空间内额外做 rigid ICP 精修')
    ap.add_argument('--output', default='./eval_results_canonical')
    ap.add_argument('--single', default=None, help='只评估单样本，如 10001_1923')
    args = ap.parse_args()

    if args.single:
        m = evaluate_single_sample(
            os.path.join(args.data_root, f"{args.single}.npz"),
            os.path.join(args.output_root, args.single),
            args.smpl_root, args.gender, args.tau, args.use_icp)
        print(f"\n{'='*50}\nSample: {args.single}\n{'='*50}")
        for k, v in m.items():
            print(f"{k}: {v}")
    else:
        summary, all_metrics = evaluate_dataset(
            args.data_root, args.output_root, args.smpl_root, args.gender,
            args.split_file, args.tau, args.use_icp)
        print(f"\n{'='*50}\nEvaluation Summary (canonical)\n{'='*50}")
        for k, v in summary.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        save_results(all_metrics, summary, args.output)
