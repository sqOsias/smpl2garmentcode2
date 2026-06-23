"""

"""

import os
import yaml
import json
import torch
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv

# ==================== 1. CloSe 标签定义 ====================

# CloSe garments 18 维二值向量 / labels 的语义索引
CLOTH_LABELS = {
    0: 'Hat', 1: 'Body', 2: 'Shirt', 3: 'TShirt', 4: 'Vest',
    5: 'Coat', 6: 'Dress', 7: 'Skirt', 8: 'Pants', 9: 'ShortPants',
    10: 'Shoes', 11: 'Hoodies', 12: 'Hair', 13: 'Swimwear',
    14: 'Underwear', 15: 'Scarf', 16: 'Jumpsuits', 17: 'Jacket'
}

# 非服装区域（身体/配饰/毛发等），计算服装真值网格时排除
NON_GARMENT_LABELS = [0, 1, 10, 12, 13, 14, 15]

# F-Score 距离阈值：论文 τ = 5mm = 0.005 m
DEFAULT_TAU = 0.005

# 采样点数：论文在每个表面均匀采样 10000 点
N_SAMPLE = 10000


# ==================== 2. 分类准确率 (Meta Acc.) ====================

# CloSe 上装细类索引
UPPER_LABELS = [2, 3, 4, 5, 11, 17]  # Shirt/TShirt/Vest/Coat/Hoodies/Jacket


def _node_v(node, default=None):
    """从 design 节点（{v:..} 或裸值）取值。"""
    if isinstance(node, dict):
        return node.get('v', default)
    return node if node is not None else default


def derive_upper_classes(design_cfg: dict) -> Optional[set]:
    """从完整 design 推导预测上装对应的 CloSe 细类集合。

    `meta.upper` 仅有 {FittedShirt, Shirt}，无法直接区分细类，故由
    sleeve / collar.component 等字段派生：
      - 风帽 (Hood2Panels)      → {Hoodies}
      - 翻领 (SimpleLapel)      → {Coat, Jacket}  （GarmentCode 无法严格区分二者）
      - 无袖 (sleeveless)       → {Vest}
      - 短袖 (sleeve.length<0.45) → {TShirt}
      - 其余                     → {Shirt}
    返回可接受类别集合；无上装时返回 None。
    """
    design = design_cfg.get('design', {})
    meta = design.get('meta', {})
    upper = _node_v(meta.get('upper'))
    if upper in [None, 'null', 'None']:
        return None

    comp_style = _node_v(design.get('collar', {}).get('component', {}).get('style'))
    sleeve = design.get('sleeve', {})
    sleeveless = _node_v(sleeve.get('sleeveless'), False) in [True, 'true', 'True']
    sleeve_len = _node_v(sleeve.get('length'))

    if comp_style == 'Hood2Panels':
        return {11}                 # Hoodies
    if comp_style == 'SimpleLapel':
        return {5, 17}              # Coat / Jacket（翻领外套类）
    if sleeveless:
        return {4}                  # Vest
    if isinstance(sleeve_len, (int, float)) and sleeve_len < 0.45:
        return {3}                  # TShirt
    return {2}                      # Shirt


def evaluate_meta_accuracy(design_cfg: dict,
                           gt_garments_binary: np.ndarray) -> Tuple[float, dict]:
    """对 meta 的 upper / bottom / 腰带(wb) / 连体(connected) 四项
    分类参数，与数据集语义标注比对，返回正确率 (correct/4) 与逐项明细。

    upper 采用细类匹配：从完整 design 派生上装 CloSe 细类，与 GT 标注的上装细类求交，交集非空即视为正确。
    """
    meta = design_cfg.get('design', {}).get('meta', {})

    def _v(key, default=None):
        return _node_v(meta.get(key), default)

    pred_bottom = _v('bottom')
    # pred_wb = _v('wb')
    pred_connected = _v('connected', False)

    g = gt_garments_binary
    gt_has_dress = g[6] == 1
    gt_has_jumpsuit = g[16] == 1
    gt_has_skirt = g[7] == 1
    gt_has_pants = g[8] == 1
    gt_has_shortpants = g[9] == 1
    gt_connected = bool(gt_has_dress or gt_has_jumpsuit)

    correct = {'upper': False, 'bottom': False, 'wb': False, 'connected': False}

    # 连体标志
    pred_connected_bool = pred_connected in [True, 'true', 'True']
    if pred_connected_bool == gt_connected:
        correct['connected'] = True

    # 上装细类匹配
    gt_upper_classes = {idx for idx in UPPER_LABELS if g[idx] == 1}
    pred_upper_classes = derive_upper_classes(design_cfg)
    if not gt_upper_classes:
        # GT 无独立上装：连体装(上装并入连体件)或纯下装。
        # 预测也无独立上装、或预测为连体装(上装模块随连体激活) 即正确。
        correct['upper'] = (pred_upper_classes is None) or gt_connected
    else:
        # GT 有上装：预测须有上装且派生细类与 GT 细类有交集
        correct['upper'] = bool(pred_upper_classes) and bool(pred_upper_classes & gt_upper_classes)

    # 下装类型（skirt / pants / 无）
    if gt_has_dress or gt_has_skirt:
        gt_bottom_type = 'skirt'
    elif gt_has_jumpsuit or gt_has_pants or gt_has_shortpants:
        gt_bottom_type = 'pants'
    else:
        gt_bottom_type = None

    pred_bottom_type = None
    if pred_bottom not in [None, 'null', 'None']:
        if 'Skirt' in str(pred_bottom):
            pred_bottom_type = 'skirt'
        elif 'Pants' in str(pred_bottom):
            pred_bottom_type = 'pants'
    if pred_bottom_type == gt_bottom_type:
        correct['bottom'] = True

    # 腰带：有下装即应有腰带 TODO 这里可能有问题，需要更仔细的判断
    # gt_has_wb = gt_bottom_type is not None
    # pred_has_wb = pred_wb not in [None, 'null', 'None']
    # if pred_has_wb == gt_has_wb:
    #     correct['wb'] = True

    accuracy = sum(correct.values()) / 3.0
    return accuracy, correct


# ==================== 3. 基于 SMPL 骨骼的刚性对齐 ====================

def rigid_align(pred_garment_v: np.ndarray,
                pred_faces: np.ndarray,
                gt_data: dict,
                pred_body_obj: Optional[str] = None) -> np.ndarray:
    """以共同 SMPL 人体为参照做刚性对齐。

    步骤：
      1) 用预测/真值 SMPL 人体把脚底高度(y)与 (x,z) 中心对齐 —— 平移；
      2) 在服装点云上做 rigid ICP（仅旋转+平移，不缩放）做精修。

    pred_garment_v: 预测服装顶点 (米)。
    pred_body_obj : 预测所用 SMPL 人体 obj 路径（A-pose, 米），用于稳健平移基准；
                    缺失时退化为用服装包围盒中心对齐。
    """
    gt_labels = gt_data['labels']
    gt_points = gt_data['points']
    gt_body = gt_points[gt_labels == 1]

    aligned = pred_garment_v.copy()
    if (np.max(aligned[:, 1]) - np.min(aligned[:, 1])) > 5.0:
        aligned = aligned / 100.0

    # ---- 1. 绝对平移对齐 ----
    if len(gt_body) > 0:
        gt_feet_y = np.percentile(gt_body[:, 1], 2)
        gt_cx = np.mean(gt_body[:, 0])
        gt_cz = np.mean(gt_body[:, 2])
    else:
        gt_feet_y = np.percentile(gt_points[:, 1], 2)
        gt_cx = np.mean(gt_points[:, 0])
        gt_cz = np.mean(gt_points[:, 2])

    pr_feet_y = np.percentile(aligned[:, 1], 2)
    pr_cx = (aligned[:, 0].max() + aligned[:, 0].min()) / 2.0
    pr_cz = (aligned[:, 2].max() + aligned[:, 2].min()) / 2.0

    # 执行绝对平移对齐
    aligned[:, 0] += (gt_cx - pr_cx)
    aligned[:, 2] += (gt_cz - pr_cz)
    aligned[:, 1] += (gt_feet_y - pr_feet_y)

    # ---- 2. rigid ICP 精修（不缩放，保持物理尺度）
    # TODO 这里做一下消融尝试:cd 升高，fscore升高（不多）----
    # try:
    #     from trimesh.registration import icp
    #     src, _ = trimesh.sample.sample_surface(trimesh.Trimesh(aligned, pred_faces), 5000)
    #     if len(gt_garment) > 20000:
    #         ref = gt_garment[np.random.choice(len(gt_garment), 20000, replace=False)]
    #     else:
    #         ref = gt_garment
    #     T, _, _ = icp(src, ref, max_iterations=40, scale=False) # 修正人物转身、身体朝向带来的旋转偏差
    #     aligned = trimesh.transformations.transform_points(aligned, T) # 通过ICP 最近点迭代配准求解最优刚性变换矩阵
    # except Exception as e:
    #     print(f"[Warning] ICP failed: {e}")

    return aligned


# ==================== 4. 倒角距离 & F-Score ====================

def compute_cd_fscore(pred_pts: np.ndarray,
                      gt_pts: np.ndarray,
                      tau: float = DEFAULT_TAU) -> Tuple[float, float]:
    P = torch.tensor(pred_pts, dtype=torch.float32).unsqueeze(0)
    Q = torch.tensor(gt_pts, dtype=torch.float32).unsqueeze(0)
    D = torch.cdist(P, Q, p=2).squeeze(0)
    pred_to_gt = D.min(dim=1).values # 计算每个预测点到最近真实点的距离
    gt_to_pred = D.min(dim=0).values # 计算每个真实点到最近预测点的距离

    # 公式：1/2 * (mean(p->q) + mean(q->p))，米 → cm
    cd_cm = 0.5 * (pred_to_gt.mean() + gt_to_pred.mean()).item() * 100.0

    precision = (pred_to_gt < tau).float().mean().item()
    recall = (gt_to_pred < tau).float().mean().item()
    f_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"[Metric] P->GT={pred_to_gt.mean().item()*100:.3f}cm "
          f"GT->P={gt_to_pred.mean().item()*100:.3f}cm | "
          f"P={precision*100:.2f}% R={recall*100:.2f}% | "
          f"CD={cd_cm:.3f}cm F@{tau*1000:.0f}mm={f_score:.4f}")
    return cd_cm, f_score


# ==================== 5. 单样本评估 ====================


def evaluate_single_sample(npz_path: str,
                           output_dir: str,
                           tau: float = DEFAULT_TAU,
                           save_debug_ply: bool = True) -> Dict:
    """评估单个样本，返回论文五项指标 + 分类明细。"""
    metrics = {
        'sample_name': Path(npz_path).stem,
        'valid_structure': 0.0,   # Val. Rate
        'sim_success': 0.0,       # SSR
        'class_acc': 0.0,         # Meta Acc.
        'upper_correct': 0.0,
        'bottom_correct': 0.0,
        # 'wb_correct': 0.0,
        'connected_correct': 0.0,
        'chamfer_distance': None,  # CD (cm)
        'f_score': None,           # F-Score @ tau
    }

    # ---- 1. 结构合法率 (Val. Rate)：spec 可被生成并通过校验 ----
    design_dir = os.path.join(output_dir, 'design')
    spec_path = os.path.join(design_dir, 'design_specification.json')
    if not os.path.exists(spec_path):
        spec_path = os.path.join(output_dir, 'design_specification.json')
    if os.path.exists(spec_path) and os.path.getsize(spec_path) > 100:
        metrics['valid_structure'] = 1.0
    else:
        return metrics

    # ---- 2. 物理模拟成功率 (SSR)：sim 网格存在且有限、非空 ----
    sim_obj_path = os.path.join(design_dir, 'design_sim.obj')
    if not os.path.exists(sim_obj_path):
        sim_obj_path = os.path.join(output_dir, 'design_sim.obj')
    pred_mesh = None
    if os.path.exists(sim_obj_path):
        try:
            pred_mesh = trimesh.load(sim_obj_path, force='mesh')
            pred_mesh.apply_scale(0.01)  # GarmentCode 输出为 cm → m
            if len(pred_mesh.vertices) > 0 and np.isfinite(pred_mesh.vertices).all():
                metrics['sim_success'] = 1.0
        except Exception as e:
            print(f"[Warning] loading sim mesh failed: {e}")
    if metrics['sim_success'] == 0.0:
        return metrics

    gt_data = np.load(npz_path)

    # ---- 3. 分类准确率 (Meta Acc.) ----
    design_yaml_path = os.path.join(output_dir, 'design.yaml')
    if not os.path.exists(design_yaml_path):
        return metrics
    with open(design_yaml_path, 'r', encoding='utf-8') as f:
        design_cfg = yaml.safe_load(f)
    acc, correct = evaluate_meta_accuracy(design_cfg, gt_data['garments'])
    metrics['class_acc'] = acc
    metrics['upper_correct'] = float(correct['upper'])
    metrics['bottom_correct'] = float(correct['bottom'])
    # metrics['wb_correct'] = float(correct['wb'])
    metrics['connected_correct'] = float(correct['connected'])

    # ---- 4. CD & F-Score ----
    gt_labels = gt_data['labels']
    gt_points = gt_data['points']


    gt_garment = gt_points[~np.isin(gt_labels, NON_GARMENT_LABELS)]
    if len(gt_garment) == 0:
        print("[Warning] no garment points in GT")
        return metrics

    pred_body_obj = os.path.join(output_dir, 'smpl.obj')
    pred_aligned_v = rigid_align(pred_mesh.vertices, pred_mesh.faces,
                                 gt_data, pred_body_obj)

    # 两表面各均匀采样 10000 点
    pred_pts, _ = trimesh.sample.sample_surface(
        trimesh.Trimesh(pred_aligned_v, pred_mesh.faces), N_SAMPLE)
    if len(gt_garment) > N_SAMPLE:
        gt_pts = gt_garment[np.random.choice(len(gt_garment), N_SAMPLE, replace=False)]
    else:
        gt_pts = gt_garment

    cd, f_score = compute_cd_fscore(pred_pts, gt_pts, tau)
    metrics['chamfer_distance'] = cd
    metrics['f_score'] = f_score

    if save_debug_ply:
        try:
            import open3d as o3d
            for name, pts, col in [('debug_pred.ply', pred_pts, [1, 0, 0]),
                                   ('debug_gt.ply', gt_pts, [0, 1, 0])]:
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(pts)
                pc.paint_uniform_color(col)
                o3d.io.write_point_cloud(os.path.join(output_dir, name), pc)
                print(f"{output_dir}/{name} saved for debugging.")
        except Exception:
            pass

    return metrics


# ==================== 6. 批量评估与汇总 ====================

def compute_summary(all_metrics: List[Dict]) -> Dict:
    if not all_metrics:
        return {}
    summary = {
        'total_samples': len(all_metrics),
        'val_rate': float(np.mean([m['valid_structure'] for m in all_metrics])),
        'sim_success_rate': float(np.mean([m['sim_success'] for m in all_metrics])),
        'meta_acc': float(np.mean([m['class_acc'] for m in all_metrics])),
        'upper_acc': float(np.mean([m['upper_correct'] for m in all_metrics])),
        'bottom_acc': float(np.mean([m['bottom_correct'] for m in all_metrics])),
        # 'wb_acc': float(np.mean([m['wb_correct'] for m in all_metrics])),
        'connected_acc': float(np.mean([m['connected_correct'] for m in all_metrics])),
    }
    cds = [m['chamfer_distance'] for m in all_metrics if m['chamfer_distance'] is not None]
    fs = [m['f_score'] for m in all_metrics if m['f_score'] is not None]
    if cds:
        summary['mean_cd'] = float(np.mean(cds))
        summary['median_cd'] = float(np.median(cds))
    if fs:
        summary['mean_fscore'] = float(np.mean(fs))
        summary['median_fscore'] = float(np.median(fs))
    return summary


def evaluate_dataset(data_root: str, output_root: str,
                     split_file: Optional[str] = None,
                     tau: float = DEFAULT_TAU) -> Tuple[Dict, List[Dict]]:
    data_root = Path(data_root)
    output_root = Path(output_root)
    if split_file and os.path.exists(split_file):
        sample_names = np.load(split_file)['names'].tolist()
    else:
        sample_names = [f.stem for f in data_root.glob('*.npz')]

    all_metrics = []
    for name in sample_names:
        npz_path = data_root / f"{name}.npz"
        output_dir = output_root / name
        if not npz_path.exists() or not output_dir.exists():
            print(f"[Skip] missing data or output: {name}")
            continue
        print(f"Evaluating {name}...")
        all_metrics.append(evaluate_single_sample(str(npz_path), str(output_dir), tau))
    return compute_summary(all_metrics), all_metrics


def save_results(all_metrics: List[Dict], summary: Dict, output_file: str):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file.with_suffix('.json'), 'w') as f:
        json.dump({'summary': summary, 'samples': all_metrics}, f, indent=2)
    if all_metrics:
        with open(output_file.with_suffix('.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
    print(f"save results to: {output_file.with_suffix('.json')}")


# ==================== 7. 主函数 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='evaluate garment reconstruction results')
    parser.add_argument('--data_root', type=str,
                        default='/root/wyc/data/CloSe/data/CloSe-Di')
    parser.add_argument('--output_root', type=str,
                        default='/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe')
    parser.add_argument('--split_file', type=str, default=None)
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU,
                        help='F-Score distance threshold (m), default 0.005 (5mm)')
    parser.add_argument('--output', type=str, default='./eval_results')
    parser.add_argument('--single', type=str, default="10001_1937",
                        help='single sample evaluation, pass sample name like 10001_1937')
    args = parser.parse_args()

    if args.single:
        npz_path = os.path.join(args.data_root, f"{args.single}.npz")
        output_dir = os.path.join(args.output_root, args.single)
        metrics = evaluate_single_sample(npz_path, output_dir, args.tau)
        print(f"\n{'='*50}\nSample: {args.single}\n{'='*50}")
        for k, v in metrics.items():
            print(f"{k}: {v}")
    else:
        summary, all_metrics = evaluate_dataset(
            args.data_root, args.output_root, args.split_file, args.tau)
        print(f"\n{'='*50}\nEvaluation Summary\n{'='*50}")
        for k, v in summary.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        save_results(all_metrics, summary, args.output)

