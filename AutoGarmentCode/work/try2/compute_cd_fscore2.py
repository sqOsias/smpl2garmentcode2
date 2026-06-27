"""将预测的garment转换到GT pose下，计算CD和F-Score指标"""

import os
import sys
import torch
import numpy as np
import trimesh
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import open3d as o3d
DEFAULT_TAU = 0.005  # F-Score阈值，单位米
NON_GARMENT_LABELS = [0, 1, 10, 12, 13, 14, 15]
N_SAMPLE = 50000
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
          f"CD={cd_cm:.3f}cm F@{DEFAULT_TAU*1000:.0f}mm={f_score:.4f}")
    return cd_cm, f_score

def close_points_to_metric(points: np.ndarray, scale: float, trans: np.ndarray) -> np.ndarray:
    """归一化扫描坐标 -> SMPL 米制坐标 (transl=0 人体所在帧)。"""
    return points / scale - trans.reshape(1, 3)

def rigid_align(pred_garment_v: np.ndarray,
                pred_faces: np.ndarray,
                gt_data: dict,
                pred_body_obj: Optional[str] = None) -> np.ndarray:
    """以 SMPL 人体躯干中心为参照做刚性对齐（均在 metric 坐标系）。

    不再使用 garment hem → body feet 对齐（对短款服装不可靠），
    改为：pred body 躯干中心 → GT body 躯干中心 对齐，保证服装穿在身上的位置一致。
    """
    gt_labels = gt_data['labels']
    gt_points = gt_data['points']
    gt_scale = gt_data['scale']
    gt_trans = gt_data['trans']
    # 将 GT body 转换到 metric 坐标系
    gt_body_raw = gt_points[gt_labels == 1]
    gt_body = gt_body_raw / gt_scale - gt_trans.reshape(1, 3)

    aligned = pred_garment_v.copy()
    if (np.max(aligned[:, 1]) - np.min(aligned[:, 1])) > 5.0:
        aligned = aligned / 100.0

    # ---- 使用身体躯干中心对齐 ----
    # GT body 躯干中心（用 Y 中段作为 torso 近似）
    gt_y_mid = (gt_body[:, 1].max() + gt_body[:, 1].min()) / 2.0
    gt_torso = gt_body[(gt_body[:, 1] > gt_y_mid - 0.2) & (gt_body[:, 1] < gt_y_mid + 0.2)]
    if len(gt_torso) > 0:
        gt_cx = np.mean(gt_torso[:, 0])
        gt_cy = np.mean(gt_torso[:, 1])
        gt_cz = np.mean(gt_torso[:, 2])
    else:
        gt_cx = np.mean(gt_body[:, 0])
        gt_cy = np.mean(gt_body[:, 1])
        gt_cz = np.mean(gt_body[:, 2])

    # Pred body 躯干中心（如果提供了 pred body OBJ）
    if pred_body_obj is not None and os.path.exists(pred_body_obj):
        pred_body = trimesh.load(pred_body_obj, process=False)
        pred_bv = np.array(pred_body.vertices, dtype=np.float64)
        # 也缩放到米
        if (np.max(pred_bv[:, 1]) - np.min(pred_bv[:, 1])) > 5.0:
            pred_bv = pred_bv / 100.0
        py_mid = (pred_bv[:, 1].max() + pred_bv[:, 1].min()) / 2.0
        pred_torso = pred_bv[(pred_bv[:, 1] > py_mid - 0.2) & (pred_bv[:, 1] < py_mid + 0.2)]
        if len(pred_torso) > 0:
            pr_cx = np.mean(pred_torso[:, 0])
            pr_cy = np.mean(pred_torso[:, 1])
            pr_cz = np.mean(pred_torso[:, 2])
        else:
            pr_cx = np.mean(pred_bv[:, 0])
            pr_cy = np.mean(pred_bv[:, 1])
            pr_cz = np.mean(pred_bv[:, 2])
    else:
        # 回退：用 garment 本身的中心
        pr_cx = (aligned[:, 0].max() + aligned[:, 0].min()) / 2.0
        pr_cy = (aligned[:, 1].max() + aligned[:, 1].min()) / 2.0
        pr_cz = (aligned[:, 2].max() + aligned[:, 2].min()) / 2.0

    # 执行平移对齐（基于躯干中心）
    aligned[:, 0] += (gt_cx - pr_cx)
    aligned[:, 1] += (gt_cy - pr_cy)
    aligned[:, 2] += (gt_cz - pr_cz)
    print(f"[Align] pred torso center=({pr_cx:.3f}, {pr_cy:.3f}, {pr_cz:.3f}) "
          f"-> GT torso center=({gt_cx:.3f}, {gt_cy:.3f}, {gt_cz:.3f}) "
          f"| shift=({gt_cx - pr_cx:.3f}, {gt_cy - pr_cy:.3f}, {gt_cz - pr_cz:.3f})")
    print(f"[Align] pred garment Y after=[{aligned[:,1].min():.3f}, {aligned[:,1].max():.3f}] "
          f"| GT garment Y=[{gt_points[~np.isin(gt_labels, NON_GARMENT_LABELS)][:,1].min()/gt_scale - gt_trans[1]:.3f}, "
          f"{gt_points[~np.isin(gt_labels, NON_GARMENT_LABELS)][:,1].max()/gt_scale - gt_trans[1]:.3f}]")
    return aligned

def compute_cd_fscore2() -> Tuple[float, float]:
    npz_path = "/root/wyc/data/CloSe/data/CloSe-Di/10001_1937.npz"
    gt_data = np.load(npz_path)
    gt_labels = gt_data['labels']
    gt_points = gt_data['points']
    gt_scale = gt_data['scale']
    gt_trans = gt_data['trans']


    gt_garment = gt_points[~np.isin(gt_labels, NON_GARMENT_LABELS)]
    gt_garment = close_points_to_metric(gt_garment, gt_scale, gt_trans)
    if len(gt_garment) == 0:
        print("[Warning] no garment points in GT")
        return 

    sim_obj_path = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/try2/output/final_result.obj"
    output_dir = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/try2/"
    pred_mesh = trimesh.load(sim_obj_path, process=False)
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


    print(f"[Info] Pred shape: {pred_pts.shape}, GT shape: {gt_pts.shape}")

    cd, f_score = compute_cd_fscore(pred_pts, gt_pts, DEFAULT_TAU)
    print(f"[Metric] CD={cd:.3f}cm F@{DEFAULT_TAU*1000:.0f}mm={f_score:.4f}")

    save_debug_ply = 1
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
                output_dir = Path(output_dir)
                o3d.io.write_point_cloud(str(output_dir / name), pc)
        except Exception as e:
            print(f"[Error] 保存调试点云失败: {e}")

if __name__ == "__main__":
    compute_cd_fscore2()