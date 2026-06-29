"""
基于 SMPL 躯干顶点对应关系的 Kabsch 刚性对齐评估脚本。
利用 SMPL 顶点的一一对应关系（同索引 = 同身体位置），
在躯干区域做最优刚性变换，应用到服装上。
"""
import os
import csv
import json
import dataclasses
from typing import Tuple, Optional

import numpy as np
import torch
import trimesh
import smplx
import open3d as o3d


# ==================== 配置 ====================

@dataclasses.dataclass
class Config:
    smpl_model_path: str = "/root/wyc/code/smpl2garmentcode2/smpl_models"
    npz_path: str = "/root/wyc/data/CloSe/data/CloSe-Di/10014_2464.npz"
    smpl_json: str = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/metric/smpl.json"
    garment_obj: str = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/metric/result/final_result.obj"
    output_dir: str = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/metric/align"
    n_sample: int = 10000
    non_garment_labels: Tuple[int, ...] = (0, 1, 10, 12, 13, 14, 15)
    gender: str = "male"
    # garment_head_offset: float = 0.22       # 服装最高点低于头顶的米数
    icp_max_corr_dist: float = 0.3
    random_seed: int = 42
    fscore_thresholds_mm: Tuple[int, ...] = (5, 10, 20, 30, 50)


# ==================== 辅助函数 ====================

def section(title: str):
    """打印分节标题。"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def rotation_angle_deg(R: np.ndarray) -> float:
    """从 3x3 旋转矩阵提取旋转角（度）。"""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# ==================== 1. 数据加载 ====================

def load_smpl_params(cfg: Config):
    """加载 GT (npz) 和 Pred (HybrIK json) 的 SMPL 参数。"""
    section("1. 加载 SMPL 参数")

    data = np.load(cfg.npz_path)
    gt_betas = data['betas'].astype(np.float32)
    gt_pose = data['pose'].astype(np.float32)
    gt_trans = data['trans'].astype(np.float32)

    with open(cfg.smpl_json) as f:
        pred_json = json.load(f)
    pred_betas = np.array(pred_json['betas'], dtype=np.float32)
    hybrik_pose = np.array(pred_json['pose'], dtype=np.float32)

    # GT pose 拆分
    gt_pose_t = torch.from_numpy(gt_pose).float().view(1, 72)
    gt_body_pose = gt_pose_t[:, 3:]
    gt_global_orient = gt_pose_t[:, :3]

    # Pred: HybrIK body pose + 零 global orient (A-pose)
    pred_body_pose = torch.from_numpy(hybrik_pose).float().view(1, 69)
    pred_global_orient = torch.zeros(1, 3)

    print(f"  GT  betas: {gt_betas.shape}, pose: {gt_pose.shape}, trans: {gt_trans}")
    print(f"  Pred betas: {pred_betas.shape}, hybrik_pose: {hybrik_pose.shape}")

    return {
        'gt_betas': gt_betas,
        'gt_body_pose': gt_body_pose,
        'gt_global_orient': gt_global_orient,
        'gt_trans': gt_trans,
        'pred_betas': pred_betas,
        'pred_body_pose': pred_body_pose,
        'pred_global_orient': pred_global_orient,
    }


# ==================== 2. SMPL 人体生成 ====================

def build_smpl_bodies(cfg: Config, smpl_params: dict):
    """用 SMPL 模型生成 GT 和 Pred 人体顶点。均不传 transl，保持在 SMPL 原生坐标系。"""
    section("2. 生成 SMPL 人体网格")

    model = smplx.create(cfg.smpl_model_path, model_type='smpl', gender=cfg.gender)

    with torch.no_grad():
        gt_out = model(
            betas=torch.from_numpy(smpl_params['gt_betas']).float().unsqueeze(0),
            body_pose=smpl_params['gt_body_pose'],
            global_orient=smpl_params['gt_global_orient'],
        )
        gt_body_v = gt_out.vertices.squeeze().numpy()

        pred_out = model(
            betas=torch.from_numpy(smpl_params['pred_betas']).float().unsqueeze(0),
            body_pose=smpl_params['pred_body_pose'],
            global_orient=smpl_params['pred_global_orient'],
        )
        pred_body_v = pred_out.vertices.squeeze().numpy()

    print(f"  GT   body: {gt_body_v.shape}, Y=[{gt_body_v[:, 1].min():.3f}, {gt_body_v[:, 1].max():.3f}]")
    print(f"  Pred body: {pred_body_v.shape}, Y=[{pred_body_v[:, 1].min():.3f}, {pred_body_v[:, 1].max():.3f}]")

    return gt_body_v, pred_body_v


# ==================== 3. Kabsch 刚性对齐 ====================

def get_torso_mask(verts: np.ndarray) -> np.ndarray:
    """返回躯干顶点的 bool mask。基于 Y 轴中段 + X 轴中心区域筛选。"""
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    y_mid = (y_max + y_min) / 2.0
    y_range = y_max - y_min
    mask_y = (verts[:, 1] > y_mid - 0.25 * y_range) & (verts[:, 1] < y_mid + 0.25 * y_range)

    torso = verts[mask_y]
    x_mid = torso[:, 0].mean()
    x_range = torso[:, 0].max() - torso[:, 0].min()
    mask_x = np.abs(verts[:, 0] - x_mid) < 0.2 * x_range

    return mask_y & mask_x


def kabsch_align(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kabsch 算法：已知一一对应关系的最优刚性变换。

    Args:
        source: (N, 3) 源点集
        target: (N, 3) 目标点集（同索引 = 对应点）

    Returns:
        R_mat: (3, 3) 旋转矩阵
        t_vec: (3,)   平移向量
    """
    src_c = source.mean(axis=0)
    tgt_c = target.mean(axis=0)

    H = (source - src_c).T @ (target - tgt_c)
    U, _, Vt = np.linalg.svd(H)

    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T

    t_vec = tgt_c - src_c @ R_mat.T
    return R_mat, t_vec


def compute_body_alignment(gt_body_v: np.ndarray, pred_body_v: np.ndarray):
    """通过躯干顶点的 Kabsch 对齐计算 Pred→GT 的刚性变换。"""
    section("3. Kabsch 对齐: Pred body → GT body (躯干对应点)")

    torso_mask = get_torso_mask(gt_body_v)
    gt_torso_v = gt_body_v[torso_mask]
    pred_torso_v = pred_body_v[torso_mask]

    print(f"  Torso verts (对应点对): {len(gt_torso_v)}")

    R_mat, t_vec = kabsch_align(pred_torso_v, gt_torso_v)

    print(f"  Kabsch 旋转角: {rotation_angle_deg(R_mat):.2f}°")
    print(f"  旋转矩阵:\n{R_mat}")
    print(f"  平移向量: {t_vec}")

    return R_mat, t_vec


# ==================== 4. 服装加载与高度修正 ====================

def load_garment(obj_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载服装 OBJ，自动检测并转换厘米→米。"""
    mesh = trimesh.load(obj_path, process=False)
    verts = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces)

    # 如果 Y 轴跨度 > 5 米，说明是厘米单位，转换为米
    if (np.max(verts[:, 1]) - np.min(verts[:, 1])) > 5.0:
        verts /= 100.0

    return verts, faces


def align_garment_height(garment_v: np.ndarray, pred_body_v: np.ndarray, head_offset: float) -> np.ndarray:
    """将服装最高点对齐到人体头顶下方 head_offset 米处。"""
    pred_body_top_y = pred_body_v[:, 1].max()
    target_top_y = pred_body_top_y - head_offset

    current_top_y = garment_v[:, 1].max()
    y_shift = target_top_y - current_top_y

    garment_v = garment_v.copy()
    garment_v[:, 1] += y_shift

    print(f"  [修正] 已将服装 Y 轴平移 {y_shift:.3f} 米，使其最高点达到 {target_top_y:.3f} 米")
    return garment_v


def apply_rigid_transform(verts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """对顶点应用刚性变换: v' = v @ R.T + t。"""
    return (verts @ R.T) + t


# ==================== 5. GT 服装网格构建 (从 NPZ) ====================

def build_gt_garment_mesh(data: dict, non_garment_labels: Tuple[int, ...]) -> trimesh.Trimesh:
    """从 CloSe NPZ 数据中提取 GT 服装网格（SMPL 原生坐标系）。"""
    labels = data['labels']
    points = data['points']
    faces = data['faces']
    scale = float(data['scale'])
    trans = data['trans']

    # 服装顶点 → metric 坐标
    gar_mask = ~np.isin(labels, non_garment_labels)
    gar_pts = points[gar_mask] / scale - trans.reshape(1, 3)

    # 服装面片（三个顶点都是服装标签）
    gar_indices = np.where(gar_mask)[0]
    gar_idx_set = set(gar_indices.tolist())
    gar_face_mask = np.array([
        (f[0] in gar_idx_set) and (f[1] in gar_idx_set) and (f[2] in gar_idx_set)
        for f in faces
    ])
    gar_faces_filtered = faces[gar_face_mask]

    # 重新映射顶点索引
    old_to_new = {old: new for new, old in enumerate(gar_indices)}
    gar_faces_remapped = np.array([
        [old_to_new[f[0]], old_to_new[f[1]], old_to_new[f[2]]]
        for f in gar_faces_filtered
    ])

    mesh = trimesh.Trimesh(gar_pts, gar_faces_remapped, process=False)
    print(f"  GT garment mesh: {len(gar_pts)} verts, {len(gar_faces_remapped)} faces")
    print(f"  GT garment Y=[{gar_pts[:, 1].min():.3f}, {gar_pts[:, 1].max():.3f}]")
    return mesh


# ==================== 6. ICP 精修 ====================

def icp_refine(source_mesh: trimesh.Trimesh,
               target_mesh: trimesh.Trimesh,
               max_corr_dist: float = 0.3,
               n_sample: int = 20000) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """对 source_mesh 做 ICP 精修对齐到 target_mesh。

    Returns:
        R_icp: (3, 3) 旋转矩阵
        t_icp: (3,)   平移向量
        fitness:       ICP fitness
        inlier_rmse:   ICP inlier RMSE (米)
    """
    src_pts, _ = trimesh.sample.sample_surface(source_mesh, n_sample)
    tgt_pts, _ = trimesh.sample.sample_surface(target_mesh, n_sample)

    src_pc = o3d.geometry.PointCloud()
    src_pc.points = o3d.utility.Vector3dVector(src_pts.astype(np.float64))

    tgt_pc = o3d.geometry.PointCloud()
    tgt_pc.points = o3d.utility.Vector3dVector(tgt_pts.astype(np.float64))

    reg = o3d.pipelines.registration.registration_icp(
        src_pc, tgt_pc,
        max_correspondence_distance=max_corr_dist,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100,
        ),
    )

    M = reg.transformation
    R_icp = M[:3, :3]
    t_icp = M[:3, 3]

    print(f"  ICP fitness={reg.fitness:.4f}  inlier_rmse={reg.inlier_rmse * 100:.2f}cm")
    print(f"  ICP 旋转={rotation_angle_deg(R_icp):.2f}deg  平移={t_icp}")

    return R_icp, t_icp, reg.fitness, reg.inlier_rmse


# ==================== 7. 评估指标 ====================

def compute_metrics(pred_pts: np.ndarray, gt_pts: np.ndarray,
                    thresholds_mm: Tuple[int, ...] = (5, 10, 20, 30, 50)):
    """计算 CD 和多阈值 F-Score。

    Returns:
        dict: 包含 cd_cm, fscore@Xmm, p2g, g2p 等
    """
    P = torch.tensor(pred_pts, dtype=torch.float32).unsqueeze(0)
    Q = torch.tensor(gt_pts, dtype=torch.float32).unsqueeze(0)
    D = torch.cdist(P, Q, p=2).squeeze(0)

    p2g = D.min(dim=1).values.numpy()  # pred → GT 最近距离
    g2p = D.min(dim=0).values.numpy()  # GT → pred 最近距离

    cd_cm = 0.5 * (p2g.mean() + g2p.mean()) * 100.0

    print(f"\n  CD      = {cd_cm:.3f} cm")
    print(f"  P->GT   mean={p2g.mean() * 100:.3f}cm  median={np.median(p2g) * 100:.3f}cm")
    print(f"  GT->P   mean={g2p.mean() * 100:.3f}cm  median={np.median(g2p) * 100:.3f}cm")

    fscores = {}
    print(f"\n  多阈值 F-Score:")
    for tau_mm in thresholds_mm:
        tau = tau_mm / 1000.0
        pr = (p2g < tau).mean()
        rc = (g2p < tau).mean()
        fs = (2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
        fscores[tau_mm] = {'fscore': fs, 'precision': pr, 'recall': rc}
        print(f"    F@{tau_mm}mm = {fs:.4f}  (P={pr * 100:.1f}%, R={rc * 100:.1f}%)")

    return {'cd_cm': cd_cm, 'p2g': p2g, 'g2p': g2p, 'fscores': fscores}


def save_metrics_to_csv(metrics: dict, output_dir: str, filename: str = "metrics.csv"):
    """将 CD 和 F-Score 保存到 CSV 文件。

    写入一行：cd_cm, F@5mm, P@5mm, R@5mm, F@10mm, ...
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)

    # 构建列名和数据行
    columns = ["cd_cm"]
    values = [f"{metrics['cd_cm']:.4f}"]

    for tau_mm, fs in sorted(metrics['fscores'].items()):
        columns += [f"F-Score@{tau_mm}mm", f"Precision@{tau_mm}mm", f"Recall@{tau_mm}mm"]
        values += [
            f"{fs['fscore']:.4f}",
            f"{fs['precision']:.4f}",
            f"{fs['recall']:.4f}",
        ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerow(values)

    print(f"\n  Metrics saved to {csv_path}")


def sample_mesh_surface(mesh: trimesh.Trimesh, n_sample: int, seed: int = 42) -> np.ndarray:
    """在 mesh 表面均匀采样 n_sample 个点。"""
    np.random.seed(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n_sample)
    return pts


# ==================== 8. 导出 ====================

def write_ply_ascii(path: str, pts: np.ndarray):
    """手动写 ASCII PLY 文件。"""
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(pts))
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write("%f %f %f\n" % (p[0], p[1], p[2]))


def export_results(pred_mesh: trimesh.Trimesh,
                   pred_pts: np.ndarray,
                   gt_pts: np.ndarray,
                   output_dir: str):
    """导出对齐后网格和采样的调试点云。"""
    section("7. 导出调试文件")

    os.makedirs(output_dir, exist_ok=True)

    # OBJ 网格
    obj_path = os.path.join(output_dir, "garment_smpl_kabsch.obj")
    pred_mesh.export(obj_path)
    print(f"  {obj_path} saved")

    # ASCII PLY 点云
    # for name, pts in [
    #     ('debug_pred_smpl_kabsch.ply', pred_pts),
    #     ('debug_gt_smpl_kabsch.ply', gt_pts),
    # ]:
    #     path = os.path.join(output_dir, name)
    #     write_ply_ascii(path, pts)
    #     print(f"  {path} saved")

    # Open3D 彩色 PLY 点云
    try:
        for name, pts, col in [
            ('debug_pred.ply', pred_pts, [1, 0, 0]),
            ('debug_gt.ply', gt_pts, [0, 1, 0]),
        ]:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(np.asarray(pts))
            pc.paint_uniform_color(col)
            o3d.io.write_point_cloud(os.path.join(output_dir, name), pc)
    except Exception as e:
        print(f"  [Error] 保存调试点云失败: {e}")


# ==================== 主流程 ====================

def main(cfg: Optional[Config] = None):
    if cfg is None:
        cfg = Config()

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- 1. 加载数据 ----
    smpl_params = load_smpl_params(cfg)
    data = np.load(cfg.npz_path)  # 保留 NPZ 用于后续构建 GT garment

    # ---- 2. 生成 SMPL 人体 ----
    gt_body_v, pred_body_v = build_smpl_bodies(cfg, smpl_params)

    # ---- 3. Kabsch 对齐 (Pred body → GT body) ----
    R_body, t_body = compute_body_alignment(gt_body_v, pred_body_v)

    # ---- 4. 加载服装、修正高度、应用 Kabsch ----
    section("4. 加载服装网格并修正初始高度")
    garment_v, garment_f = load_garment(cfg.garment_obj)
    # garment_v = align_garment_height(garment_v, pred_body_v, cfg.garment_head_offset)
    garment_v = apply_rigid_transform(garment_v, R_body, t_body)
    print(f"  garment 粗对齐后: Y=[{garment_v[:, 1].min():.3f}, {garment_v[:, 1].max():.3f}]")

    # ---- 4.5. ICP 精修: garment → GT garment ----
    section("4.5 ICP 精修: garment → GT garment")
    gt_mesh = build_gt_garment_mesh(data, cfg.non_garment_labels)
    garment_mesh = trimesh.Trimesh(garment_v, garment_f)
    R_icp, t_icp, fitness, rmse = icp_refine(garment_mesh, gt_mesh,
                                              max_corr_dist=cfg.icp_max_corr_dist)
    garment_v = apply_rigid_transform(garment_v, R_icp, t_icp)
    print(f"  garment ICP后: Y=[{garment_v[:, 1].min():.3f}, {garment_v[:, 1].max():.3f}]")

    # ---- 5. GT 服装网格（复用上面已构建的 gt_mesh） ----
    section("5. 构建 GT 服装网格")

    # ---- 6. 采样 & 评估 ----
    section("6. 评估指标")
    pred_mesh = trimesh.Trimesh(garment_v, garment_f, process=False)
    pred_pts = sample_mesh_surface(pred_mesh, cfg.n_sample, cfg.random_seed)
    gt_pts = sample_mesh_surface(gt_mesh, cfg.n_sample, cfg.random_seed)

    print(f"  Pred pts: {pred_pts.shape}, Y=[{pred_pts[:, 1].min():.3f}, {pred_pts[:, 1].max():.3f}]")
    print(f"  GT   pts: {gt_pts.shape},   Y=[{gt_pts[:, 1].min():.3f}, {gt_pts[:, 1].max():.3f}]")

    metrics = compute_metrics(pred_pts, gt_pts, cfg.fscore_thresholds_mm)

    # ---- 6.5. 保存指标到 CSV ----
    save_metrics_to_csv(metrics, cfg.output_dir)

    # ---- 7. 导出 ----
    export_results(pred_mesh, pred_pts, gt_pts, cfg.output_dir)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
