import os
import yaml
import json
import torch
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import smplx
import open3d as o3d

SMPL_MODEL_PATH = '/root/wyc/code/smpl2garmentcode2/smpl_models'

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
ICP_MAX_CORR_DIST = 0.3
RANDOM_SEED = 42
# CloSe 上装细类索引
UPPER_LABELS = [2, 3, 4, 5, 11, 17]  # Shirt/TShirt/Vest/Coat/Hoodies/Jacket

def close_points_to_metric(points: np.ndarray, scale: float, trans: np.ndarray) -> np.ndarray:
    """归一化扫描坐标 -> SMPL 米制坐标 (transl=0 人体所在帧)。"""
    return points / scale - trans.reshape(1, 3)


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
      - 翻领 (SimpleLapel)      → {Coat, Jacket}  
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

    accuracy = sum(correct.values()) / 3.0
    return accuracy, correct

def section(title: str):
    """打印分节标题。"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ==================== 1. 数据加载 ====================

def load_smpl_params(smpl_json: str,npz_path: str):
    """加载 GT (npz) 和 Pred (HybrIK json) 的 SMPL 参数。"""
    section("1. 加载 SMPL 参数")

    data = np.load(npz_path)
    gt_betas = data['betas'].astype(np.float32)
    gt_pose = data['pose'].astype(np.float32)
    gt_trans = data['trans'].astype(np.float32)

    with open(smpl_json) as f:
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

def build_smpl_bodies(smpl_params: dict,gender: str ):
    """用 SMPL 模型生成 GT 和 Pred 人体顶点。均不传 transl，保持在 SMPL 原生坐标系。"""
    section("2. 生成 SMPL 人体网格")

    model = smplx.create(model_path=SMPL_MODEL_PATH, model_type='smpl', gender=gender)

    with torch.no_grad():
        gt_out = model(
            betas=torch.from_numpy(smpl_params['gt_betas']).float().unsqueeze(0),# TODO gt beatas is zero
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
def body_kabsch_align(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
def rotation_angle_deg(R: np.ndarray) -> float:
    """从 3x3 旋转矩阵提取旋转角（度）。"""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

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


def get_cd_fscore(pred_pts: np.ndarray, gt_pts: np.ndarray,
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

def compute_cd_fscore(smpl_json: str, npz_path: str, garment_obj: str,gender:str,
                           output_dir: Optional[str] = None) -> Dict:
    """完整评估管线: SMPL参数 → 身体生成 → Kabsch → 服装加载 → ICP → CD/F-Score。
    与 compute_fscore_cd.py 计算流程完全相同。"""

    # 1. 加载 SMPL 参数
    smpl_params = load_smpl_params(smpl_json, npz_path)

    # 2. 生成 SMPL 身体
    gt_body, pred_body = build_smpl_bodies(smpl_params,gender)

    # 3. Kabsch 躯干对齐 (Pred body → GT body)
    R_body, t_body = body_kabsch_align(gt_body, pred_body)

    # 4. 加载服装 + 应用 Kabsch 变换
    garment_v, garment_f = load_garment(garment_obj)
    garment_v = apply_rigid_transform(garment_v, R_body, t_body)

    # 5. 构建 GT 服装网格
    gt_mesh = build_gt_garment_mesh(smpl_params['data'])

    # 6. ICP 精修 (garment → GT garment)
    R_icp, t_icp = icp_refine(trimesh.Trimesh(garment_v, garment_f), gt_mesh,
                                max_corr_dist=ICP_MAX_CORR_DIST)
    garment_v = apply_rigid_transform(garment_v, R_icp, t_icp)

    # 7. 采样 & 评估
    np.random.seed(RANDOM_SEED)
    pred_mesh = trimesh.Trimesh(garment_v, garment_f, process=False)
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, N_SAMPLE)
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, N_SAMPLE)

    metrics = get_cd_fscore(pred_pts, gt_pts)
    save_metrics_to_csv(metrics, output_dir)
    export_results(pred_mesh, pred_pts, gt_pts, output_dir)
    print(f"F-SCORE and CD results saved to {output_dir}")
    return metrics


def evaluate_single_sample(args) -> Dict:
    """评估单个样本，返回五项指标 + 分类明细。"""
    metrics = {
        'sample_name': args.sample,
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

    npz_path = os.path.join(args.data_root, f"{args.sample}.npz")
    output_dir = os.path.join(args.output_root, args.sample)
    smpl_json = os.path.join(args.output_root,"hybrik","smpl.json")
    driven_garment_obj = os.path.join(args.output_root,"driven","final_result.obj")
    gender = args.gender


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
    cd_fscore = compute_cd_fscore(smpl_json,
                                  npz_path,
                                  driven_garment_obj,
                                  gender,
                                  output_dir) 
    metrics['chamfer_distance'] = cd_fscore['cd_cm']
    metrics['f_score'] = cd_fscore['f_score']['0.01'] #TODO

    return metrics




# ==================== 7. 主函数 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='evaluate garment reconstruction results')
    parser.add_argument('--data_root', type=str,
                        default='/root/wyc/data/CloSe/data/CloSe-Di')
    parser.add_argument('--output_root', type=str,
                        default='/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe')
    parser.add_argument('--sample', type=str, default="10010_2314",
                        help='sample evaluation, pass sample name like 10010_2314')
    parser.add_argument('--gender', type=str)
    args = parser.parse_args()


    metrics = evaluate_single_sample(args)
    print(f"\n{'='*50}\nSample: {args.sample}\n{'='*50}")
    for k, v in metrics.items():
        print(f"{k}: {v}")
