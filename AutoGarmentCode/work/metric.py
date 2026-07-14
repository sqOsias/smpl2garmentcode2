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
from scipy.spatial import cKDTree

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

# F-Score 正式距离阈值：论文勘误后采用 τ = 10mm = 0.010m
DEFAULT_TAU = 0.010
NORMALIZED_TAU = 0.01

# 论文协议：在两个网格表面分别均匀采样 10000 点
N_SAMPLE =10000
RANDOM_SEED = 42
# CloSe 上装细类索引
UPPER_LABELS = [2, 3, 4, 5, 11, 17]  # Shirt/TShirt/Vest/Coat/Hoodies/Jacket

def apply_similarity_transform(verts: np.ndarray, scale: float,
                               R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """应用行向量形式的相似变换 ``v' = s * v @ R.T + t``。"""
    return scale * (verts @ R.T) + t


def umeyama_align(source: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """求无反射的最小二乘相似变换 source → target。"""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Similarity inputs must have matching (N,3) shape, got {source.shape}, {target.shape}")
    if len(source) < 3:
        raise ValueError("At least three correspondences are required for similarity alignment")

    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src_centered = source - src_mean
    dst_centered = target - dst_mean
    covariance = (src_centered.T @ dst_centered) / len(source)
    U, singular_values, Vt = np.linalg.svd(covariance)
    correction = np.ones(3, dtype=np.float64)
    if np.linalg.det(Vt.T @ U.T) < 0:
        correction[-1] = -1.0
    R_mat = Vt.T @ np.diag(correction) @ U.T
    src_variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    if src_variance <= 1e-12:
        raise ValueError("Degenerate scan correspondences: near-zero source variance")
    sim_scale = float(np.dot(singular_values, correction) / src_variance)
    t_vec = dst_mean - sim_scale * (src_mean @ R_mat.T)
    return sim_scale, R_mat, t_vec


def estimate_scan_to_smpl_transform(data: dict, gt_body_v: np.ndarray,
                                    gt_template_v: np.ndarray):
    """用 canon_pose 对应和 Body 标签估计 scan-centered → SMPL-native。

    不显式恢复缺失的 ``centers``，也不混用 NPZ 的包围盒归一化 scale
    与原 registration scale；二者的影响由相似变换统一吸收。
    """
    npz_scale_values = np.asarray(data['scale'], dtype=np.float64).reshape(-1)
    if npz_scale_values.size != 1:
        raise ValueError(
            f"CloSe scale must be scalar, got shape {np.asarray(data['scale']).shape}"
        )
    npz_scale = float(npz_scale_values[0])
    if not np.isfinite(npz_scale) or npz_scale <= 0.0:
        raise ValueError(f"CloSe scale must be finite and positive, got {npz_scale}")

    points_metric = np.asarray(data['points'], dtype=np.float64) / npz_scale
    canon_pose = np.asarray(data['canon_pose'], dtype=np.float64)
    labels = np.asarray(data['labels'])
    gt_body_v = np.asarray(gt_body_v, dtype=np.float64)
    gt_template_v = np.asarray(gt_template_v, dtype=np.float64)

    if len(points_metric) != len(canon_pose) or len(points_metric) != len(labels):
        raise ValueError("points, canon_pose and labels must contain the same number of vertices")

    canon_dist, smpl_idx = cKDTree(gt_template_v).query(canon_pose, k=1, workers=-1)
    valid = ((labels == 1) & np.isfinite(points_metric).all(axis=1)
             & np.isfinite(canon_pose).all(axis=1) & np.isfinite(canon_dist))
    if int(valid.sum()) < 100:
        raise ValueError(f"Too few Body correspondences for scan alignment: {int(valid.sum())}")

    # 模板版本可能产生微小差异；保留匹配误差最低的 95%，随后再按拟合残差裁剪。
    canon_limit = np.quantile(canon_dist[valid], 0.95)
    valid &= canon_dist <= canon_limit
    source = points_metric[valid]
    target = gt_body_v[smpl_idx[valid]]

    keep = np.ones(len(source), dtype=bool)
    for _ in range(3):
        sim_scale, R_mat, t_vec = umeyama_align(source[keep], target[keep])
        aligned = apply_similarity_transform(source, sim_scale, R_mat, t_vec)
        residual = np.linalg.norm(aligned - target, axis=1)
        residual_limit = np.quantile(residual[keep], 0.80)
        keep &= residual <= residual_limit
        if int(keep.sum()) < 100:
            raise ValueError("Robust scan alignment rejected too many Body correspondences")

    sim_scale, R_mat, t_vec = umeyama_align(source[keep], target[keep])
    aligned = apply_similarity_transform(source[keep], sim_scale, R_mat, t_vec)
    residual = np.linalg.norm(aligned - target[keep], axis=1)
    diagnostics = {
        'num_body_correspondences': int(keep.sum()),
        'scale': sim_scale,
        'rotation_deg': rotation_angle_deg(R_mat),
        'translation': t_vec,
        'canon_match_median_mm': float(np.median(canon_dist[valid]) * 1000.0),
        'canon_match_p95_mm': float(np.quantile(canon_dist[valid], 0.95) * 1000.0),
        'median_residual_cm': float(np.median(residual) * 100.0),
        'p95_residual_cm': float(np.quantile(residual, 0.95) * 100.0),
        'npz_scale': npz_scale,
        # CloSe normalized the full scan longest side to one.  The Umeyama
        # scale maps that original scan length into the current SMPL frame.
        'normalization_length_m': sim_scale / npz_scale,
    }
    return points_metric, sim_scale, R_mat, t_vec, diagnostics


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
    # hybrik_pose = np.array(pred_json['pose'], dtype=np.float32)

    # GT pose 拆分
    gt_pose_t = torch.from_numpy(gt_pose).float().view(1, 72)
    gt_body_pose = gt_pose_t[:, 3:]
    gt_global_orient = gt_pose_t[:, :3]

    # Pred: HybrIK body pose + 零 global orient (A-pose)
    # pred_body_pose = torch.from_numpy(hybrik_pose).float().view(1, 69)
    # Pred pose: build_default_pose from export_smpl_mesh.py (match smpl.obj A-pose)
    angle = np.pi / 4.0
    angle2 = np.pi / 20.0
    apose = torch.zeros((1, 24, 3), dtype=torch.float32)
    apose[0, 16, 2] = -angle     # left elbow Z
    apose[0, 17, 2] = angle      # right elbow Z
    apose[0, 16, 1] = -angle2    # left elbow Y
    apose[0, 17, 1] = angle2     # right elbow Y

    pred_body_pose = apose.view(1, 72)[:, 3:]  # body_pose (69 dims)
    pred_global_orient = torch.zeros(1, 3)

    print(f"  GT  betas: {gt_betas.shape}, pose: {gt_pose.shape}, trans: {gt_trans}")
    print(f"  Pred betas: {pred_betas.shape}, pred_body_pose: {pred_body_pose.shape}")

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

def build_gt_garment_mesh(data: dict, non_garment_labels: Tuple[int, ...],
                          gt_body_v: np.ndarray,
                          gt_template_v: np.ndarray) -> Tuple[trimesh.Trimesh, dict]:
    """提取 GT 服装并通过人体对应关系变换到 GT SMPL 原生坐标系。"""
    labels = data['labels']
    faces = data['faces']
    points_metric, sim_scale, R_scan, t_scan, alignment = estimate_scan_to_smpl_transform(
        data, gt_body_v, gt_template_v
    )
    print(f"  Scan→SMPL scale: {alignment['scale']:.6f}")
    print(f"  Scan→SMPL rotation: {alignment['rotation_deg']:.3f} deg")
    print(f"  Scan→SMPL translation: {alignment['translation']}")
    print(f"  Body correspondences: {alignment['num_body_correspondences']}")
    print(f"  Canon match median/P95: {alignment['canon_match_median_mm']:.3f}/"
          f"{alignment['canon_match_p95_mm']:.3f} mm")
    print(f"  Body residual median/P95: {alignment['median_residual_cm']:.3f}/"
          f"{alignment['p95_residual_cm']:.3f} cm")

    # GT 服装与 GT SMPL 人体统一到同一个 native target-pose frame。
    gar_mask = ~np.isin(labels, non_garment_labels)
    gar_pts = apply_similarity_transform(
        points_metric[gar_mask], sim_scale, R_scan, t_scan
    )

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
    ], dtype=np.int64).reshape(-1, 3)

    # 仅做与预测无关的确定性清理：移除非有限/退化面和未引用顶点。
    # 不按预测结果删除 GT 点，也不只保留最大分量（上下装可为独立分量）。
    removed_faces = 0
    if len(gar_faces_remapped) > 0:
        tri = gar_pts[gar_faces_remapped]
        finite_faces = np.isfinite(tri).all(axis=(1, 2))
        double_area = np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        )
        valid_faces = finite_faces & (double_area > 1e-12)
        removed_faces = int((~valid_faces).sum())
        gar_faces_remapped = gar_faces_remapped[valid_faces]

    mesh = trimesh.Trimesh(gar_pts, gar_faces_remapped, process=False)
    mesh.remove_unreferenced_vertices()
    print(f"  GT garment mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print(f"  Removed invalid/degenerate GT faces: {removed_faces}")
    if len(mesh.vertices) > 0:
        print(f"  GT garment Y=[{mesh.vertices[:, 1].min():.3f}, {mesh.vertices[:, 1].max():.3f}]")
    return mesh, alignment


# ==================== 6. 距离指标 ====================
def rotation_angle_deg(R: np.ndarray) -> float:
    """从 3x3 旋转矩阵提取旋转角（度）。"""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

def normalize_close_point_pair(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    normalization_length_m: float,
    common_center_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize both point sets with the same CloSe full-scan transform.

    ``normalization_length_m`` is the original full-scan longest side after
    Scan-to-SMPL registration.  Never derive separate scales from the two
    garment bounding boxes: doing so would remove real size errors.
    """
    pred_pts = np.asarray(pred_pts, dtype=np.float64)
    gt_pts = np.asarray(gt_pts, dtype=np.float64)
    length = float(normalization_length_m)
    center = np.asarray(common_center_m, dtype=np.float64).reshape(-1)

    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"normalization_length_m must be finite and positive, got {length}")
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError(f"common_center_m must contain three finite values, got {center}")

    return (pred_pts - center) / length, (gt_pts - center) / length


def get_normalized_cd_fscore(
    pred_pts_norm: np.ndarray,
    gt_pts_norm: np.ndarray,
    tau: float = NORMALIZED_TAU,
) -> Dict:
    """Compute linear-distance CD/F-Score in the shared normalized frame."""
    pred_pts_norm = np.asarray(pred_pts_norm, dtype=np.float64)
    gt_pts_norm = np.asarray(gt_pts_norm, dtype=np.float64)
    tau = float(tau)

    for name, points in (("pred_pts_norm", pred_pts_norm), ("gt_pts_norm", gt_pts_norm)):
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(f"{name} must be a non-empty (N, 3) array, got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"Normalized tau must be finite and positive, got {tau}")

    p2g = cKDTree(gt_pts_norm).query(pred_pts_norm, k=1, workers=-1)[0]
    g2p = cKDTree(pred_pts_norm).query(gt_pts_norm, k=1, workers=-1)[0]
    cd = 0.5 * (p2g.mean() + g2p.mean())
    precision = float((p2g <= tau).mean())
    recall = float((g2p <= tau).mean())
    fscore = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0 else 0.0
    )

    print(f"\n  CloSe normalized CD = {cd:.6f}")
    print(
        f"  CloSe normalized F@{tau:g} = {fscore:.4f}  "
        f"(P={precision * 100:.1f}%, R={recall * 100:.1f}%)"
    )
    return {
        'cd': float(cd),
        'tau': tau,
        'fscore': float(fscore),
        'precision': precision,
        'recall': recall,
    }


def get_cd_fscore(pred_pts: np.ndarray, gt_pts: np.ndarray,
                    thresholds_mm: Tuple[int, ...] = (5, 10, 20, 30, 50)):
    """计算 CD 和多阈值 F-Score。

    Returns:
        dict: 包含 cd_cm, fscore@Xmm, p2g, g2p 等
    """
    pred_pts = np.asarray(pred_pts, dtype=np.float64)
    gt_pts = np.asarray(gt_pts, dtype=np.float64)
    if pred_pts.ndim != 2 or pred_pts.shape[1] != 3 or len(pred_pts) == 0:
        raise ValueError(f"pred_pts must be a non-empty (N, 3) array, got {pred_pts.shape}")
    if gt_pts.ndim != 2 or gt_pts.shape[1] != 3 or len(gt_pts) == 0:
        raise ValueError(f"gt_pts must be a non-empty (N, 3) array, got {gt_pts.shape}")
    if not np.isfinite(pred_pts).all() or not np.isfinite(gt_pts).all():
        raise ValueError("Point clouds contain NaN or Inf")

    # KD-tree 避免构造 N×M 的完整距离矩阵。
    p2g = cKDTree(gt_pts).query(pred_pts, k=1, workers=-1)[0]
    g2p = cKDTree(pred_pts).query(gt_pts, k=1, workers=-1)[0]

    cd_cm = 0.5 * (p2g.mean() + g2p.mean()) * 100.0

    print(f"\n  CD      = {cd_cm:.3f} cm")
    print(f"  P->GT   mean={p2g.mean() * 100:.3f}cm  median={np.median(p2g) * 100:.3f}cm")
    print(f"  GT->P   mean={g2p.mean() * 100:.3f}cm  median={np.median(g2p) * 100:.3f}cm")

    fscores = {}
    print(f"\n  多阈值 F-Score:")
    for tau_mm in thresholds_mm:
        tau = tau_mm / 1000.0
        pr = (p2g <= tau).mean()
        rc = (g2p <= tau).mean()
        fs = (2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
        fscores[tau_mm] = {'fscore': fs, 'precision': pr, 'recall': rc}
        print(f"    F@{tau_mm}mm = {fs:.4f}  (P={pr * 100:.1f}%, R={rc * 100:.1f}%)")

    return {'cd_cm': cd_cm, 'p2g': p2g, 'g2p': g2p, 'fscores': fscores}


def save_metrics_to_csv(metrics: dict, output_dir: str, filename: str = "metrics.csv"):
    """将 CD 和 F-Score 保存到 CSV 文件。"""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "cd_cm",
            "normalized_cd",
            "normalized_tau",
            "normalized_effective_tau_mm",
            "normalized_fscore",
            "normalized_precision",
            "normalized_recall",
        ] + [f"F-Score@{t}mm" for t in (5,10,20,30,50)])
        fs = metrics['fscores']
        normalized = metrics['normalized']
        writer.writerow([
            f"{metrics['cd_cm']:.4f}",
            f"{normalized['cd']:.6f}",
            f"{normalized['tau']:.6f}",
            f"{normalized['effective_tau_mm']:.4f}",
            f"{normalized['fscore']:.4f}",
            f"{normalized['precision']:.4f}",
            f"{normalized['recall']:.4f}",
        ] + [f"{fs[t]['fscore']:.4f}" for t in (5,10,20,30,50)])
    print(f"  CD/F-Score saved to {csv_path}")


def save_metrics(full_metrics: dict, output_dir: str):
    """保存所有评估参数: 结构合法率、仿真成功率、分类准确率、CD、F-Score。

    写入两个文件:
      eval_summary.json  — 完整指标 (dict)
      eval_summary.csv   — 扁平化单行 CSV
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- JSON ----
    json_path = os.path.join(output_dir, "eval_summary.json")
    json_out = {}
    for k, v in full_metrics.items():
        if isinstance(v, (float, int, str, type(None))):
            json_out[k] = v
        elif isinstance(v, np.ndarray):
            json_out[k] = v.tolist() if v.size < 100 else f"<ndarray shape={v.shape}>"
        elif isinstance(v, dict):
            json_out[k] = {str(kk): (vv if not isinstance(vv, np.ndarray) else f"<ndarray shape={vv.shape}>") for kk, vv in v.items()}
        else:
            json_out[k] = str(v)
    with open(json_path, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"  Full metrics saved to {json_path}")

    # ---- CSV ----
    csv_path = os.path.join(output_dir, "eval_summary.csv")
    columns = [
        "sample_name",
        "valid_structure",
        "sim_success",
        "class_acc",
        "upper_correct",
        "bottom_correct",
        "connected_correct",
        "chamfer_distance_cm",
        "normalized_cd",
        "normalized_tau",
        "normalized_effective_tau_mm",
        "normalized_fscore",
        "normalized_precision",
        "normalized_recall",
        "f_score_protocol",
    ]
    # 添加多阈值 F-Score
    thresholds = [5, 10, 20, 30, 50]
    for t in thresholds:
        columns += [f"F-Score@{t}mm", f"Precision@{t}mm", f"Recall@{t}mm"]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        normalized = full_metrics.get('normalized_metrics', {})
        row = [
            full_metrics.get('sample_name', ''),
            f"{full_metrics.get('valid_structure', 0):.4f}",
            f"{full_metrics.get('sim_success', 0):.4f}",
            f"{full_metrics.get('class_acc', 0):.4f}",
            f"{full_metrics.get('upper_correct', 0):.4f}",
            f"{full_metrics.get('bottom_correct', 0):.4f}",
            f"{full_metrics.get('connected_correct', 0):.4f}",
            f"{full_metrics.get('chamfer_distance', 0):.4f}" if full_metrics.get('chamfer_distance') is not None else "",
            f"{normalized.get('cd', 0):.6f}" if normalized else "",
            f"{normalized.get('tau', 0):.6f}" if normalized else "",
            f"{normalized.get('effective_tau_mm', 0):.4f}" if normalized else "",
            f"{normalized.get('fscore', 0):.4f}" if normalized else "",
            f"{normalized.get('precision', 0):.4f}" if normalized else "",
            f"{normalized.get('recall', 0):.4f}" if normalized else "",
            full_metrics.get('f_score_protocol', ''),
        ]
        for t in thresholds:
            fs = full_metrics.get('fscores', {}).get(t, {})
            row += [
                f"{fs.get('fscore', 0):.4f}" if fs else "",
                f"{fs.get('precision', 0):.4f}" if fs else "",
                f"{fs.get('recall', 0):.4f}" if fs else "",
            ]
        writer.writerow(row)
    print(f"  Summary CSV saved to {csv_path}")


def export_results(pred_mesh: trimesh.Trimesh,
                   gt_mesh: trimesh.Trimesh,
                   pred_pts: np.ndarray,
                   gt_pts: np.ndarray,
                   output_dir: str):
    """Export aligned garment meshes and debug point clouds."""
    section("7. Export debug files")

    os.makedirs(output_dir, exist_ok=True)

    # Pred garment mesh (aligned)
    obj_path = os.path.join(output_dir, "pred_garment_kabsch.obj")
    pred_mesh.export(obj_path)
    print(f"  {obj_path} saved")

    # GT garment mesh (aligned, from NPZ)
    gt_obj_path = os.path.join(output_dir, "gt_garment.obj")
    gt_mesh.export(gt_obj_path)
    print(f"  {gt_obj_path} saved")

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

def evaluate_driven_garment(
    driven_verts_m: np.ndarray,
    driven_faces: np.ndarray,
    pred_target_body_verts_m: np.ndarray,
    gt_target_body_verts_m: np.ndarray,
    gt_template_verts_m: np.ndarray,
    gt_data: dict,
    torso_mask: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """以内存数据评估驱动服装，不使用 GT 服装执行 ICP。

    驱动服装与 ``pred_target_body_verts_m`` 必须处于同一坐标系。通过具有
    相同 SMPL 顶点索引的目标姿态人体求 Pred→GT 刚性变换，再将同一变换
    应用于服装。GT 服装仅参与最终距离计算，不参与对齐。
    """
    garment_v = np.asarray(driven_verts_m, dtype=np.float64)
    garment_f = np.asarray(driven_faces, dtype=np.int64)
    pred_body = np.asarray(pred_target_body_verts_m, dtype=np.float64)
    gt_body = np.asarray(gt_target_body_verts_m, dtype=np.float64)

    if garment_v.ndim != 2 or garment_v.shape[1] != 3 or len(garment_v) == 0:
        raise ValueError(f"driven_verts_m must be a non-empty (N, 3) array, got {garment_v.shape}")
    if garment_f.ndim != 2 or garment_f.shape[1] != 3 or len(garment_f) == 0:
        raise ValueError(f"driven_faces must be a non-empty (F, 3) array, got {garment_f.shape}")
    if pred_body.shape != gt_body.shape or pred_body.ndim != 2 or pred_body.shape[1] != 3:
        raise ValueError(
            "Predicted and GT SMPL bodies must have matching (N, 3) topology, "
            f"got {pred_body.shape} and {gt_body.shape}"
        )
    if not all(np.isfinite(x).all() for x in (garment_v, pred_body, gt_body)):
        raise ValueError("Driven garment or SMPL body contains NaN or Inf")
    if garment_f.min() < 0 or garment_f.max() >= len(garment_v):
        raise ValueError("driven_faces contains an out-of-range vertex index")

    section("3. SMPL torso alignment: Pred body → GT body")
    if torso_mask is None:
        torso_mask = get_torso_mask(gt_body)
    else:
        torso_mask = np.asarray(torso_mask, dtype=bool).reshape(-1)
        if len(torso_mask) != len(gt_body):
            raise ValueError(
                f"torso_mask length {len(torso_mask)} does not match SMPL vertices {len(gt_body)}"
            )
    if int(torso_mask.sum()) < 3:
        raise ValueError("Unable to select enough torso vertices for rigid alignment")
    R_body, t_body = kabsch_align(pred_body[torso_mask], gt_body[torso_mask])
    garment_v = apply_rigid_transform(garment_v, R_body, t_body)
    print(f"  Torso verts: {int(torso_mask.sum())}")
    print(f"  Rotation: {rotation_angle_deg(R_body):.3f} deg")
    print(f"  Translation: {t_body}")

    gt_mesh, scan_alignment = build_gt_garment_mesh(
        gt_data, tuple(NON_GARMENT_LABELS), gt_body, gt_template_verts_m
    )
    if len(gt_mesh.vertices) == 0 or len(gt_mesh.faces) == 0:
        raise ValueError("GT garment mesh is empty after semantic filtering")

    np.random.seed(RANDOM_SEED)
    pred_mesh = trimesh.Trimesh(garment_v, garment_f, process=False)
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, N_SAMPLE)
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, N_SAMPLE)

    print(f"\n  Pred garment Y=[{pred_pts[:, 1].min():.3f}, {pred_pts[:, 1].max():.3f}]")
    print(f"  GT garment Y=[{gt_pts[:, 1].min():.3f}, {gt_pts[:, 1].max():.3f}]")

    # Apply one shared full-scan normalization before computing the formal
    # CloSe metric.  Physical-unit metrics are retained below as diagnostics.
    normalization_length_m = float(scan_alignment['normalization_length_m'])
    normalization_center_m = np.asarray(scan_alignment['translation'], dtype=np.float64)
    pred_pts_norm, gt_pts_norm = normalize_close_point_pair(
        pred_pts,
        gt_pts,
        normalization_length_m=normalization_length_m,
        common_center_m=normalization_center_m,
    )
    normalized = get_normalized_cd_fscore(
        pred_pts_norm,
        gt_pts_norm,
        tau=NORMALIZED_TAU,
    )
    effective_tau_mm = normalized['tau'] * normalization_length_m * 1000.0
    normalized['effective_tau_mm'] = float(effective_tau_mm)
    normalized['normalization_length_m'] = normalization_length_m
    normalized['npz_scale'] = float(scan_alignment['npz_scale'])
    normalized['scan_to_smpl_scale'] = float(scan_alignment['scale'])
    normalized['protocol'] = 'close_full_scan_normalized_linear_distance'
    print(f"  Full-scan normalization length = {normalization_length_m:.6f} m")
    print(f"  Effective physical tau = {effective_tau_mm:.3f} mm")

    metrics = get_cd_fscore(pred_pts, gt_pts)
    metrics['normalized'] = normalized
    metrics['fscore_normalized_0p01'] = normalized['fscore']
    metrics['fscore_10mm'] = metrics['fscores'][10]['fscore']
    metrics['alignment'] = {'R': R_body, 't': t_body}
    metrics['scan_alignment'] = scan_alignment
    if output_dir is not None:
        save_metrics_to_csv(metrics, output_dir)
        export_results(pred_mesh, gt_mesh, pred_pts, gt_pts, output_dir)
        print(f"F-SCORE and CD results saved to {output_dir}")
    return metrics


def compute_cd_fscore(smpl_json: str, npz_path: str, garment_obj: str, gender: str,
                      output_dir: Optional[str] = None) -> Dict:
    """兼容历史文件评估入口；正式全流程应调用 ``evaluate_driven_garment``。

    ``smpl_json`` 和 ``gender`` 为兼容旧调用保留。驱动目录必须同时包含
    ``final_result.obj`` 与 ``target_body.obj``，以保证服装和预测人体处于
    完全相同的驱动坐标系。
    """
    del smpl_json
    garment_v, garment_f = load_garment(garment_obj)
    target_body_path = os.path.join(os.path.dirname(garment_obj), "target_body.obj")
    if not os.path.exists(target_body_path):
        raise FileNotFoundError(
            f"Missing driven target body required for SMPL alignment: {target_body_path}"
        )
    pred_body_mesh = trimesh.load(target_body_path, process=False, force='mesh')
    pred_body = np.asarray(pred_body_mesh.vertices, dtype=np.float64)

    with np.load(npz_path) as npz:
        gt_data = {key: npz[key] for key in npz.files}
    gt_pose = torch.from_numpy(np.asarray(gt_data['pose'], dtype=np.float32)).view(1, 72)
    # CloSe registration 在缺少逐样本 gender 时使用 neutral SMPL。
    gt_model = smplx.create(
        model_path=SMPL_MODEL_PATH, model_type='smpl', gender='neutral'
    )
    with torch.no_grad():
        gt_output = gt_model(
            betas=torch.from_numpy(np.asarray(gt_data['betas'], dtype=np.float32)).view(1, -1),
            body_pose=gt_pose[:, 3:],
            global_orient=gt_pose[:, :3],
        )
    gt_body = gt_output.vertices.squeeze(0).cpu().numpy()
    gt_template = gt_model.v_template.detach().cpu().numpy()
    dominant_joint = gt_model.lbs_weights.argmax(dim=1).cpu().numpy()
    torso_mask = np.isin(dominant_joint, [0, 3, 6, 9])
    return evaluate_driven_garment(
        garment_v, garment_f, pred_body, gt_body, gt_template, gt_data,
        torso_mask=torso_mask, output_dir=output_dir,
    )


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
    smpl_json = os.path.join(args.output_root,args.sample,"hybrik","smpl.json")
    # driven_garment_obj = os.path.join(args.output_root,args.sample,"driven","final_result.obj")
    # driven_garment_obj = os.path.join(args.output_root,args.sample,"design","design_sim.obj")
    driven_garment_obj = os.path.join(args.output_root,args.sample,"design","driven","final_result.obj")
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
    metrics['f_score'] = cd_fscore['normalized']['fscore']
    metrics['f_score_protocol'] = cd_fscore['normalized']['protocol']
    metrics['normalized_metrics'] = cd_fscore['normalized']
    metrics['fscore_10mm'] = cd_fscore['fscore_10mm']
    metrics['fscores'] = cd_fscore['fscores']
    metrics['scan_alignment'] = cd_fscore['scan_alignment']

    # ---- 5. 保存所有指标 ----
    save_metrics(metrics, output_dir)

    return metrics




# ==================== 7. 主函数 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='evaluate garment reconstruction results')
    parser.add_argument('--data_root', type=str,
                        default='/root/wyc/data/CloSe/data/CloSe-Di')
    parser.add_argument('--output_root', type=str,
                        default='/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe')
    parser.add_argument('--sample', type=str, default="10027_3297",
                        help='sample evaluation, pass sample name like 10010_2314')
    parser.add_argument('--gender', type=str,default="female")
    args = parser.parse_args()


    metrics = evaluate_single_sample(args)
    print(f"\n{'='*50}\nSample: {args.sample}\n{'='*50}")
    for k, v in metrics.items():
        print(f"{k}: {v}")
