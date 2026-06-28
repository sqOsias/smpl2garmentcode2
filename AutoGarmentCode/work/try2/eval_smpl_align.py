
"""
基于 SMPL 躯干顶点对应关系的 Kabsch 刚性对齐评估脚本。
利用 SMPL 顶点的一一对应关系（同索引 = 同身体位置），
在躯干区域做最优刚性变换，应用到服装上。
"""
import os, sys, json
import numpy as np
import torch
import trimesh
import smplx
import open3d as o3d


# ==================== 配置 ====================
SMPL_MODEL_PATH = "/root/wyc/code/smpl2garmentcode2/smpl_models"
NPZ_PATH  = "/root/wyc/data/CloSe/data/CloSe-Di/10001_1937.npz"
SMPL_JSON = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe/10001_1937/smpl.json"
GARMENT_OBJ = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/try2/output/final_result.obj"  # Warp 驱动到 GT pose 后的网格
OUTPUT_DIR = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/try2/align"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

N_SAMPLE = 50000
DEFAULT_TAU = 0.005  # 5mm
NON_GARMENT_LABELS = [0, 1, 10, 12, 13, 14, 15]
GENDER = "male"

# ==================== 1. 加载数据 ====================
print("=" * 60)
print("1. 加载 SMPL 参数")
print("=" * 60)

# GT: npz
data = np.load(NPZ_PATH)
gt_betas = data['betas'].astype(np.float32)
gt_pose  = data['pose'].astype(np.float32)   # 72 维: global_orient(3) + body_pose(69)
gt_trans = data['trans'].astype(np.float32)  # SMPL transl

# Pred: HybrIK smpl.json
with open(SMPL_JSON) as f:
    pred_json = json.load(f)
pred_betas = np.array(pred_json['betas'], dtype=np.float32)
hybrik_pose = np.array(pred_json['pose'], dtype=np.float32)  # 69 维 body pose

# 构建 A-pose: 用 HybrIK body pose + 零 global orient
angle = np.pi / 4.0
angle2 = np.pi / 20.0
pred_body_pose = torch.from_numpy(hybrik_pose).float().view(1, 69)
pred_global_orient = torch.zeros(1, 3)  # A-pose 无全局旋转

# GT pose 拆分
gt_pose_t = torch.from_numpy(gt_pose).float().view(1, 72)
gt_body_pose = gt_pose_t[:, 3:]
gt_global_orient = gt_pose_t[:, :3]

print(f"  GT  betas: {gt_betas.shape}, pose: {gt_pose.shape}, trans: {gt_trans}")
print(f"  Pred betas: {pred_betas.shape}, hybrik_pose: {hybrik_pose.shape}")

# ==================== 2. SMPL 模型生成人体顶点 ====================
print("\n" + "=" * 60)
print("2. 生成 SMPL 人体网格")
print("=" * 60)

model = smplx.create(SMPL_MODEL_PATH, model_type='smpl', gender=GENDER)

# GT 人体: 使用 npz 的 transl 参数
# 注意: GT trans 来自 CloSe 的归一化空间，直接用作 SMPL transl 会导致
# 人体位置偏移。这里不传 transl，让 GT 和 Pred 都在 SMPL 原生坐标系下。
with torch.no_grad():
    gt_out = model(
        betas=torch.from_numpy(gt_betas).float().unsqueeze(0),
        body_pose=gt_body_pose,
        global_orient=gt_global_orient,
    )
    gt_body_v = gt_out.vertices.squeeze().numpy()

    pred_out = model(
        betas=torch.from_numpy(pred_betas).float().unsqueeze(0),
        body_pose=pred_body_pose,
        global_orient=pred_global_orient,
    )
    pred_body_v = pred_out.vertices.squeeze().numpy()

print(f"  GT   body: {gt_body_v.shape}, Y=[{gt_body_v[:,1].min():.3f}, {gt_body_v[:,1].max():.3f}]")
print(f"  Pred body: {pred_body_v.shape}, Y=[{pred_body_v[:,1].min():.3f}, {pred_body_v[:,1].max():.3f}]")

# ==================== 3. ICP 计算变换矩阵 ====================
print("\n" + "=" * 60)
print("3. ICP 对齐: Pred body → GT body")
print("=" * 60)

gt_pc = o3d.geometry.PointCloud()
gt_pc.points = o3d.utility.Vector3dVector(gt_body_v.astype(np.float64))

pred_pc = o3d.geometry.PointCloud()
pred_pc.points = o3d.utility.Vector3dVector(pred_body_v.astype(np.float64))

# SMPL 全身顶点有已知的一一对应关系（同索引 = 同身体位置）
# 用躯干顶点做 Kabsch 算法（已知对应关系的 Procrustes），比 ICP 更精确
def get_torso_mask(verts):
    """返回躯干顶点的 bool mask"""
    y_mid = (verts[:, 1].max() + verts[:, 1].min()) / 2
    y_range = verts[:, 1].max() - verts[:, 1].min()
    mask_y = (verts[:, 1] > y_mid - 0.25 * y_range) & (verts[:, 1] < y_mid + 0.25 * y_range)
    torso = verts[mask_y]
    x_mid = torso[:, 0].mean()
    x_range = torso[:, 0].max() - torso[:, 0].min()
    mask_x = np.abs(verts[:, 0] - x_mid) < 0.2 * x_range
    return mask_y & mask_x

torso_mask = get_torso_mask(gt_body_v)
gt_torso_v = gt_body_v[torso_mask]
pred_torso_v = pred_body_v[torso_mask]  # 同索引 → 一一对应
print(f"  Torso verts (对应点对): {len(gt_torso_v)}")

# Kabsch 算法：已知对应关系的最优刚性变换
gt_c = gt_torso_v.mean(axis=0)
pred_c = pred_torso_v.mean(axis=0)
H = (pred_torso_v - pred_c).T @ (gt_torso_v - gt_c)
U, _, Vt = np.linalg.svd(H)
R_mat = Vt.T @ U.T
if np.linalg.det(R_mat) < 0:
    Vt[-1, :] *= -1
    R_mat = Vt.T @ U.T
t_vec = gt_c - pred_c @ R_mat.T

print(f"  Kabsch 旋转角: {np.degrees(np.arccos(np.clip((np.trace(R_mat)-1)/2, -1, 1))):.2f}°")
print(f"  平移向量: {t_vec}")

print(f"  旋转矩阵:\n{R_mat}")
print(f"  平移向量: {t_vec}")

# ==================== 4. 加载服装并应用变换 ====================
print("\n" + "=" * 60)
print("4. 加载服装网格并应用 ICP 变换")
print("=" * 60)

garment_mesh = trimesh.load(GARMENT_OBJ, process=False)
garment_v = np.array(garment_mesh.vertices, dtype=np.float64)  # 已经是米制 (Warp输出)
garment_f = np.array(garment_mesh.faces)

# 撤销 export_smpl_mesh.py 的 Y 偏移，使服装回到 SMPL 原生坐标系
min_y_pred = pred_body_v[:, 1].min()
garment_v[:, 1] += min_y_pred  # min_y_pred 为负值，将服装下移

print(f"  garment 原始 (cm→m, Y偏移已撤销): Y=[{garment_v[:,1].min():.3f}, {garment_v[:,1].max():.3f}]")

# 应用 ICP 变换
garment_v_aligned = (garment_v @ R_mat.T) + t_vec

print(f"  garment 对齐后: Y=[{garment_v_aligned[:,1].min():.3f}, {garment_v_aligned[:,1].max():.3f}]")

# ==================== 4.5. Garment→GT ICP 精修 ====================
print("\n" + "=" * 60)
print("4.5 ICP 精修: garment → GT garment")
print("=" * 60)

# 先构建 GT 服装网格 (用于 ICP target)
gt_labels_tmp = data['labels']
gt_points_tmp = data['points']
gt_faces_tmp = data['faces']
scale_tmp = float(data['scale'])
trans_tmp = data['trans']
gar_mask_tmp = ~np.isin(gt_labels_tmp, NON_GARMENT_LABELS)
gt_gar_pts_tmp = gt_points_tmp[gar_mask_tmp] / scale_tmp - trans_tmp.reshape(1, 3)
gar_indices_tmp = np.where(gar_mask_tmp)[0]
gar_idx_set_tmp = set(gar_indices_tmp.tolist())
gar_face_mask_tmp = np.array([
    (f[0] in gar_idx_set_tmp) and (f[1] in gar_idx_set_tmp) and (f[2] in gar_idx_set_tmp)
    for f in gt_faces_tmp
])
gar_faces_f_tmp = gt_faces_tmp[gar_face_mask_tmp]
old_to_new_tmp = {old: new for new, old in enumerate(gar_indices_tmp)}
gar_faces_r_tmp = np.array([
    [old_to_new_tmp[f[0]], old_to_new_tmp[f[1]], old_to_new_tmp[f[2]]]
    for f in gar_faces_f_tmp
])
gt_mesh_tmp = trimesh.Trimesh(gt_gar_pts_tmp, gar_faces_r_tmp, process=False)

# 在 garment 表面采点做 ICP source
garment_mesh_aligned = trimesh.Trimesh(garment_v_aligned, garment_f)
src_pts, _ = trimesh.sample.sample_surface(garment_mesh_aligned, 20000)
src_pc = o3d.geometry.PointCloud()
src_pc.points = o3d.utility.Vector3dVector(src_pts.astype(np.float64))

# GT garment 表面采点做 ICP target
tgt_pts, _ = trimesh.sample.sample_surface(gt_mesh_tmp, 20000)
tgt_pc = o3d.geometry.PointCloud()
tgt_pc.points = o3d.utility.Vector3dVector(tgt_pts.astype(np.float64))

reg = o3d.pipelines.registration.registration_icp(
    src_pc, tgt_pc, max_correspondence_distance=0.3,
    init=np.eye(4),
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100))

M_icp = reg.transformation
garment_v_aligned = (garment_v_aligned @ M_icp[:3,:3].T) + M_icp[:3,3]
print(f"  ICP fitness={reg.fitness:.4f}  inlier_rmse={reg.inlier_rmse*100:.2f}cm")
print(f"  ICP旋转={np.degrees(np.arccos(np.clip((np.trace(M_icp[:3,:3])-1)/2,-1,1))):.2f}deg  平移={M_icp[:3,3]}")
print(f"  garment ICP后: Y=[{garment_v_aligned[:,1].min():.3f}, {garment_v_aligned[:,1].max():.3f}]")

# ==================== 5. GT 服装网格 ====================
print("\n" + "=" * 60)
print("5. 构建 GT 服装网格")
print("=" * 60)

gt_labels = data['labels']
gt_points = data['points']
gt_faces = data['faces']
scale = float(data['scale'])
trans_npz = data['trans']

# GT 服装点 → metric (SMPL 原生坐标系)
gar_mask = ~np.isin(gt_labels, NON_GARMENT_LABELS)
gt_gar_pts = gt_points[gar_mask] / scale - trans_npz.reshape(1, 3)

# GT 服装面片
gar_indices = np.where(gar_mask)[0]
gar_idx_set = set(gar_indices.tolist())
gar_face_mask = np.array([
    (f[0] in gar_idx_set) and (f[1] in gar_idx_set) and (f[2] in gar_idx_set)
    for f in gt_faces
])
gar_faces_filtered = gt_faces[gar_face_mask]
old_to_new = {old: new for new, old in enumerate(gar_indices)}
gar_faces_remapped = np.array([
    [old_to_new[f[0]], old_to_new[f[1]], old_to_new[f[2]]]
    for f in gar_faces_filtered
])

gt_mesh = trimesh.Trimesh(gt_gar_pts, gar_faces_remapped, process=False)
print(f"  GT garment mesh: {len(gt_gar_pts)} verts, {len(gar_faces_remapped)} faces")
print(f"  GT garment Y=[{gt_gar_pts[:,1].min():.3f}, {gt_gar_pts[:,1].max():.3f}]")

# ==================== 6. 采样并计算指标 ====================
print("\n" + "=" * 60)
print("6. 评估指标")
print("=" * 60)

pred_mesh = trimesh.Trimesh(garment_v_aligned, garment_f)

np.random.seed(42)
pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, N_SAMPLE)
gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, N_SAMPLE)

print(f"  Pred pts: {pred_pts.shape}, Y=[{pred_pts[:,1].min():.3f}, {pred_pts[:,1].max():.3f}]")
print(f"  GT   pts: {gt_pts.shape},   Y=[{gt_pts[:,1].min():.3f}, {gt_pts[:,1].max():.3f}]")

# CD & F-Score (point-to-POINT, 论文方式)
P = torch.tensor(pred_pts, dtype=torch.float32).unsqueeze(0)
Q = torch.tensor(gt_pts, dtype=torch.float32).unsqueeze(0)
D = torch.cdist(P, Q, p=2).squeeze(0)
p2g = D.min(dim=1).values.numpy()
g2p = D.min(dim=0).values.numpy()

cd = 0.5 * (p2g.mean() + g2p.mean()) * 100
pr_5 = (p2g < 0.005).mean()
rc_5 = (g2p < 0.005).mean()
fs_5 = (2 * pr_5 * rc_5 / (pr_5 + rc_5)) if (pr_5 + rc_5) > 0 else 0

print(f"\n  CD      = {cd:.3f} cm")
print(f"  F@5mm   = {fs_5:.4f}  (P={pr_5*100:.1f}%, R={rc_5*100:.1f}%)")
print(f"  P->GT   mean={p2g.mean()*100:.3f}cm  median={np.median(p2g)*100:.3f}cm")
print(f"  GT->P   mean={g2p.mean()*100:.3f}cm  median={np.median(g2p)*100:.3f}cm")

# 多 tau F-score
print(f"\n  多阈值 F-Score:")
for tau_mm in [5, 10, 20, 30, 50]:
    tau = tau_mm / 1000.0
    pr = (p2g < tau).mean()
    rc = (g2p < tau).mean()
    fs = (2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0
    print(f"    F@{tau_mm}mm = {fs:.4f}  (P={pr*100:.1f}%, R={rc*100:.1f}%)")

# ==================== 7. 导出调试文件 ====================
print("\n" + "=" * 60)
print("7. 导出调试文件")
print("=" * 60)

# 导出对齐后的 garment mesh
aligned_mesh_path = os.path.join(OUTPUT_DIR, "garment_smpl_kabsch.obj")
pred_mesh.export(aligned_mesh_path)
print(f"  {aligned_mesh_path} saved")

# 导出采样点云为 ply (手动写)
for name, pts in [
    ('debug_pred_smpl_kabsch.ply', pred_pts),
    ('debug_gt_smpl_kabsch.ply', gt_pts),
]:
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(pts))
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write("%f %f %f\n" % (p[0], p[1], p[2]))
    print(f"  {path} saved")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

try:
    import open3d as o3d
    from pathlib import Path
    for name, pts, col in [('debug_pred.ply', pred_pts, [1, 0, 0]),
                            ('debug_gt.ply', gt_pts, [0, 1, 0])]:

        # 打印两个点云坐标
        # print(f"[Debug] {name} 点云坐标: {pts[:5]}")

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(pts))
        pc.paint_uniform_color(col)
        output_dir = Path(OUTPUT_DIR)
        o3d.io.write_point_cloud(str(output_dir / name), pc)
except Exception as e:
    print(f"[Error] 保存调试点云失败: {e}")