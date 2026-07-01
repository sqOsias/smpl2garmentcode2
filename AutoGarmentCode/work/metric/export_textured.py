"""
从 Warp 仿真输出的驱动顶点 + 原始 design_sim.obj，导出带 UV 的 OBJ 并渲染。
反merge方案：驱动顶点映射回原始拓扑(18123v)，顶点数=UV数=面索引范围，无错位。
"""
import os, sys, shutil
import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial import KDTree
from collections import defaultdict

import platform
if platform.system() == 'Linux':
    os.environ["PYOPENGL_PLATFORM"] = "egl"

sys.path.insert(0, "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode")
from pygarment.meshgen.render.pythonrender import load_meshes, render

# ========================== 配置 ==========================
SAMPLE = "10072_7073"
ORIG_GARMENT = f"/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe/{SAMPLE}/design/design_sim.obj"
NPZ_PATH = f"/root/wyc/data/CloSe/data/CloSe-Di/{SAMPLE}.npz"
SMPL_JSON = f"/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe/{SAMPLE}/hybrik/smpl.json"
DRIVEN_MERGED_OBJ = f"/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe/{SAMPLE}/driven/final_result.obj"
APOSE_REF_OBJ = f"/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe/{SAMPLE}/driven/apose_merged_reference.obj"
PRED_SMPL_OBJ = f"/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe/{SAMPLE}/smpl.obj"
OUTPUT_DIR   = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/metric/result"
SMPL_MODEL_PATH = "/root/wyc/code/smpl2garmentcode2/smpl_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================== 1. 解析 OBJ UV/面 ==========================
def parse_obj_uv(path):
    uvs = []
    faces_v, faces_vt = [], []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('vt '):
                p = line.strip().split(); uvs.append([float(p[1]), float(p[2])])
            elif line.startswith('f '):
                p = line.strip().split()[1:]
                fv, fvt = [], []
                for x in p:
                    idx = x.split('/')
                    fv.append(int(idx[0]))
                    fvt.append(int(idx[1]) if len(idx)>1 and idx[1] else 1)
                faces_v.append(fv); faces_vt.append(fvt)
    return {'vt': np.array(uvs, dtype=np.float32), 'faces_v': faces_v, 'faces_vt': faces_vt}

# ========================== 2. 构建 merge 映射 ==========================
def build_merge_map(orig_obj_path, apose_merged_path):
    """用 driven_garment_mesh.py 导出的 A-pose merged 基准顶点做空间最近邻匹配。
    不再依赖 round 分组——直接用 KDTree 建立 原始顶点→merged顶点 的精确映射。"""
    orig_mesh = trimesh.load(orig_obj_path, process=False)
    orig_verts = np.array(orig_mesh.vertices)  # cm

    apose_merged = trimesh.load(apose_merged_path, process=False)
    apose_verts = np.array(apose_merged.vertices)  # m → cm
    if apose_verts[:, 1].max() < 5.0:
        apose_verts *= 100.0

    tree = KDTree(apose_verts)
    _, orig_to_merged = tree.query(orig_verts)
    return orig_to_merged, len(orig_verts), len(apose_verts)

# ========================== 3. 导出 OBJ (merged 拓扑 + merged UV) ==========================
def export_merged_obj(out_path, driven_merged_cm, apose_merged_path, orig_obj_path):
    """用 merged 拓扑 (25293v) + merged面 + 重建 UV。
    不再反merge到原始拓扑——避免 KDTree 多对一映射导致面索引错乱。"""
    # 原始 OBJ 的 UV 和面数据
    orig = trimesh.load(orig_obj_path, process=False)
    orig_verts = np.array(orig.vertices)
    orig_uv = orig.visual.uv.copy()

    # A-pose merged 面索引 (25293 顶点 → 面)
    apose = trimesh.load(apose_merged_path, process=False)
    apose_v = np.array(apose.vertices)
    apose_f = np.array(apose.faces)
    if apose_v[:,1].max() < 5.0:
        apose_v *= 100.0  # m → cm

    # 为每个 merged 顶点分配 UV：找最近的原始顶点位置，取其 UV
    n_merged = len(apose_v)
    merged_uv = np.zeros((n_merged, 2), dtype=np.float32)
    tree = KDTree(orig_verts)
    for mi in range(n_merged):
        _, oi = tree.query(apose_v[mi])
        merged_uv[mi] = orig_uv[oi]

    with open(out_path, 'w') as f:
        f.write("mtllib design_material.mtl\n")
        for v in driven_merged_cm:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for u in merged_uv:
            f.write(f"vt {u[0]:.6f} {u[1]:.6f}\n")
        for face in apose_f:
            a, b, c = int(face[0])+1, int(face[1])+1, int(face[2])+1
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

# ========================== 4. 生成 GT pose 身体 + 渲染 ==========================
def render_driven_gtpose(driven_obj_path, npz_path, smpl_json, out_dir, sample_name):
    """生成 GT pose 的 SMPL 身体并渲染（与驱动后服装姿态一致）"""
    import json, torch, smplx

    data = np.load(npz_path)
    with open(smpl_json) as f:
        pred = json.load(f)

    # GT body: npz betas + GT pose
    model = smplx.create(SMPL_MODEL_PATH, model_type='smpl', gender='male')
    gt_betas = torch.from_numpy(data['betas']).float().unsqueeze(0)
    gt_pose = torch.from_numpy(data['pose']).float().view(1, 72)
    with torch.no_grad():
        gt_body = model(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3]).vertices.squeeze().numpy()
    # 应用与 driven_garment_mesh.py 相同的 offset 逻辑
    pred_betas_t = torch.from_numpy(np.array(pred['betas'], dtype=np.float32)).unsqueeze(0)
    a_pose = torch.zeros(1, 69)
    with torch.no_grad():
        pred_body_raw = model(betas=pred_betas_t, body_pose=a_pose, global_orient=torch.zeros(1,3)).vertices.squeeze().numpy()
    pred_body_ref = np.array(trimesh.load(PRED_SMPL_OBJ, process=False).vertices, dtype=np.float32)
    align_offset = pred_body_ref.mean(axis=0) - pred_body_raw.mean(axis=0)
    gt_body = gt_body + align_offset

    # 导出临时 OBJ 供 load_meshes 加载
    body_obj = os.path.join(out_dir, "_gt_body_temp.obj")
    trimesh.Trimesh(gt_body, model.faces).export(body_obj)
    body_v = gt_body * 100.0  # m → cm (load_meshes 会 /100)
    body_f = np.array(model.faces, dtype=np.int32)

    out_path = Path(out_dir)
    class RP: pass
    rp = RP()
    rp.out_el = out_path; rp.g_sim = Path(driven_obj_path)
    rp.sim_tag = f"{sample_name}_driven"; rp.name = sample_name
    rp.render_path = lambda side: str(out_path / f"{sample_name}_driven_{side}.png")

    render_props = {'sides': ['front', 'back'], 'resolution': [1080, 1080],
                     'front_camera_location': [0, 0.97, 4.15]}

    for side in ['front', 'back']:
        py_garm, py_body = load_meshes(rp, body_v, body_f)  # 每次重新加载，避免 Mesh already bound
        render(py_garm, py_body, side, rp, render_props)
        print(f"  {rp.render_path(side)}")
    os.remove(body_obj)

# ========================== main ==========================
if __name__ == "__main__":
    print("1. 加载 A-pose merged 参考")
    apose_ref = trimesh.load(APOSE_REF_OBJ, process=False)
    n_merged = len(apose_ref.vertices)
    print(f"   merged verts count: {n_merged}")

    print("\n2. 加载驱动顶点")
    driven = trimesh.load(DRIVEN_MERGED_OBJ, process=False)
    driven_cm = np.array(driven.vertices, dtype=np.float64)
    if driven_cm[:,1].max() < 5.0:
        driven_cm *= 100.0  # m → cm
    print(f"   driven verts: {len(driven_cm)} (期望 {n_merged})")

    print("\n4. 导出 merged 拓扑 OBJ")
    driven_obj = os.path.join(OUTPUT_DIR, f"{SAMPLE}_driven_merged.obj")
    export_merged_obj(driven_obj, driven_cm, APOSE_REF_OBJ, ORIG_GARMENT)

    design_dir = os.path.dirname(ORIG_GARMENT)
    for f in ["design_material.mtl", "design_texture.png", "design_texture_fabric.png"]:
        src = os.path.join(design_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUTPUT_DIR, f))
    print(f"   {driven_obj}")

    # 验证
    tex = trimesh.load(driven_obj, process=False)
    deg = sum(1 for f in tex.faces if np.linalg.norm(np.cross(
        tex.vertices[f[0]]-tex.vertices[f[1]], tex.vertices[f[0]]-tex.vertices[f[2]])) < 1e-6)
    print(f"   顶点={tex.vertices.shape}, UV={tex.visual.uv.shape}, 退化面={deg}")

    print("\n5. 渲染 (GT pose 身体)")
    render_driven_gtpose(driven_obj, NPZ_PATH, SMPL_JSON, OUTPUT_DIR, SAMPLE)
    print("\nDONE")
