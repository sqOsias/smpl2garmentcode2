"""
从 Warp 仿真输出的驱动顶点 + 原始 design_sim.obj，导出带 UV 的 OBJ 并渲染。
不改动现有仿真脚本，独立运行。
写 16832 merged 顶点 + 重映射面索引（不展开到原始拓扑，避免重复顶点导致 trimesh 加载出错）。
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
SAMPLE = "10014_2464"
ORIG_GARMENT = f"/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe/{SAMPLE}/design/design_sim.obj"
SMPL_OBJ     = f"/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe/{SAMPLE}/smpl.obj"
DRIVEN_MERGED_OBJ = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/metric/result/final_result.obj"
OUTPUT_DIR   = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/metric/result"
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
def build_merge_map(orig_obj_path):
    orig_mesh = trimesh.load(orig_obj_path, process=False)
    orig_verts = np.array(orig_mesh.vertices)

    groups = defaultdict(list)
    for i in range(len(orig_verts)):
        groups[tuple(orig_verts[i].round(3))].append(i)

    merged_mesh = orig_mesh.copy()
    merged_mesh.merge_vertices(merge_tex=True, merge_norm=True)
    merged_verts = np.array(merged_mesh.vertices)

    tree = KDTree(merged_verts)
    orig_to_merged = np.zeros(len(orig_verts), dtype=np.int32)
    for key, members in groups.items():
        _, mi = tree.query(np.array(key))
        for orig_i in members:
            orig_to_merged[orig_i] = mi

    return orig_to_merged, len(orig_verts), len(merged_verts)

# ========================== 3. 导出 merged 顶点 OBJ ==========================
def export_merged_obj(out_path, driven_merged_cm, orig_to_merged, uv_data):
    orig_vt = uv_data['vt']
    orig_fv = uv_data['faces_v']; orig_fvt = uv_data['faces_vt']

    with open(out_path, 'w') as f:
        f.write("mtllib design_material.mtl\n")
        # 顶点: 16832 merged 位置
        for v in driven_merged_cm:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # UV: 保留全部 17668 个原始 UV (面UV索引仍然有效)
        for u in orig_vt:
            f.write(f"vt {u[0]:.6f} {u[1]:.6f}\n")
        # 面: 顶点ID重映射, UV ID 保持不变(参考原始17668 UV数组)
        for i in range(len(orig_fv)):
            oa, ob, oc = orig_fv[i][0], orig_fv[i][1], orig_fv[i][2]
            ma = orig_to_merged[oa-1] + 1
            mb = orig_to_merged[ob-1] + 1
            mc = orig_to_merged[oc-1] + 1
            ta, tb, tc = orig_fvt[i]
            f.write(f"f {ma}/{ta} {mb}/{tb} {mc}/{tc}\n")

# ========================== 4. 渲染 ==========================
def render_driven(driven_obj_path, body_obj_path, out_dir, sample_name):
    out_path = Path(out_dir)
    class RP: pass
    rp = RP()
    rp.out_el = out_path; rp.g_sim = Path(driven_obj_path)
    rp.sim_tag = f"{sample_name}_driven"; rp.name = sample_name
    rp.render_path = lambda side: str(out_path / f"{sample_name}_driven_{side}.png")

    body_mesh = trimesh.load(body_obj_path, process=False)
    body_v = np.array(body_mesh.vertices, dtype=np.float32)
    body_f = np.array(body_mesh.faces, dtype=np.int32)

    render_props = {'sides': ['front', 'back'], 'resolution': [1080, 1080],
                     'front_camera_location': [0, 0.97, 4.15]}

    py_garm, py_body = load_meshes(rp, body_v, body_f)
    for side in ['front', 'back']:
        render(py_garm, py_body, side, rp, render_props)
        print(f"  {rp.render_path(side)}")

# ========================== main ==========================
if __name__ == "__main__":
    print("1. 解析原始 OBJ")
    uv_data = parse_obj_uv(ORIG_GARMENT)
    print(f"   UV: {len(uv_data['vt'])}, faces: {len(uv_data['faces_v'])}")

    print("\n2. 构建 merge 映射")
    orig_to_merged, n_orig, n_merged = build_merge_map(ORIG_GARMENT)
    print(f"   {n_orig} -> {n_merged}")

    print("\n3. 加载驱动顶点")
    driven = trimesh.load(DRIVEN_MERGED_OBJ, process=False)
    driven_cm = np.array(driven.vertices, dtype=np.float64)
    if driven_cm[:,1].max() < 5.0:
        driven_cm *= 100.0  # m -> cm
    print(f"   merged verts: {len(driven_cm)}")
    assert len(driven_cm) == n_merged, f"顶点数不匹配: {len(driven_cm)} vs {n_merged}"

    print("\n4. 导出 merged OBJ")
    driven_obj = os.path.join(OUTPUT_DIR, f"{SAMPLE}_driven_merged.obj")
    export_merged_obj(driven_obj, driven_cm, orig_to_merged, uv_data)

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
    print(f"   退化面: {deg} (期望0)")

    print("\n5. 渲染")
    render_driven(driven_obj, SMPL_OBJ, OUTPUT_DIR, SAMPLE)
    print("\nDONE")
