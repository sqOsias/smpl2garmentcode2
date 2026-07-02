import os, sys, shutil, json
import numpy as np
import trimesh
import torch
import smplx
from pathlib import Path

import platform
if platform.system() == 'Linux':
    os.environ["PYOPENGL_PLATFORM"] = "egl"

sys.path.insert(0, "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode")
from pygarment.meshgen.render.pythonrender import load_meshes, render

SAMPLE = "10072_7073"
ORIG_GARMENT = f"/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe/{SAMPLE}/design/design_sim.obj"
NPZ_PATH = f"/root/wyc/data/CloSe/data/CloSe-Di/{SAMPLE}.npz"
SMPL_JSON = f"/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe/{SAMPLE}/hybrik/smpl.json"
DRIVEN_MERGED_OBJ = f"/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe/{SAMPLE}/driven/final_result.obj"
PRED_SMPL_OBJ = f"/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe/{SAMPLE}/smpl.obj"
OUTPUT_DIR = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/metric/result"
SMPL_MODEL_PATH = "/root/wyc/code/smpl2garmentcode2/smpl_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_correct_uv_obj(out_path, driven_cm, orig_obj_path):
    """merge_vertices(merge_tex=True) preserves UVs per merged vertex."""
    orig = trimesh.load(orig_obj_path, process=False)
    orig.merge_vertices(merge_tex=True, merge_norm=True)
    merged_uv = orig.visual.uv
    merged_faces = orig.faces

    with open(out_path, 'w') as f:
        f.write("mtllib design_material.mtl\n")
        for v in driven_cm:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for u in merged_uv:
            f.write(f"vt {u[0]:.6f} {u[1]:.6f}\n")
        for face in merged_faces:
            a, b, c = int(face[0])+1, int(face[1])+1, int(face[2])+1
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")


def render_driven_gtpose(driven_obj_path, npz_path, smpl_json, out_dir, sample_name):
    """直接使用 pyrender 渲染，不做 trimesh process(避免打乱 UV)。"""
    import pyrender as pyr
    from PIL import Image as PILImage

    data = np.load(npz_path)
    with open(smpl_json) as f:
        pred = json.load(f)

    model = smplx.create(SMPL_MODEL_PATH, model_type='smpl', gender='male')
    gt_betas = torch.from_numpy(data['betas']).float().unsqueeze(0)
    gt_pose = torch.from_numpy(data['pose']).float().view(1, 72)
    with torch.no_grad():
        gt_body = model(betas=gt_betas, body_pose=gt_pose[:,3:],
                        global_orient=gt_pose[:,:3]).vertices.squeeze().numpy()
    pred_betas_t = torch.from_numpy(np.array(pred['betas'], dtype=np.float32)).unsqueeze(0)
    a_pose = torch.zeros(1, 69)
    with torch.no_grad():
        pred_body_native = model(betas=pred_betas_t, body_pose=a_pose,
                                 global_orient=torch.zeros(1,3)).vertices.squeeze().numpy()
    pred_body_ref = np.array(trimesh.load(PRED_SMPL_OBJ, process=False).vertices, dtype=np.float32)
    smpl_offset = pred_body_ref.mean(axis=0) - pred_body_native.mean(axis=0)
    gt_body = gt_body + smpl_offset
    print(f"  GT body Y=[{gt_body[:,1].min():.2f},{gt_body[:,1].max():.2f}] m")

    # Body mesh (米制)
    body_mesh = trimesh.Trimesh(gt_body, model.faces, process=False)
    body_mat = pyr.MetallicRoughnessMaterial(baseColorFactor=(0.62,0.58,0.55,1.0), metallicFactor=0., roughnessFactor=0.8)
    py_body = pyr.Mesh.from_trimesh(body_mesh, material=body_mat)

    # Garment mesh (从 OBJ 加载, cm→m)
    garm_mesh = trimesh.load_mesh(driven_obj_path)
    garm_mesh.vertices /= 100.0
    mat = garm_mesh.visual.material.to_pbr()
    mat.baseColorFactor = [1.,1.,1.,1.]; mat.doubleSided = True
    white = PILImage.new('RGBA', mat.baseColorTexture.size, (255,255,255,255))
    white.paste(mat.baseColorTexture)
    mat.baseColorTexture = white.convert('RGB')
    garm_mesh.visual.material = mat
    py_garm = pyr.Mesh.from_trimesh(garm_mesh, smooth=True)

    def rot_y(m, deg):
        a = np.radians(deg)
        R = np.array([[np.cos(a),0,np.sin(a),0],[0,1,0,0],[-np.sin(a),0,np.cos(a),0],[0,0,0,1]])
        return R @ m
    def rot_x(m, deg):
        a = np.radians(deg)
        R = np.array([[1,0,0,0],[0,np.cos(a),-np.sin(a),0],[0,np.sin(a),np.cos(a),0],[0,0,0,1]])
        return R @ m

    cam_loc = [0, 0.97, 4.15]
    for side in ['front', 'back']:
        scene = pyr.Scene(bg_color=(0.95,0.95,0.95,1.0))
        scene.add(py_garm); scene.add(py_body)
        cam_pose = np.eye(4); cam_pose[:3,3] = cam_loc
        cam_pose = rot_x(cam_pose, -15); cam_pose = rot_y(cam_pose, 20)
        if side == 'back': cam_pose = rot_y(cam_pose, 180)
        scene.add(pyr.PerspectiveCamera(yfov=np.pi/6.), pose=cam_pose)
        for lp in [[1.6,1.5,1.2],[1.3,1.9,-2.5],[-2.8,1.3,2.3],[0.2,1.8,3.5],[-2.7,1.4,-1.3]]:
            light = pyr.PointLight(color=[1,1,1], intensity=80.)
            lp_m = np.eye(4); lp_m[:3,3] = lp; scene.add(light, pose=lp_m)
        r = pyr.OffscreenRenderer(1080, 1080)
        color, _ = r.render(scene, flags=pyr.RenderFlags.RGBA)
        out = os.path.join(out_dir, f"{sample_name}_driven_{side}.png")
        PILImage.fromarray(color).save(out)
        r.delete()
        print(f"  {out}")


if __name__ == "__main__":
    print("1. load apose merged reference")
    driven = trimesh.load(DRIVEN_MERGED_OBJ, process=False)
    driven_cm = np.array(driven.vertices, dtype=np.float64)
    if driven_cm[:,1].max() < 5.0:
        driven_cm *= 100.0
    print(f"   driven verts: {len(driven_cm)}")

    print("\n2. export merged OBJ with correct UVs")
    driven_obj = os.path.join(OUTPUT_DIR, f"{SAMPLE}_driven_merged.obj")
    export_correct_uv_obj(driven_obj, driven_cm, ORIG_GARMENT)

    design_dir = os.path.dirname(ORIG_GARMENT)
    for f in ["design_material.mtl", "design_texture.png", "design_texture_fabric.png"]:
        src = os.path.join(design_dir, f)
        if os.path.exists(src): shutil.copy(src, os.path.join(OUTPUT_DIR, f))
    print(f"   {driven_obj}")

    tex = trimesh.load(driven_obj, process=False)
    print(f"   verts={tex.vertices.shape}, UV={tex.visual.uv.shape}")

    print("\n3. render with GT pose body")
    render_driven_gtpose(driven_obj, NPZ_PATH, SMPL_JSON, OUTPUT_DIR, SAMPLE)
    print("\nDONE")
