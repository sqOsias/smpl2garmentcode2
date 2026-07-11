#!/usr/bin/env python3
"""Export interpolated SMPL body meshes for each frame of pose transition.

Initial: smpl.json body_pose(69) + global_orient=0 (same as garment_driver.py L533)
Target:  NPZ pose(72) (same as garment_driver.py L538)

Slerp: scipy.spatial.transform.Slerp per joint, exactly matching garment_driver.py.

Usage:
    python work/export_interp_body.py 10001_1924 male
"""

import sys, os, json, scipy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch, trimesh, smplx
from scipy.spatial.transform import Rotation as R

SAMPLE   = sys.argv[1] if len(sys.argv) > 1 else '10001_1924'
GENDER   = sys.argv[2] if len(sys.argv) > 2 else 'male'
N_FRAMES = 80

BASE       = f'output/CloSe/{SAMPLE}'
SMPL_MODEL = '/root/wyc/code/smpl2garmentcode2/smpl_models'
NPZ        = f'/root/wyc/data/CloSe/data/CloSe-Di/{SAMPLE}.npz'
JSON       = f'{BASE}/hybrik/smpl.json'
OUT        = f'{BASE}/interp_body_frames'
os.makedirs(OUT, exist_ok=True)

print(f"=== Export Interpolated SMPL Body Frames ===")
print(f"  Sample: {SAMPLE}  Gender: {GENDER}  Frames: {N_FRAMES}")

# 1. Load
model = smplx.create(SMPL_MODEL, model_type='smpl', gender=GENDER)
with open(JSON) as f: smpl_data = json.load(f)
npz_data = np.load(NPZ)
pred_betas = torch.from_numpy(np.array(smpl_data['betas'],dtype=np.float32)).unsqueeze(0)

# betas from HybrIK, pose = build_default_pose (matching smpl.obj A-pose)
angle = np.pi/4.0; angle2 = np.pi/20.0
apose = torch.zeros((1,24,3),dtype=torch.float32)
apose[0,16,2] = -angle; apose[0,17,2] = angle
apose[0,16,1] = -angle2; apose[0,17,1] = angle2
base_go = torch.zeros(1,3)
base_bp = apose.view(1,72)[:,3:]
print(f"  Init: go=[0,0,0]  A-pose (build_default_pose, elbows=+-45deg)")

target_72 = torch.from_numpy(npz_data['pose'].astype(np.float32)).view(1,72)
target_go = target_72[:,:3]; target_bp = target_72[:,3:]
print(f"  Target: go={target_go.squeeze().numpy()}  bp range=[{target_bp.min():.3f},{target_bp.max():.3f}]")

# 2. Align
ref = np.array(trimesh.load(f'{BASE}/smpl.obj',process=False).vertices,dtype=np.float32)
with torch.no_grad():
    bn = model(betas=pred_betas,body_pose=base_bp,global_orient=base_go).vertices.squeeze().numpy()
offset = ref.mean(axis=0)-bn.mean(axis=0)
faces = model.faces.astype(np.int32)
print(f"  Offset: {offset}")

# 3. Export initial/target
with torch.no_grad():
    bi = model(betas=pred_betas,body_pose=base_bp,global_orient=base_go).vertices.squeeze().numpy()+offset
    bt = model(betas=pred_betas,body_pose=target_bp,global_orient=target_go).vertices.squeeze().numpy()+offset
trimesh.Trimesh(vertices=bi,faces=faces,process=False).export(os.path.join(OUT,"body_initial_apose.obj"))
trimesh.Trimesh(vertices=bt,faces=faces,process=False).export(os.path.join(OUT,"body_target_pose.obj"))
print(f"  body_initial_apose.obj / body_target_pose.obj saved")
print(f"  target disp: max={np.linalg.norm(bt-bi,axis=1).max()*100:.1f}cm")

# 4. Slerp frames
bb = base_bp.numpy().reshape(-1,3); tb = target_bp.numpy().reshape(-1,3)
bg = base_go.numpy().reshape(-1,3); tg = target_go.numpy().reshape(-1,3)
rb=R.from_rotvec(bb); rt=R.from_rotvec(tb)
rbg=R.from_rotvec(bg); rtg=R.from_rotvec(tg)

print(f"\n  Exporting {N_FRAMES} frames...")
prev=bi.copy()
for f in range(N_FRAMES):
    a = f/max(N_FRAMES-1,1)
    ibp=np.zeros_like(bb)
    for j in range(bb.shape[0]):
        rots=R.concatenate([rb[j],rt[j]])
        ibp[j]=scipy.spatial.transform.Slerp([0,1],rots)([a])[0].as_rotvec()
    igp=scipy.spatial.transform.Slerp([0,1],R.concatenate([rbg[0],rtg[0]]))([a])[0].as_rotvec().reshape(1,3)
    ib=torch.from_numpy(ibp.reshape(1,69)).float(); ig=torch.from_numpy(igp).float()
    with torch.no_grad():
        bv=model(betas=pred_betas,body_pose=ib,global_orient=ig).vertices.squeeze().numpy()+offset
    trimesh.Trimesh(vertices=bv,faces=faces,process=False).export(os.path.join(OUT,f"body_{f:04d}.obj"))
    if f%10==0:
        da=np.max(np.linalg.norm(bv-bi,axis=1)); dp=np.max(np.linalg.norm(bv-prev,axis=1))
        print(f"  [{f:3d}/{N_FRAMES}] a={a:.3f}  from_init={da*100:.1f}cm  step={dp*100:.1f}cm")
    prev=bv.copy()
print(f"\nDone: {OUT}/")
