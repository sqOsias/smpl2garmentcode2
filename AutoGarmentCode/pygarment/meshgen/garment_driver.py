"""
SMPL 姿态驱动的服装网格变形模块。

从 driven_garment_mesh.py 提取核心逻辑，封装为可复用的函数接口。
通过 Warp 物理仿真将服装驱动到目标 SMPL 姿态，输出带 UV 的 OBJ 和渲染图。
"""

import os
import json
import shutil
import yaml
import numpy as np
import torch
import trimesh
import warp as wp
import warp.sim
import smplx
from scipy.spatial.transform import Rotation as R
import scipy

# 初始化 Warp (惰性初始化，避免 import 时就必须有 CUDA)
_wp_initialized = False

def _ensure_wp_init():
    global _wp_initialized
    if not _wp_initialized:
        wp.init()
        wp.set_device("cuda:0")
        _wp_initialized = True


# ======================================
# 1. 工具函数
# ======================================

def load_betas_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    if "betas" not in data:
        raise ValueError(f"Missing 'betas' field in {json_path}")
    betas = np.asarray(data["betas"], dtype=np.float32)
    if betas.ndim != 1:
        raise ValueError(f"'betas' must be a 1D array, got shape {betas.shape}")
    return betas


def load_pose_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    if "pose" not in data:
        raise ValueError(f"Missing 'pose' field in {json_path}")
    pose = np.asarray(data["pose"], dtype=np.float32)
    if pose.ndim != 1:
        raise ValueError(f"'pose' must be a 1D array, got shape {pose.shape}")
    return pose


def load_sim_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["sim"]["config"]


def _resolve_obj_index(raw_index, count, field_name, obj_path):
    """Resolve a one-based (or negative) OBJ index to a zero-based index."""
    index = int(raw_index)
    resolved = index - 1 if index > 0 else count + index
    if resolved < 0 or resolved >= count:
        raise ValueError(
            f"Invalid {field_name} index {index} in {obj_path}; "
            f"the file contains {count} {field_name} entries"
        )
    return resolved


def _load_obj_topology(obj_path):
    """Load OBJ positions, UVs and their independent face indices.

    BoxMesh has already collapsed actual pattern stitches before serializing the
    OBJ.  Reading the position indices directly therefore preserves the sewn
    physical topology, while keeping the per-panel UV seams separate.  This
    avoids Trimesh expanding vertices for UV seams and a later coordinate-based
    merge accidentally welding unrelated, coincident garment layers.
    """
    vertices = []
    uvs = []
    faces = []
    face_uvs = []

    with open(obj_path, "r", encoding="utf-8") as obj_file:
        for line_number, line in enumerate(obj_file, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            record = fields[0]

            if record == "v" and len(fields) >= 4:
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif record == "vt" and len(fields) >= 3:
                uvs.append([float(fields[1]), float(fields[2])])
            elif record == "f":
                if len(fields) < 4:
                    raise ValueError(f"Invalid face at {obj_path}:{line_number}")

                position_indices = []
                texture_indices = []
                for corner in fields[1:]:
                    components = corner.split("/")
                    if not components[0]:
                        raise ValueError(
                            f"Missing position index at {obj_path}:{line_number}"
                        )
                    position_indices.append(
                        _resolve_obj_index(components[0], len(vertices), "vertex", obj_path)
                    )
                    if len(components) > 1 and components[1]:
                        texture_indices.append(
                            _resolve_obj_index(components[1], len(uvs), "texture", obj_path)
                        )
                    else:
                        texture_indices.append(-1)

                # BoxMesh writes triangles, but fan triangulation keeps the
                # loader well-defined if a compatible OBJ contains polygons.
                for corner_idx in range(1, len(position_indices) - 1):
                    tri = [0, corner_idx, corner_idx + 1]
                    faces.append([position_indices[idx] for idx in tri])
                    face_uvs.append([texture_indices[idx] for idx in tri])

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    uvs = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
    face_uvs = np.asarray(face_uvs, dtype=np.int32).reshape(-1, 3)

    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"OBJ contains no valid vertices: {obj_path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"OBJ contains no valid triangle faces: {obj_path}")

    return vertices, faces, uvs, face_uvs


def _load_stitched_garment_topology(garment_obj_path, boxmesh_obj_path):
    """Load one shared BoxMesh topology for simulation and UV export."""
    garment_vertices, garment_faces, _, _ = _load_obj_topology(garment_obj_path)
    template_vertices, template_faces, template_uvs, template_face_uvs = (
        _load_obj_topology(boxmesh_obj_path)
    )

    if len(garment_vertices) != len(template_vertices):
        raise ValueError(
            "Garment and BoxMesh template vertex counts differ: "
            f"{len(garment_vertices)} != {len(template_vertices)}. "
            "They must originate from the same BoxMesh serialization."
        )
    if garment_faces.shape != template_faces.shape or not np.array_equal(
        garment_faces, template_faces
    ):
        raise ValueError(
            "Garment and BoxMesh template position topology differ. "
            "Refusing to drive with an ambiguous vertex/UV mapping."
        )
    if len(template_uvs) == 0 or np.any(template_face_uvs < 0):
        raise ValueError(f"BoxMesh template has incomplete UV indices: {boxmesh_obj_path}")

    return garment_vertices, garment_faces, template_uvs, template_face_uvs


def build_cloth_springs(verts, faces):
    """从服装三角网格构建弹簧系统。返回结构边、弯曲边及其原长。"""
    num_verts = len(verts)
    edges_set = set()
    struct_edges = []

    for f in faces:
        for i in range(3):
            v0, v1 = f[i], f[(i + 1) % 3]
            if v0 > v1:
                v0, v1 = v1, v0
            if (v0, v1) not in edges_set:
                edges_set.add((v0, v1))
                struct_edges.append([v0, v1])

    struct_edges = np.array(struct_edges, dtype=np.int32)
    struct_rest_lengths = np.linalg.norm(
        verts[struct_edges[:, 0]] - verts[struct_edges[:, 1]], axis=1
    ).astype(np.float32)

    edge_face_map = {}
    for face_idx, f in enumerate(faces):
        for i in range(3):
            v0, v1 = f[i], f[(i + 1) % 3]
            if v0 > v1:
                v0, v1 = v1, v0
            key = (v0, v1)
            if key not in edge_face_map:
                edge_face_map[key] = []
            edge_face_map[key].append(face_idx)

    bend_edges = []
    for edge, face_indices in edge_face_map.items():
        if len(face_indices) == 2:
            f0, f1 = faces[face_indices[0]], faces[face_indices[1]]
            v0, v1 = edge
            opp0 = [v for v in f0 if v != v0 and v != v1][0]
            opp1 = [v for v in f1 if v != v0 and v != v1][0]
            bend_edges.append([opp0, opp1])

    bend_edges = np.array(bend_edges, dtype=np.int32)
    bend_rest_lengths = np.linalg.norm(
        verts[bend_edges[:, 0]] - verts[bend_edges[:, 1]], axis=1
    ).astype(np.float32)

    return struct_edges, struct_rest_lengths, bend_edges, bend_rest_lengths


def compute_particle_masses(verts, faces, density):
    """根据面密度计算每个粒子的质量（面积加权分配）。"""
    num_verts = len(verts)
    masses = np.zeros(num_verts, dtype=np.float32)

    for f in faces:
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        face_mass = area * density
        masses[f[0]] += face_mass / 3.0
        masses[f[1]] += face_mass / 3.0
        masses[f[2]] += face_mass / 3.0

    return masses


def find_attachment_points(garment_verts, body_verts):
    """找服装顶点中距离人体极近的点（肩/胸部区域），作为绑定点。"""
    attachment_idx = []
    attachment_body_idx = []

    threshold = 0.015
    body_max_y = body_verts[:, 1].max()
    safe_y_min = body_max_y - 0.35

    for i, gv in enumerate(garment_verts):
        if gv[1] < safe_y_min:
            continue
        dists = np.linalg.norm(body_verts - gv, axis=1)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        if min_dist < threshold:
            attachment_idx.append(i)
            attachment_body_idx.append(min_idx)

    return np.array(attachment_idx), np.array(attachment_body_idx)


# ======================================
# 2. Warp GPU Kernels
# ======================================

@wp.kernel
def body_collision_kernel(
    cloth_pos: wp.array(dtype=wp.vec3),
    cloth_vel: wp.array(dtype=wp.vec3),
    body_mesh_id: wp.uint64,
    collision_thickness: float,
    friction: float,
    query_dist: float,
):
    tid = wp.tid()
    p = cloth_pos[tid]
    v = cloth_vel[tid]

    query = wp.mesh_query_point_sign_normal(body_mesh_id, p, query_dist)
    if not query.result:
        return

    cp = wp.mesh_eval_position(body_mesh_id, query.face, query.u, query.v)
    delta = p - cp
    dist_sq = wp.dot(delta, delta)
    if dist_sq <= 1e-12:
        return
    dist = wp.sqrt(dist_sq)

    n = (delta / dist) * query.sign
    signed_dist = dist * query.sign

    if signed_dist < collision_thickness:
        push = collision_thickness - signed_dist
        push = wp.min(push, collision_thickness * 0.8)
        cloth_pos[tid] = p + n * push

        vn = wp.dot(v, n) * n
        vt = v - vn
        cloth_vel[tid] = vt * (1.0 - friction)


@wp.kernel
def attachment_kernel(
    cloth_pos: wp.array(dtype=wp.vec3),
    cloth_vel: wp.array(dtype=wp.vec3),
    cloth_idx: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3),
    strength: float,
    velocity_scale: float,
):
    tid = wp.tid()
    idx = cloth_idx[tid]
    p = cloth_pos[idx]
    t = target_pos[tid]
    cloth_pos[idx] = p + (t - p) * strength
    cloth_vel[idx] = cloth_vel[idx] * velocity_scale


# ======================================
# 3. SMPL 人体驱动
# ======================================

class SMPLDriver:
    """通过 SMPL 模型生成不同姿态的人体顶点，并与参考人体坐标系对齐。"""

    def __init__(self, model_path, gender='male',
                 base_betas=None, base_body_pose=None, base_global_orient=None):
        self.model = smplx.create(model_path, model_type='smpl', gender=gender, ext='npz')
        self.faces = self.model.faces.astype(np.int32)

        with torch.no_grad():
            output = self.model(
                betas=base_betas,
                body_pose=base_body_pose,
                global_orient=base_global_orient,
            )
            self.base_verts = output.vertices.squeeze().numpy().astype(np.float32)

        self.align_offset = np.zeros(3, dtype=np.float32)

    def align_to_reference(self, reference_verts):
        self.align_offset = (
            reference_verts.mean(axis=0) - self.base_verts.mean(axis=0)
        ).astype(np.float32)
        return self.align_offset

    def get_body_verts(self, betas=None, body_pose=None, global_orient=None):
        with torch.no_grad():
            output = self.model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
            )
        verts = output.vertices.squeeze().numpy().astype(np.float32)
        return verts + self.align_offset


# ======================================
# 4. Warp 布料仿真器
# ======================================

class WarpClothSimulator:
    """基于 Warp 弹簧-质点模型的布料仿真器，带人体碰撞检测。"""

    def __init__(self, garment_verts, garment_faces, body_verts, body_faces, sim_config):
        _ensure_wp_init()
        self.device = "cuda:0"
        self.config = sim_config
        self.num_cloth_verts = len(garment_verts)

        # ---------- 弹簧 ----------
        struct_edges, struct_rest, bend_edges, bend_rest = build_cloth_springs(
            garment_verts, garment_faces
        )
        self.struct_edges = struct_edges
        self.bend_edges = bend_edges

        # ---------- 质量 ----------
        density = sim_config["material"]["fabric_density"]
        particle_masses = compute_particle_masses(garment_verts, garment_faces, density)
        MASS_FLOOR = 1e-3
        particle_masses = np.maximum(particle_masses, MASS_FLOOR)
        self.min_mass = float(particle_masses.min())

        # ---------- 稳定性上限 ----------
        self.num_substeps = int(sim_config.get("num_substeps", 120))
        sub_dt = (1.0 / 60.0) / self.num_substeps
        _VALENCE = 12.0
        _SAFETY_KE = 0.1
        _SAFETY_KD = 0.25
        ke_cap = _SAFETY_KE * self.min_mass / (_VALENCE * sub_dt * sub_dt)
        kd_cap = _SAFETY_KD * 2.0 * self.min_mass / (_VALENCE * sub_dt)

        # ---------- Model builder ----------
        builder = wp.sim.ModelBuilder()

        for i in range(self.num_cloth_verts):
            builder.add_particle(
                pos=wp.vec3(*garment_verts[i]),
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=float(particle_masses[i]),
            )

        # 保持原驱动器的参数语义：garment_edge_* 用于网格结构边，
        # spring_* 用于跨共享边的对角弹簧。不要与 Warp cloth triangle
        # 模型的 tri_* 参数直接互换，两种模型的刚度含义并不等价。
        edge_ke = min(max(float(sim_config["material"]["garment_edge_ke"]), 1000.0), ke_cap)
        edge_kd = min(float(sim_config["material"]["garment_edge_kd"]), kd_cap)
        for i in range(len(struct_edges)):
            a, b = struct_edges[i]
            builder.add_spring(int(a), int(b), ke=edge_ke, kd=edge_kd, control=0.0)

        bend_ke = min(float(sim_config["material"]["spring_ke"]), ke_cap) * 0.5
        bend_kd = min(float(sim_config["material"]["spring_kd"]), kd_cap) * 2.0
        for i in range(len(bend_edges)):
            a, b = bend_edges[i]
            builder.add_spring(int(a), int(b), ke=bend_ke, kd=bend_kd, control=0.0)

        print(f"  [driver] 弹簧刚度钳制: struct ke={edge_ke:.1f} kd={edge_kd:.2f} | "
              f"bend ke={bend_ke:.1f} kd={bend_kd:.2f} (cap ke={ke_cap:.1f} kd={kd_cap:.2f})")

        builder.gravity = wp.vec3(0.0, -9.8, 0.0)
        self.model = builder.finalize()
        self.model.ground = sim_config["ground"]

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.integrator = wp.sim.SemiImplicitIntegrator()

        # ---------- 人体碰撞网格 ----------
        self.body_mesh = wp.Mesh(
            points=wp.array(body_verts, dtype=wp.vec3, device=self.device),
            indices=wp.array(body_faces.flatten(), dtype=int, device=self.device),
        )
        raw_thickness_cm = sim_config["options"]["body_collision_thickness"]
        self.collision_thickness = float(min(raw_thickness_cm * 0.01, 0.005))
        self.body_friction = sim_config["options"]["body_friction"]
        self.query_dist = float(max(self.collision_thickness * 20.0, 0.1))

        # ---------- 绑定约束 ----------
        self.enable_attachment = sim_config["options"]["enable_attachment_constraint"]
        if self.enable_attachment:
            self.attach_cloth_idx, self.attach_body_idx = find_attachment_points(
                garment_verts, body_verts
            )
            print(f"  [driver] 绑定点数量: {len(self.attach_cloth_idx)}")
            if len(self.attach_cloth_idx) == 0:
                self.enable_attachment = False
            else:
                self.attach_cloth_idx_wp = wp.array(
                    self.attach_cloth_idx, dtype=wp.int32, device=self.device
                )
            self.attach_stiffness = sim_config["options"]["attachment_stiffness"][0]
            self.attach_damping = sim_config["options"]["attachment_damping"][0]
            self.attach_frames = sim_config["options"]["attachment_frames"]
            self.current_frame = 0

    def update_body_collider(self, body_verts_np):
        self.body_mesh.points.assign(
            wp.array(body_verts_np, dtype=wp.vec3, device=self.device)
        )
        self.body_mesh.refit()

    def update_attachment_body_idx(self, body_verts_np):
        if not self.enable_attachment:
            return
        from scipy.spatial import KDTree
        tree = KDTree(body_verts_np)
        cloth_pos = self.state_0.particle_q.numpy()
        new_body_idx = []
        for ci in self.attach_cloth_idx:
            _, ni = tree.query(cloth_pos[ci])
            new_body_idx.append(ni)
        self.attach_body_idx = np.array(new_body_idx, dtype=np.int32)

    def _apply_attachment(self, body_verts_np, dt):
        if not self.enable_attachment:
            return
        alpha = min(1.0, self.current_frame / max(1, self.attach_frames))
        current_stiff = self.attach_stiffness * alpha
        # attachment_stiffness 来自原 Warp 约束配置，不能直接作为单帧
        # 位置插值系数。旧公式在 current_stiff=600 时突然钳到 1.0，
        # 会把绑定点瞬间硬投影到人体并使相邻弹簧爆炸。这里将其映射为
        # 连续时间响应率，并限制每帧最多修正 25% 的剩余距离。
        response_rate = current_stiff * 0.01
        strength = float(min(0.25, 1.0 - np.exp(-response_rate * dt)))
        if strength <= 0.0:
            return
        target_pos = body_verts_np[self.attach_body_idx].astype(np.float32)
        target_pos_wp = wp.array(target_pos, dtype=wp.vec3, device=self.device)
        velocity_scale = float(max(0.0, 1.0 - self.attach_damping * dt))
        wp.launch(
            kernel=attachment_kernel,
            dim=len(self.attach_cloth_idx),
            inputs=[
                self.state_0.particle_q, self.state_0.particle_qd,
                self.attach_cloth_idx_wp, target_pos_wp, strength, velocity_scale,
            ],
            device=self.device,
        )

    def step(self, body_verts_np, dt=1.0 / 60.0, gravity_enabled=True):
        self.current_frame += 1

        if not gravity_enabled:
            self.model.gravity = wp.vec3(0.0, 0.0, 0.0)
        else:
            self.model.gravity = wp.vec3(0.0, -9.8, 0.0)

        sub_dt = dt / self.num_substeps

        self._apply_attachment(body_verts_np, dt)

        for _ in range(self.num_substeps):
            self.state_0.clear_forces()
            self.integrator.simulate(self.model, self.state_0, self.state_1, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(
                kernel=body_collision_kernel,
                dim=self.num_cloth_verts,
                inputs=[
                    self.state_0.particle_q, self.state_0.particle_qd,
                    self.body_mesh.id, self.collision_thickness,
                    self.body_friction, self.query_dist,
                ],
                device=self.device,
            )

        # 速度阻尼 + 钳制
        damping = self.config["options"]["global_damping_factor"]
        max_vel = self.config["options"]["global_max_velocity"] * 0.01  # cm/s → m/s
        vel_np = self.state_0.particle_qd.numpy()
        pos_np = self.state_0.particle_q.numpy()
        if not np.isfinite(pos_np).all() or not np.isfinite(vel_np).all():
            raise FloatingPointError(
                f"Cloth solver produced NaN/Inf at simulation frame {self.current_frame}"
            )
        if damping > 0:
            vel_np *= (1.0 - damping)
        norms = np.linalg.norm(vel_np, axis=1, keepdims=True)
        mask = (norms > max_vel).squeeze(-1)
        if np.any(mask):
            vel_np[mask] = vel_np[mask] / norms[mask] * max_vel
        self.state_0.particle_qd.assign(wp.array(vel_np, dtype=wp.vec3, device=self.device))

        return pos_np


# ======================================
# 5. UV 恢复工具
# ======================================

def _export_driven_obj_with_uv(
    out_path, driven_verts_m, garment_faces, template_uvs, template_face_uvs
):
    """使用 BoxMesh 的独立位置/UV 索引导出驱动后的 OBJ。

    Args:
        out_path: 输出 OBJ 路径
        driven_verts_m: 驱动后顶点 (米)
        garment_faces: BoxMesh 已缝合的物理面索引
        template_uvs: BoxMesh 模板的纹理坐标
        template_face_uvs: 每个物理面的独立纹理坐标索引
    """
    if len(garment_faces) != len(template_face_uvs):
        raise ValueError(
            "Physical face count and UV face count differ during OBJ export: "
            f"{len(garment_faces)} != {len(template_face_uvs)}"
        )
    if len(garment_faces) and int(np.max(garment_faces)) >= len(driven_verts_m):
        raise ValueError("Physical face index exceeds driven garment vertex count")
    if len(template_face_uvs) and int(np.max(template_face_uvs)) >= len(template_uvs):
        raise ValueError("Texture face index exceeds BoxMesh UV count")

    driven_cm = driven_verts_m * 100.0  # m → cm (匹配 OBJ 约定)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("mtllib design_material.mtl\n")
        f.write("usemtl panels_texture\n")
        for v in driven_cm:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for u in template_uvs:
            f.write(f"vt {u[0]:.6f} {u[1]:.6f}\n")
        for face, face_uv in zip(garment_faces, template_face_uvs):
            corners = [
                f"{int(vertex_idx) + 1}/{int(uv_idx) + 1}"
                for vertex_idx, uv_idx in zip(face, face_uv)
            ]
            f.write(f"f {' '.join(corners)}\n")


def _copy_material_files(src_dir, dst_dir):
    """复制 MTL 和纹理文件到目标目录。"""
    for fname in ["design_material.mtl", "design_texture.png", "design_texture_fabric.png"]:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dst_dir, fname))


# ======================================
# 6. 主接口：驱动服装到目标姿态
# ======================================

def drive_garment(
    garment_obj_path,
    boxmesh_obj_path,
    body_obj_path,
    smpl_model_path,
    base_smpl_json,
    target_pose_npz,
    target_data=None,
    gender='male',
    output_dir=None,
    sim_config_path=None,
    save_intermediate=True,
    drive_to_gt_shape=False,
):
    """将服装从基准姿态驱动到目标 SMPL 姿态。

    Args:
        garment_obj_path:  仿真后的服装 OBJ (cm)
        boxmesh_obj_path:  boxmesh 模板 OBJ (含 UV/face，cm)
        body_obj_path:     参考人体 OBJ (米)
        smpl_model_path:   SMPL 模型目录
        base_smpl_json:    基准 SMPL 参数 JSON {betas, pose}
        target_pose_npz:   目标姿态 NPZ 路径；target_data 未提供时使用
        target_data:       已加载的目标 NPZ 字典，用于与后续指标计算复用
        gender:            'male' / 'female'
        output_dir:        输出目录 (默认在 garment 同级创建 'driven/')
        sim_config_path:   仿真配置 YAML 路径
        save_intermediate: 是否保存中间结果 OBJ
        drive_to_gt_shape: 是否将碰撞人体同时过渡到 CloSe GT neutral shape

    Returns:
        dict:
            driven_obj:     驱动后带 UV 的 OBJ 路径
            driven_verts_m: 驱动后顶点 (米)
            garment_faces:  面片索引
            target_body_v_m: 目标姿态人体顶点 (米)
            target_body_f:  人体面片
    """
    _ensure_wp_init()

    # ---------- 默认路径 ----------
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(garment_obj_path), 'driven')
    os.makedirs(output_dir, exist_ok=True)

    if sim_config_path is None:
        sim_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'assets', 'Sim_props', 'default_sim_props.yaml'
        )

    # ---------- 仿真参数 ----------
    DT = 1.0 / 60.0
    GARMENT_SCALE = 0.01  # cm → m

    # ---------- 加载仿真配置 ----------
    sim_config = load_sim_config(sim_config_path)
    print("[driver] 仿真配置加载完成")

    # ---------- 加载服装 (cm → m) ----------
    # BoxMesh 已按纸样缝合关系折叠物理顶点。直接复用 OBJ 的 position
    # indices，并单独保存 UV indices；不再执行基于坐标的全局 merge。
    garment_verts_cm, garment_faces, template_uvs, template_face_uvs = (
        _load_stitched_garment_topology(garment_obj_path, boxmesh_obj_path)
    )
    garment_verts_m = garment_verts_cm * GARMENT_SCALE
    print(f"[driver] 服装: {len(garment_verts_m)} 顶点, {len(garment_faces)} 面")
    print("[driver] 拓扑: 复用 BoxMesh 缝合 position indices，UV indices 独立保留")

    # ---------- 加载参考人体 (米) ----------
    body_mesh = trimesh.load(body_obj_path, process=False)
    body_verts_m = np.array(body_mesh.vertices, dtype=np.float32)
    body_faces = np.array(body_mesh.faces, dtype=np.int32)
    print(f"[driver] 参考人体: {len(body_verts_m)} 顶点, {len(body_faces)} 面")

    # ---------- 基准 SMPL 参数 ----------
    # betas: HybrIK (smpl.json)
    pred_betas = torch.from_numpy(load_betas_from_json(base_smpl_json)).float().unsqueeze(0)
    # pose: build_default_pose from export_smpl_mesh.py (match smpl.obj A-pose)
    angle = np.pi / 4.0
    angle2 = np.pi / 20.0
    apose = torch.zeros((1, 24, 3), dtype=torch.float32)
    apose[0, 16, 2] = -angle     # left shoulder Z
    apose[0, 17, 2] = angle      # right shoulder Z
    apose[0, 16, 1] = -angle2    # left shoulder Y
    apose[0, 17, 1] = angle2     # right shoulder Y
    base_global_orient = torch.zeros(1, 3)
    base_body_pose = apose.view(1, 72)[:, 3:]  # body_pose (69 dims)

    # ---------- 目标姿态 ----------
    if target_data is None:
        if target_pose_npz is None:
            raise ValueError("target_pose_npz or target_data must be provided")
        with np.load(target_pose_npz) as npz:
            target_data = {key: npz[key] for key in npz.files}
    if "pose" not in target_data or "betas" not in target_data:
        raise ValueError("target_data must contain 'pose' and 'betas'")
    pose_full = torch.from_numpy(np.asarray(target_data["pose"], dtype=np.float32)).view(1, 72)
    target_global_orient = pose_full[:, :3]
    target_body_pose = pose_full[:, 3:]

    # ---------- SMPL 驱动器 ----------
    smpl_driver = SMPLDriver(
        model_path=smpl_model_path,
        gender=gender,
        base_betas=pred_betas,
        base_body_pose=base_body_pose,
        base_global_orient=base_global_orient,
    )
    align_offset = smpl_driver.align_to_reference(body_verts_m)
    print(f"[driver] SMPL 对齐偏移: {align_offset}")

    # CloSe registration 在缺少逐样本 gender 时默认使用 neutral SMPL。
    # 预测驱动人体仍使用调用方指定 gender；评估参考人体与 canon_pose
    # 必须使用同一个 neutral template。
    gt_betas = torch.from_numpy(
        np.asarray(target_data["betas"], dtype=np.float32)
    ).view(1, -1)
    gt_model = smplx.create(
        smpl_model_path, model_type='smpl', gender='neutral', ext='npz'
    )
    with torch.no_grad():
        gt_body_output = gt_model(
            betas=gt_betas,
            body_pose=target_body_pose,
            global_orient=target_global_orient,
        )
    gt_target_body_verts = gt_body_output.vertices.squeeze(0).cpu().numpy().astype(np.float32)
    gt_template_verts = gt_model.v_template.detach().cpu().numpy().astype(np.float32)
    dominant_joint = gt_model.lbs_weights.argmax(dim=1).cpu().numpy()
    torso_mask = np.isin(dominant_joint, [0, 3, 6, 9])
    gt_target_body_aligned = gt_target_body_verts + align_offset

    # ---------- 布料仿真器 ----------
    sim = WarpClothSimulator(
        garment_verts=garment_verts_m,
        garment_faces=garment_faces,
        body_verts=body_verts_m,
        body_faces=body_faces,
        sim_config=sim_config,
    )
    print("[driver] 布料仿真器初始化完成")

    # ========== 阶段一：零重力松弛 ==========
    zero_gravity_steps = sim_config["zero_gravity_steps"]
    print(f"[driver] 阶段一: 零重力松弛 ({zero_gravity_steps} 步)")
    for step in range(zero_gravity_steps):
        cloth_verts_m = sim.step(body_verts_m, dt=DT, gravity_enabled=False)

    if save_intermediate:
        relaxed_mesh = trimesh.Trimesh(vertices=cloth_verts_m, faces=garment_faces, process=False)
        relaxed_mesh.export(os.path.join(output_dir, "00_relaxed.obj"))

    # ========== 阶段二：姿态线性过渡 ==========
    transition_frames = 100
    print(f"[driver] 阶段二: 姿态过渡 ({transition_frames} 帧)")

    base_body_np = base_body_pose.numpy().reshape(-1, 3)
    target_body_np = target_body_pose.numpy().reshape(-1, 3)
    base_global_np = base_global_orient.numpy().reshape(-1, 3)
    target_global_np = target_global_orient.numpy().reshape(-1, 3)

    rot_base_body = R.from_rotvec(base_body_np)
    rot_target_body = R.from_rotvec(target_body_np)
    rot_base_global = R.from_rotvec(base_global_np)
    rot_target_global = R.from_rotvec(target_global_np)

    for frame in range(transition_frames):
        alpha = (frame + 1) / transition_frames

        interp_body_np = np.zeros_like(base_body_np)
        for j in range(base_body_np.shape[0]):
            rots = R.concatenate([rot_base_body[j], rot_target_body[j]])
            slerp = scipy.spatial.transform.Slerp([0, 1], rots)
            interp_body_np[j] = slerp([alpha])[0].as_rotvec()

        interp_global_np = np.zeros_like(base_global_np)
        rots_g = R.concatenate([rot_base_global[0], rot_target_global[0]])
        slerp_g = scipy.spatial.transform.Slerp([0, 1], rots_g)
        interp_global_np[0] = slerp_g([alpha])[0].as_rotvec()

        interp_body = torch.from_numpy(interp_body_np).float().view(1, 69)
        interp_global = torch.from_numpy(interp_global_np).float().view(1, 3)

        pred_interp_body_verts = smpl_driver.get_body_verts(
            betas=pred_betas, body_pose=interp_body, global_orient=interp_global
        )
        if drive_to_gt_shape:
            with torch.no_grad():
                gt_interp_output = gt_model(
                    betas=gt_betas,
                    body_pose=interp_body,
                    global_orient=interp_global,
                )
            gt_interp_body_verts = (
                gt_interp_output.vertices.squeeze(0).cpu().numpy().astype(np.float32)
                + align_offset
            )
            interp_body_verts = (
                (1.0 - alpha) * pred_interp_body_verts
                + alpha * gt_interp_body_verts
            ).astype(np.float32)
        else:
            interp_body_verts = pred_interp_body_verts
        sim.update_body_collider(interp_body_verts)
        cloth_verts_m = sim.step(interp_body_verts, dt=DT, gravity_enabled=True)

        if save_intermediate and frame % 50 == 0:
            mesh = trimesh.Trimesh(vertices=cloth_verts_m, faces=garment_faces, process=False)
            mesh.export(os.path.join(output_dir, f"trans_{frame:04d}.obj"))
            print(f"  [driver] Transition {frame}/{transition_frames}")

    if save_intermediate:
        trans_final = trimesh.Trimesh(vertices=cloth_verts_m, faces=garment_faces, process=False)
        trans_final.export(os.path.join(output_dir, "01_transition_end.obj"))

    # ========== 阶段三：目标姿态稳定 ==========
    max_sim_steps = min(sim_config["max_sim_steps"], 2500)
    static_threshold = sim_config["static_threshold"] * GARMENT_SCALE  # cm → m
    print(f"[driver] 阶段三: 目标姿态稳定 (≤{max_sim_steps} 步)")

    if drive_to_gt_shape:
        target_body_verts = gt_target_body_aligned
    else:
        target_body_verts = smpl_driver.get_body_verts(
            betas=pred_betas, body_pose=target_body_pose, global_orient=target_global_orient
        )

    prev_verts = cloth_verts_m.copy()
    static_frames = 0
    required_static_frames = 30
    for step in range(max_sim_steps):
        sim.update_body_collider(target_body_verts)
        cloth_verts_m = sim.step(target_body_verts, dt=DT, gravity_enabled=True)

        max_displacement = np.max(np.linalg.norm(cloth_verts_m - prev_verts, axis=1))
        prev_verts = cloth_verts_m.copy()

        if step % 100 == 0:
            mesh = trimesh.Trimesh(vertices=cloth_verts_m, faces=garment_faces, process=False)
            mesh.export(os.path.join(output_dir, f"sim_{step:04d}.obj"))
            print(f"  [driver] Step {step}, max displacement: {max_displacement:.6f}")

        if max_displacement < static_threshold:
            static_frames += 1
        else:
            static_frames = 0

        if static_frames >= required_static_frames and step > 50:
            print(f"[driver] 仿真在第 {step} 步收敛")
            break

    # ---------- 保存最终结果 (m → cm, 带 UV) ----------
    final_obj_path = os.path.join(output_dir, "final_result.obj")
    _export_driven_obj_with_uv(
        final_obj_path,
        cloth_verts_m,
        garment_faces,
        template_uvs,
        template_face_uvs,
    )
    _copy_material_files(os.path.dirname(garment_obj_path), output_dir)
    print(f"[driver] 带 UV 的最终 OBJ → {final_obj_path}")

    # 保存简化版 (米, 纯顶点)
    final_mesh = trimesh.Trimesh(vertices=cloth_verts_m, faces=garment_faces, process=False)
    final_mesh.export(os.path.join(output_dir, "final_result_meters.obj"))

    # ---------- 导出目标姿态人体 OBJ ----------
    target_body_obj_path = os.path.join(output_dir, "target_body.obj")
    body_mesh_target = trimesh.Trimesh(vertices=target_body_verts, faces=body_faces, process=False)
    body_mesh_target.export(target_body_obj_path)
    print(f"[driver] 目标姿态人体 OBJ → {target_body_obj_path}")

    # ---------- 导出 HybrIK 预测的基准人体 OBJ (A-pose) ----------
    base_body_verts = smpl_driver.get_body_verts(
        betas=pred_betas, body_pose=base_body_pose, global_orient=base_global_orient
    )
    base_body_obj_path = os.path.join(output_dir, "base_body.obj")
    trimesh.Trimesh(vertices=base_body_verts, faces=body_faces, process=False).export(base_body_obj_path)
    print(f"[driver] HybrIK 基准人体 (A-pose) OBJ → {base_body_obj_path}")

    # ---------- 导出穿衣人体 OBJ (garment + body 合并) ----------
    dressed_obj_path = os.path.join(output_dir, "dressed_body.obj")
    # 服装在 cm 单位，人体在 m 单位；统一到 m 单位合并
    driven_cm = cloth_verts_m * 100.0
    body_cm = target_body_verts * 100.0
    # 合并：先写人体顶点，再写服装顶点（面片索引偏移）
    with open(dressed_obj_path, 'w') as f:
        f.write("# Dressed body: target body + driven garment\n")
        f.write(f"# Body vertices: {len(body_cm)}, Garment vertices: {len(driven_cm)}\n")
        for v in body_cm:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for v in driven_cm:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in body_faces:
            a, b, c = int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1
            f.write(f"f {a} {b} {c}\n")
        face_offset = len(body_cm)
        for face in garment_faces:
            a, b, c = (int(face[0]) + 1 + face_offset,
                       int(face[1]) + 1 + face_offset,
                       int(face[2]) + 1 + face_offset)
            f.write(f"f {a} {b} {c}\n")
    print(f"[driver] 穿衣人体 OBJ → {dressed_obj_path}")

    return {
        'driven_obj': final_obj_path,
        'driven_verts_m': cloth_verts_m,
        'garment_faces': garment_faces,
        'target_body_v_m': target_body_verts,
        'gt_target_body_v_m': gt_target_body_verts,
        'gt_template_v_m': gt_template_verts,
        'smpl_torso_mask': torso_mask,
        'target_body_f': body_faces,
        'align_offset_m': align_offset,
        'target_data': target_data,
        'drive_to_gt_shape': drive_to_gt_shape,
        'target_body_obj': target_body_obj_path,
        'base_body_obj': base_body_obj_path,
        'dressed_body_obj': dressed_obj_path,
    }
