import os
import json
import yaml
import numpy as np
import torch
import trimesh
import warp as wp
import warp.sim
import smplx

# 初始化 Warp
wp.init()
wp.set_device("cuda:0")

# ======================================
# 1. 工具函数：配置加载、弹簧构建、绑定计算
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
    """加载 YAML 仿真配置"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["sim"]["config"]

def build_cloth_springs(verts, faces):
    """
    从服装三角网格构建弹簧系统
    返回：结构边索引、弯曲边索引、对应原长
    """
    num_verts = len(verts)
    edges_set = set()
    struct_edges = []
    
    # 1. 构建结构弹簧（每条边对应一根拉伸弹簧）
    for f in faces:
        for i in range(3):
            v0, v1 = f[i], f[(i+1)%3]
            if v0 > v1:
                v0, v1 = v1, v0
            if (v0, v1) not in edges_set:
                edges_set.add((v0, v1))
                struct_edges.append([v0, v1])
    
    struct_edges = np.array(struct_edges, dtype=np.int32)
    struct_rest_lengths = np.linalg.norm(
        verts[struct_edges[:, 0]] - verts[struct_edges[:, 1]], axis=1
    ).astype(np.float32)

    # 2. 构建弯曲弹簧（共享边的两个对角顶点）
    edge_face_map = {}
    for face_idx, f in enumerate(faces):
        for i in range(3):
            v0, v1 = f[i], f[(i+1)%3]
            if v0 > v1:
                v0, v1 = v1, v0
            key = (v0, v1)
            if key not in edge_face_map:
                edge_face_map[key] = []
            edge_face_map[key].append(face_idx)
    
    bend_edges = []
    for edge, face_indices in edge_face_map.items():
        if len(face_indices) == 2:
            # 找到两个面中不共边的两个顶点
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
    """根据面密度计算每个粒子的质量（面积加权分配）"""
    num_verts = len(verts)
    masses = np.zeros(num_verts, dtype=np.float32)
    
    for f in faces:
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        # 每个三角面的质量均匀分配给3个顶点
        face_mass = area * density
        masses[f[0]] += face_mass / 3.0
        masses[f[1]] += face_mass / 3.0
        masses[f[2]] += face_mass / 3.0
    
    return masses

def find_attachment_points(garment_verts, body_verts, attachment_labels=None):
    attachment_idx = []
    attachment_body_idx = []
    threshold = 0.005  # 收紧阈值，只选最贴身的顶点
    
    # 只保留腰部和领口区域（Y轴范围根据你的人体尺寸调整）
    waist_y_min, waist_y_max = 0.85, 1.05
    collar_y_min, collar_y_max = 1.45, 1.65
    
    for i, gv in enumerate(garment_verts):
        v_y = gv[1]
        # 不在目标区域直接跳过
        if not ((waist_y_min < v_y < waist_y_max) or (collar_y_min < v_y < collar_y_max)):
            continue
        dists = np.linalg.norm(body_verts - gv, axis=1)
        min_dist = dists.min()
        if min_dist < threshold:
            attachment_idx.append(i)
            attachment_body_idx.append(np.argmin(dists))
    
    return np.array(attachment_idx, dtype=np.int32), np.array(attachment_body_idx, dtype=np.int32)
# ======================================
# 2. SMPL 人体驱动模块
# ======================================

class SMPLDriver:
    def __init__(self, model_path, gender='male',
                 base_betas=None, base_body_pose=None, base_global_orient=None):
        self.model = smplx.create(model_path, model_type='smpl', gender=gender, ext='npz')
        self.faces = self.model.faces.astype(np.int32)
        
        # 生成基准姿态人体（A-pose）
        with torch.no_grad():
            output = self.model(
                betas=base_betas,
                body_pose=base_body_pose,
                global_orient=base_global_orient
            )
            self.base_verts = output.vertices.squeeze().numpy().astype(np.float32)

        # 对齐偏移：SMPL 输出未含 trans，与服装所在参考人体存在固定平移。
        # 调用 align_to_reference 后，所有输出都会平移到参考人体坐标系。
        self.align_offset = np.zeros(3, dtype=np.float32)

    def align_to_reference(self, reference_verts):
        """用参考人体（服装所贴合的人体）计算固定平移，使驱动结果与服装同一坐标系"""
        self.align_offset = (
            reference_verts.mean(axis=0) - self.base_verts.mean(axis=0)
        ).astype(np.float32)
        return self.align_offset

    def get_body_verts(self, betas=None, body_pose=None, global_orient=None):
        """获取当前姿态下的人体顶点"""
        with torch.no_grad():
            output = self.model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient
            )
        verts = output.vertices.squeeze().numpy().astype(np.float32)
        return verts + self.align_offset

# ======================================
# 3. 自定义 Kernel：动态人体碰撞检测
# ======================================
@wp.kernel
def body_collision_kernel(
    cloth_pos: wp.array(dtype=wp.vec3),
    cloth_vel: wp.array(dtype=wp.vec3),
    body_mesh_id: wp.uint64,
    collision_thickness: float,
    friction: float,
    query_dist: float
):
    tid = wp.tid()
    p = cloth_pos[tid]
    v = cloth_vel[tid]
    
    query = wp.mesh_query_point_sign_normal(body_mesh_id, p, query_dist)
    if not query.result:
        return
    
    # 计算人体表面最近点坐标
    cp = wp.mesh_eval_position(body_mesh_id, query.face, query.u, query.v)
    delta = p - cp
    dist_sq = wp.dot(delta, delta)
    if dist_sq <= 1e-12:
        return
    dist = wp.sqrt(dist_sq)
    
    # 核心修正：用 sign 统一法线方向，永远指向人体外部
    # sign > 0: 点在人体外，几何方向本身朝外
    # sign < 0: 点在人体内，几何方向朝内，乘以 sign 后翻转朝外
    n = (delta / dist) * query.sign
    signed_dist = dist * query.sign
    
    if signed_dist < collision_thickness:
        # 沿朝外法线推离，绝对不会把布料推到人体内部
        push = collision_thickness - signed_dist
        push = wp.min(push, collision_thickness * 0.2)  # 限制单帧推离强度
        cloth_pos[tid] = p + n * push
        
        # 法向速度归零 + 切向摩擦
        vn = wp.dot(v, n) * n
        vt = v - vn
        cloth_vel[tid] = vt * (1.0 - friction)
@wp.kernel
def attachment_kernel(
    cloth_pos: wp.array(dtype=wp.vec3),
    cloth_idx: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3),
    strength: float
):
    # PBD-style: pull attachment points toward body targets
    tid = wp.tid()
    idx = cloth_idx[tid]
    p = cloth_pos[idx]
    t = target_pos[tid]
    cloth_pos[idx] = p + (t - p) * strength
            
# ======================================
# 4. Warp 官方 sim 布料仿真器
# ======================================

class WarpClothSimulator:
    def __init__(self, garment_verts, garment_faces, body_verts, body_faces, sim_config):
        self.device = "cuda:0"
        self.config = sim_config
        self.num_cloth_verts = len(garment_verts)
        
        # ---------- 1. 构建布料弹簧 ----------
        struct_edges, struct_rest, bend_edges, bend_rest = build_cloth_springs(
            garment_verts, garment_faces
        )
        self.struct_edges = struct_edges
        self.bend_edges = bend_edges
        
        # ---------- 2. 计算粒子质量 ----------
        density = sim_config["material"]["fabric_density"]
        particle_masses = compute_particle_masses(garment_verts, garment_faces, density)
        # 质量下限：三角面积极小时 m~1e-5kg，显式积分(SemiImplicit)会立即发散。
        # 抬高到下限可大幅提高稳定步长——不改变方法，仅保证数值可解。
        MASS_FLOOR = 1e-3
        particle_masses = np.maximum(particle_masses, MASS_FLOOR)
        self.min_mass = float(particle_masses.min())

        # ---------- 子步数 & 显式积分稳定性上限 ----------
        # 一帧(1/60s)拆成多个子步以满足显式积分稳定性
        self.num_substeps = int(sim_config.get("num_substeps", 240))
        sub_dt = (1.0 / 60.0) / self.num_substeps
        # 显式法稳定条件（按单节点累计）：刚度 k < m/(N*sub_dt^2)，阻尼 kd < 2m/(N*sub_dt)
        # N 为单节点弹簧数（结构+弯曲叠加，三角网格约 10~12），SAFETY 为安全余量
        _VALENCE = 12.0
        _SAFETY_KE = 0.1
        _SAFETY_KD = 0.25
        ke_cap = _SAFETY_KE * self.min_mass / (_VALENCE * sub_dt * sub_dt)
        kd_cap = _SAFETY_KD * 2.0 * self.min_mass / (_VALENCE * sub_dt)

        # ---------- 3. 初始化 Warp sim 模型 ----------
        builder = wp.sim.ModelBuilder()

        # 添加布料粒子
        for i in range(self.num_cloth_verts):
            builder.add_particle(
                pos=wp.vec3(*garment_verts[i]),
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=float(particle_masses[i])
            )

        # 添加结构弹簧（拉伸）——刚度/阻尼按稳定性上限钳制
        edge_ke = min(float(sim_config["material"]["garment_edge_ke"]), ke_cap)
        edge_ke = max(edge_ke, 800.0) 
        edge_kd = min(float(sim_config["material"]["garment_edge_kd"]), kd_cap)
        for i in range(len(struct_edges)):
            a, b = struct_edges[i]
            builder.add_spring(
                int(a), int(b),
                ke=edge_ke,
                kd=edge_kd,
                control=0.0
            )

        # 添加弯曲弹簧——同样钳制（原配置 ke=5e4/kd=10 会发散）
        bend_ke = min(float(sim_config["material"]["spring_ke"]), ke_cap) * 0.02
        bend_kd = min(float(sim_config["material"]["spring_kd"]), kd_cap)
        for i in range(len(bend_edges)):
            a, b = bend_edges[i]
            builder.add_spring(
                int(a), int(b),
                ke=bend_ke,
                kd=bend_kd,
                control=0.0
            )
        print(f"弹簧刚度钳制: struct ke={edge_ke:.1f} kd={edge_kd:.2f} | "
              f"bend ke={bend_ke:.1f} kd={bend_kd:.2f} (cap ke={ke_cap:.1f} kd={kd_cap:.2f})")

        # 全局参数
        builder.gravity = wp.vec3(0.0, -9.8, 0.0)
        self.model = builder.finalize()
        self.model.ground = sim_config["ground"]

        # ---------- 4. 初始化仿真状态 ----------
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.integrator = wp.sim.SemiImplicitIntegrator()
        
        # ---------- 5. 初始化人体碰撞网格 ----------
        self.body_mesh = wp.Mesh(
            points=wp.array(body_verts, dtype=wp.vec3, device=self.device),
            indices=wp.array(body_faces.flatten(), dtype=int, device=self.device)
        )
        # body_collision_thickness 配置值(0.25m)是按其它单位标定的，对本场景过大：
        # 服装贴体间距仅~2cm，若按 25cm 推离会每帧注入巨大能量并与绑定约束对抗。
        # 这里钳制到合理的贴体厚度（约 0.5cm）。
        raw_thickness = sim_config["options"]["body_collision_thickness"]
        self.collision_thickness = float(min(raw_thickness, 0.01))
        self.body_friction = sim_config["options"]["body_friction"]
        # mesh 查询半径：需大于碰撞厚度，保证能命中体表最近点
        self.query_dist = float(max(self.collision_thickness * 20.0, 0.1))
        
        # ---------- 6. 绑定约束（Attachment）----------
        self.enable_attachment = sim_config["options"]["enable_attachment_constraint"]
        if self.enable_attachment:
            self.attach_cloth_idx, self.attach_body_idx = find_attachment_points(
                garment_verts, body_verts
            )
            print(f"绑定点数量: {len(self.attach_cloth_idx)}")
            # 没有匹配到绑定点则关闭约束，避免空 kernel 启动
            if len(self.attach_cloth_idx) == 0:
                self.enable_attachment = False
            else:
                # cloth 索引在仿真过程中固定，预先上传为 warp 数组供 kernel 使用
                self.attach_cloth_idx_wp = wp.array(
                    self.attach_cloth_idx, dtype=wp.int32, device=self.device
                )
            self.attach_stiffness = sim_config["options"]["attachment_stiffness"][0]
            self.attach_damping = sim_config["options"]["attachment_damping"][0]
            self.attach_frames = sim_config["options"]["attachment_frames"]
            self.current_frame = 0
    
    def update_body_collider(self, body_verts_np):
        """更新人体碰撞网格顶点（每帧姿态变化后调用）"""
        self.body_mesh.points.assign(
            wp.array(body_verts_np, dtype=wp.vec3, device=self.device)
        )
        # 重建加速结构（动态变形必须更新）
        self.body_mesh.refit()
    
    def _apply_attachment(self, body_verts_np, dt):
        """应用绑定约束：将绑定点拉向对应人体顶点"""
        if not self.enable_attachment:
            return
        
        # 渐进式增加刚度
        alpha = min(1.0, self.current_frame / self.attach_frames)
        current_stiff = self.attach_stiffness * alpha
        
        # 获取当前绑定目标位置
        strength = float(min(1.0, current_stiff * dt * 0.1))
        if strength <= 0.0:
            return
        target_pos = body_verts_np[self.attach_body_idx].astype(np.float32)
        target_pos_wp = wp.array(target_pos, dtype=wp.vec3, device=self.device)
        
        # 在 kernel 内做位置修正（host 端不支持对 wp.array 逐元素索引）
        wp.launch(
            kernel=attachment_kernel,
            dim=len(self.attach_cloth_idx),
            inputs=[
                self.state_0.particle_q,
                self.attach_cloth_idx_wp,
                target_pos_wp,
                strength
            ],
            device=self.device
        )
    
    def step(self, body_verts_np, dt=1.0/60.0, gravity_enabled=True):
        self.current_frame += 1

        # 1. 重力开关
        if not gravity_enabled:
            self.model.gravity = wp.vec3(0.0, 0.0, 0.0)
        else:
            self.model.gravity = wp.vec3(0.0, -9.8, 0.0)

        sub_dt = dt / self.num_substeps

        # 2. 绑定约束（每帧一次即可，人体姿态不变）
        self._apply_attachment(body_verts_np, dt)

        # 3. 积分步进 + 定期碰撞修正
        collision_every = 10  # 每20个子步做一次碰撞，平衡性能与稳定性
        for sub_step in range(self.num_substeps):
            # 定期执行碰撞修正
            if sub_step % collision_every == 0:
                wp.launch(
                    kernel=body_collision_kernel,
                    dim=self.num_cloth_verts,
                    inputs=[
                        self.state_0.particle_q,
                        self.state_0.particle_qd,
                        self.body_mesh.id,
                        self.collision_thickness,
                        self.body_friction,
                        self.query_dist
                    ],
                    device=self.device
                )
            
            self.state_0.clear_forces()
            self.integrator.simulate(self.model, self.state_0, self.state_1, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # 4. 全局阻尼 + 速度钳制
        damping = max(self.config["options"]["global_damping_factor"], 0.06)
        max_vel = self.config["options"]["global_max_velocity"]
        vel_np = self.state_0.particle_qd.numpy()
        if damping > 0:
            vel_np *= (1.0 - damping)
        norms = np.linalg.norm(vel_np, axis=1, keepdims=True)
        mask = (norms > max_vel).squeeze(-1)
        if np.any(mask):
            vel_np[mask] = vel_np[mask] / norms[mask] * max_vel
        self.state_0.particle_qd.assign(wp.array(vel_np, dtype=wp.vec3, device=self.device))

        # NaN 检测
        pos_np = self.state_0.particle_q.numpy()
        nan_mask = np.isnan(pos_np).any(axis=1)
        if np.any(nan_mask):
            raise RuntimeError(f"仿真发散：检测到 {np.sum(nan_mask)} 个异常顶点，仿真终止")

        return self.state_0.particle_q.numpy()

# ======================================
# 5. 主函数：完整仿真流程
# ======================================

if __name__ == "__main__":
    # ---------- 配置路径（请根据实际路径修改） ----------
    SMPL_MODEL_PATH = "/root/wyc/code/smpl2garmentcode2/smpl_models"
    GARMENT_OBJ_PATH = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/try3/design_sim.obj"
    PRED_SMPL_OBJ_PATH = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/try3/pred_smpl.obj"
    TARGET_POSE_NPZ = "/root/wyc/data/CloSe/data/CloSe-Di/10001_1937.npz"
    SIM_CONFIG_PATH = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/assets/Sim_props/default_sim_props.yaml"
    OUTPUT_DIR = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/try3/output"
    PRED_SMPL_JSON_PATH = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/work/try3/smpl.json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    GENDER = "male"
    DT = 1.0 / 60.0  # 仿真步长 60fps
    
    # ---------- 1. 加载仿真配置 ----------
    sim_config = load_sim_config(SIM_CONFIG_PATH)
    print("仿真配置加载完成")
    
    # ---------- 2. 加载基准姿态资源 ----------
    # 加载服装网格
    # 注意：服装 OBJ 单位为厘米，SMPL 人体单位为米，需统一到米后再仿真
    GARMENT_SCALE = 0.01
    garment_mesh = trimesh.load(GARMENT_OBJ_PATH, process=False)
    garment_verts = np.array(garment_mesh.vertices, dtype=np.float32) * GARMENT_SCALE
    garment_faces = np.array(garment_mesh.faces, dtype=np.int32)
    print(f"服装网格加载完成：{len(garment_verts)} 顶点，{len(garment_faces)} 面 (已缩放到米)")
    
    # 加载基准人体（A-pose）
    pred_body_mesh = trimesh.load(PRED_SMPL_OBJ_PATH, process=False)
    pred_body_verts = np.array(pred_body_mesh.vertices, dtype=np.float32)
    pred_body_faces = np.array(pred_body_mesh.faces, dtype=np.int32)
    print(f"基准人体加载完成：{len(pred_body_verts)} 顶点, {len(pred_body_faces)} 面")

    pred_betas = torch.from_numpy(load_betas_from_json(PRED_SMPL_JSON_PATH)).float().unsqueeze(0)
    pred_pose = load_pose_from_json(PRED_SMPL_JSON_PATH)
    base_global_orient = torch.zeros(1, 3)
    pred_body_pose = torch.from_numpy(pred_pose).float().view(1, 69)


    # ---------- 3. 初始化 SMPL 驱动器 ----------
    # 从 npz 中读取目标姿态参数
    data = np.load(TARGET_POSE_NPZ)
    betas_np = data["betas"].astype(np.float32) #TODO beta修改
    pose_full = torch.from_numpy(data["pose"]).float().view(1, 72)
    
    target_global_orient = pose_full[:, :3]
    target_body_pose = pose_full[:, 3:]
    
    # 初始化 SMPL（基准姿态用零姿态 + betas，实际应替换为你的 A-pose 参数）
    # base_betas = torch.from_numpy(betas_np).float().unsqueeze(0)
    # base_body_pose = torch.zeros(1, 69)  # 替换为你的 A-pose body_pose
    # base_global_orient = torch.zeros(1, 3)
    
    smpl_driver = SMPLDriver(
        model_path=SMPL_MODEL_PATH,
        gender=GENDER,
        base_betas=pred_betas,
        base_body_pose=pred_body_pose,
        base_global_orient=base_global_orient
    )
    print("SMPL 驱动器初始化完成")

    # 将驱动器输出对齐到参考人体（服装贴合的人体）坐标系
    offset = smpl_driver.align_to_reference(pred_body_verts)
    print(f"SMPL 输出对齐偏移: {offset}")
    
    # ---------- 4. 初始化布料仿真器 ----------
    base_body_aligned = smpl_driver.base_verts + smpl_driver.align_offset
    base_body_faces = smpl_driver.faces

    sim = WarpClothSimulator(
        garment_verts=garment_verts,
        garment_faces=garment_faces,
        body_verts=base_body_aligned,  # 统一用SMPL基准人体
        body_faces=base_body_faces,    # 用SMPL的面拓扑
        sim_config=sim_config
    )
    print("布料仿真器初始化完成")
    
    # ---------- 5. 阶段一：零重力松弛 ----------
    zero_gravity_steps = sim_config["zero_gravity_steps"]
    print(f"零重力松弛阶段：{zero_gravity_steps} 步")
    for step in range(20):
        cloth_verts = sim.step(base_body_aligned, dt=DT, gravity_enabled=False)

    # # 保存松弛后结果
    relaxed_mesh = trimesh.Trimesh(vertices=cloth_verts, faces=garment_faces)
    relaxed_mesh.export(os.path.join(OUTPUT_DIR, "00_relaxed.obj"))
    



    # ---------- 6. 阶段二：姿态线性过渡 ----------
    transition_frames = 100  # 过渡帧数
    print(f"姿态过渡阶段：{transition_frames} 帧")
    for frame in range(transition_frames):
        alpha = frame / transition_frames
        # 姿态线性插值
        interp_global = base_global_orient * (1-alpha) + target_global_orient * alpha
        interp_body = pred_body_pose * (1-alpha) + target_body_pose * alpha
        
        current_body_verts = smpl_driver.get_body_verts(
            betas=pred_betas,
            body_pose=interp_body,
            global_orient=interp_global
        )

        diff_max = np.max(np.linalg.linalg.norm(current_body_verts - base_body_aligned, axis=1))
        print(f"与基准人体顶点最大差异: {diff_max:.6f} 米")
        # 更新碰撞体并步进
        sim.update_body_collider(current_body_verts)
        cloth_verts = sim.step(current_body_verts, dt=DT, gravity_enabled=True)
        
        # 每隔 10 帧保存
        if frame % 10 == 0:
            mesh = trimesh.Trimesh(vertices=cloth_verts, faces=garment_faces)
            mesh.export(os.path.join(OUTPUT_DIR, f"trans_{frame:04d}.obj"))
    
    # ---------- 7. 阶段三：正式仿真（目标姿态下动力学） ----------
    max_sim_steps = min(sim_config["max_sim_steps"], 2500)  # 限制步数防止过长
    static_threshold = sim_config["static_threshold"]
    print(f"正式仿真阶段：最多 {max_sim_steps} 步")
    
    target_body_verts = smpl_driver.get_body_verts(
        betas=pred_betas,
        body_pose=target_body_pose,
        global_orient=target_global_orient
    )
    
    prev_verts = cloth_verts.copy()
    for step in range(max_sim_steps):
        sim.update_body_collider(target_body_verts)
        cloth_verts = sim.step(target_body_verts, dt=DT, gravity_enabled=True)
        
        # 静态收敛检测
        max_displacement = np.max(np.linalg.norm(cloth_verts - prev_verts, axis=1))
        prev_verts = cloth_verts.copy()
        
        if step % 10 == 0:
            mesh = trimesh.Trimesh(vertices=cloth_verts, faces=garment_faces)
            mesh.export(os.path.join(OUTPUT_DIR, f"sim_{step:04d}.obj"))
            print(f"Step {step}, max displacement: {max_displacement:.6f}")
        
        # 收敛则提前结束
        if max_displacement < static_threshold and step > 50:
            print(f"仿真在第 {step} 步收敛")
            break
    
    # 保存最终结果
    final_mesh = trimesh.Trimesh(vertices=cloth_verts, faces=garment_faces)
    final_mesh.export(os.path.join(OUTPUT_DIR, "final_result.obj"))
    print(f"仿真完成，结果保存在 {OUTPUT_DIR}")