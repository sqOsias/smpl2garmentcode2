import open3d as o3d
import numpy as np
import numpy as np
import os

def load_and_separate_canon_pose(npz_path):
    """
    读取 .npz 文件，解析 canon_pose 并根据 labels 分离人体与衣服
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"文件未找到: {npz_path}")
    
    # 1. 加载数据
    data = np.load(npz_path)
    
    # 2. 获取所需字段
    # canon_pose: (N, 3), labels: (N,)
    canon_pose = data['canon_pose']
    labels = data['labels']

    
    # 3. 定义非服装标签 (根据 CloSe 数据集规范)
    # 1 是 Body (人体)
    # 0, 10, 12, 13, 14, 15 是配饰、毛发、皮肤等非衣服类
    NON_GARMENT_LABELS = [0, 1, 10, 12, 13, 14, 15]
    
    # 4. 创建掩码 (Mask)
    # 人体掩码：标签为 1
    body_mask = (labels == 1)
    
    # 服装掩码：标签不在非服装列表中
    garment_mask = ~np.isin(labels, NON_GARMENT_LABELS)
    
    # 5. 分离点云
    body_points = canon_pose[body_mask]
    garment_points = canon_pose[garment_mask]
    
    print(f"原始点总数: {canon_pose.shape[0]}")
    print(f"提取人体点数: {body_points.shape[0]}")
    print(f"提取服装点数: {garment_points.shape[0]}")
    
    return body_points, garment_points

# 使用示例
# npz_file = "path/to/your/data.npz"
# body, garment = load_and_separate_canon_pose(npz_file)
import numpy as np
import open3d as o3d
import os

def export_debug_pointclouds(npz_path, output_dir="debug_output"):
    """
    导出完整的 canon_pose、掩码后的 canon_pose，以及原始扫描点云进行对比
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在读取: {npz_path}")
    data = np.load(npz_path)

    # 提取所需字段
    points = data['points']           # 原始扫描的真实 3D 点云 (包含姿态、褶皱)
    canon_pose = data['canon_pose']   # 投影到 T-pose 人体上的点
    labels = data['labels']           # 语义标签
    colors = data.get('colors', None) # RGB 颜色 (如果有)

    # 定义非服装标签，提取服装掩码
    NON_GARMENT_LABELS = [0, 1, 10, 12, 13, 14, 15]
    garment_mask = ~np.isin(labels, NON_GARMENT_LABELS)

    print(f"总点数: {len(labels)}")
    print(f"服装点数: {np.sum(garment_mask)}")

    # ---------------------------------------------------------
    # 1. 完整的 canon_pose (你之前看到的“人体”)
    # ---------------------------------------------------------
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(canon_pose)
    o3d.io.write_point_cloud(os.path.join(output_dir, "1_canon_pose_all.ply"), pcd1)
    print("导出: 1_canon_pose_all.ply")

    # ---------------------------------------------------------
    # 2. 掩码处理后的 canon_pose (投影在皮肤上的衣服区域)
    # ---------------------------------------------------------
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(canon_pose[garment_mask])
    o3d.io.write_point_cloud(os.path.join(output_dir, "2_canon_pose_garment.ply"), pcd2)
    print("导出: 2_canon_pose_garment.ply ")

    # ---------------------------------------------------------
    # 3. 原始 Scan 的完整点云 (真实世界的扫描结果，包含人、衣服、头发等)
    # ---------------------------------------------------------
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd3.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(output_dir, "3_scan_points_all.ply"), pcd3)
    print("导出: 3_scan_points_all.ply")

    # ---------------------------------------------------------
    # 4. 原始 Scan 掩码处理后的点云 (真正的 3D 衣服真值！)
    # ---------------------------------------------------------
    pcd4 = o3d.geometry.PointCloud()
    pcd4.points = o3d.utility.Vector3dVector(points[garment_mask])
    if colors is not None:
        pcd4.colors = o3d.utility.Vector3dVector(colors[garment_mask])
    o3d.io.write_point_cloud(os.path.join(output_dir, "4_scan_points_garment.ply"), pcd4)
    print("导出: 4_scan_points_garment.ply ")
    print("-" * 50)

# 使用示例 (请替换为你的实际路径)
npz_file = "/root/wyc/data/CloSe/data/CloSe-Di/10014_2464.npz"
export_debug_pointclouds(npz_file, output_dir="/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe/10014_2464/viz_output/close_debug_plys")
