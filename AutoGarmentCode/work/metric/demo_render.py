"""
Demo script: Re-render garment images from existing simulation data

This script demonstrates how to re-render the design_render_front.png and 
design_render_back.png images using the existing simulation output data.

Required input data (all available in the output directory):
- smpl.obj: SMPL body mesh
- design/design_sim.obj: Simulated garment mesh (with texture)
- design/sim_props.yaml: Render configuration (resolution, camera position, etc.)
"""

import os
import sys
import platform
import numpy as np
import trimesh
from PIL import Image
import yaml
from pathlib import Path

# IMPORTANT: Set OpenGL platform BEFORE importing pyrender
# This is required for headless Linux servers without display
if platform.system() == 'Linux':
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender

# Add the project root to the path
project_root = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/"
sys.path.insert(0, str(project_root))


def load_body_mesh(body_obj_path, visible=True):
    """Load SMPL body mesh from .obj file (already in meters)"""
    body_mesh = trimesh.load(body_obj_path)
    # smpl.obj 已经是米制，不需要缩放
    
    if visible:
        # Use a light gray color for better visibility
        body_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.0, 0.0, 0.0),  # Light gray
            metallicFactor=0.3,
            roughnessFactor=0.8
        )
    else:
        # Make body invisible (for garment-only rendering)
        body_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.0, 0.0, 0.0, 0.0),  # Transparent
            metallicFactor=0.0,
            roughnessFactor=0.5
        )
    
    pyrender_body_mesh = pyrender.Mesh.from_trimesh(body_mesh, material=body_material)
    
    return pyrender_body_mesh, body_mesh.vertices, body_mesh.faces


def load_garment_mesh(garment_obj_path):
    """Load garment mesh from .obj file (includes texture)"""
    garm_mesh = trimesh.load_mesh(garment_obj_path)
    # Convert to meters
    garm_mesh.vertices = garm_mesh.vertices / 100
    
    # Material adjustments
    material = garm_mesh.visual.material.to_pbr()
    material.baseColorFactor = [1., 1., 1., 1.]
    material.doubleSided = True  # Color both face sides
    
    # Remove transparency - add white background
    white_back = Image.new('RGBA', material.baseColorTexture.size, color=(255, 255, 255, 255))
    white_back.paste(material.baseColorTexture)
    material.baseColorTexture = white_back.convert('RGB')
    
    garm_mesh.visual.material = material
    pyrender_garm_mesh = pyrender.Mesh.from_trimesh(garm_mesh, smooth=True)
    
    return pyrender_garm_mesh


def load_render_config(sim_props_path):
    """Load render configuration from sim_props.yaml"""
    with open(sim_props_path, 'r') as f:
        props = yaml.safe_load(f)
    
    render_props = props['render']['config']
    return render_props


def rotate_matrix_y(matrix, angle_deg):
    """Rotate matrix around Y axis"""
    rotation_angle = angle_deg * (np.pi / 180)
    rotation_matrix = np.array([
        [np.cos(rotation_angle), 0, np.sin(rotation_angle), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation_angle), 0, np.cos(rotation_angle), 0],
        [0, 0, 0, 1]
    ])
    return np.dot(rotation_matrix, matrix)


def rotate_matrix_x(matrix, angle_deg):
    """Rotate matrix around X axis"""
    rotation_angle = angle_deg * (np.pi / 180)
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [0, np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 0, 1]
    ])
    return np.dot(rotation_matrix, matrix)


def create_camera(pyrender_body_mesh, side, camera_location=None):
    """Create camera for rendering"""
    y_fov = np.pi / 6.
    camera = pyrender.PerspectiveCamera(yfov=y_fov)
    
    if camera_location is None:
        # Auto-calculate camera position based on body mesh
        fov = 50
        bounding_box_center = pyrender_body_mesh.bounds.mean(axis=0)
        diagonal_length = np.linalg.norm(pyrender_body_mesh.bounds[1] - pyrender_body_mesh.bounds[0])
        distance = 1.5 * diagonal_length / (2 * np.tan(np.radians(fov / 2)))
        
        camera_location = bounding_box_center
        camera_location[-1] += distance
    
    # Create camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_location
    
    # Apply rotations
    camera_pose = rotate_matrix_x(camera_pose, -15)
    camera_pose = rotate_matrix_y(camera_pose, 20)
    
    # For back view, rotate 180 degrees
    if side == 'back':
        camera_pose = rotate_matrix_y(camera_pose, 180)
    
    return camera, camera_pose


def create_lights(scene, intensity=80.0):
    """Add lights to the scene"""
    light_positions = [
        np.array([1.60614, 1.5341, 1.23701]),
        np.array([1.31844, 1.92831, -2.52238]),
        np.array([-2.80522, 1.2594, 2.34624]),
        np.array([0.160261, 1.81789, 3.52215]),
        np.array([-2.65752, 1.41194, -1.26328])
    ]
    
    for i in range(5):
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        light_pose = np.eye(4)
        light_pose[:3, 3] = light_positions[i]
        scene.add(light, pose=light_pose)


def render_single_view(pyrender_garm_mesh, pyrender_body_mesh, side, 
                       output_path, render_props):
    """Render a single view (front or back)"""
    # Get resolution
    view_width, view_height = render_props.get('resolution', (1080, 1080))
    
    # Create scene with transparent background
    scene = pyrender.Scene(bg_color=(1., 1., 1., 0.))
    
    # Add meshes
    scene.add(pyrender_garm_mesh)
    scene.add(pyrender_body_mesh)
    
    # Create and add camera
    camera_location = render_props.get('front_camera_location', None)
    camera, camera_pose = create_camera(pyrender_body_mesh, side, camera_location)
    scene.add(camera, pose=camera_pose)
    
    # Add lights
    create_lights(scene)
    
    # Create renderer
    renderer = pyrender.OffscreenRenderer(
        viewport_width=view_width, 
        viewport_height=view_height
    )
    
    # Render
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    
    # Save image
    image = Image.fromarray(color)
    image.save(output_path, "PNG")
    print(f"Saved: {output_path}")
    
    # Clean up
    renderer.delete()


def main():
    """Main rendering pipeline"""
    # ===== Configuration =====
    # Input directory
    base_dir = Path(project_root) / "output" / "CloSe" / "10001_1924"
    
    # Input files
    body_obj_path = base_dir / "smpl.obj"
    garment_obj_path = base_dir / "design" / "design_sim.obj"
    sim_props_path = base_dir / "design" / "sim_props.yaml"
    
    # Output directory (can be the same or different)
    output_dir = base_dir / "design" / "rerendered"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Garment Re-rendering Demo")
    print("=" * 60)
    
    # Step 1: Load render configuration
    print("\n[Step 1] Loading render configuration...")
    render_props = load_render_config(sim_props_path)
    print(f"  Resolution: {render_props.get('resolution', 'default')}")
    print(f"  Sides: {render_props.get('sides', ['front', 'back'])}")
    print(f"  Camera location: {render_props.get('front_camera_location', 'auto')}")
    
    # Step 2: Load body mesh
    print("\n[Step 2] Loading body mesh...")
    pyrender_body_mesh, body_v, body_f = load_body_mesh(body_obj_path)
    print(f"  Body vertices: {body_v.shape}")
    print(f"  Body faces: {body_f.shape}")
    
    # Step 3: Load garment mesh
    print("\n[Step 3] Loading garment mesh...")
    pyrender_garm_mesh = load_garment_mesh(garment_obj_path)
    print(f"  Garment vertices: {pyrender_garm_mesh.primitives[0].positions.shape}")
    
    # Step 4: Render each view
    print("\n[Step 4] Rendering views...")
    sides = render_props.get('sides', ['front', 'back'])
    
    for side in sides:
        print(f"\n  Rendering {side} view...")
        output_path = output_dir / f"design_render_{side}.png"
        render_single_view(
            pyrender_garm_mesh, 
            pyrender_body_mesh, 
            side, 
            output_path, 
            render_props
        )
    
    print("\n" + "=" * 60)
    print("Rendering complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()