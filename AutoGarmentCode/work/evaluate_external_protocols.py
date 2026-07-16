#!/usr/bin/env python3
"""Evaluate one fixed mesh pair with Mesh R-CNN and SiTH-compatible protocols.

The preferred project entry point is ``--sample``.  It reads the aligned meshes
already exported by ``work/metric.py`` from the sample output directory.  The
explicit ``--pred``/``--gt`` mode is retained for evaluating arbitrary mesh
pairs and must always be supplied as a complete pair.

References:
https://github.com/facebookresearch/meshrcnn/blob/main/meshrcnn/utils/metrics.py
https://github.com/SiTH-Diffusion/SiTH/blob/main/tools/evaluate.py
"""

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree


MESHR_CNN_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)
MESHR_CNN_TARGET_LONG_EDGE = 10.0
SITH_TAU_M = 0.01
DEFAULT_OUTPUT_ROOT = Path(
    "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe"
)
PRED_ALIGNED_FILENAME = "pred_garment_kabsch.obj"
GT_ALIGNED_FILENAME = "gt_garment.obj"
EXPECTED_INTERNAL_PROTOCOL = "close_full_scan_normalized_linear_distance"
PAIR_MTIME_WARNING_SECONDS = 300.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an aligned prediction/GT garment pair with external protocols."
    )
    parser.add_argument(
        "--sample",
        help=(
            "CloSe sample name. Loads <output-root>/<sample>/"
            f"{PRED_ALIGNED_FILENAME} and {GT_ALIGNED_FILENAME}."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="CloSe output root used by --sample mode.",
    )
    parser.add_argument("--pred", type=Path, help="Explicit aligned prediction mesh")
    parser.add_argument("--gt", type=Path, help="Explicit aligned GT garment mesh")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <pred-parent>/external_protocols.",
    )
    parser.add_argument(
        "--input-unit",
        choices=("meter", "centimeter"),
        default="meter",
        help="Unit of both input mesh files; exported evaluation meshes use meters.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meshrcnn-samples", type=int, default=10000)
    parser.add_argument("--sith-samples", type=int, default=100000)
    args = parser.parse_args()

    has_pred = args.pred is not None
    has_gt = args.gt is not None
    if has_pred != has_gt:
        parser.error("--pred and --gt must be supplied together")
    if args.sample is not None and has_pred:
        parser.error("Use either --sample or the explicit --pred/--gt pair, not both")
    if args.sample is None and not has_pred:
        parser.error("Specify --sample or provide both --pred and --gt")
    if args.sample is not None and args.input_unit != "meter":
        parser.error("Meshes exported by metric.py are in meters; --sample requires --input-unit meter")
    return args


def resolve_inputs(args) -> Tuple[Optional[str], Path, Path, Path, Optional[Path]]:
    """Resolve sample or explicit input mode without loading mesh contents."""
    if args.sample is not None:
        sample = str(args.sample).strip()
        if not sample or Path(sample).name != sample or sample in (".", ".."):
            raise ValueError(f"Invalid sample name: {args.sample!r}")
        sample_dir = args.output_root.expanduser().resolve() / sample
        pred_path = sample_dir / PRED_ALIGNED_FILENAME
        gt_path = sample_dir / GT_ALIGNED_FILENAME
        summary_path = sample_dir / "eval_summary.json"
        default_output_dir = sample_dir / "external_protocols"
        return sample, pred_path, gt_path, default_output_dir, summary_path

    pred_path = args.pred.expanduser().resolve()
    gt_path = args.gt.expanduser().resolve()
    return None, pred_path, gt_path, pred_path.parent / "external_protocols", None


def file_metadata(path: Path) -> Dict[str, Any]:
    """Return reproducibility metadata for one existing input file."""
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "modified_unix": float(stat.st_mtime),
    }


def inspect_sample_artifacts(
    sample: Optional[str],
    pred_path: Path,
    gt_path: Path,
    summary_path: Optional[Path],
) -> Tuple[list, Optional[Dict[str, Any]]]:
    """Warn about stale or mismatched sample artifacts without rejecting them."""
    warnings = []
    if sample is None:
        return warnings, None

    if not pred_path.is_file() or not gt_path.is_file():
        return warnings, None

    pred_mtime = pred_path.stat().st_mtime
    gt_mtime = gt_path.stat().st_mtime
    pair_gap = abs(pred_mtime - gt_mtime)
    if pair_gap > PAIR_MTIME_WARNING_SECONDS:
        warnings.append(
            "Aligned prediction and GT mesh modification times differ by "
            f"{pair_gap:.1f} seconds; they may come from different evaluations."
        )

    if summary_path is None or not summary_path.is_file():
        warnings.append(
            "eval_summary.json is missing; the aligned mesh pair cannot be tied "
            "to the current internal evaluation protocol."
        )
        return warnings, None

    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Unable to read eval_summary.json: {exc}")
        return warnings, None

    recorded_sample = summary.get("sample_name")
    if recorded_sample is not None and str(recorded_sample) != sample:
        warnings.append(
            f"eval_summary.json records sample {recorded_sample!r}, expected {sample!r}."
        )
    recorded_protocol = summary.get("f_score_protocol")
    if recorded_protocol != EXPECTED_INTERNAL_PROTOCOL:
        warnings.append(
            "eval_summary.json does not record the current normalized alignment "
            "protocol; the mesh pair may be a legacy result."
        )

    summary_gap = abs(summary_path.stat().st_mtime - max(pred_mtime, gt_mtime))
    if summary_gap > PAIR_MTIME_WARNING_SECONDS:
        warnings.append(
            "eval_summary.json and aligned meshes differ in modification time by "
            f"{summary_gap:.1f} seconds; provenance may be inconsistent."
        )
    return warnings, {
        "path": str(summary_path),
        "sample_name": recorded_sample,
        "f_score_protocol": recorded_protocol,
    }


def load_triangle_mesh(path: Path, unit: str) -> trimesh.Trimesh:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = trimesh.load(str(path), process=False, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Expected one triangle mesh: {path}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"Invalid vertices in {path}: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"Invalid triangle faces in {path}: {faces.shape}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"Mesh contains NaN or Inf vertices: {path}")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"Mesh contains out-of-range face indices: {path}")
    if unit == "centimeter":
        vertices = vertices / 100.0
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def sample_pair(
    pred_mesh: trimesh.Trimesh,
    gt_mesh: trimesh.Trimesh,
    count: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if count <= 0:
        raise ValueError(f"Sample count must be positive, got {count}")
    np.random.seed(seed)
    pred_points, pred_face_index = trimesh.sample.sample_surface(pred_mesh, count)
    gt_points, gt_face_index = trimesh.sample.sample_surface(gt_mesh, count)
    pred_normals = np.asarray(pred_mesh.face_normals[pred_face_index], dtype=np.float64)
    gt_normals = np.asarray(gt_mesh.face_normals[gt_face_index], dtype=np.float64)
    for name, values in (
        ("pred_points", pred_points),
        ("gt_points", gt_points),
        ("pred_normals", pred_normals),
        ("gt_normals", gt_normals),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"Surface sampling produced NaN or Inf in {name}")
    return pred_points, gt_points, pred_normals, gt_normals


def directed_nearest(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    distances, indices = cKDTree(target_points).query(
        source_points,
        k=1,
        workers=-1,
    )
    return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64)


def normalized_normals(normals: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError("Sampled mesh contains zero-length face normals")
    return normals / lengths


def normal_scores(
    pred_normals: np.ndarray,
    gt_normals: np.ndarray,
    pred_to_gt_index: np.ndarray,
    gt_to_pred_index: np.ndarray,
) -> Tuple[float, float]:
    pred_normals = normalized_normals(pred_normals)
    gt_normals = normalized_normals(gt_normals)
    pred_to_gt_dot = np.sum(pred_normals * gt_normals[pred_to_gt_index], axis=1)
    gt_to_pred_dot = np.sum(gt_normals * pred_normals[gt_to_pred_index], axis=1)
    signed = 0.5 * (pred_to_gt_dot.mean() + gt_to_pred_dot.mean())
    absolute = 0.5 * (
        np.abs(pred_to_gt_dot).mean() + np.abs(gt_to_pred_dot).mean()
    )
    return float(signed), float(absolute)


def fscore_percent(
    pred_to_gt: np.ndarray,
    gt_to_pred: np.ndarray,
    threshold: float,
    strict: bool,
) -> Dict[str, float]:
    if strict:
        precision = float((pred_to_gt < threshold).mean() * 100.0)
        recall = float((gt_to_pred < threshold).mean() * 100.0)
    else:
        precision = float((pred_to_gt <= threshold).mean() * 100.0)
        recall = float((gt_to_pred <= threshold).mean() * 100.0)
    fscore = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0 else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "fscore": float(fscore),
    }


def evaluate_meshrcnn(
    pred_mesh_m: trimesh.Trimesh,
    gt_mesh_m: trimesh.Trimesh,
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    gt_extent_m = np.asarray(gt_mesh_m.bounds[1] - gt_mesh_m.bounds[0], dtype=np.float64)
    gt_long_edge_m = float(gt_extent_m.max())
    if not math.isfinite(gt_long_edge_m) or gt_long_edge_m <= 0.0:
        raise ValueError(f"Invalid GT bounding-box longest edge: {gt_long_edge_m}")
    common_scale = MESHR_CNN_TARGET_LONG_EDGE / gt_long_edge_m

    pred_scaled = trimesh.Trimesh(
        vertices=np.asarray(pred_mesh_m.vertices) * common_scale,
        faces=np.asarray(pred_mesh_m.faces),
        process=False,
    )
    gt_scaled = trimesh.Trimesh(
        vertices=np.asarray(gt_mesh_m.vertices) * common_scale,
        faces=np.asarray(gt_mesh_m.faces),
        process=False,
    )
    pred_points, gt_points, pred_normals, gt_normals = sample_pair(
        pred_scaled,
        gt_scaled,
        samples,
        seed,
    )
    pred_to_gt, pred_to_gt_index = directed_nearest(pred_points, gt_points)
    gt_to_pred, gt_to_pred_index = directed_nearest(gt_points, pred_points)
    normal_consistency, absolute_normal_consistency = normal_scores(
        pred_normals,
        gt_normals,
        pred_to_gt_index,
        gt_to_pred_index,
    )

    fscores = {
        f"{threshold:g}": fscore_percent(
            pred_to_gt,
            gt_to_pred,
            threshold,
            strict=True,
        )
        for threshold in MESHR_CNN_THRESHOLDS
    }
    return {
        "protocol": "meshrcnn_compatible",
        "samples_per_mesh": samples,
        "gt_bbox_extent_m": gt_extent_m.tolist(),
        "gt_bbox_longest_edge_m": gt_long_edge_m,
        "target_longest_edge": MESHR_CNN_TARGET_LONG_EDGE,
        "common_scale": common_scale,
        "chamfer_l2_sum": float(
            np.square(pred_to_gt).mean() + np.square(gt_to_pred).mean()
        ),
        "pred_to_gt_l1_mean": float(pred_to_gt.mean()),
        "gt_to_pred_l1_mean": float(gt_to_pred.mean()),
        "normal_consistency": normal_consistency,
        "absolute_normal_consistency": absolute_normal_consistency,
        "fscores_percent": fscores,
    }


def bbox_iou(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> float:
    minimum = np.maximum(mesh_a.bounds[0], mesh_b.bounds[0])
    maximum = np.minimum(mesh_a.bounds[1], mesh_b.bounds[1])
    intersection_extent = np.maximum(maximum - minimum, 0.0)
    intersection = float(np.prod(intersection_extent))
    volume_a = float(np.prod(mesh_a.bounds[1] - mesh_a.bounds[0]))
    volume_b = float(np.prod(mesh_b.bounds[1] - mesh_b.bounds[0]))
    union = volume_a + volume_b - intersection
    return intersection / (union + 1e-11)


def evaluate_sith(
    pred_mesh_m: trimesh.Trimesh,
    gt_mesh_m: trimesh.Trimesh,
    samples: int,
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    icp_matrix, pred_vertices_icp, icp_cost = trimesh.registration.icp(
        np.asarray(pred_mesh_m.vertices),
        np.asarray(gt_mesh_m.vertices),
    )
    pred_icp = trimesh.Trimesh(
        vertices=np.asarray(pred_vertices_icp, dtype=np.float64),
        faces=np.asarray(pred_mesh_m.faces, dtype=np.int64),
        process=False,
    )
    pred_icp_path = output_dir / "pred_sith_icp.obj"
    pred_icp.export(str(pred_icp_path))

    pred_points, gt_points, pred_normals, gt_normals = sample_pair(
        pred_icp,
        gt_mesh_m,
        samples,
        seed,
    )
    pred_to_gt, pred_to_gt_index = directed_nearest(pred_points, gt_points)
    gt_to_pred, gt_to_pred_index = directed_nearest(gt_points, pred_points)
    _, absolute_normal_consistency = normal_scores(
        pred_normals,
        gt_normals,
        pred_to_gt_index,
        gt_to_pred_index,
    )
    fscore = fscore_percent(
        pred_to_gt,
        gt_to_pred,
        SITH_TAU_M,
        strict=False,
    )
    return {
        "protocol": "sith_compatible",
        "samples_per_mesh": samples,
        "input_unit": "meter",
        "icp_matrix": np.asarray(icp_matrix, dtype=np.float64).tolist(),
        "icp_cost": float(icp_cost),
        "icp_prediction_mesh": str(pred_icp_path),
        "pred_to_gt_mm": float(pred_to_gt.mean() * 1000.0),
        "gt_to_pred_mm": float(gt_to_pred.mean() * 1000.0),
        "tau_m": SITH_TAU_M,
        "fscore_percent": fscore,
        "absolute_normal_consistency": absolute_normal_consistency,
        "bbox_iou": bbox_iou(pred_icp, gt_mesh_m),
    }


def flatten(prefix: str, value: Any, rows: list) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), item, rows)
    elif isinstance(value, (list, tuple)):
        rows.append((prefix, json.dumps(value, ensure_ascii=False)))
    else:
        rows.append((prefix, value))


def main() -> int:
    args = parse_args()
    sample, pred_path, gt_path, default_output_dir, summary_path = resolve_inputs(args)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir
    )

    pred_mesh_m = load_triangle_mesh(pred_path, args.input_unit)
    gt_mesh_m = load_triangle_mesh(gt_path, args.input_unit)
    artifact_warnings, summary_info = inspect_sample_artifacts(
        sample,
        pred_path,
        gt_path,
        summary_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "input_mode": "sample" if sample is not None else "explicit_mesh_pair",
        "sample": sample,
        "prediction": str(pred_path),
        "gt": str(gt_path),
        "source_files": {
            "prediction": file_metadata(pred_path),
            "gt": file_metadata(gt_path),
        },
        "internal_summary": summary_info,
        "artifact_warnings": artifact_warnings,
        "input_unit": args.input_unit,
        "working_unit": "meter",
        "pred_vertices": int(len(pred_mesh_m.vertices)),
        "pred_faces": int(len(pred_mesh_m.faces)),
        "gt_vertices": int(len(gt_mesh_m.vertices)),
        "gt_faces": int(len(gt_mesh_m.faces)),
        "random_seed": args.seed,
        "meshrcnn": evaluate_meshrcnn(
            pred_mesh_m,
            gt_mesh_m,
            args.meshrcnn_samples,
            args.seed,
        ),
        "sith": evaluate_sith(
            pred_mesh_m,
            gt_mesh_m,
            args.sith_samples,
            args.seed,
            output_dir,
        ),
    }

    json_path = output_dir / "external_protocol_metrics.json"
    csv_path = output_dir / "external_protocol_metrics.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    rows = []
    flatten("", results, rows)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    print(f"Mesh R-CNN CD-L2: {results['meshrcnn']['chamfer_l2_sum']:.8f}")
    print(
        "SiTH directed Chamfer: "
        f"{results['sith']['pred_to_gt_mm']:.4f} mm, "
        f"{results['sith']['gt_to_pred_mm']:.4f} mm"
    )
    for warning in artifact_warnings:
        print(f"WARNING: {warning}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
