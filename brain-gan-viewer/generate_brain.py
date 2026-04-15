"""
generate_brain.py

Full preprocessing pipeline: GAN MRI slices → clean 3D volume → realistic mesh + slice exports.
"""

import argparse
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from natsort import natsorted
from scipy import ndimage
from skimage import filters, measure, morphology, exposure
import trimesh

warnings.filterwarnings("ignore")


# ─────────────────────────── 1. LOAD SLICES ───────────────────────────────

def load_slices(input_dir: Path) -> np.ndarray:
    """Natural-sort all PNG/JPG files and stack into uint8 3-D volume."""
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = natsorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in exts],
        key=lambda p: p.name,
    )
    if not files:
        raise ValueError(f"No image files found in {input_dir}")
    print(f"[load] Found {len(files)} slices in {input_dir}")

    slices = []
    ref_shape = None
    for f in files:
        img = Image.open(f).convert("L")          # force grayscale
        arr = np.array(img, dtype=np.uint8)
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise ValueError(
                f"Inconsistent slice size: {f.name} is {arr.shape}, expected {ref_shape}"
            )
        slices.append(arr)

    volume = np.stack(slices, axis=0)             # (Z, H, W)  uint8
    print(f"[load] Volume shape: {volume.shape}")
    return volume, files


# ─────────────────────────── 2. NORMALISE ─────────────────────────────────

def normalise_volume(volume: np.ndarray, p_low: float = 1, p_high: float = 99,
                     sigma: float = 0.8) -> np.ndarray:
    """
    Percentile clip → rescale [0,255] → mild Gaussian blur to suppress
    GAN checkerboard artefacts while keeping anatomical contrast.
    """
    lo = np.percentile(volume, p_low)
    hi = np.percentile(volume, p_high)
    clipped = np.clip(volume.astype(np.float32), lo, hi)
    scaled = (clipped - lo) / max(hi - lo, 1e-6) * 255.0

    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        scaled = gaussian_filter(scaled, sigma=sigma)

    return np.clip(scaled, 0, 255).astype(np.uint8)


# ─────────────────────────── 3. BRAIN MASK ────────────────────────────────

def build_brain_mask(volume: np.ndarray,
                     closing_radius: int = 4,
                     dilation_radius: int = 1) -> np.ndarray:
    """
    Otsu threshold on the full 3-D volume → morphological closing →
    hole filling → largest connected component → slight dilation.
    Returns boolean mask same shape as volume.
    """
    print("[mask] Computing Otsu threshold …")
    thresh = filters.threshold_otsu(volume)
    binary = volume > thresh

    print("[mask] Binary closing …")
    struct_close = morphology.ball(closing_radius)
    binary = ndimage.binary_closing(binary, structure=struct_close)

    print("[mask] Filling holes …")
    binary = ndimage.binary_fill_holes(binary)

    print("[mask] Keeping largest connected component …")
    labelled = measure.label(binary)
    if labelled.max() == 0:
        raise RuntimeError("No foreground voxels found after thresholding.")
    props = measure.regionprops(labelled)
    largest = max(props, key=lambda r: r.area)
    binary = labelled == largest.label

    if dilation_radius > 0:
        binary = ndimage.binary_dilation(binary, structure=morphology.ball(dilation_radius))

    print(f"[mask] Mask volume: {binary.sum()} voxels")
    return binary.astype(np.uint8)


# ─────────────── 3b. MASK BOUNDING BOX ────────────────────────────────────

def compute_mask_bbox(mask: np.ndarray, pad: int = 3):
    """
    Tight voxel bounding box of the brain mask, with a small padding.
    mask shape: (Z, H, W)
    Returns (z0, z1, y0, y1, x0, x1) — all inclusive, clamped to volume.
    """
    Z, H, W = mask.shape
    proj_z = np.any(mask, axis=(1, 2))
    proj_y = np.any(mask, axis=(0, 2))
    proj_x = np.any(mask, axis=(0, 1))

    z_idx = np.where(proj_z)[0]
    y_idx = np.where(proj_y)[0]
    x_idx = np.where(proj_x)[0]

    if z_idx.size == 0 or y_idx.size == 0 or x_idx.size == 0:
        return 0, Z - 1, 0, H - 1, 0, W - 1

    return (
        max(0,     int(z_idx[0])  - pad),
        min(Z - 1, int(z_idx[-1]) + pad),
        max(0,     int(y_idx[0])  - pad),
        min(H - 1, int(y_idx[-1]) + pad),
        max(0,     int(x_idx[0])  - pad),
        min(W - 1, int(x_idx[-1]) + pad),
    )


# ─────────────────────────── 4. BUILD MESH ────────────────────────────────

def _taubin_smooth(mesh: trimesh.Trimesh,
                   iterations: int = 20,
                   lamb: float = 0.5,
                   mu: float = -0.53) -> trimesh.Trimesh:
    """
    Taubin (1995) smoothing: alternates positive and negative Laplacian steps
    so volume is approximately preserved while noise is suppressed.
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces

    adj = [set() for _ in range(len(vertices))]
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[f[i]].add(f[j])

    for _ in range(iterations):
        for sign, factor in [(1, lamb), (1, mu)]:
            new_v = vertices.copy()
            for i, nb in enumerate(adj):
                if nb:
                    nb_list = list(nb)
                    delta = vertices[nb_list].mean(axis=0) - vertices[i]
                    new_v[i] = vertices[i] + factor * delta
            vertices = new_v

    mesh.vertices = vertices
    return mesh


def build_mesh(mask: np.ndarray,
               volume_norm: np.ndarray,
               pixel_spacing: float = 1.0,
               slice_thickness: float = 1.0,
               step_size: int = 1,
               smooth_sigma: float = 1.5,
               taubin_iterations: int = 25,
               decimate_fraction: float = 0.85) -> trimesh.Trimesh:
    """
    Smoothed scalar field → marching cubes → Taubin smooth → clean → decimate.
    """
    print("[mesh] Pre-smoothing mask field …")
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(mask.astype(np.float32), sigma=smooth_sigma)

    print("[mesh] Marching cubes …")
    verts, faces, normals, _ = measure.marching_cubes(
        field, level=0.5, step_size=step_size
    )
    # Apply voxel spacing: (Z, Y, X) → scale each axis
    spacing = np.array([slice_thickness, pixel_spacing, pixel_spacing])
    verts = verts * spacing

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    print(f"[mesh] Raw mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    print("[mesh] Taubin smoothing …")
    mesh = _taubin_smooth(mesh, iterations=taubin_iterations)

    print("[mesh] Cleaning …")
    trimesh.repair.fix_normals(mesh)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    # Fill small holes
    trimesh.repair.fill_holes(mesh)

    # Decimate
    if decimate_fraction < 1.0:
        target = max(500, int(len(mesh.faces) * decimate_fraction))
        print(f"[mesh] Decimating to {target} faces …")
        try:
            mesh = mesh.simplify_quadric_decimation(target)
        except Exception as e:
            print(f"[mesh] Decimation warning: {e} — skipping")

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)

    # ── Vertex colours from MRI intensity ─────────────────────────────────
    # Sample volume_norm at each vertex via trilinear interpolation so the
    # GLB carries per-vertex colour, giving the cortex realistic MRI texture.
    print("[mesh] Sampling MRI intensity at vertices …")
    coords = np.stack([
        mesh.vertices[:, 0] / max(slice_thickness, 1e-6),   # Z_voxel
        mesh.vertices[:, 1] / max(pixel_spacing,   1e-6),   # Y_voxel
        mesh.vertices[:, 2] / max(pixel_spacing,   1e-6),   # X_voxel
    ], axis=0)  # shape (3, N)

    intensity = ndimage.map_coordinates(
        volume_norm.astype(np.float32),
        coords, order=1, mode='nearest', prefilter=False,
    )
    intensity = np.clip(intensity, 0, 255)

    # Greyscale: percentile-normalise so the full 0-255 range is used,
    # then apply mild gamma (0.75) to lift mid-tones and reveal sulcal detail.
    p_lo = float(np.percentile(intensity, 2))
    p_hi = float(np.percentile(intensity, 98))
    t    = np.clip((intensity - p_lo) / max(p_hi - p_lo, 1.0), 0.0, 1.0)
    t    = np.power(t, 0.75)
    val  = np.clip(t * 255, 0, 255).astype(np.uint8)
    a    = np.full(len(val), 255, dtype=np.uint8)
    vertex_colors = np.column_stack([val, val, val, a])   # pure greyscale RGBA
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)

    print(f"[mesh] Final mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


# ─────────────────────────── 5. EXPORT SLICES ─────────────────────────────

def export_slices(volume_norm: np.ndarray,
                  mask: np.ndarray,
                  output_dir: Path,
                  axis: str = "axial",
                  max_slices: int = 128,
                  pixel_spacing: float = 1.0,
                  slice_thickness: float = 1.0,
                  bbox: tuple = None) -> dict:
    """
    Export cropped MRI slice JPGs into output_dir/slices/{axis}/slice_NNN.jpg.

    Volume layout: (Z, H, W)
      axial    → slice along Z  → raw shape (H, W), cropped to [y0:y1, x0:x1]
      coronal  → slice along H  → raw shape (Z, W), cropped to [z0:z1, x0:x1]
      sagittal → slice along W  → raw shape (Z, H), cropped to [z0:z1, y0:y1]

    Slices are cropped to the tight brain bounding box so there is no black
    border — the exported image fills the entire quad in the viewer.

    Returns metadata with world-space quad dimensions and crop center.
    """
    slices_dir = output_dir / "slices" / axis
    slices_dir.mkdir(parents=True, exist_ok=True)

    axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}
    ax = axis_map.get(axis, 0)

    if bbox is None:
        bbox = compute_mask_bbox(mask)
    z0, z1, y0, y1, x0, x1 = bbox

    n_total = volume_norm.shape[ax]
    indices = np.linspace(0, n_total - 1, min(max_slices, n_total), dtype=int)

    print(f"[slices] Exporting {len(indices)} {axis} slices (cropped to brain bbox) …")

    # Global contrast from brain voxels only
    brain_voxels = volume_norm[mask > 0]
    g_lo = float(np.percentile(brain_voxels, 1))  if brain_voxels.size else 0.0
    g_hi = float(np.percentile(brain_voxels, 99)) if brain_voxels.size else 255.0
    g_range = max(g_hi - g_lo, 1.0)

    for out_i, src_i in enumerate(indices):
        # Extract raw 2-D slice
        if ax == 0:
            sl = volume_norm[src_i, y0:y1+1, x0:x1+1].astype(np.float32)
            mk = mask[src_i, y0:y1+1, x0:x1+1]
        elif ax == 1:
            sl = volume_norm[z0:z1+1, src_i, x0:x1+1].astype(np.float32)
            mk = mask[z0:z1+1, src_i, x0:x1+1]
        else:
            sl = volume_norm[z0:z1+1, y0:y1+1, src_i].astype(np.float32)
            mk = mask[z0:z1+1, y0:y1+1, src_i]

        # Greyscale intensity (background already 0 from mask multiply)
        sl = sl * mk.astype(np.float32)
        sl = np.clip((sl - g_lo) / g_range * 255.0, 0, 255).astype(np.uint8)

        # Alpha channel: Gaussian-feathered mask so brain edges blend smoothly
        alpha = ndimage.gaussian_filter(mk.astype(np.float32), sigma=1.5)
        alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)

        # ── Axis-specific spatial transform ───────────────────────────────
        # After Three.js rotation the plane's local axes map to world axes:
        #   axial    rotation.y=+π/2 → local X → world −Z, local Y → world +Y
        #   coronal  rotation.x=−π/2 → local X → world +X, local Y → world −Z
        #   sagittal no rotation     → local X → world +X, local Y → world +Y
        #
        # The raw slice shape is:
        #   axial    (Y_count, X_count): cols=X_voxels, rows=Y_voxels
        #   coronal  (Z_count, X_count): cols=X_voxels, rows=Z_voxels
        #   sagittal (Z_count, Y_count): cols=Y_voxels, rows=Z_voxels
        #
        # PlaneGeometry UV: u goes left→right (local X), v goes bottom→top (local Y).
        # Three.js flipY=true: UV.v=0 → image last row, UV.v=1 → image first row.
        #
        # Required mapping for correct alignment:
        #   axial:    image cols → world −Z (need fliplr) + image rows → world Y (need flipud) = rot90(2)
        #   coronal:  image cols must become Z_voxels (world X) → transpose;  rows→X_voxels (world −Z) OK
        #   sagittal: image cols must become Z_voxels (world X) → transpose;  rows→Y_voxels (world Y) + flipud
        if ax == 0:                          # axial: 180° rotation fixes both axes
            sl    = np.rot90(sl,    2)
            alpha = np.rot90(alpha, 2)
        elif ax == 1:                        # coronal: transpose only
            sl    = np.ascontiguousarray(sl.T)
            alpha = np.ascontiguousarray(alpha.T)
        else:                                # sagittal: transpose then flipud
            sl    = np.flipud(sl.T)
            alpha = np.flipud(alpha.T)

        # RGBA PNG — background pixels are fully transparent
        rgba = np.stack([sl, sl, sl, alpha], axis=-1)
        Image.fromarray(rgba, mode="RGBA") \
             .save(slices_dir / f"slice_{out_i:03d}.png", optimize=False)

    # ── Physical (world-space) quad dimensions ────────────────────────────
    # Three.js X = Z_voxel*dz,  Y = H_voxel*dy,  Z = W_voxel*dx
    # For each axis the quad's width = local X extent, height = local Y extent.
    # After rotation:
    #   axial:    local X → world −Z,   local Y → world +Y
    #   coronal:  local X → world +X,   local Y → world −Z
    #   sagittal: local X → world +X,   local Y → world +Y
    dz, dy, dx = slice_thickness, pixel_spacing, pixel_spacing

    if ax == 0:    # axial: width covers world-Z (X_voxels), height covers world-Y
        world_w = (x1 - x0 + 1) * dx      # Three.js Z extent
        world_h = (y1 - y0 + 1) * dy      # Three.js Y extent
        center_world = [
            None,
            (y0 + y1) / 2.0 * dy,
            (x0 + x1) / 2.0 * dx,
        ]
    elif ax == 1:  # coronal: width covers world-X (Z_voxels), height covers world-Z (X_voxels)
        world_w = (z1 - z0 + 1) * dz      # Three.js X extent  ← was X extent before (wrong)
        world_h = (x1 - x0 + 1) * dx      # Three.js Z extent  ← was Z extent before (wrong)
        center_world = [
            (z0 + z1) / 2.0 * dz,
            None,
            (x0 + x1) / 2.0 * dx,
        ]
    else:          # sagittal: width covers world-X (Z_voxels), height covers world-Y (Y_voxels)
        world_w = (z1 - z0 + 1) * dz      # Three.js X extent  ← was Y extent before (wrong)
        world_h = (y1 - y0 + 1) * dy      # Three.js Y extent  ← was X extent before (wrong)
        center_world = [
            (z0 + z1) / 2.0 * dz,
            (y0 + y1) / 2.0 * dy,
            None,
        ]

    return {
        "count":        int(len(indices)),
        "world_w":      float(world_w),
        "world_h":      float(world_h),
        "center_world": [v if v is not None else 0.0 for v in center_world],
        "source_indices": [int(i) for i in indices],
    }


def export_all_slices(volume_norm: np.ndarray,
                      mask: np.ndarray,
                      output_dir: Path,
                      max_slices: int = 128,
                      pixel_spacing: float = 1.0,
                      slice_thickness: float = 1.0) -> dict:
    """Export slices for all three axes with shared crop bbox."""
    bbox = compute_mask_bbox(mask)
    result = {}
    for axis in ("axial", "coronal", "sagittal"):
        result[axis] = export_slices(
            volume_norm, mask, output_dir,
            axis=axis, max_slices=max_slices,
            pixel_spacing=pixel_spacing, slice_thickness=slice_thickness,
            bbox=bbox,
        )
    return result


# ─────────────────────────── 6. METADATA ──────────────────────────────────

def save_metadata(output_dir: Path,
                  volume: np.ndarray,
                  mask: np.ndarray,
                  mesh: trimesh.Trimesh,
                  slices_by_axis: dict,
                  pixel_spacing: float,
                  slice_thickness: float,
                  primary_axis: str) -> None:
    """
    Write viewer/assets/volume_meta.json.

    slices_by_axis: { "axial": {...}, "coronal": {...}, "sagittal": {...} }

    Coordinate mapping stored so the viewer can verify:
      Three.js X  ←  verts[:,0] = Z_voxel * slice_thickness  (axial depth)
      Three.js Y  ←  verts[:,1] = H_voxel * pixel_spacing    (height / coronal)
      Three.js Z  ←  verts[:,2] = W_voxel * pixel_spacing    (width  / sagittal)
    """
    bbox_min = mesh.bounds[0].tolist()
    bbox_max = mesh.bounds[1].tolist()
    center = mesh.centroid.tolist()

    meta = {
        "volume": {
            "shape": list(volume.shape),   # [Z, H, W]
            "primary_axis": primary_axis,
            "pixel_spacing_mm": pixel_spacing,
            "slice_thickness_mm": slice_thickness,
            # axis→Three.js dim mapping (for documentation / viewer reference)
            "threejs_axis_map": {
                "axial":    "x",   # Three.js X = Z_voxel * dz
                "coronal":  "y",   # Three.js Y = H_voxel * dy
                "sagittal": "z",   # Three.js Z = W_voxel * dx
            },
        },
        "mesh": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "center": center,
        },
        "slices": slices_by_axis,
    }

    out = output_dir / "volume_meta.json"
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[meta] Saved {out}")


# ─────────────────────────── 7. PREVIEWS ──────────────────────────────────

def save_previews(volume_norm: np.ndarray,
                  mask: np.ndarray,
                  output_dir: Path) -> None:
    """Save a few orthogonal mid-plane previews for quick visual QC."""
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    Z, Y, X = volume_norm.shape
    for label, sl in [
        ("axial_mid",    volume_norm[Z // 2, :, :]),
        ("coronal_mid",  volume_norm[:, Y // 2, :]),
        ("sagittal_mid", volume_norm[:, :, X // 2]),
    ]:
        img = Image.fromarray(sl, mode="L")
        img.save(preview_dir / f"{label}.png")

    # also save mask projections
    for label, proj in [
        ("mask_axial",    mask.max(axis=0)),
        ("mask_coronal",  mask.max(axis=1)),
        ("mask_sagittal", mask.max(axis=2)),
    ]:
        img = Image.fromarray((proj * 255).astype(np.uint8), mode="L")
        img.save(preview_dir / f"{label}.png")

    print(f"[preview] Saved previews to {preview_dir}")


# ─────────────────────────── MAIN ─────────────────────────────────────────

def run(args):
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir) / "viewer" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    volume_raw, _ = load_slices(input_dir)

    # 2. Normalise
    volume_norm = normalise_volume(volume_raw, sigma=args.smooth_sigma)

    # 3. Mask
    mask = build_brain_mask(volume_norm,
                            closing_radius=args.closing_radius,
                            dilation_radius=args.dilation_radius)

    # 4. Mesh
    mesh = build_mesh(mask, volume_norm,
                      pixel_spacing=args.pixel_spacing,
                      slice_thickness=args.slice_thickness,
                      smooth_sigma=args.field_sigma,
                      taubin_iterations=args.taubin_iter,
                      decimate_fraction=args.decimate_fraction)

    glb_path = output_dir / "brain_surface.glb"
    mesh.export(str(glb_path))
    print(f"[mesh] Exported {glb_path}")

    # 5. Slices — export all three axes so the viewer supports any cut direction
    slices_by_axis = export_all_slices(volume_norm, mask, output_dir,
                                       max_slices=args.max_slices,
                                       pixel_spacing=args.pixel_spacing,
                                       slice_thickness=args.slice_thickness)

    # 6. Metadata
    save_metadata(output_dir, volume_norm, mask, mesh, slices_by_axis,
                  args.pixel_spacing, args.slice_thickness, args.axis)

    # 7. Previews
    save_previews(volume_norm, mask, output_dir)

    print("\n[done] Preprocessing complete.")
    print(f"       Assets written to: {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="GAN MRI → Brain mesh pipeline")
    p.add_argument("--input_dir",          default="data/gan_slices")
    p.add_argument("--output_dir",         default=".")
    p.add_argument("--axis",               default="axial",
                   choices=["axial", "coronal", "sagittal"])
    p.add_argument("--pixel_spacing",      type=float, default=1.0)
    p.add_argument("--slice_thickness",    type=float, default=1.0)
    p.add_argument("--smooth_sigma",       type=float, default=0.8,
                   help="Pre-normalisation Gaussian sigma (GAN artefact reduction)")
    p.add_argument("--field_sigma",        type=float, default=1.5,
                   help="Pre-marching-cubes field smoothing sigma")
    p.add_argument("--closing_radius",     type=int,   default=4)
    p.add_argument("--dilation_radius",    type=int,   default=1)
    p.add_argument("--taubin_iter",        type=int,   default=25)
    p.add_argument("--decimate_fraction",  type=float, default=0.85)
    p.add_argument("--max_slices",         type=int,   default=128,
                   help="Max slice PNGs exported for the viewer")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
