"""
run_pipeline.py

Master runner: preprocessing → asset validation → GitHub Pages prep.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[error] Step '{label}' failed (exit {result.returncode})")
        sys.exit(result.returncode)


def validate_inputs(input_dir: Path) -> None:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    images = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    if not images:
        print(f"[error] No images found in {input_dir}")
        sys.exit(1)
    print(f"[validate] {len(images)} input slices found in {input_dir}")


def validate_outputs(output_dir: Path) -> bool:
    assets = output_dir / "viewer" / "assets"
    required = [
        assets / "brain_surface.glb",
        assets / "volume_meta.json",
    ]
    all_ok = True
    print("\n[validate] Checking output assets …")
    for p in required:
        if p.exists():
            size_kb = p.stat().st_size // 1024
            print(f"  [ok]  {p.relative_to(output_dir)}  ({size_kb} KB)")
        else:
            print(f"  [!!]  MISSING: {p.relative_to(output_dir)}")
            all_ok = False

    slices = list((assets / "slices").glob("slice_*.jpg")) if (assets / "slices").exists() else []
    if slices:
        print(f"  [ok]  {len(slices)} slice JPGs")
    else:
        print("  [!!]  No slice JPGs found")
        all_ok = False

    return all_ok


def print_final_instructions(args, deploy_dst: Path) -> None:
    repo  = args.project_repo_name or "<YOUR_REPO>"
    sub   = args.github_pages_subpath or f"/{repo}/brain-viewer/"
    print(f"""
╔══════════════════════════════════════════════════════════╗
║              PIPELINE COMPLETE                           ║
╠══════════════════════════════════════════════════════════╣
║  Static viewer built at:                                 ║
║    {str(deploy_dst):<52} ║
╠══════════════════════════════════════════════════════════╣
║  To publish on GitHub Pages:                             ║
║                                                          ║
║  1. Copy  docs/brain-viewer/  into your main repo        ║
║  2. Repo Settings → Pages → Source: main / /docs         ║
║  3. Viewer URL:                                          ║
║    https://<USER>.github.io/{repo}/brain-viewer/   ║
╠══════════════════════════════════════════════════════════╣
║  To run locally:                                         ║
║    cd docs/brain-viewer && python -m http.server 8080    ║
║    open http://localhost:8080                            ║
╚══════════════════════════════════════════════════════════╝
""")


def parse_args():
    p = argparse.ArgumentParser(description="Full GAN-MRI brain viewer pipeline")
    p.add_argument("--input_dir",              default="data/gan_slices")
    p.add_argument("--output_dir",             default=".")
    p.add_argument("--axis",                   default="axial",
                   choices=["axial", "coronal", "sagittal"])
    p.add_argument("--pixel_spacing",          type=float, default=1.0)
    p.add_argument("--slice_thickness",        type=float, default=1.0)
    p.add_argument("--project_repo_name",      default="")
    p.add_argument("--github_pages_subpath",   default="")
    p.add_argument("--deploy_dst",             default="docs/brain-viewer")
    # pass-through to generate_brain.py
    p.add_argument("--smooth_sigma",           type=float, default=0.8)
    p.add_argument("--field_sigma",            type=float, default=1.5)
    p.add_argument("--closing_radius",         type=int,   default=4)
    p.add_argument("--dilation_radius",        type=int,   default=1)
    p.add_argument("--taubin_iter",            type=int,   default=25)
    p.add_argument("--decimate_fraction",      type=float, default=0.85)
    p.add_argument("--max_slices",             type=int,   default=128)
    return p.parse_args()


def main():
    args = parse_args()
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    deploy_dst = output_dir / args.deploy_dst

    print("[pipeline] Validating inputs …")
    validate_inputs(input_dir)

    # ── Step 1: Generate mesh + slices
    gen_cmd = [
        sys.executable, "generate_brain.py",
        "--input_dir",         str(input_dir),
        "--output_dir",        str(output_dir),
        "--axis",              args.axis,
        "--pixel_spacing",     str(args.pixel_spacing),
        "--slice_thickness",   str(args.slice_thickness),
        "--smooth_sigma",      str(args.smooth_sigma),
        "--field_sigma",       str(args.field_sigma),
        "--closing_radius",    str(args.closing_radius),
        "--dilation_radius",   str(args.dilation_radius),
        "--taubin_iter",       str(args.taubin_iter),
        "--decimate_fraction", str(args.decimate_fraction),
        "--max_slices",        str(args.max_slices),
    ]
    run_step(gen_cmd, "Generate brain mesh + export slices")

    # ── Step 2: Validate outputs
    if not validate_outputs(output_dir):
        print("[pipeline] Output validation failed. Exiting.")
        sys.exit(1)

    # ── Step 3: Prepare GitHub Pages
    pages_cmd = [
        sys.executable, "prepare_github_pages.py",
        "--viewer_src", str(output_dir / "viewer"),
        "--deploy_dst", str(deploy_dst),
        "--base_path",  args.github_pages_subpath,
    ]
    run_step(pages_cmd, "Prepare GitHub Pages build")

    print_final_instructions(args, deploy_dst)


if __name__ == "__main__":
    main()
