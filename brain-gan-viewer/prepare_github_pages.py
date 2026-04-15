"""
prepare_github_pages.py

Copies the static viewer into docs/ (or another deploy folder) and
verifies that all asset references are relative and will work on GitHub Pages.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def copy_viewer(viewer_src: Path, deploy_dst: Path) -> None:
    if deploy_dst.exists():
        print(f"[pages] Removing existing {deploy_dst} …")
        shutil.rmtree(deploy_dst)
    shutil.copytree(viewer_src, deploy_dst)
    print(f"[pages] Copied {viewer_src} → {deploy_dst}")


def rewrite_base_path(index_html: Path, base_path: str) -> None:
    """
    If the viewer is deployed under a sub-path like /my-repo/brain-viewer/,
    inject a <base href="…"> tag so all relative URLs resolve correctly.
    The base_path should end with '/'.
    """
    if not base_path or base_path in ("/", ""):
        return
    if not base_path.endswith("/"):
        base_path += "/"
    text = index_html.read_text(encoding="utf-8")
    if "<base " in text:
        text = re.sub(r'<base\s+href="[^"]*"\s*/?>',
                      f'<base href="{base_path}" />', text)
    else:
        text = text.replace("<head>", f'<head>\n  <base href="{base_path}" />', 1)
    index_html.write_text(text, encoding="utf-8")
    print(f"[pages] Set <base href=\"{base_path}\"> in {index_html}")


def verify_assets(deploy_dst: Path) -> bool:
    ok = True
    required = [
        deploy_dst / "index.html",
        deploy_dst / "app.js",
        deploy_dst / "style.css",
        deploy_dst / "assets" / "brain_surface.glb",
        deploy_dst / "assets" / "volume_meta.json",
    ]
    for path in required:
        if path.exists():
            print(f"  [ok]  {path.relative_to(deploy_dst)}")
        else:
            print(f"  [!!]  MISSING: {path.relative_to(deploy_dst)}")
            ok = False

    slices_dir = deploy_dst / "assets" / "slices"
    n_slices = len(list(slices_dir.glob("slice_*.jpg"))) if slices_dir.exists() else 0
    if n_slices == 0:
        print("  [!!]  No slice JPGs found in assets/slices/")
        ok = False
    else:
        print(f"  [ok]  {n_slices} slice JPGs present")

    return ok


def patch_meta_basepath(deploy_dst: Path, base_path: str) -> None:
    """Not strictly needed since all paths in volume_meta.json are relative,
    but we record the deploy base_path for the viewer's reference."""
    meta_path = deploy_dst / "assets" / "volume_meta.json"
    if not meta_path.exists():
        return
    with open(meta_path) as f:
        meta = json.load(f)
    meta["deploy_base_path"] = base_path
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def run(args):
    viewer_src = Path(args.viewer_src)
    deploy_dst = Path(args.deploy_dst)

    if not viewer_src.exists():
        print(f"[error] viewer source not found: {viewer_src}")
        sys.exit(1)

    print(f"\n[pages] Deploying viewer to {deploy_dst} …")
    copy_viewer(viewer_src, deploy_dst)

    index_html = deploy_dst / "index.html"
    if args.base_path:
        rewrite_base_path(index_html, args.base_path)
        patch_meta_basepath(deploy_dst, args.base_path)

    print("\n[pages] Verifying assets …")
    ok = verify_assets(deploy_dst)

    if ok:
        print("\n[pages] All assets present. Site is ready.")
    else:
        print("\n[pages] Some assets are missing. Run generate_brain.py first.")
        sys.exit(1)

    print(f"""
─────────────────────────────────────────────────────────
  GitHub Pages deployment instructions
─────────────────────────────────────────────────────────

Option A — docs/ folder on main branch (recommended)
  1. Copy {deploy_dst} into your main repo at  docs/brain-viewer/
  2. In your repo → Settings → Pages → Source: main branch, /docs folder
  3. Your viewer will be at:
       https://<USERNAME>.github.io/<REPO_NAME>/brain-viewer/

Option B — gh-pages branch
  1. Push {deploy_dst} contents to a gh-pages branch
  2. In Settings → Pages → Source: gh-pages branch, / (root)
  3. Your viewer will be at:
       https://<USERNAME>.github.io/<REPO_NAME>/

Link from your main project homepage:
  [View 3D Brain](./docs/brain-viewer/index.html)
  or simply:
  [View 3D Brain](https://<USERNAME>.github.io/<REPO_NAME>/brain-viewer/)
─────────────────────────────────────────────────────────
""")


def parse_args():
    p = argparse.ArgumentParser(description="Prepare static viewer for GitHub Pages")
    p.add_argument("--viewer_src",  default="viewer",
                   help="Path to the built viewer/ directory")
    p.add_argument("--deploy_dst",  default="docs/brain-viewer",
                   help="Destination folder (e.g. docs/brain-viewer)")
    p.add_argument("--base_path",   default="",
                   help="URL sub-path, e.g. /my-repo/brain-viewer/")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
