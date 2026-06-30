from __future__ import annotations

from pathlib import Path
from typing import Any


def install_gsplat_root(gsplat_root: Path | str) -> None:
    import sys

    root = Path(gsplat_root).expanduser().resolve()
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


def render_splats_gsplat(
    splats: Any,
    *,
    viewmat: Any,
    K: Any,
    width: int,
    height: int,
    device: str,
):
    import torch
    import gsplat

    viewmat_t = torch.as_tensor(viewmat, dtype=torch.float32, device=device).contiguous()
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device).contiguous()
    with torch.no_grad():
        image, _alpha, _meta = gsplat.rasterization(
            means=splats.means,
            quats=splats.quats,
            scales=splats.scales,
            opacities=splats.opacities,
            colors=splats.colors,
            viewmats=viewmat_t.unsqueeze(0),
            Ks=K_t.unsqueeze(0),
            sh_degree=int(splats.sh_degree),
            width=int(width),
            height=int(height),
        )
    return image[0].permute(2, 0, 1).clamp(0.0, 1.0).contiguous()
