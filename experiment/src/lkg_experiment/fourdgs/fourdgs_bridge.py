from __future__ import annotations

import re
import sys
import types
from argparse import Namespace
from dataclasses import dataclass
from importlib import import_module
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from lkg_experiment.coherent_default.coherent_gsplat_bridge import scale_intrinsics_to_panel


DEFAULT_4DGS_CODE_ROOT = Path("/home/ysj/GS/4DGaussians")


_PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass
class FourDGSCheckpoint:
    static_splats: Mapping[str, Any]
    sh_degree: int
    iteration: int
    iteration_dir: Path
    cfg_args: Namespace
    deformation: Optional[Any] = None
    device: str = "cuda"

    @property
    def gs_count(self) -> int:
        return int(self.static_splats["means"].shape[0])

    def splats_at(self, time_value: float, *, stage: str = "fine") -> dict[str, Any]:
        if stage not in {"fine", "coarse"}:
            raise ValueError("stage must be 'fine' or 'coarse'")
        splats = {key: value for key, value in self.static_splats.items()}
        if self.deformation is None or stage == "coarse":
            return {key: value.contiguous() for key, value in splats.items()}

        import torch

        means = splats["means"]
        scales = splats["scales"]
        quats = splats["quats"]
        opacities = splats["opacities"].reshape(-1, 1)
        shs = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        times = torch.full((means.shape[0], 1), float(time_value), dtype=means.dtype, device=means.device)
        with torch.no_grad():
            means_t, scales_t, quats_t, opacities_t, shs_t = self.deformation(
                means,
                scales,
                quats,
                opacities,
                shs,
                times,
            )
        return {
            "means": means_t.contiguous(),
            "opacities": opacities_t.reshape(-1).contiguous(),
            "quats": quats_t.contiguous(),
            "scales": scales_t.contiguous(),
            "sh0": shs_t[:, :1, :].contiguous(),
            "shN": shs_t[:, 1:, :].contiguous(),
        }


def parse_4dgs_cfg_args(path: Path | str) -> Namespace:
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"4DGS cfg_args file not found: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8").strip()
    value = eval(text, {"__builtins__": {}, "Namespace": Namespace}, {})
    if not isinstance(value, Namespace):
        raise ValueError(f"4DGS cfg_args did not evaluate to argparse.Namespace: {cfg_path}")
    return value


def resolve_4dgs_iteration_dir(model_path: Path | str, iteration: Optional[int] = None) -> Path:
    root = Path(model_path).expanduser()
    if root.name.startswith("iteration_") and root.is_dir():
        if iteration is not None and _iteration_number(root) != int(iteration):
            raise ValueError(f"iteration dir {root} does not match requested iteration {iteration}")
        return root

    point_cloud_dir = root / "point_cloud"
    if not point_cloud_dir.is_dir():
        raise FileNotFoundError(f"4DGS point_cloud directory not found: {point_cloud_dir}")

    if iteration is not None:
        selected = point_cloud_dir / f"iteration_{int(iteration)}"
        if not selected.is_dir():
            raise FileNotFoundError(f"4DGS iteration directory not found: {selected}")
        return selected

    candidates = [
        path
        for path in point_cloud_dir.iterdir()
        if path.is_dir() and _iteration_number(path) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"no 4DGS iteration_* directories found in {point_cloud_dir}")
    return max(candidates, key=lambda path: int(_iteration_number(path) or -1))


def load_4dgs_checkpoint(
    model_path: Path | str,
    *,
    code_root: Path | str | None = None,
    iteration: Optional[int] = None,
    device: str = "cuda",
    load_deformation: bool = True,
) -> FourDGSCheckpoint:
    root = Path(model_path).expanduser()
    cfg_path = root / "cfg_args"
    cfg_args = parse_4dgs_cfg_args(cfg_path) if cfg_path.is_file() else Namespace(sh_degree=3)
    sh_degree = int(getattr(cfg_args, "sh_degree", 3))
    iteration_dir = resolve_4dgs_iteration_dir(root, iteration=iteration)
    iteration_value = int(_iteration_number(iteration_dir) or getattr(cfg_args, "iterations", -1))
    static_splats = load_4dgs_static_splats(iteration_dir / "point_cloud.ply", sh_degree=sh_degree, device=device)

    deformation = None
    if load_deformation:
        deformation_path = iteration_dir / "deformation.pth"
        if not deformation_path.is_file():
            raise FileNotFoundError(f"4DGS deformation checkpoint not found: {deformation_path}")
        import torch

        deform_network = load_4dgs_deform_network_class(code_root or DEFAULT_4DGS_CODE_ROOT)
        deformation = deform_network(cfg_args).to(device)
        state = torch.load(str(deformation_path), map_location=device)
        deformation.load_state_dict(state)
        deformation.eval()

    return FourDGSCheckpoint(
        static_splats=static_splats,
        sh_degree=sh_degree,
        iteration=iteration_value,
        iteration_dir=iteration_dir,
        cfg_args=cfg_args,
        deformation=deformation,
        device=str(device),
    )


def load_4dgs_static_splats(path: Path | str, *, sh_degree: int, device: str = "cuda") -> dict[str, Any]:
    import torch

    columns = _read_ply_vertex_columns(Path(path).expanduser())
    sh_count = (int(sh_degree) + 1) ** 2
    rest_count = 3 * sh_count - 3

    means = _stack_columns(columns, ("x", "y", "z"))
    opacities = _column(columns, "opacity")
    scales = _stack_columns(columns, ("scale_0", "scale_1", "scale_2"))
    quats = _stack_columns(columns, ("rot_0", "rot_1", "rot_2", "rot_3"))
    sh0 = _stack_columns(columns, ("f_dc_0", "f_dc_1", "f_dc_2"))[:, None, :]
    if rest_count:
        rest_names = tuple(f"f_rest_{idx}" for idx in range(rest_count))
        rest_flat = _stack_columns(columns, rest_names)
        shn = rest_flat.reshape(rest_flat.shape[0], 3, sh_count - 1).transpose(0, 2, 1)
    else:
        shn = np.empty((means.shape[0], 0, 3), dtype=np.float32)

    return {
        "means": torch.as_tensor(means, dtype=torch.float32, device=device).contiguous(),
        "opacities": torch.as_tensor(opacities, dtype=torch.float32, device=device).reshape(-1).contiguous(),
        "quats": torch.as_tensor(quats, dtype=torch.float32, device=device).contiguous(),
        "scales": torch.as_tensor(scales, dtype=torch.float32, device=device).contiguous(),
        "sh0": torch.as_tensor(sh0, dtype=torch.float32, device=device).contiguous(),
        "shN": torch.as_tensor(shn, dtype=torch.float32, device=device).contiguous(),
    }


def install_4dgs_code_root(code_root: Path | str) -> Path:
    root = Path(code_root).expanduser().resolve()
    if not (root / "scene" / "deformation.py").is_file():
        raise FileNotFoundError(f"4DGaussians code root not found or incomplete: {root}")
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def load_4dgs_deform_network_class(code_root: Path | str):
    root = install_4dgs_code_root(code_root)
    _ensure_namespace_package("scene", root / "scene")
    _ensure_namespace_package("utils", root / "utils")
    module = import_module("scene.deformation")
    try:
        return module.deform_network
    except AttributeError as exc:
        raise RuntimeError(f"4DGaussians deformation module has no deform_network: {root}") from exc


def _ensure_namespace_package(name: str, directory: Path) -> None:
    if not directory.is_dir():
        return
    module = sys.modules.get(name)
    directory_s = str(directory)
    if module is not None:
        search_locations = getattr(module, "__path__", None)
        if search_locations is not None and directory_s in [str(path) for path in search_locations]:
            return

    package = types.ModuleType(name)
    package.__package__ = name
    package.__path__ = [directory_s]
    spec = ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [directory_s]
    package.__spec__ = spec
    sys.modules[name] = package


def has_4dgs_dynerf_camera(cfg_args: Any) -> bool:
    source_path = getattr(cfg_args, "source_path", None)
    if not source_path:
        return False
    return (Path(str(source_path)).expanduser() / "poses_bounds.npy").is_file()


def load_4dgs_dynerf_camera(
    cfg_args: Any,
    args: Any,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    source_path_value = getattr(cfg_args, "source_path", None)
    if not source_path_value:
        raise ValueError("4DGS cfg_args has no source_path for dynerf camera loading")
    source_path = Path(str(source_path_value)).expanduser()
    poses_path = source_path / "poses_bounds.npy"
    if not poses_path.is_file():
        raise FileNotFoundError(f"4DGS dynerf poses_bounds.npy not found: {poses_path}")

    poses_arr = np.load(poses_path)
    if poses_arr.ndim != 2 or poses_arr.shape[1] < 17:
        raise ValueError(f"poses_bounds.npy must have shape (N,17+): {poses_path}")
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    if poses.shape[0] == 0:
        raise ValueError(f"poses_bounds.npy contains no cameras: {poses_path}")

    pose_height, pose_width, pose_focal = [float(value) for value in poses[0, :, -1]]
    downsample = max(pose_width / 1352.0, 1e-8)
    native_width = int(round(pose_width / downsample))
    native_height = int(round(pose_height / downsample))
    focal = pose_focal / downsample

    converted = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], axis=-1)
    requested_split = "val" if args.camera_split == "auto" else str(args.camera_split)
    eval_index = 0
    if requested_split in {"val", "test"}:
        camera_indices = [eval_index]
    elif requested_split == "train":
        camera_indices = [index for index in range(converted.shape[0]) if index != eval_index]
    else:
        raise ValueError("4DGS dynerf camera supports --camera-split auto, val, test, or train")
    if not camera_indices:
        raise ValueError(f"4DGS dynerf split {requested_split!r} has no cameras")
    if args.camera_index < 0 or args.camera_index >= len(camera_indices):
        raise IndexError(
            f"--camera-index {args.camera_index} outside {requested_split} split count {len(camera_indices)}"
        )

    all_c2ws = np.stack([_dynerf_pose_to_c2w(converted[index]) for index in range(converted.shape[0])], axis=0)
    selected_camera_index = int(camera_indices[int(args.camera_index)])
    c2w = all_c2ws[selected_camera_index].astype(np.float32)
    K_original = np.array(
        [[focal, 0.0, native_width / 2.0], [0.0, focal, native_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    K_scaled = scale_intrinsics_to_panel(
        K_original,
        original_height=native_height,
        original_width=native_width,
        target_height=int(height),
        target_width=int(width),
        crop_to_fill=not args.no_crop_to_fill,
    )
    scene_center = all_c2ws[:, :3, 3].mean(axis=0).astype(np.float32)
    label = f"fourdgs-dynerf:{source_path}:{requested_split}[{args.camera_index}]/cam{selected_camera_index:02d}"
    return c2w, K_scaled.astype(np.float32), scene_center, label


def _dynerf_pose_to_c2w(pose: np.ndarray) -> np.ndarray:
    R = np.array(pose[:3, :3], dtype=np.float64, copy=True)
    R = -R
    R[:, 0] = -R[:, 0]
    T = -np.asarray(pose[:3, 3], dtype=np.float64).dot(R)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = T
    return np.linalg.inv(w2c).astype(np.float32)


def _iteration_number(path: Path) -> Optional[int]:
    match = re.match(r"^iteration_(\d+)$", path.name)
    return int(match.group(1)) if match else None


def _column(columns: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in columns:
        raise KeyError(f"PLY missing required property {name!r}")
    return np.asarray(columns[name], dtype=np.float32)


def _stack_columns(columns: Mapping[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    return np.stack([_column(columns, name) for name in names], axis=1).astype(np.float32, copy=False)


def _read_ply_vertex_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        fmt, count, properties = _read_ply_header(f)
        if fmt == "ascii":
            values = np.loadtxt(f, dtype=np.float32, max_rows=count)
            values = np.atleast_2d(values)
            if values.shape != (count, len(properties)):
                raise ValueError(f"PLY vertex body shape {values.shape} does not match header {(count, len(properties))}")
            return {name: values[:, idx] for idx, (_, name) in enumerate(properties)}
        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {fmt}")

        dtype = np.dtype([(name, "<" + _PLY_DTYPES[type_name]) for type_name, name in properties])
        data = np.fromfile(f, dtype=dtype, count=count)
        if int(data.shape[0]) != int(count):
            raise ValueError(f"PLY binary body has {data.shape[0]} vertices, expected {count}")
        return {name: np.asarray(data[name]) for _, name in properties}


def _read_ply_header(file_obj) -> tuple[str, int, list[tuple[str, str]]]:
    first = file_obj.readline().decode("ascii", errors="strict").strip()
    if first != "ply":
        raise ValueError("not a PLY file")
    fmt = ""
    vertex_count = None
    vertex_properties: list[tuple[str, str]] = []
    current_element = None

    while True:
        raw = file_obj.readline()
        if not raw:
            raise ValueError("PLY header ended before end_header")
        line = raw.decode("ascii", errors="strict").strip()
        if line == "end_header":
            break
        if not line or line.startswith("comment "):
            continue
        parts = line.split()
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                raise ValueError("PLY list properties are not supported for vertices")
            if parts[1] not in _PLY_DTYPES:
                raise ValueError(f"Unsupported PLY property type: {parts[1]}")
            vertex_properties.append((parts[1], parts[2]))

    if not fmt:
        raise ValueError("PLY header missing format")
    if vertex_count is None:
        raise ValueError("PLY header missing vertex element")
    return fmt, int(vertex_count), vertex_properties
