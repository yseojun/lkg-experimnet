from __future__ import annotations

import lzma
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FtgsSplats:
    means: Any
    quats: Any
    scales: Any
    opacities: Any
    colors: Any
    sh_degree: int


def peek_checkpoint_keys(path: Path | str) -> set[str]:
    payload = _load_checkpoint_payload(path)
    return set(str(key) for key in payload.keys())


def load_dynamic_gaussians(
    path: Path | str,
    *,
    device: str = "cuda",
) -> DynamicGaussians:
    payload = _load_checkpoint_payload(path)
    model = DynamicGaussians(device=device)
    model.decode(payload)
    return model


def _load_checkpoint_payload(path: Path | str) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"OMG4-FTGS checkpoint not found: {checkpoint_path}")
    try:
        with lzma.open(str(checkpoint_path), "rb") as f:
            payload = pickle.load(f)
    except ModuleNotFoundError as exc:
        if exc.name == "dahuffman":
            raise RuntimeError("dahuffman is required to read OMG4-FTGS checkpoints") from exc
        raise
    if not isinstance(payload, dict):
        raise ValueError(f"OMG4-FTGS checkpoint must contain a dict: {checkpoint_path}")
    return payload


class DynamicGaussians:
    def __init__(self, *, device: str = "cuda") -> None:
        self.device = str(device)
        self.sh_degree = 3
        self.time_duration = [0.0, 10.0]

    def decode(self, payload: dict[str, Any]) -> None:
        import torch
        from torch import nn

        self._ensure_tcnn()
        means = payload["means"]
        times = payload["times"]
        self.means = nn.Parameter(torch.from_numpy(means).to(device=self.device, dtype=torch.float32), requires_grad=False)
        self.times = nn.Parameter(torch.from_numpy(times).to(device=self.device, dtype=torch.float32), requires_grad=False)

        self.scales = nn.Parameter(self._decode_codebooks(payload, "scale").float(), requires_grad=False)
        self.quats = nn.Parameter(self._decode_codebooks(payload, "rotation").float(), requires_grad=False)
        self.durations = nn.Parameter(self._decode_codebooks(payload, "durations").float(), requires_grad=False)
        self.velocities = nn.Parameter(self._decode_codebooks(payload, "velocities").float(), requires_grad=False)
        appearance = self._decode_codebooks(payload, "app").float()

        self._construct_net()
        self.mlp_cont.params = nn.Parameter(
            torch.as_tensor(payload["MLP_cont"], device=self.device, dtype=torch.float16),
            requires_grad=False,
        )
        self.mlp_dc.params = nn.Parameter(
            torch.as_tensor(payload["MLP_dc"], device=self.device, dtype=torch.float16),
            requires_grad=False,
        )
        self.mlp_view.params = nn.Parameter(
            torch.as_tensor(payload["MLP_sh"], device=self.device, dtype=torch.float16),
            requires_grad=False,
        )
        self.mlp_opacity.params = nn.Parameter(
            torch.as_tensor(payload["MLP_opacity"], device=self.device, dtype=torch.float16),
            requires_grad=False,
        )
        self._features_static = nn.Parameter(appearance[:, 0:3].clone().detach(), requires_grad=False)
        self._features_view = nn.Parameter(appearance[:, 3:6].clone().detach(), requires_grad=False)

    def materialize(self, timestamp: float) -> FtgsSplats:
        import torch

        t = float(timestamp)
        means_t = self.means + (t - self.times) * self.velocities
        scales = torch.exp(self.scales)
        temporal_opacity = torch.exp(-0.5 * ((t - self.times) / self.durations.exp()) ** 2)
        contracted = self.contract_to_unisphere(
            means_t.clone().detach(),
            torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=self.device),
        )
        cont_feature = self.mlp_cont(contracted)
        space_feature = torch.cat([cont_feature, self._features_static], dim=-1)
        view_feature = torch.cat([cont_feature, self._features_view], dim=-1)
        shs = self.mlp_view(view_feature).reshape(-1, 47, 3).float()
        dc = self.mlp_dc(space_feature).reshape(-1, 1, 3).float()
        opacities = torch.sigmoid(self.mlp_opacity(space_feature).float())
        colors = torch.cat([dc, shs], dim=1)
        return FtgsSplats(
            means=means_t.contiguous(),
            quats=self.quats.contiguous(),
            scales=scales.contiguous(),
            opacities=(opacities * temporal_opacity).reshape(-1).contiguous(),
            colors=colors.contiguous(),
            sh_degree=int(self.sh_degree),
        )

    def _decode_codebooks(self, payload: dict[str, Any], prefix: str):
        import numpy as np
        import torch
        from dahuffman import HuffmanCodec

        decoded = []
        for encoded, table, codebook in zip(
            payload[f"{prefix}_index"],
            payload[f"{prefix}_htable"],
            payload[f"{prefix}_code"],
        ):
            labels = np.asarray(HuffmanCodec(code_table=table).decode(encoded), dtype=np.uint16)
            decoded.append(torch.as_tensor(codebook[labels], device=self.device))
        return torch.cat(decoded, dim=-1)

    def _ensure_tcnn(self) -> None:
        try:
            import tinycudann  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "tinycudann is required for OMG4-FTGS decode; ensure CUDA is visible and use the rtgs-coherent-cu121 environment"
            ) from exc

    def _construct_net(self) -> None:
        import tinycudann as tcnn

        self.mlp_cont = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=13,
            encoding_config={"otype": "Frequency", "n_frequencies": 16},
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        self.mlp_view = tcnn.Network(
            n_input_dims=16,
            n_output_dims=3 * 47,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        self.mlp_dc = tcnn.Network(
            n_input_dims=16,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        self.mlp_opacity = tcnn.Network(
            n_input_dims=16,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

    def contract_to_unisphere(self, x: Any, aabb: Any, ord: int = 2, eps: float = 1e-6):
        import torch

        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2.0 - 1.0
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1.0
        x = x.clone()
        x[mask] = (2.0 - 1.0 / mag[mask]) * (x[mask] / mag[mask])
        return x / 4.0 + 0.5
