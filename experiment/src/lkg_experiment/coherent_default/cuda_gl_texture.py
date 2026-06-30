from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable


TEXTURE_UPLOAD_AUTO = "auto"
TEXTURE_UPLOAD_CPU = "cpu"
TEXTURE_UPLOAD_CUDA_GL = "cuda-gl"
TEXTURE_UPLOAD_CHOICES = (TEXTURE_UPLOAD_AUTO, TEXTURE_UPLOAD_CPU, TEXTURE_UPLOAD_CUDA_GL)


class CudaGlTextureError(RuntimeError):
    pass


def _warn_stderr(message: str) -> None:
    print(f"TextureUpload: {message}", file=sys.stderr, flush=True)


def _cuda_tensor_device_index(tensor: Any) -> int:
    if not bool(getattr(tensor, "is_cuda", False)):
        raise CudaGlTextureError("CUDA-GL texture upload requires a CUDA tensor")
    device = getattr(tensor, "device", None)
    device_type = getattr(device, "type", None)
    if device_type is not None and str(device_type) != "cuda":
        raise CudaGlTextureError(f"CUDA-GL texture upload requires a CUDA tensor, got {device!r}")
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)
    try:
        import torch

        return int(torch.cuda.current_device())
    except Exception as exc:
        raise CudaGlTextureError(f"failed to resolve CUDA tensor device: {exc}") from exc


class CpuTextureUploader:
    def __init__(
        self,
        *,
        texture: int,
        width: int,
        height: int,
        rgba_converter: Callable[[Any], Any] | None = None,
        gl_uploader: Callable[[int, int, int, Any], None] | None = None,
    ) -> None:
        self.texture = int(texture)
        self.width = int(width)
        self.height = int(height)
        self._rgba_converter = rgba_converter
        self._gl_uploader = gl_uploader
        self.mode = TEXTURE_UPLOAD_CPU

    def upload(self, tensor: Any) -> None:
        converter = self._rgba_converter
        gl_uploader = self._gl_uploader
        if converter is None or gl_uploader is None:
            from lkg_experiment.coherent_default.render_looking_glass import rendered_to_rgba, upload_direct

            converter = converter or rendered_to_rgba
            gl_uploader = gl_uploader or upload_direct
        rgba = converter(tensor)
        gl_uploader(self.texture, self.width, self.height, rgba)

    def close(self) -> None:
        return None


class CudaGlTextureUploader:
    def __init__(
        self,
        *,
        texture: int,
        width: int,
        height: int,
        torch_extensions_dir: str | Path | None = None,
        extension_loader: Callable[[], Any] | None = None,
    ) -> None:
        self.texture = int(texture)
        self.width = int(width)
        self.height = int(height)
        self.mode = TEXTURE_UPLOAD_CUDA_GL
        self._resource: int | None = None
        self._closed = False
        loader = extension_loader or (lambda: load_cuda_gl_texture_extension(torch_extensions_dir=torch_extensions_dir))
        try:
            self._extension = loader()
            self._gl_cuda_devices = [int(device) for device in self._extension.gl_cuda_devices()]
            if not self._gl_cuda_devices:
                raise CudaGlTextureError("current GL context does not expose any CUDA devices")
            self._gl_cuda_device = int(self._gl_cuda_devices[0])
            self._resource = int(self._extension.register_gl_texture(self.texture, self._gl_cuda_device))
        except CudaGlTextureError:
            raise
        except Exception as exc:
            raise CudaGlTextureError(f"failed to initialize CUDA-GL texture uploader: {exc}") from exc

    def upload(self, tensor: Any) -> None:
        if self._closed or self._resource is None:
            raise CudaGlTextureError("CUDA-GL texture uploader is closed")
        upload_tensor = self._tensor_on_gl_device(tensor)
        try:
            self._extension.upload_cuda_tensor(self._resource, upload_tensor, self.width, self.height)
        except CudaGlTextureError:
            raise
        except Exception as exc:
            raise CudaGlTextureError(f"CUDA-GL texture upload failed: {exc}") from exc

    def _tensor_on_gl_device(self, tensor: Any) -> Any:
        device_index = _cuda_tensor_device_index(tensor)
        if device_index == self._gl_cuda_device:
            return tensor
        target = f"cuda:{self._gl_cuda_device}"
        try:
            return tensor.to(device=target, non_blocking=True)
        except TypeError:
            return tensor.to(target, non_blocking=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        resource = self._resource
        self._resource = None
        if resource is None:
            return
        self._extension.unregister_gl_texture(int(resource))


class AutoTextureUploader:
    def __init__(
        self,
        *,
        cuda_uploader: CudaGlTextureUploader,
        cpu_uploader: CpuTextureUploader,
        warn: Callable[[str], None] | None = None,
    ) -> None:
        self._cuda_uploader: CudaGlTextureUploader | None = cuda_uploader
        self._cpu_uploader = cpu_uploader
        self._warn = warn or _warn_stderr

    @property
    def mode(self) -> str:
        return self._cuda_uploader.mode if self._cuda_uploader is not None else self._cpu_uploader.mode

    def upload(self, tensor: Any) -> None:
        if self._cuda_uploader is not None:
            try:
                self._cuda_uploader.upload(tensor)
                return
            except CudaGlTextureError as exc:
                self._warn(f"CUDA-GL texture upload failed; falling back to CPU upload: {exc}")
                self._close_cuda_uploader()
        self._cpu_uploader.upload(tensor)

    def close(self) -> None:
        self._close_cuda_uploader()
        self._cpu_uploader.close()

    def _close_cuda_uploader(self) -> None:
        if self._cuda_uploader is None:
            return
        try:
            self._cuda_uploader.close()
        finally:
            self._cuda_uploader = None


def create_panel_texture_uploader(
    *,
    mode: str,
    texture: int,
    width: int,
    height: int,
    torch_extensions_dir: str | Path | None = None,
    extension_loader: Callable[[], Any] | None = None,
    warn: Callable[[str], None] | None = None,
) -> CpuTextureUploader | CudaGlTextureUploader | AutoTextureUploader:
    normalized = str(mode or TEXTURE_UPLOAD_AUTO)
    if normalized not in TEXTURE_UPLOAD_CHOICES:
        raise ValueError(f"unsupported texture upload mode: {mode!r}")

    cpu_uploader = CpuTextureUploader(texture=texture, width=width, height=height)
    if normalized == TEXTURE_UPLOAD_CPU:
        return cpu_uploader

    try:
        cuda_uploader = CudaGlTextureUploader(
            texture=texture,
            width=width,
            height=height,
            torch_extensions_dir=torch_extensions_dir,
            extension_loader=extension_loader,
        )
    except CudaGlTextureError as exc:
        if normalized == TEXTURE_UPLOAD_CUDA_GL:
            raise
        (warn or _warn_stderr)(f"CUDA-GL texture uploader unavailable; falling back to CPU upload: {exc}")
        return cpu_uploader

    if normalized == TEXTURE_UPLOAD_CUDA_GL:
        return cuda_uploader
    return AutoTextureUploader(cuda_uploader=cuda_uploader, cpu_uploader=cpu_uploader, warn=warn)


def load_cuda_gl_texture_extension(*, torch_extensions_dir: str | Path | None = None) -> Any:
    _configure_cuda_extension_environment(torch_extensions_dir=torch_extensions_dir)
    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).resolve().with_name("cuda_gl_texture_ext")
    sources = [
        str(source_dir / "cuda_gl_texture_ext.cpp"),
        str(source_dir / "cuda_gl_texture_ext_kernel.cu"),
    ]
    return load(
        name="lkg_cuda_gl_texture_ext_v2",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_include_paths=_cuda_include_paths(),
        extra_ldflags=_cuda_library_flags(),
        with_cuda=True,
        verbose=bool(os.environ.get("LKG_CUDA_GL_VERBOSE")),
    )


def _configure_cuda_extension_environment(*, torch_extensions_dir: str | Path | None) -> None:
    if torch_extensions_dir is not None:
        root = Path(torch_extensions_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_EXTENSIONS_DIR"] = str(root)
    if os.environ.get("CUDA_HOME"):
        return
    prefix = Path(sys.prefix)
    if (prefix / "bin" / "nvcc").exists() and (prefix / "include" / "cuda_runtime_api.h").exists():
        os.environ["CUDA_HOME"] = str(prefix)
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        try:
            import torch

            device_count = int(torch.cuda.device_count())
        except Exception:
            device_count = 0
        if device_count == 0:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6+PTX"


def _cuda_include_paths() -> list[str]:
    paths: list[str] = []
    prefix_include = Path(sys.prefix) / "include"
    if prefix_include.exists():
        paths.append(str(prefix_include))
    return paths


def _cuda_library_flags() -> list[str]:
    prefix_lib = Path(sys.prefix) / "lib"
    if prefix_lib.exists():
        return [f"-L{prefix_lib}", "-lcudart"]
    return ["-lcudart"]
