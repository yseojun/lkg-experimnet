from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from lkg_experiment.coherent_default import cuda_gl_texture


class FakeCudaGlExtension:
    def __init__(self, *, devices: list[int] | None = None) -> None:
        self.devices = [0] if devices is None else list(devices)
        self.registered: list[int] = []
        self.unregistered: list[int] = []
        self.uploads: list[tuple[int, object, int, int]] = []

    def gl_cuda_devices(self) -> list[int]:
        return list(self.devices)

    def register_gl_texture(self, texture: int, cuda_device: int | None = None) -> int:
        self.registered.append(int(texture))
        self.register_device = cuda_device
        return int(texture) + 1000

    def unregister_gl_texture(self, resource: int) -> None:
        self.unregistered.append(int(resource))

    def upload_cuda_tensor(self, resource: int, tensor: object, width: int, height: int) -> None:
        self.uploads.append((int(resource), tensor, int(width), int(height)))


class CudaGlTextureUploaderTest(unittest.TestCase):
    def test_cpu_uploader_uses_existing_rgba_upload_path(self) -> None:
        converter = mock.Mock(return_value="rgba")
        gl_upload = mock.Mock()
        uploader = cuda_gl_texture.CpuTextureUploader(
            texture=7,
            width=1440,
            height=2560,
            rgba_converter=converter,
            gl_uploader=gl_upload,
        )

        uploader.upload("tensor")

        converter.assert_called_once_with("tensor")
        gl_upload.assert_called_once_with(7, 1440, 2560, "rgba")
        self.assertEqual(uploader.mode, "cpu")

    def test_cuda_gl_uploader_registers_uploads_and_unregisters_texture(self) -> None:
        extension = FakeCudaGlExtension()
        tensor = SimpleNamespace(is_cuda=True, device=SimpleNamespace(type="cuda", index=0))
        uploader = cuda_gl_texture.CudaGlTextureUploader(
            texture=11,
            width=1440,
            height=2560,
            extension_loader=lambda: extension,
        )

        uploader.upload(tensor)
        uploader.close()

        self.assertEqual(extension.registered, [11])
        self.assertEqual(extension.register_device, 0)
        self.assertEqual(extension.uploads, [(1011, tensor, 1440, 2560)])
        self.assertEqual(extension.unregistered, [1011])
        self.assertEqual(uploader.mode, "cuda-gl")

    def test_cuda_gl_uploader_rejects_cpu_tensor_before_extension_upload(self) -> None:
        extension = FakeCudaGlExtension()
        tensor = SimpleNamespace(is_cuda=False, device=SimpleNamespace(type="cpu", index=None))
        uploader = cuda_gl_texture.CudaGlTextureUploader(
            texture=11,
            width=1440,
            height=2560,
            extension_loader=lambda: extension,
        )
        try:
            with self.assertRaisesRegex(cuda_gl_texture.CudaGlTextureError, "CUDA tensor"):
                uploader.upload(tensor)
        finally:
            uploader.close()

        self.assertEqual(extension.uploads, [])

    def test_cuda_gl_uploader_copies_mismatched_cuda_tensor_to_gl_device(self) -> None:
        extension = FakeCudaGlExtension(devices=[2])
        copied = SimpleNamespace(is_cuda=True, device=SimpleNamespace(type="cuda", index=2))

        class FakeTensor:
            is_cuda = True
            device = SimpleNamespace(type="cuda", index=0)

            def to(self, *, device: object, non_blocking: bool) -> object:
                self.copy_device = device
                self.copy_non_blocking = non_blocking
                return copied

        tensor = FakeTensor()
        uploader = cuda_gl_texture.CudaGlTextureUploader(
            texture=11,
            width=1440,
            height=2560,
            extension_loader=lambda: extension,
        )
        try:
            uploader.upload(tensor)
        finally:
            uploader.close()

        self.assertEqual(extension.register_device, 2)
        self.assertEqual(str(tensor.copy_device), "cuda:2")
        self.assertTrue(tensor.copy_non_blocking)
        self.assertEqual(extension.uploads, [(1011, copied, 1440, 2560)])

    def test_auto_mode_falls_back_to_cpu_when_cuda_gl_registration_fails(self) -> None:
        warnings: list[str] = []

        def fail_loader() -> object:
            raise cuda_gl_texture.CudaGlTextureError("register failed")

        uploader = cuda_gl_texture.create_panel_texture_uploader(
            mode="auto",
            texture=5,
            width=1440,
            height=2560,
            extension_loader=fail_loader,
            warn=warnings.append,
        )

        self.assertIsInstance(uploader, cuda_gl_texture.CpuTextureUploader)
        self.assertEqual(uploader.mode, "cpu")
        self.assertEqual(len(warnings), 1)
        self.assertIn("register failed", warnings[0])

    def test_cuda_gl_mode_propagates_registration_error(self) -> None:
        def fail_loader() -> object:
            raise cuda_gl_texture.CudaGlTextureError("register failed")

        with self.assertRaisesRegex(cuda_gl_texture.CudaGlTextureError, "register failed"):
            cuda_gl_texture.create_panel_texture_uploader(
                mode="cuda-gl",
                texture=5,
                width=1440,
                height=2560,
                extension_loader=fail_loader,
            )


if __name__ == "__main__":
    unittest.main()
