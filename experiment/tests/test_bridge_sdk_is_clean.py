import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_ROOT = REPO_ROOT.parent
BRIDGE_SDK_ROOT = CLEAN_ROOT / "Bridge-Python-SDK-Lab"


class BridgeSdkCleanTest(unittest.TestCase):
    def test_clean_bridge_sdk_checkout_has_no_local_changes(self):
        status = subprocess.check_output(
            ["git", "-C", str(BRIDGE_SDK_ROOT), "status", "--short"],
            text=True,
        )

        self.assertEqual(status, "")

    def test_experiment_modules_are_not_written_into_bridge_sdk(self):
        forbidden_paths = [
            BRIDGE_SDK_ROOT / "src/bridge_python_sdk/Examples/RunCoherentRasterExperiment.py",
            BRIDGE_SDK_ROOT / "src/bridge_python_sdk/Examples/OpenCoherentRasterExperiment.py",
            BRIDGE_SDK_ROOT / "src/bridge_python_sdk/Examples/BuildCoherentRasterWebIndex.py",
            BRIDGE_SDK_ROOT / "src/bridge_python_sdk/Rendering/CoherentRasterExperiment.py",
            BRIDGE_SDK_ROOT / "src/bridge_python_sdk/Rendering/CoherentRaster.py",
            BRIDGE_SDK_ROOT / "src/bridge_python_sdk/Rendering/CoherentGsplatBridge.py",
        ]

        for path in forbidden_paths:
            self.assertFalse(path.exists(), str(path))
