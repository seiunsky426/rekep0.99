import os
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from rekpiper_perception.sam_automatic import SAMAutomaticSegmenter


class SAMSourceTest(unittest.TestCase):
    def test_configured_vendor_source_is_used_and_other_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            vendor = Path(directory)
            root = vendor / "segment-anything"
            (root / ".git").mkdir(parents=True)
            (root / ".git/HEAD").write_text(
                "6fdee8f2727f4506cfbbe553e23b895e27956588\n")
            checkpoint = vendor / "sam.pth"
            checkpoint.touch()
            package = ModuleType("segment_anything")
            package.__file__ = str(root / "segment_anything/__init__.py")
            factory = Mock()
            package.sam_model_registry = {"vit_h": factory}
            package.SamAutomaticMaskGenerator = Mock()
            with patch.dict(os.environ, {"REKPIPER_VENDOR_ROOT": str(vendor)}), \
                    patch.dict("sys.modules", {"segment_anything": package}):
                SAMAutomaticSegmenter(checkpoint, "cpu")
                factory.assert_called_once_with(checkpoint=str(checkpoint))
                package.__file__ = str(vendor / "unrelated/__init__.py")
                with self.assertRaisesRegex(RuntimeError, "outside the pinned source"):
                    SAMAutomaticSegmenter(checkpoint, "cpu")
                package.__file__ = str(root / "segment_anything/__init__.py")
                (root / ".git/HEAD").write_text("0" * 40)
                with self.assertRaisesRegex(RuntimeError, "source commit mismatch"):
                    SAMAutomaticSegmenter(checkpoint, "cpu")


if __name__ == "__main__":
    unittest.main()
