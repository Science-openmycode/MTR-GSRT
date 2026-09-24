from __future__ import annotations

import runpy
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ManifestLineEndingsTest(unittest.TestCase):
    def test_text_hash_is_independent_of_checkout_line_endings(self) -> None:
        canonical_bytes = runpy.run_path(str(ROOT / "commands/reproduce.py"))["canonical_bytes"]
        with tempfile.TemporaryDirectory() as directory:
            text = Path(directory) / "sample.md"
            text.write_bytes(b"first\nsecond\n")
            lf = canonical_bytes(text)
            text.write_bytes(b"first\r\nsecond\r\n")
            self.assertEqual(canonical_bytes(text), lf)

            binary = Path(directory) / "sample.pkl"
            binary.write_bytes(b"first\r\nsecond\r\n")
            self.assertNotEqual(canonical_bytes(binary), lf)


if __name__ == "__main__":
    unittest.main()
