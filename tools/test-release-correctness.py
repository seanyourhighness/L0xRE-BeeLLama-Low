#!/usr/bin/env python3
"""Unit checks for the release-correctness answer verifier."""
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

verify = runpy.run_path(str(Path(__file__).with_name("release-correctness.py")))
needle_answer_matches = verify["needle_answer_matches"]


class NeedleAnswerTests(unittest.TestCase):
    def test_accepts_exact_answer_with_surrounding_whitespace(self):
        self.assertTrue(needle_answer_matches("\nLARKSPUR-7412 \n", "LARKSPUR-7412"))

    def test_rejects_code_embedded_in_explanation(self):
        self.assertFalse(needle_answer_matches("The code is LARKSPUR-7412.", "LARKSPUR-7412"))

    def test_rejects_echoed_prompt_or_context(self):
        self.assertFalse(needle_answer_matches("Prompt text ... LARKSPUR-7412 ...", "LARKSPUR-7412"))

    def test_rejects_empty_answer(self):
        self.assertFalse(needle_answer_matches("  \n", "LARKSPUR-7412"))


class DoctorManifestTests(unittest.TestCase):
    def test_repository_manifest_mismatch_is_actionable(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / "doctor.py")],
            cwd=root, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("expected the package-specific SM120 MANIFEST.json", result.stderr)
        self.assertNotIn("KeyError", result.stderr)


class ReleaseCorrectnessCliTests(unittest.TestCase):
    def test_model_paths_are_explicit_and_validated(self):
        script = Path(__file__).with_name("release-correctness.py")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = subprocess.run(
                [sys.executable, str(script), "--launcher", str(root / "launcher"),
                 "--workspace", str(root / "workspace"), "--out", str(root / "out.json"),
                 "--model-e3", str(root / "E3 model.gguf"),
                 "--model-w2", str(root / "W2 model.gguf"),
                 "--model-native", str(root / "IQ3 control.gguf"),
                 "--draft", str(root / "DFlash2 draft.gguf")],
                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("e3 model file does not exist", result.stderr)
        self.assertNotIn("/home/sean/", result.stderr)


if __name__ == "__main__":
    unittest.main()
