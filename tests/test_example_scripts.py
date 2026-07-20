"""Regression contracts for the copyable public example runners."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import support


EXAMPLE_NAMES = ("two-hermes", "two-claude", "three-mixed")
EXAMPLES = support.REPO_ROOT / "examples"


def _selected_python_bin(executable: str) -> str:
    return str(Path(executable).parent)


class InterpreterSelectionTests(unittest.TestCase):
    def test_virtualenv_symlink_keeps_virtualenv_bin_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_python = root / "base" / "python3"
            selected_python = root / "venv" / "bin" / "python3"
            base_python.parent.mkdir(parents=True)
            selected_python.parent.mkdir(parents=True)
            base_python.touch()
            selected_python.symlink_to(base_python)

            self.assertEqual(
                _selected_python_bin(str(selected_python)),
                str(selected_python.parent),
            )


class ExampleScriptSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.results: dict[str, subprocess.CompletedProcess[str]] = {}
        cls.roots: dict[str, Path] = {}
        cls.parents: dict[str, Path] = {}
        clean_env = dict(os.environ)
        clean_env.pop("PYTHONPATH", None)
        selected_python_bin = _selected_python_bin(sys.executable)
        clean_env["PATH"] = os.pathsep.join(
            [
                selected_python_bin,
                "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            ]
        )
        for name in EXAMPLE_NAMES:
            parent = Path(cls._tmp.name) / name
            root = parent / "chosen-root"
            parent.mkdir(parents=True)
            cls.parents[name] = parent
            cls.roots[name] = root
            cls.results[name] = subprocess.run(
                ["bash", str(EXAMPLES / name / "run.sh"), str(root)],
                cwd=support.REPO_ROOT,
                env=clean_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=120,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_custom_path_becomes_exact_isolated_repository_root(self) -> None:
        for name in EXAMPLE_NAMES:
            with self.subTest(example=name):
                result = self.results[name]
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertTrue((self.roots[name] / ".git").is_dir())
                self.assertFalse((self.parents[name] / ".git").exists())
                self.assertTrue((self.roots[name] / "dialogue" / "state.json").is_file())

    def test_wrong_actor_check_proves_protocol_rejection_without_spawn(self) -> None:
        for name in EXAMPLE_NAMES:
            with self.subTest(example=name):
                result = self.results[name]
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn(
                    "wrong actor rejected before runtime spawn; state unchanged",
                    result.stdout,
                )
                self.assertNotIn("invalid choice: 'claim'", result.stdout)
                marker = (
                    self.roots[name]
                    / "dialogue"
                    / "work"
                    / "wrong-actor-spawn-marker"
                )
                self.assertTrue(marker.is_file())
                self.assertEqual(marker.read_text(encoding="utf-8"), "")


class PublicExampleReferenceTests(unittest.TestCase):
    def test_claude_examples_reference_a_shipped_fake_profile(self) -> None:
        profile = EXAMPLES / "fakes" / "fable-profile.toml"
        self.assertTrue(profile.is_file())
        for name in ("two-claude", "three-mixed"):
            with self.subTest(example=name):
                text = (EXAMPLES / name / "run.sh").read_text(encoding="utf-8")
                self.assertIn("examples/fakes/fable-profile.toml", text)
                self.assertNotIn("agent-context/profiles/fable-5.toml", text)

    def test_documented_layout_lists_only_shipped_paths(self) -> None:
        reference = (support.REPO_ROOT / "docs" / "technical-reference.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("docs/plans/", reference)


if __name__ == "__main__":
    unittest.main()
