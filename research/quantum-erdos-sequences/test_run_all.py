"""Runner contract tests; only temporary stdlib fixtures are executed, never the corpus."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import run_all as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.sequences = self.root / "sequences"
        self.sequences.mkdir()
        self.runs = self.root / "runs"
        for name, value in {
            "SEQUENCES_DIR": self.sequences,
            "RECEIPTS_DIR": self.runs,
            "RESULTS_JSON": self.root / "RESULTS.json",
            "RESULTS_MD": self.root / "RESULTS.md",
        }.items():
            self.enterContext(mock.patch.object(runner, name, value))

    def lane(self, source="print('PASS')\n", number=1):
        path = self.sequences / f"problem_{number}.py"
        path.write_text(source, encoding="utf-8")
        return path

    def run_output(self, stdout=b"PASS\n", stderr=b"", code=0):
        path = self.lane()
        with mock.patch.object(runner.subprocess, "run", return_value=
                               subprocess.CompletedProcess([], code, stdout, stderr)):
            return runner.run_lane(path)

    def main(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return runner.main(list(args))

    def test_supported_whole_line_verdicts(self):
        for output in (
            "PASS", " \n PASS \n", "RESULT: PASS", "PASS: checked the demo",
            "PASS (toy instance only)", "PASS/FAIL: PASS (toy mechanism matches;",
            "Quantum result [1, 2] matches classical answer [1, 2]: PASS",
        ):
            with self.subTest(output=output):
                self.assertEqual(runner.parse_verdict(output), ("PASS", "valid"))
        for output in (
            "FAIL", "RESULT: FAIL", "FAIL: toy mismatch", "PASS/FAIL: FAIL (toy mismatch)",
            "Quantum result [1] does NOT match classical answer [2]: FAIL",
            "Overall verdict for Erdos problem #906: FAIL",
        ):
            with self.subTest(output=output):
                self.assertEqual(runner.parse_verdict(output), ("FAIL", "valid"))

    def test_narrative_cannot_supply_or_replace_verdict(self):
        notes = ("NOTE: PASS/FAIL describes the toy circuit only", "Threshold for PASS: 0.8",
                 "NOT PASS", "BYPASS", "some check: PASS", "PASS/FAIL", "PASS FAIL",
                 "PASSING", "FAILURE", "PASS was expected", "\x1b[32mPASS\x1b[0m")
        for note in notes:
            with self.subTest(note=note):
                self.assertEqual(runner.parse_verdict(note), (None, "missing_verdict"))
                self.assertEqual(runner.parse_verdict("FAIL\n" + note), ("FAIL", "valid"))
        result = self.run_output(b"FAIL\nNOTE: PASS/FAIL only tests the toy circuit\n")
        self.assertTrue(result["ran_ok"])
        self.assertFalse(result["verified_against_classical"])

    def test_contradictions_fail_closed_in_either_order(self):
        for output in (b"FAIL\nPASS\n", b"PASS\nRESULT: FAIL\n", b"PASS: actually FAIL\n"):
            with self.subTest(output=output):
                result = self.run_output(output)
                self.assertFalse(result["ran_ok"])
                self.assertEqual(result["protocol_status"], "contradictory_verdicts")
                self.assertEqual(result["hypothesis_status"], "not_established")

    def test_exit_codes_are_consistent_with_reported_outcome(self):
        for code, verdict, clean, passed in (
            (0, "PASS", True, True), (0, "FAIL", True, False),
            (1, "FAIL", True, False), (1, "PASS", False, False),
            (2, "PASS", False, False), (2, "FAIL", False, False),
            (-9, "PASS", False, False), (1, "", False, False),
        ):
            with self.subTest(code=code, verdict=verdict):
                result = self.run_output(verdict.encode(), code=code)
                self.assertEqual(result["ran_ok"], clean)
                self.assertEqual(result["self_reported_demo_pass"], passed)
                self.assertEqual(result["verified_against_classical"], passed)
                if not clean:
                    self.assertNotEqual(result["verdict"], "PASS")
                    self.assertEqual(result["reported_verdict"], verdict or None)

    def test_traceback_in_either_stream_prevents_pass(self):
        trace = b"Traceback (most recent call last):\n"
        for stdout, stderr in ((trace + b"PASS\n", b""), (b"PASS\n", trace)):
            with self.subTest(stderr=stderr):
                result = self.run_output(stdout, stderr)
                self.assertFalse(result["ran_ok"])
                self.assertEqual(result["execution_status"], "failed")

    def test_no_verdict_is_protocol_failure_even_on_exit_zero(self):
        result = self.run_output(b"measurement complete\n")
        self.assertEqual(result["execution_status"], "clean")
        self.assertEqual(result["protocol_status"], "missing_verdict")
        self.assertFalse(result["ran_ok"])

    def test_real_crash_preserves_traceback_and_no_pass(self):
        result = runner.run_lane(self.lane("print('PASS', flush=True)\nraise RuntimeError('fixture')\n"))
        self.assertEqual(result["returncode"], 1)
        self.assertFalse(result["ran_ok"])
        receipt = Path(result["receipt_directory"])
        self.assertIn(b"RuntimeError: fixture", (receipt / "stderr.log").read_bytes())
        self.assertEqual(json.loads((receipt / "receipt.json").read_text())["execution_status"], "failed")

    def test_real_timeout_has_partial_raw_logs_and_error_receipt(self):
        path = self.lane("import time\nprint('partial output', flush=True)\ntime.sleep(30)\n")
        with mock.patch.object(runner, "TIMEOUT_SECONDS", 1):
            result = runner.run_lane(path)
        self.assertEqual(result["execution_status"], "timeout")
        self.assertEqual(result["verdict"], "TIMEOUT")
        self.assertIsNone(result["returncode"])
        self.assertFalse(result["ran_ok"])
        receipt = Path(result["receipt_directory"])
        self.assertIn(b"partial output", (receipt / "stdout.log").read_bytes())
        self.assertTrue(json.loads((receipt / "receipt.json").read_text())["error"])

    def test_launch_oserror_has_durable_error_receipt(self):
        path = self.lane()
        with mock.patch.object(runner.subprocess, "run", side_effect=OSError("fixture launch failure")):
            result = runner.run_lane(path)
        self.assertEqual(result["execution_status"], "launch_error")
        receipt = Path(result["receipt_directory"])
        self.assertEqual((receipt / "stdout.log").read_bytes(), b"")
        self.assertIn("fixture launch failure", (receipt / "receipt.json").read_text())

    def test_timeout_after_printing_pass_is_still_timeout(self):
        path = self.lane()
        with mock.patch.object(runner.subprocess, "run", side_effect=
                               subprocess.TimeoutExpired(["fixture"], 120, output=b"PASS\n")):
            result = runner.run_lane(path)
        self.assertEqual(result["verdict"], "TIMEOUT")
        self.assertEqual(result["reported_verdict"], "PASS")
        self.assertFalse(result["self_reported_demo_pass"])

    def test_source_change_during_execution_invalidates_pass(self):
        path = self.lane()

        def changed_source(*args, **kwargs):
            path.write_text("print('FAIL')\n", encoding="utf-8")
            return subprocess.CompletedProcess([], 0, b"PASS\n", b"")

        with mock.patch.object(runner.subprocess, "run", side_effect=changed_source):
            result = runner.run_lane(path)
        self.assertEqual(result["execution_status"], "source_changed")
        self.assertFalse(result["ran_ok"])

    def test_missing_source_has_error_receipt_without_execution(self):
        with mock.patch.object(runner.subprocess, "run") as execute:
            result = runner.run_lane(self.sequences / "problem_1.py")
        execute.assert_not_called()
        self.assertEqual(result["execution_status"], "source_error")
        self.assertTrue((Path(result["receipt_directory"]) / "receipt.json").is_file())

    def test_raw_logs_are_byte_exact_and_hashed(self):
        stdout, stderr = b"PASS\r\nopaque: \xff\x00\r\n", b"warning: \xfe\r\n"
        result = self.run_output(stdout, stderr)
        for filename, expected in (("stdout.log", stdout), ("stderr.log", stderr)):
            self.assertEqual((Path(result["receipt_directory"]) / filename).read_bytes(), expected)
            self.assertEqual(result["logs"][filename]["sha256"], hashlib.sha256(expected).hexdigest())

    def test_empty_incomplete_and_missing_selected_corpus_fail_before_execution(self):
        for args in ((), ("--lane", "1")):
            with self.subTest(args=args), mock.patch.object(runner.subprocess, "run") as execute:
                self.assertEqual(self.main(*args), 2)
                execute.assert_not_called()
        self.lane()
        with mock.patch.object(runner.subprocess, "run") as execute:
            self.assertEqual(self.main(), 2)
            execute.assert_not_called()
        for manifest in self.runs.glob("*/manifest.json"):
            data = json.loads(manifest.read_text())
            self.assertEqual(data["status"], "error")
            self.assertFalse(data["success"])
            self.assertTrue(data["completed_utc"])

    def test_selected_lane_is_bounded_and_failed_demo_makes_main_nonzero(self):
        self.lane("print('FAIL')\nraise SystemExit(1)\n")
        self.lane("raise AssertionError('unselected lane ran')\n", number=2)
        self.assertEqual(self.main("--lane", "1"), 1)
        results = json.loads((self.root / "RESULTS.json").read_text())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["hypothesis_status"], "self_reported_fail")
        self.assertTrue(results[0]["ran_ok"])
        self.assertIn("Self-reported demo", (self.root / "RESULTS.md").read_text())

    def test_history_is_not_overwritten_and_hashes_bind_each_source_version(self):
        path = self.lane("print('PASS')\n")
        first_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(self.main("--lane", "1"), 0)
        first_dir = next(self.runs.iterdir())
        before = {p.relative_to(first_dir): p.read_bytes() for p in first_dir.rglob("*") if p.is_file()}
        path.write_text("print('FAIL')\n", encoding="utf-8")
        self.assertEqual(self.main("--lane", "1"), 1)
        self.assertEqual(len(list(self.runs.iterdir())), 2)
        after = {p.relative_to(first_dir): p.read_bytes() for p in first_dir.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        first = json.loads((first_dir / "RESULTS.json").read_text())[0]
        latest = json.loads((self.root / "RESULTS.json").read_text())[0]
        self.assertEqual(first["source"]["sha256"], first_hash)
        self.assertNotEqual(first["source"]["sha256"], latest["source"]["sha256"])
        manifest = json.loads((first_dir / "manifest.json").read_text())
        for key, source in (("runner", runner.RUNNER), ("requirements", runner.REQUIREMENTS)):
            self.assertEqual(manifest[key]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertIn("version", manifest["python"])
        self.assertEqual(set(manifest["dependency_versions"]), {"qiskit", "qiskit-aer", "numpy"})
        self.assertEqual(manifest["reviewer_status"], "not_independently_reviewed")
        self.assertIn("unseeded", manifest["seed_status"])

    def test_duplicate_attempt_in_same_run_cannot_overwrite_receipt(self):
        run_dir, _ = runner.create_run_directory()
        path = self.lane()
        first = runner.run_lane(path, run_dir)
        receipt = Path(first["receipt_directory"]) / "receipt.json"
        before = receipt.read_bytes()
        with self.assertRaises(FileExistsError):
            runner.run_lane(path, run_dir)
        self.assertEqual(receipt.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
