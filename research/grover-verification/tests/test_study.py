"""Pure arithmetic and mocked execution tests. Never runs Qiskit/Aer experiments."""
import contextlib
import importlib.util
import io
import json
import math
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import study


def integer_fixture(distribution):
    raw = {key: value * 4096 for key, value in distribution.items()}
    counts = {key: math.floor(value) for key, value in raw.items()}
    remaining = 4096 - sum(counts.values())
    for key in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:remaining]:
        counts[key] += 1
    return counts


def fixture():
    reference = study.analytic_reference()
    result = {}
    for arm, target in (("positive", 13), ("wrong_oracle", 12)):
        result[arm] = {"counts": integer_fixture(reference[arm]), "shots": 4096,
                       "oracle_target": target, "qubits": 4, "classical_bits": 4,
                       "seed_simulator": 1729, "seed_transpiler": 1729,
                       "qasm_format": "OpenQASM 2.0", "qasm_strict_parse": True}
        for name in ("exact_source", "exact_transpiled", "exact_qasm_roundtrip"):
            result[arm][name] = dict(reference[arm])
    result["uniform_classical"] = {"counts": {key: 256 for key in study.LABELS},
                                    "shots": 4096, "seed_classical": 1729}
    return result


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)

    def test_analytic_reference_matches_exact_rational_and_normalizes(self):
        reference = study.analytic_reference()
        self.assertAlmostEqual(reference["positive"]["1101"], 63001 / 65536, places=14)
        self.assertAlmostEqual(reference["positive"]["0000"], 169 / 65536, places=14)
        for arm in ("positive", "wrong_oracle", "uniform_classical"):
            self.assertEqual(len(reference[arm]), 16)
            self.assertAlmostEqual(sum(reference[arm].values()), 1.0)
        self.assertEqual(reference["wrong_oracle"]["1100"], reference["positive"]["1101"])

    def test_full_synthetic_fixture_meets_predeclared_rules(self):
        protocol, hypothesis, _ = study.evaluate(fixture(), study.analytic_reference())
        self.assertEqual((protocol, hypothesis), ("passes", "passes"))

    def test_uniform_positive_counts_cannot_self_agree(self):
        data = fixture()
        data["positive"]["counts"] = {key: 256 for key in study.LABELS}
        protocol, hypothesis, checks = study.evaluate(data, study.analytic_reference())
        self.assertEqual(protocol, "passes")
        self.assertEqual(hypothesis, "fails")
        self.assertFalse(checks["hypothesis_checks"]["positive_sampling"])

    def test_wrong_oracle_cannot_be_mislabeled_as_positive(self):
        data = fixture()
        data["positive"]["counts"] = dict(data["wrong_oracle"]["counts"])
        self.assertEqual(study.evaluate(data, study.analytic_reference())[1], "fails")

    def test_full_exact_distribution_is_checked_not_only_target(self):
        data = fixture()
        data["positive"]["exact_source"]["0000"] += 0.001
        protocol, hypothesis, checks = study.evaluate(data, study.analytic_reference())
        self.assertEqual(protocol, "passes")
        self.assertEqual(hypothesis, "fails")
        self.assertFalse(checks["hypothesis_checks"]["positive_exact_source"])

    def test_missing_empty_wrong_seed_and_bad_counts_fail_protocol(self):
        variants = []
        data = fixture(); del data["wrong_oracle"]; variants.append(data)
        data = fixture(); data["positive"]["counts"] = {}; variants.append(data)
        data = fixture(); data["positive"]["counts"]["1101"] -= 1; variants.append(data)
        data = fixture(); data["positive"]["seed_simulator"] = 1; variants.append(data)
        data = fixture(); data["positive"]["qasm_strict_parse"] = False; variants.append(data)
        for data in variants:
            with self.subTest(data=data):
                self.assertEqual(study.evaluate(data, study.analytic_reference())[:2], ("fails", "not_evaluated"))

    def manifest(self):
        path = self.root / "manifest.json"
        study.write_json(path, {"analytic_reference": study.analytic_reference()})
        return path, study.sha(path)

    def fake_experiment(self, data):
        def run(directory):
            for arm in data:
                study.write_json(directory / f"{arm}.json", data[arm])
                if arm != "uniform_classical":
                    study.write_text(directory / f"{arm}.circuit.txt", "unit-test fixture only\n")
                    study.write_text(directory / f"{arm}.qasm", "unit-test fixture only\n")
            return data
        return run

    def test_manifest_tampering_stops_before_experiment(self):
        path, digest = self.manifest()
        path.write_text("{}")
        experiment = mock.Mock()
        with contextlib.redirect_stdout(io.StringIO()):
            code, directory = study.execute(path, digest, self.root / "runs", experiment)
        experiment.assert_not_called()
        receipt = json.loads((directory / "receipt.json").read_text())
        self.assertEqual(code, 1)
        self.assertEqual(receipt["execution_status"], "not_run")
        self.assertEqual(receipt["protocol_status"], "fails")

    def test_source_drift_stops_before_experiment(self):
        path, digest = self.manifest()
        experiment = mock.Mock()
        with mock.patch.object(study, "check_manifest", side_effect=ValueError("source drift")), \
             contextlib.redirect_stdout(io.StringIO()):
            code, directory = study.execute(path, digest, self.root / "runs", experiment)
        experiment.assert_not_called()
        self.assertEqual(code, 1)
        self.assertTrue((directory / "receipt.json").exists())

    def test_real_manifest_validator_rejects_source_and_dependency_changes(self):
        original = self.root / "original.py"
        original.write_text("# unchanged historical fixture\n")
        manifest = {"schema_version": 1, "specification": study.SPEC,
                    "analytic_reference": study.analytic_reference(),
                    "source_sha256": {"study.py": "frozen"},
                    "original_source": {"path": str(original), "sha256": study.sha(original)},
                    "dependencies": {"fixed": "dependency"}}
        with mock.patch.object(study, "source_snapshot", return_value={"study.py": "frozen"}), \
             mock.patch.object(study, "dependency_snapshot", return_value={"fixed": "dependency"}):
            study.check_manifest(manifest)
            with mock.patch.object(study, "source_snapshot", return_value={"study.py": "changed"}):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    study.check_manifest(manifest)
            with mock.patch.object(study, "dependency_snapshot", return_value={"fixed": "changed"}):
                with self.assertRaisesRegex(ValueError, "dependencies changed"):
                    study.check_manifest(manifest)

    def test_failed_hypothesis_is_retained_and_next_run_preserves_bytes(self):
        path, digest = self.manifest()
        data = fixture()
        data["positive"]["counts"] = {key: 256 for key in study.LABELS}
        with mock.patch.object(study, "check_manifest"), contextlib.redirect_stdout(io.StringIO()):
            code, first = study.execute(path, digest, self.root / "runs", self.fake_experiment(data))
            before = {p.name: p.read_bytes() for p in first.iterdir()}
            second_code, second = study.execute(path, digest, self.root / "runs", self.fake_experiment(fixture()))
        self.assertEqual(code, 1)
        self.assertEqual(second_code, 0)
        self.assertNotEqual(first, second)
        self.assertEqual(before, {p.name: p.read_bytes() for p in first.iterdir()})
        receipt = json.loads((first / "receipt.json").read_text())
        self.assertEqual(receipt["execution_status"], "complete")
        self.assertEqual(receipt["hypothesis_status"], "fails")
        self.assertEqual(receipt["independent_review_status"], "pending")
        self.assertEqual(receipt["T9_acceptance"], "not_assessed")

    def test_missing_durable_artifacts_cannot_pass(self):
        path, digest = self.manifest()
        with mock.patch.object(study, "check_manifest"), contextlib.redirect_stdout(io.StringIO()):
            code, directory = study.execute(path, digest, self.root / "runs", lambda _: fixture())
        self.assertEqual(code, 1)
        receipt = json.loads((directory / "receipt.json").read_text())
        self.assertEqual(receipt["execution_status"], "failed")
        self.assertEqual(receipt["protocol_status"], "fails")

    def test_import_never_imports_qiskit_or_starts_experiment(self):
        spec = importlib.util.spec_from_file_location("import_probe", HERE / "study.py")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"qiskit": None, "qiskit_aer": None}), \
             mock.patch("pathlib.Path.mkdir", side_effect=AssertionError("no import writes")):
            spec.loader.exec_module(module)

    def test_exclusive_writes_preserve_prior_evidence(self):
        path = self.root / "receipt.json"
        study.write_json(path, {"first": True})
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            study.write_json(path, {"first": False})
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
