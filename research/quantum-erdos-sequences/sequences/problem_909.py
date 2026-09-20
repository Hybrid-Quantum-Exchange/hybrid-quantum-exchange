"""
Erdos problem #909 -- quantum-testable-sequence lane (best-effort, limited).

Source record (erdosproblems.com data, /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: \"909\""):
    prize: no
    informal_status: proved (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["analysis", "topology"]

LIMITATION (read this before trusting the PASS below): problem #909 carries NO
OEIS sequence id -- its `oeis` field is literally ["N/A"] -- and its tags
("analysis", "topology") describe a proved analytic/topological statement, not
a combinatorial sequence with a small finite computable membership property.
There is therefore no genuine small-instance decision problem to derive from
problem #909 itself that a quantum circuit could search or verify. Fabricating
a "property of the sequence" here would misrepresent the problem, which the
task instructions explicitly forbid.

Rather than fake a connection, this script does the next-most-honest thing:
it builds a REAL, fully verified Grover search circuit on AerSimulator for a
genuine, independently-checkable finite arithmetic property -- "which 3-bit
integers N in [0, 7] are divisible by 3" (i.e. membership in OEIS A008587,
multiples of 3) -- and documents plainly that this instance is a stand-in
witness circuit, NOT a property of Erdos problem #909's (nonexistent) OEIS
sequence. The classical answer set {0, 3, 6} is computed from first principles
in this script (by direct division, not copied from OEIS), and the quantum
result is compared against it.

Honest bottom line for problem #909: no genuine quantum circuit could be
constructed FOR THIS PROBLEM'S OWN SEQUENCE, because it has no OEIS id and no
finite computable sequence-membership property. ran_ok / verified_against_classical
below describe only the stand-in circuit's own internal correctness, not any
claim about problem #909's mathematical content.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate
import numpy as np


def classical_multiples_of_3(n_bits: int) -> list[int]:
    """Ground truth: integers in [0, 2**n_bits - 1] divisible by 3, computed
    directly by division (first principles), not looked up."""
    N = 2 ** n_bits
    return [x for x in range(N) if x % 3 == 0]


def build_oracle(n_bits: int, targets: list[int]) -> QuantumCircuit:
    """Phase oracle flipping the sign of each target computational basis state."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for t in targets:
        bits = format(t, f"0{n_bits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
            qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def run_grover(n_bits: int, targets: list[int], shots: int = 4096):
    N = 2 ** n_bits
    M = len(targets)
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, targets)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_bits))
        qc.append(diffuser.to_instruction(), range(n_bits))

    qc.measure(range(n_bits), range(n_bits))
    qc = qc.decompose(reps=3)

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_bits = 3  # search space [0, 7]
    classical_targets = classical_multiples_of_3(n_bits)
    print(f"Classical multiples of 3 in [0, {2**n_bits - 1}]: {classical_targets}")

    counts, iterations = run_grover(n_bits, classical_targets)
    print(f"Grover iterations used: {iterations}")
    print("Measurement counts:", counts)

    shots = sum(counts.values())
    threshold = 0.05 * shots  # ignore noise-floor outcomes
    measured_ints = sorted(
        int(bitstring[::-1], 2)
        for bitstring, c in counts.items()
        if c >= threshold
    )

    print(f"Quantum-measured high-probability set: {measured_ints}")
    print(f"Classical target set:                  {classical_targets}")

    passed = measured_ints == classical_targets
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
