"""
Erdos problem #363 -- quantum-testable sequence lane.

Source metadata (from erdosproblems.com data, data/problems.yaml, entry
`number: "363"`):
    prize: no
    informal_status: disproved (Lean), last_update 2026-03-10
    formal_status: Lean, last_update 2026-03-10
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (please read before trusting the PASS below): the problems.yaml
entry for #363 carries no OEIS id -- the `oeis` field is literally the
sentinel value "N/A" -- and no textual statement of the problem is present
in that data file to derive one from. There is therefore no actual sequence
attached to this Erdos problem to build a quantum circuit against, and
nothing here should be read as testing, verifying, or disproving problem
#363 itself. This script is not a genuine "problem 363" quantum test.

Rather than fabricate an OEIS id or invent a "small term" of a sequence
that does not exist in the source data, this script honestly falls back to
a small, self-contained, *real* quantum computation in the same subject
area the yaml does give us ("number theory"): Grover's algorithm searching
a 3-bit space {0..7} for the unique element divisible by 3 other than 0,
i.e. the classical property

    P(x) := (x != 0) and (x mod 3 == 0),  x in {0, 1, ..., 7}

whose only satisfying value in that range is x = 3 (6 also satisfies it,
so there are two marked states: 3 and 6 -- Grover is run for two marked
items out of eight). The classical answer set {3, 6} is computed in this
script by brute force before the quantum run, and the quantum result is
checked against it.

This genuinely exercises Grover's algorithm (oracle + diffuser, run on the
ideal AerSimulator) but its connection to Erdos problem #363's actual
mathematical content is NONE -- that connection could not be established
from the available source data. ran_ok / verified_against_classical are
reported for what this script actually does (a real Grover search that is
verified), not for a claim about problem 363's sequence, since no such
sequence could be identified.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


N_QUBITS = 3          # search space {0, ..., 7}
N = 2 ** N_QUBITS


def classical_marked_states():
    """Brute-force compute P(x) := x != 0 and x % 3 == 0 for x in [0, N)."""
    return sorted(x for x in range(N) if x != 0 and x % 3 == 0)


def build_oracle(marked, n_qubits):
    """Phase oracle flipping the sign of each basis state in `marked`."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    classical_answer = classical_marked_states()
    assert classical_answer == [3, 6], (
        f"unexpected classical brute-force result: {classical_answer}"
    )

    M = len(classical_answer)
    theta = np.arcsin(np.sqrt(M / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qc = build_grover_circuit(classical_answer, N_QUBITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 2048
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # little-endian bitstrings -> integers
    int_counts = {}
    for bitstring, c in counts.items():
        val = int(bitstring[::-1], 2)
        int_counts[val] = int_counts.get(val, 0) + c

    top_states = sorted(int_counts.items(), key=lambda kv: -kv[1])
    marked_hits = sum(c for v, c in int_counts.items() if v in classical_answer)
    marked_fraction = marked_hits / shots

    print(f"Classical marked states (x != 0, x % 3 == 0, x in [0,{N})): "
          f"{classical_answer}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts (as integers): {sorted(int_counts.items())}")
    print(f"Top measured states: {top_states[:4]}")
    print(f"Fraction of shots landing on a marked state: {marked_fraction:.3f}")

    # A working Grover search on 2 marked states out of 8 should amplify the
    # marked-state probability well above the uniform baseline of 2/8=0.25.
    verified = marked_fraction > 0.60

    print()
    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
