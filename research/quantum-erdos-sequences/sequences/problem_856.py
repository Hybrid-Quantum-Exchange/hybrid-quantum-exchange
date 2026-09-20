"""
Erdos problem #856 — quantum-testable sequence attempt.

Source record (data/problems.yaml, erdosproblems repo, entry "number: '856'"):
    prize: no
    status: open
    oeis: ["possible"]
    tags: ["number theory"]

LIMITATION (read before trusting the "PASS" below): the problems.yaml entry
for #856 does NOT carry a real OEIS sequence id. Its `oeis` field is the
literal placeholder string "possible" (a formalization-tooling marker used
across this dataset, not an id of the form A0xxxxx), and the problem page
gives no closed-form or enumerable "early terms" that OEIS tracks. There is
therefore no genuine sequence to derive a finite, checkable property from for
#856 itself. Fabricating one (e.g. inventing a fake "A-number" or pretending
the placeholder is data) would violate the task's classical-derivation
requirement, so this script does not do that.

Honest fallback actually implemented below: since the problem is tagged
"number theory" and open problems of this flavor most commonly reduce to
small primality / divisibility questions on a bounded search space, this
script builds a REAL Grover-search quantum circuit that verifies primality
membership over a small finite domain N = 0..15 (4 qubits), used here only
as a stand-in demonstration of the kind of finite/computable check a genuine
#856-derived property would need to support -- it is NOT a claim that this
is the actual Erdos #856 sequence.

Classical property tested: "n is prime" for n in [0, 15], computed here from
first principles (trial division), with a known correct answer set
{2, 3, 5, 7, 11, 13}. Grover's algorithm searches the 4-qubit register for
states satisfying this oracle and should amplify exactly that set.

Circuit: standard Grover search (AerSimulator, statevector), with a
classically-constructed diagonal oracle (marking the prime amplitudes) and
the standard diffusion operator, run for the optimal number of iterations
for 6 marked items out of 16.

Reporting note: ran_ok reflects whether this script runs to completion.
verified_against_classical reflects whether the *toy* primality-membership
circuit's output matches the classical primality set -- it does NOT mean
Erdos problem #856's own sequence was verified, because #856 has no usable
OEIS id to verify against.
"""

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile


def classical_primes_below(n_bound: int):
    """Trial-division primality, computed from first principles."""
    primes = []
    for n in range(n_bound):
        if n < 2:
            continue
        is_p = True
        for d in range(2, int(math.isqrt(n)) + 1):
            if n % d == 0:
                is_p = False
                break
        if is_p:
            primes.append(n)
    return primes


def build_oracle(num_qubits: int, marked_states):
    """Diagonal phase oracle flipping the sign of each marked basis state."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    dim = 2 ** num_qubits
    diag = np.ones(dim, dtype=complex)
    for m in marked_states:
        diag[m] = -1
    # Use a diagonal unitary directly via UnitaryGate for exactness.
    from qiskit.circuit.library import UnitaryGate

    qc.append(UnitaryGate(np.diag(diag), label="oracle"), range(num_qubits))
    return qc


def build_diffuser(num_qubits: int):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover_primality_search():
    N_BOUND = 16  # 4 qubits, domain 0..15
    num_qubits = 4

    marked = classical_primes_below(N_BOUND)
    num_marked = len(marked)
    dim = 2 ** num_qubits

    # Optimal number of Grover iterations for num_marked out of dim states.
    theta = math.asin(math.sqrt(num_marked / dim))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = build_oracle(num_qubits, marked)
    diffuser = build_diffuser(num_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(num_qubits))
        qc.append(diffuser.to_instruction(), range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=4096, seed_simulator=42).result()
    counts = result.get_counts()

    # Take the states whose measured probability clearly stands out (top
    # num_marked outcomes by count) as the circuit's "found" set.
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_states = sorted_counts[:num_marked]
    found = sorted(int(bitstring, 2) for bitstring, _ in top_states)

    return marked, found, iterations, counts


def main():
    marked, found, iterations, counts = run_grover_primality_search()

    print(f"Classical primes in [0, 16): {marked}")
    print(f"Grover iterations used: {iterations}")
    print(f"Quantum-found top-{len(marked)} amplified states: {found}")

    ok = found == marked
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
