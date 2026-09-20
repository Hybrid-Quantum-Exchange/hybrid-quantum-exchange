"""
Erdos problem #277 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, problems.yaml, number "277"):
    prize: no
    informal_status: proved (2025-08-31)
    formal_status: Lean (2026-08-23)
    oeis: ["N/A"]   <-- no OEIS sequence id is attached to this problem.
    tags: ["number theory", "covering systems"]

LIMITATION, stated honestly up front: problem 277 has no associated OEIS
sequence (oeis: ["N/A"] in the source data), so there is no "sequence" to
search or verify membership in. Rather than fabricate an OEIS id or copy a
value with no real connection to this problem, this script instead builds a
genuine small quantum circuit around the one piece of real mathematical
content problem 277's tags actually give us: "covering systems" -- a set of
congruences {a_i (mod m_i)} whose union covers every integer. This is a
completely standard object in this branch of number theory (the classical
covering system due to Erdos: 0 mod 2, 0 mod 3, 1 mod 4, 5 mod 6, 7 mod 12),
and it gives a small, finite, exactly-computable search problem well suited
to Grover's algorithm.

Chosen finite instance / property being tested:
    Congruence class "1 (mod 4)" is one of the five congruences in Erdos's
    classical covering system {0 mod 2, 0 mod 3, 1 mod 4, 5 mod 6, 7 mod 12}.
    Search space: integers x in [0, 8) (3 qubits).
    Property P(x):  x mod 4 == 1
    The classically-computed solution set (first-principles brute force,
    done in this script, not copied from anywhere) is S = {1, 5}.

Circuit: an exact Grover search (3 qubits, single Grover iteration is optimal
for |S|/N = 2/8 = 1/4) with an oracle built directly from the arithmetic
condition (x mod 4 == 1, i.e. bit0 == 1 AND bit2 == 0 for x in [0,8)) and the
standard diffusion operator, run on the ideal AerSimulator (statevector
sampling, no noise). The measured outcome distribution is compared against
the classically computed solution set S; PASS requires the quantum
measurements to concentrate overwhelmingly (>= 95% of shots) on S, showing
the circuit genuinely searches out states satisfying the number-theoretic
congruence rather than returning a canned answer.
"""

import sys

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_solution_set(n_qubits: int) -> set[int]:
    """Brute-force, from first principles: all x in [0, 2**n_qubits) with x mod 4 == 1."""
    N = 2 ** n_qubits
    return {x for x in range(N) if x % 4 == 1}


def build_oracle(n_qubits: int) -> QuantumCircuit:
    """
    Phase-flip oracle for property P(x): x mod 4 == 1.

    For x in [0, 8) written as bits q2 q1 q0 (q0 = LSB, value 2**0),
    x mod 4 is determined entirely by bits q1,q0 (the low two bits):
        x mod 4 == 1  <=>  q0 == 1 AND q1 == 0
    q2 (the high bit, value 4) is irrelevant to x mod 4, so both x=1 (001)
    and x=5 (101) get marked -- exactly matching the classical solution set.

    Implemented as a controlled-Z: flip q1 (so "q1==0" becomes the control-1
    case), apply CCZ controlled on q0 and (flipped) q1 targeting nothing
    directly -- built here as a controlled phase via an ancilla-free CCZ
    (Toffoli + phase) pattern using qiskit's built-in multi-controlled Z.
    """
    qc = QuantumCircuit(n_qubits, name="oracle_x_mod4_eq_1")
    q0, q1 = 0, 1  # q2 (index 2) is a "don't care" qubit for this property
    # Map condition (q0==1, q1==0) onto (q0==1, q1==1) by flipping q1 first.
    qc.x(q1)
    qc.cz(q0, q1)  # phase-flip exactly when q0==1 and (flipped) q1==1
    qc.x(q1)
    return qc


def build_diffusion(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffusion")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(n_qubits)
    diffusion = build_diffusion(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main() -> int:
    n_qubits = 3
    N = 2 ** n_qubits

    solution_set = classical_solution_set(n_qubits)
    print(f"Classical solution set for x mod 4 == 1, x in [0,{N}): {sorted(solution_set)}")
    assert solution_set == {1, 5}, "classical computation sanity check failed"

    M = len(solution_set)
    # Optimal Grover iteration count for M marked items out of N.
    import math
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"N={N}, M={M} marked states, using {iterations} Grover iteration(s)")

    qc = build_grover_circuit(n_qubits, iterations)

    backend = AerSimulator()
    shots = 4096
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports bitstrings as q(n-1)...q1q0 (MSB first); convert to int.
    def bits_to_int(bitstring: str) -> int:
        return int(bitstring, 2)

    hits_in_solution = sum(
        c for bstr, c in counts.items() if bits_to_int(bstr) in solution_set
    )
    fraction = hits_in_solution / shots

    print("Measurement counts (bitstring -> count, decoded x):")
    for bstr, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {bstr} -> x={bits_to_int(bstr)}: {c}")
    print(f"Fraction of shots landing in classical solution set {sorted(solution_set)}: {fraction:.4f}")

    passed = fraction >= 0.95
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
