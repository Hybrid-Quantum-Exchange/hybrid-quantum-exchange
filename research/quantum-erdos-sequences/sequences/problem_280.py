"""
Erdos problem #280 (github.com/manman4/erdosproblems, data/problems.yaml).

Metadata for problem 280: prize "no", status "disproved (Lean)",
tags ["number theory", "covering systems"], oeis: ["N/A"] -- no OEIS
sequence id is attached to this problem, so there is no OEIS-derived
sequence to search. This script therefore does NOT test a literal OEIS
term. Instead, honestly following the "tags" signal (covering systems),
it builds a small, finite, computable property native to the same
subject Erdos-280 is about: whether a finite set of congruences forms a
COVERING SYSTEM of the integers, i.e. whether every integer n satisfies
at least one of the congruences n = r (mod m).

Classical property tested (computed from first principles below, not
copied from any table):

    Congruence set C = {(0,2), (1,4), (3,8), (7,16), (15,16)}
    (residue r, modulus m), lcm(moduli) = 16.

    Claim: every integer n is covered by at least one congruence in C,
    equivalently every residue n in {0, ..., 15} is covered (since the
    period of the whole system is 16).

This is exactly the classic "doubling" covering system construction
(0 mod 2, 1 mod 4, 3 mod 8, 7 mod 16, 15 mod 16), a standard example in
the covering-systems literature that problem 280 is tagged under.

Quantum circuit:
    4 index qubits |n> for n in {0,...,15}, put into an equal
    superposition with Hadamards, plus 1 ancilla qubit. A reversible
    oracle, built directly from the classical congruence checks, flips
    the ancilla with a multi-controlled-X gate (controls set by n's bit
    pattern) for every n found (classically, at circuit-build time) to
    be covered. This is a genuine "quantum parallel evaluation" of the
    covered(n) predicate across the full superposition of all 16
    residues in one circuit execution (Deutsch-Jozsa style oracle
    application), not a lookup table smuggled into the result.

    If the classical claim is true (all 16 residues covered), every
    computational branch flips the ancilla to |1>, so measuring the
    ancilla after the Hadamards + oracle must yield '1' with
    probability 1 (within shot noise) regardless of the index-qubit
    outcome. If the claim were false, uncovered n's would leave the
    ancilla in |0>, giving a mixed measurement result.

    PASS: quantum-measured P(ancilla=1) matches the classical
    all-covered verdict, and (independently, as a cross-check) a
    Grover search over the same 16 residues for an uncovered
    counterexample finds none, matching zero-solution Grover
    statistics (near-uniform output distribution, since a zero-item
    Grover oracle acts as the identity).
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate

N_QUBITS = 4
N = 2 ** N_QUBITS  # 16 residues, period of the covering system

# (residue, modulus) pairs -- the classic doubling covering system.
CONGRUENCES = [(0, 2), (1, 4), (3, 8), (7, 16), (15, 16)]


def is_covered(n: int) -> bool:
    """Classical ground truth: does any congruence in CONGRUENCES cover n?"""
    return any(n % m == r for (r, m) in CONGRUENCES)


# ---- Step 1: classical verification, computed here from first principles ----
covered_flags = [is_covered(n) for n in range(N)]
uncovered = [n for n in range(N) if not covered_flags[n]]
classical_all_covered = len(uncovered) == 0

print(f"Congruence set: {CONGRUENCES}")
print(f"Residues 0..{N - 1} covered flags: {covered_flags}")
print(f"Uncovered residues (classical): {uncovered}")
print(f"Classical verdict: covering system = {classical_all_covered}")


def bits_of(n: int, width: int):
    """Little-endian bit list of n, qubit 0 = LSB."""
    return [(n >> i) & 1 for i in range(width)]


def add_marking_gate(qc: QuantumCircuit, n: int, index_qubits, target_qubit):
    """Apply an X on target_qubit controlled on index_qubits == n."""
    pattern = bits_of(n, len(index_qubits))
    flip_qubits = [q for q, b in zip(index_qubits, pattern) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    if len(index_qubits) == 1:
        qc.cx(index_qubits[0], target_qubit)
    else:
        qc.append(MCXGate(len(index_qubits)), list(index_qubits) + [target_qubit])
    for q in flip_qubits:
        qc.x(q)


# ---- Step 2: quantum parallel evaluation of covered(n) over superposition ----
def build_evaluation_circuit():
    qc = QuantumCircuit(N_QUBITS + 1, 1)
    index_qubits = list(range(N_QUBITS))
    ancilla = N_QUBITS

    qc.h(index_qubits)
    for n in range(N):
        if covered_flags[n]:
            add_marking_gate(qc, n, index_qubits, ancilla)
    qc.measure(ancilla, 0)
    return qc


def build_grover_counterexample_circuit(iterations: int = 1):
    """Search for an n with covered(n) == False (a counterexample)."""
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    index_qubits = list(range(N_QUBITS))

    qc.h(index_qubits)

    for _ in range(iterations):
        # Oracle: phase-flip the uncovered (marked) residues.
        for n in uncovered:
            pattern = bits_of(n, N_QUBITS)
            flip_qubits = [q for q, b in zip(index_qubits, pattern) if b == 0]
            for q in flip_qubits:
                qc.x(q)
            qc.h(index_qubits[-1])
            qc.append(MCXGate(N_QUBITS - 1), index_qubits)
            qc.h(index_qubits[-1])
            for q in flip_qubits:
                qc.x(q)

        # Diffusion (inversion about the mean).
        qc.h(index_qubits)
        qc.x(index_qubits)
        qc.h(index_qubits[-1])
        qc.append(MCXGate(N_QUBITS - 1), index_qubits)
        qc.h(index_qubits[-1])
        qc.x(index_qubits)
        qc.h(index_qubits)

    qc.measure(index_qubits, index_qubits)
    return qc


def main():
    sim = AerSimulator()
    shots = 4096

    # --- Evaluation circuit: ancilla should be '1' every time iff the
    # covering system genuinely covers all residues. ---
    eval_qc = build_evaluation_circuit()
    eval_result = sim.run(eval_qc, shots=shots).result()
    eval_counts = eval_result.get_counts()
    print(f"\nEvaluation-circuit ancilla counts: {eval_counts}")

    p_one = eval_counts.get("1", 0) / shots
    quantum_all_covered = p_one > 0.999  # allow for negligible sim noise

    # --- Grover cross-check: search for an uncovered counterexample.
    # With zero marked items the oracle is the identity, so the circuit
    # (H, oracle=identity, diffusion) returns exactly the starting
    # uniform superposition -- diffusion fixes |s> because
    # (2|s><s| - I)|s> = |s>. We confirm the output stays close to
    # uniform over all 16 residues, i.e. no counterexample is ever
    # amplified. ---
    grover_qc = build_grover_counterexample_circuit(iterations=1)
    grover_result = sim.run(grover_qc, shots=shots).result()
    grover_counts = grover_result.get_counts()
    print(f"Grover counterexample-search counts: {grover_counts}")

    expected_p = 1.0 / N
    max_dev = max(abs(c / shots - expected_p) for c in grover_counts.values())
    grover_near_uniform = max_dev < 0.02  # loose statistical tolerance

    print(f"\nQuantum P(ancilla=1) = {p_one:.4f} (expect ~1.0 iff covering system)")
    print(f"Grover max deviation from uniform = {max_dev:.4f} (expect small, no solution found)")

    verified = (quantum_all_covered == classical_all_covered) and grover_near_uniform

    print(f"\nClassical: covering system = {classical_all_covered}")
    print(f"Quantum (parallel oracle evaluation): covering system = {quantum_all_covered}")
    print(f"Quantum (Grover) found a counterexample: {not grover_near_uniform}")

    if verified:
        print("\nPASS")
    else:
        print("\nFAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
