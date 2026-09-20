"""
Erdos problem #273 -- quantum-testable instance.

Source metadata (erdosproblems.com data, /home/user/manman4/erdosproblems/data/
problems.yaml, entry "number: \"273\""):
    tags: ["number theory", "covering systems"]
    oeis: ["N/A"]   -- no OEIS sequence id is attached to this problem.

LIMITATION: problem #273 has no OEIS id in the source data, so there is no
"sequence" from it to build a Grover/QPE circuit over in the sense the other
lanes use (a numeric OEIS sequence and a term-membership test). Faking an
OEIS id or a fabricated sequence would violate the task's own instructions.
Instead this script builds a genuine, small, finite, computable property
that comes directly from problem #273's own subject matter -- covering
systems -- and tests it with a real quantum circuit. This is the "best
honest attempt" path the task explicitly allows when no OEIS id exists.

The classical property tested
------------------------------
A covering system is a finite set of congruences (r_i mod m_i) such that
every integer satisfies at least one of them. Erdos problem #273 is about
covering systems. We use the well-known exact covering system of the
integers mod 8:

    0 (mod 2)
    1 (mod 4)
    3 (mod 8)
    7 (mod 8)

Because 1/2 + 1/4 + 1/8 + 1/8 = 1, this covering system is in fact an exact
partition of Z/8Z: every residue x in {0,...,7} satisfies EXACTLY ONE of the
four congruences. This is verified classically in this script by brute
force before any quantum circuit runs (see `classical_is_covered` /
`classical_answer`).

The quantum circuit
--------------------
We build a 3-qubit reversible oracle (with one ancilla) that computes, for
a computational basis state |x> (x in 0..7, the 3 qubits are the bits of
x), whether x is covered by the system above, by toggling the ancilla
qubit with a sequence of (X-padded) multi-controlled-X gates -- one per
congruence, each targeting the ancilla. Because the four congruences
partition Z/8Z, each basis state triggers exactly one of the four
multi-controlled-X gates, so the ancilla ends in |1> if and only if x is
covered.

We then run this oracle, as an actual circuit on the ideal AerSimulator,
once for each of the 8 possible inputs x = 0..7 (prepared as basis states),
measure the ancilla, and check that the quantum circuit reports "covered"
(ancilla == 1) for every single x -- exactly matching the classical
brute-force result that this covering system covers all of Z/8Z.

PASS means: for all x in 0..7, quantum_is_covered(x) == classical_is_covered(x)
(and classically, every x is covered, confirming the covering-system claim).
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# The covering system under test: list of (residue, modulus) congruences.
# All moduli are powers of two dividing 8, so each can be checked by
# inspecting a fixed number of low bits of x (given as a fixed-width binary
# pattern, most-significant bit of the *relevant* prefix first).
# ---------------------------------------------------------------------------
CONGRUENCES = [
    (0, 2),  # x mod 2 == 0
    (1, 4),  # x mod 4 == 1
    (3, 8),  # x mod 8 == 3
    (7, 8),  # x mod 8 == 7
]

N_BITS = 3  # x ranges over 0..7


def classical_is_covered(x: int) -> bool:
    """Brute-force classical check: does x satisfy some congruence?"""
    return any(x % m == r for (r, m) in CONGRUENCES)


def classical_answer():
    """Classical ground truth for all x in 0..7, computed from first principles."""
    results = {x: classical_is_covered(x) for x in range(2 ** N_BITS)}
    all_covered = all(results.values())
    # sanity: verify it is an exact partition (densities sum to 1 and no
    # residue is covered twice), which is what makes this covering system
    # interesting rather than a trivial redundant one.
    coverers = {x: [(r, m) for (r, m) in CONGRUENCES if x % m == r] for x in range(2 ** N_BITS)}
    exactly_one = all(len(v) == 1 for v in coverers.values())
    return results, all_covered, exactly_one


def bits_of(x: int, n: int):
    """Return list of n bits of x, index 0 = least significant bit."""
    return [(x >> i) & 1 for i in range(n)]


def add_congruence_oracle(qc: QuantumCircuit, data_qubits, ancilla, residue: int, modulus: int):
    """
    Append gates that flip `ancilla` iff the basis state currently held on
    `data_qubits` (LSB-first, N_BITS qubits encoding x in 0..7) satisfies
    x mod modulus == residue.

    Only the low log2(modulus) bits matter; those bits must match the bits
    of `residue`. We X-pad the "0" bits so a plain multi-controlled-X fires
    exactly when the relevant low bits equal `residue`, then undo the X-pad.
    """
    k = modulus.bit_length() - 1  # number of relevant low bits (modulus is a power of 2)
    relevant_qubits = data_qubits[:k]
    target_bits = bits_of(residue, k)

    flipped = []
    for qubit, bit in zip(relevant_qubits, target_bits):
        if bit == 0:
            qc.x(qubit)
            flipped.append(qubit)

    if k == 0:
        # modulus == 1: always true, unconditionally flip ancilla (not used here)
        qc.x(ancilla)
    elif k == 1:
        qc.cx(relevant_qubits[0], ancilla)
    else:
        qc.mcx(relevant_qubits, ancilla)

    for qubit in flipped:
        qc.x(qubit)


def build_circuit_for_x(x: int) -> QuantumCircuit:
    """
    Build the full reversible "is x covered?" circuit for a specific
    fixed input x: prepare |x> on the data qubits, apply the OR-of-
    congruences oracle (implemented as sequential toggles, valid because
    the congruences partition Z/8Z so each x triggers exactly one term),
    then measure the ancilla.
    """
    data = list(range(N_BITS))
    ancilla = N_BITS
    qc = QuantumCircuit(N_BITS + 1, 1)

    # Prepare |x> on the data qubits.
    for i, bit in enumerate(bits_of(x, N_BITS)):
        if bit == 1:
            qc.x(data[i])

    # Apply one toggle per congruence.
    for (r, m) in CONGRUENCES:
        add_congruence_oracle(qc, data, ancilla, r, m)

    qc.measure(ancilla, 0)
    return qc


def quantum_is_covered(x: int, simulator: AerSimulator, shots: int = 256) -> bool:
    qc = build_circuit_for_x(x)
    tqc = qc.copy()  # AerSimulator can run the circuit directly (no transpile needed for this gate set)
    result = simulator.run(tqc, shots=shots).result()
    counts = result.get_counts()
    # Deterministic circuit: expect a single outcome across all shots.
    assert len(counts) == 1, f"non-deterministic oracle output for x={x}: {counts}"
    outcome = next(iter(counts))
    return outcome == "1"


def main():
    classical_results, classical_all_covered, exact_partition = classical_answer()

    print("Erdos problem #273 -- covering systems (no OEIS id available; see docstring)")
    print(f"Congruences tested: {CONGRUENCES} (modulus 8)")
    print(f"Classical: exact partition of Z/8Z = {exact_partition}")
    print(f"Classical: all residues 0..7 covered = {classical_all_covered}")
    print(f"Classical per-residue coverage: {classical_results}")

    simulator = AerSimulator()

    quantum_results = {}
    mismatches = []
    for x in range(2 ** N_BITS):
        q_covered = quantum_is_covered(x, simulator)
        quantum_results[x] = q_covered
        if q_covered != classical_results[x]:
            mismatches.append(x)

    print(f"Quantum per-residue coverage:   {quantum_results}")

    verified = (not mismatches) and classical_all_covered and exact_partition

    if verified:
        print("PASS")
    else:
        print(f"FAIL (mismatches at x={mismatches}, "
              f"classical_all_covered={classical_all_covered}, exact_partition={exact_partition})")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
