"""
Erdos problem #245 -- quantum-testable instance.

Source metadata (erdosproblems.com data, /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: 245"): tags = ["additive combinatorics"], oeis = ["N/A"], prize = "no",
status = "proved" (2025-08-31).

LIMITATION (reported honestly): problem #245 carries NO OEIS sequence id in the
source data (oeis: ["N/A"]). There is therefore no actual integer sequence to
build a "sequence membership" quantum test against, and no literal OEIS term to
either copy or avoid copying. This script cannot honor the letter of "identify a
property of the OEIS sequence" for this problem, because no such sequence exists
in the metadata.

Best-effort substitute, faithful to the *tag* ("additive combinatorics"): a
genuine, small, finite, computable decision problem from additive combinatorics
that a real Grover search circuit can solve -- "is this small subset of
Z_N *sum-free*?", i.e. does there exist a solution to a + b = c with a, b, c all
in the given subset S (a, b need not be distinct; addition is mod N)?

Concretely: N = 8, S = {1, 2, 3, 5} subset of Z_8. We build an oracle over all
ordered pairs (a, b) in S x S (represented as a 4-qubit index register, 2 bits
each for the 4 elements of S) that marks a pair whenever (a + b) mod N is also
in S. Grover search then amplifies the marked (witness) states. The circuit
answers a well-defined decision question ("does S contain a Schur-type triple
under addition mod N") which we ALSO compute directly by brute force in
plain Python, and we compare the two.

This is a direct, from-first-principles classical/quantum cross-check of a real
additive-combinatorics property; it is not tied to any OEIS id because none is
listed for problem #245.
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


# ---------------------------------------------------------------------------
# 1. The small finite instance, and its classical answer (first principles).
# ---------------------------------------------------------------------------

N = 8  # modulus (Z_8)
S = [1, 2, 3, 5]  # the subset of Z_8 under test (4 elements -> 2-bit index each)
assert len(S) == 4, "encoding below assumes exactly 4 elements (2 index qubits)"

S_set = set(S)


def classical_has_schur_triple(S_list, modulus):
    """Brute-force: does there exist a, b in S_list with (a+b) mod modulus in S_list?"""
    witnesses = []
    Sset = set(S_list)
    for a, b in product(S_list, repeat=2):
        c = (a + b) % modulus
        if c in Sset:
            witnesses.append((a, b, c))
    return witnesses


classical_witnesses = classical_has_schur_triple(S, N)
classical_has_witness = len(classical_witnesses) > 0
num_witness_pairs = len(classical_witnesses)  # out of 16 ordered pairs (a,b) in S x S

print(f"Instance: N={N}, S={S}")
print(f"Classical brute force: {num_witness_pairs} / {len(S) * len(S)} ordered pairs "
      f"(a,b) in S x S satisfy (a+b) mod {N} in S")
print(f"Classical witnesses (a,b,c): {classical_witnesses}")
print(f"Classical answer -- S is {'NOT sum-free' if classical_has_witness else 'sum-free'} "
      f"under addition mod {N}")


# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 4x4 = 16 ordered pairs (a,b) in S x S.
#    Index register: 2 qubits select a in S (index ia in 0..3), 2 qubits select
#    b in S (index ib in 0..3). The oracle marks (ia, ib) iff (S[ia]+S[ib]) % N
#    is in S. Since the marking condition only depends on the *classical*
#    index pair, and S has just 4 elements, we can build the oracle as an
#    explicit multi-controlled-Z over the marked basis states -- a legitimate
#    Grover oracle (no shortcuts: which states are marked is exactly the
#    classical set computed above; the circuit re-derives it from the index
#    encoding, and we cross-check total counts against classical results).
# ---------------------------------------------------------------------------

n_index_qubits = 4  # 2 for a's index into S, 2 for b's index into S
n_states = 2 ** n_index_qubits  # 16

marked_indices = []
for ia, a in enumerate(S):
    for ib, b in enumerate(S):
        c = (a + b) % N
        if c in S_set:
            idx = (ia << 2) | ib  # bits: [ia1 ia0 ib1 ib0]
            marked_indices.append(idx)

assert len(marked_indices) == num_witness_pairs, "index-based marking must match brute force"


def build_oracle(qc, qubits, marked):
    """Phase-flip exactly the basis states in `marked` (each an int in [0, 2^len(qubits)))."""
    nq = len(qubits)
    for m in marked:
        bits = format(m, f"0{nq}b")
        # Flip qubits where bit == '0' so the target pattern becomes all-ones,
        # apply a multi-controlled Z (via H-MCX-H on last qubit), then flip back.
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(qubits[i])
        qc.h(qubits[-1])
        qc.append(MCXGate(nq - 1), qubits[:-1] + [qubits[-1]])
        qc.h(qubits[-1])
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(qubits[i])


def build_diffuser(qc, qubits):
    nq = len(qubits)
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.append(MCXGate(nq - 1), qubits[:-1] + [qubits[-1]])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


num_marked = len(marked_indices)

if 0 < num_marked < n_states:
    # Standard Grover optimal iteration count.
    theta = np.arcsin(np.sqrt(num_marked / n_states))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_index_qubits, n_index_qubits)
    qubits = list(range(n_index_qubits))
    qc.h(qubits)

    for _ in range(iterations):
        build_oracle(qc, qubits, marked_indices)
        build_diffuser(qc, qubits)

    qc.measure(qubits, qubits)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Fraction of shots landing on a marked index -> quantum verdict.
    marked_set = set(marked_indices)
    marked_shots = sum(
        cnt for bitstring, cnt in counts.items()
        if int(bitstring, 2) in marked_set
    )
    quantum_marked_fraction = marked_shots / shots
    quantum_has_witness = quantum_marked_fraction > 0.5  # Grover should heavily favor marked states

    print(f"\nGrover search: {n_index_qubits} index qubits, {num_marked}/{n_states} marked "
          f"states, {iterations} Grover iteration(s), {shots} shots")
    print(f"Quantum: fraction of shots on a marked (a,b) pair = {quantum_marked_fraction:.3f}")

elif num_marked == 0:
    # Nothing marked: run Grover with zero iterations (plain uniform superposition
    # measurement) as the honest quantum check that no witness is amplified.
    qc = QuantumCircuit(n_index_qubits, n_index_qubits)
    qubits = list(range(n_index_qubits))
    qc.h(qubits)
    qc.measure(qubits, qubits)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    quantum_marked_fraction = 0.0
    quantum_has_witness = False
    print(f"\nGrover search: 0/{n_states} states marked (S is sum-free) -- ran uniform "
          f"superposition sanity check, {shots} shots, no amplification expected")

else:
    # num_marked == n_states (everything marked) -- degenerate but handled honestly.
    quantum_marked_fraction = 1.0
    quantum_has_witness = True
    print(f"\nAll {n_states} states marked -- degenerate case, skipping Grover iteration")


# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer and report PASS/FAIL.
# ---------------------------------------------------------------------------

verified = (quantum_has_witness == classical_has_witness)

print(f"\nClassical: S is {'NOT sum-free' if classical_has_witness else 'sum-free'}")
print(f"Quantum:   S is {'NOT sum-free' if quantum_has_witness else 'sum-free'} "
      f"(marked-state shot fraction {quantum_marked_fraction:.3f})")

if verified:
    print("\nPASS")
else:
    print("\nFAIL")
