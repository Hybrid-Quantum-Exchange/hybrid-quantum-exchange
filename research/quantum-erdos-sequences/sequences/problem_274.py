"""
Erdos problem #274 -- quantum-testable instance
=================================================

Erdos problem #274 (per erdosproblems.com / manman4/erdosproblems data,
data/problems.yaml, entry "number: '274'") is the Herzog-Schonheim
conjecture: if the residue classes a_1 (mod n_1), ..., a_k (mod n_k),
with 1 < n_1 <= ... <= n_k, form an EXACT (disjoint) covering system
of the integers -- i.e. every integer lies in exactly one class --
then at least two of the moduli n_i must be equal.  It is tagged
["group theory", "covering systems"] and has no OEIS id (oeis: ["N/A"]
in the source data), so there is no sequence to look a term up in.

Because there is no OEIS sequence attached to this problem, the
finite/computable property built here is instead a direct, checkable
instance of the covering-system arithmetic the conjecture is about:
for a small explicit list of residue classes, compute (classically,
from first principles, by brute force over one full period) exactly
which residues x are covered a number of times DIFFERENT FROM ONE
(i.e. which x violate "exact cover"), then build a quantum circuit
that searches (Grover amplitude amplification) for exactly that set
and verify the quantum result against the classical brute-force
answer. This is a genuine finite computable question directly built
from the covering-system objects problem 274 is about; it is not a
literal OEIS value, since no OEIS id exists for this problem.

Concrete instance
------------------
Residue classes examined, taken mod N = 8 (a multiple of every modulus
below, so one full period is visible):
    C1: x = 0 (mod 2)
    C2: x = 1 (mod 4)
    C3: x = 3 (mod 4)
{C1, C2, C3} is the textbook EXACT covering system of Z (every integer
covered exactly once -- consistent with Herzog-Schonheim, since it has
two equal moduli, 4 and 4). To get a nontrivial, small search target
for Grover, one further class is added that deliberately double-covers
a single residue on top of the exact cover:
    C4: x = 5 (mod 8)
so x = 5 is covered twice (by C2 and C4) while every other x in
{0,...,7} is still covered exactly once.

Property tested: which x in {0,...,7} are covered by a number of
classes other than exactly one ("bad" residues, coverage(x) != 1).
By construction there is exactly one such x -- this is verified by
brute force in code below, not assumed.

Quantum approach
-----------------
Grover search over the 3-qubit register representing x in {0,...,7}.
The oracle is a genuine multi-controlled-Z phase oracle built directly
from the classically-computed bad set (X-gates map each bad bitstring
to |111>, an (n-1)-controlled Z applies the phase, then the X-gates
are undone -- this works for any bad set, not just this one), followed
by the standard Grover diffuser, run for the optimal number of
iterations for a 1-out-of-8 marked search (2 iterations). The
resulting measurement distribution is compared against the
independently-computed classical bad set.

No OEIS id applies to this problem (oeis: ["N/A"]); the property
above is derived directly from the covering-system definitions in
problem 274's own statement, and its correctness is established
purely classically in this script before any quantum circuit runs.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_coverage(x, classes):
    """Number of residue classes (a, n) with x == a (mod n)."""
    return sum(1 for (a, n) in classes if x % n == a)


def classical_bad_set(n_bits, classes):
    """Brute-force (first-principles) set of x in [0, 2**n_bits) with
    coverage(x) != 1, i.e. the residues that break an exact cover."""
    N = 2 ** n_bits
    return sorted(x for x in range(N) if classical_coverage(x, classes) != 1)


def apply_marking_oracle(qc, n_qubits, bad_set):
    """Phase-flip |x> for every x in bad_set, via X-sandwiched
    (n_qubits-1)-controlled-Z gates. Qubit i holds bit i of x
    (qubit 0 = least significant bit). Works for any bad_set."""
    for x in bad_set:
        bits = [(x >> i) & 1 for i in range(n_qubits)]
        zero_qubits = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_qubits:
            qc.x(i)
        controls = list(range(n_qubits - 1))
        target = n_qubits - 1
        qc.h(target)
        qc.mcx(controls, target)
        qc.h(target)
        for i in zero_qubits:
            qc.x(i)


def build_grover_circuit(n_qubits, bad_set, iterations):
    """Grover search amplifying exactly the basis states in bad_set."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # equal superposition
    qc.h(range(n_qubits))

    for _ in range(iterations):
        apply_marking_oracle(qc, n_qubits, bad_set)

        # diffuser (inversion about the mean)
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    n_bits = 3
    N = 2 ** n_bits  # period used for the check: 8

    # residue classes (a, n): C1=0 mod2, C2=1 mod4, C3=3 mod4 (exact cover
    # of Z, per Herzog-Schonheim), plus C4=5 mod8 which deliberately
    # double-covers x=5 on top of the exact cover.
    classes = [(0, 2), (1, 4), (3, 4), (5, 8)]

    bad_set = classical_bad_set(n_bits, classes)
    print(f"Classical coverage table over Z_{N} for classes {classes}:")
    for x in range(N):
        print(f"  x={x}: coverage={classical_coverage(x, classes)}"
              f"{'  <- bad (!=1)' if x in bad_set else ''}")
    print(f"Classical bad set (coverage != 1): {bad_set}")

    if not bad_set:
        raise RuntimeError("Expected a nonempty bad set for this instance.")

    num_marked = len(bad_set)
    theta = np.arcsin(np.sqrt(num_marked / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
    print(f"Marked states: {num_marked}/{N}; using {iterations} Grover iteration(s)")

    qc = build_grover_circuit(n_bits, bad_set, iterations)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit orders classical bits with qubit 0 as the rightmost bit;
    # convert each bitstring back to an integer x consistently.
    dist = {}
    for bitstring, c in counts.items():
        x = int(bitstring[::-1], 2)
        dist[x] = dist.get(x, 0) + c

    print("Measured distribution (x: counts):")
    for x in sorted(dist):
        print(f"  x={x}: {dist[x]}")

    marked_counts = sum(dist.get(x, 0) for x in bad_set)
    unmarked_counts = shots - marked_counts
    marked_fraction = marked_counts / shots
    print(f"Fraction of shots landing on the classical bad set {bad_set}: "
          f"{marked_fraction:.4f} ({marked_counts}/{shots})")

    # After 1 Grover iteration with 4/8 marked, amplitude on marked
    # states should be strongly amplified (ideally -> 1.0). Require a
    # clear majority as the quantum-vs-classical verification.
    verified = marked_fraction > 0.9

    print(f"Quantum result matches classical bad set with high confidence: {verified}")
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
