"""
Erdos problem #552 -- quantum-testable instance.

OEIS: A006672
  a(n) = smallest m such that every red/blue edge-coloring of the complete
  graph K_m contains either a red C4 (4-cycle) or a blue star K_{1,n}.
  This is the Ramsey number r(C4, K_{1,n}).
  Known values (n = 1, 2, 3, ...): 4, 4, 6, 7, 8, 9, 11, ...
  a(2) = 4 in particular.

Classical property tested (small, finite, exhaustively computable):
  Consider the complete graph K_4 on vertices {1,2,3,4}, with its 6 edges
  indexed 0..5 as (1,2),(1,3),(1,4),(2,3),(2,4),(3,4). A subset S of these
  6 edges is represented as a 6-bit string (bit i = 1 iff edge i is in S,
  interpreted as "red"). The property being searched for:

      Does edge subset S form a 4-cycle (Hamiltonian cycle) of K_4?

  This is exactly the "red C4" alternative that appears in the definition
  of the Ramsey number r(C4, K_{1,n}) = A006672(n) above: a red/blue
  coloring of K_4 fails to witness a(2) > 4 precisely when its red edges
  contain (as an exact 4-edge subset here) one of K_4's Hamiltonian
  4-cycles.

  K_4 has exactly 3 distinct 4-cycles (each omits a perfect matching of
  2 edges). The script:
    1. Enumerates all 2^6 = 64 edge subsets of K_4 classically and checks,
       from first principles (a subset is a 4-cycle iff it has exactly 4
       edges forming a single cycle touching every vertex exactly twice),
       which ones are 4-cycles. This gives the ground truth: exactly 3 of
       the 64 subsets are 4-cycles.
    2. As a second classical fact tied to a(2) = 4: it also enumerates all
       2^6 colorings of K_4 and confirms that NONE of them avoids both a
       red C4 and a blue K_{1,2} (a vertex with >=2 blue edges) -- i.e.
       there is no counterexample coloring of K_4, consistent with
       a(2) = 4 (K_4 already forces one of the two patterns).
    3. Builds a genuine Grover search circuit over the 6 edge-qubits whose
       oracle phase-flips exactly the 3 classically-identified 4-cycle
       edge subsets (computed in step 1, not copied from OEIS).
    4. Runs the Grover circuit on the ideal AerSimulator with the
       theoretically optimal number of iterations for 3 marked states out
       of 64, and checks that the measured distribution concentrates on
       exactly those 3 classical 4-cycle bitstrings.
    5. Prints PASS if the quantum search's high-probability outcomes
       exactly equal the classical 4-cycle set, else FAIL.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: enumerate all edge subsets of K_4 and find which
#    are exactly a 4-cycle (Hamiltonian cycle on all 4 vertices).
# ---------------------------------------------------------------------------

VERTICES = [1, 2, 3, 4]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges, index 0..5
EDGE_INDEX = {frozenset(e): i for i, e in enumerate(EDGES)}
N_EDGES = len(EDGES)  # 6

VERTEX_EDGES = {
    v: [i for i, e in enumerate(EDGES) if v in e] for v in VERTICES
}


def is_four_cycle(bits):
    """bits: tuple of 6 ints (0/1) selecting a subset of K_4's edges.

    True iff the selected edge subset is exactly a Hamiltonian 4-cycle:
    exactly 4 edges are selected, and every vertex has exactly 2 of its
    3 incident edges selected, and the selected edges form a single
    connected cycle (not two disjoint 2-edge components, which can't
    happen once every vertex has degree exactly 2 among 4 edges on 4
    vertices -- degree-2-everywhere on a connected graph on 4 vertices
    with 4 edges is automatically a single 4-cycle).
    """
    if sum(bits) != 4:
        return False
    for v in VERTICES:
        deg = sum(bits[i] for i in VERTEX_EDGES[v])
        if deg != 2:
            return False
    return True


all_subsets = list(itertools.product([0, 1], repeat=N_EDGES))
four_cycles = [s for s in all_subsets if is_four_cycle(s)]
four_cycle_ints = sorted(
    sum(bit << i for i, bit in enumerate(s)) for s in four_cycles
)

print(f"K_4 edges (index: pair): {list(enumerate(EDGES))}")
print(f"Total edge subsets of K_4: {len(all_subsets)}")
print(f"Subsets that are exactly a 4-cycle: {four_cycles}")
print(f"As integers (bit i = edge i selected): {four_cycle_ints}")
assert len(four_cycles) == 3, "K_4 has exactly 3 distinct Hamiltonian 4-cycles"


# Second classical fact: no red/blue coloring of K_4 avoids both a red C4
# and a blue K_{1,2}. (This is the a(2) = 4 witness from A006672: K_4
# already forces one of the two patterns, unlike K_3.)
def has_red_c4(bits):
    return any(all(bits[i] == 1 for i in [EDGE_INDEX[frozenset(c)] for c in cyc])
               for cyc in _four_cycle_edge_lists())


def _four_cycle_edge_lists():
    # Rebuild the 3 four-cycles as lists of vertex-pair edges, from the
    # classically found bit patterns above (edges selected = the cycle).
    result = []
    for s in four_cycles:
        result.append([EDGES[i] for i, b in enumerate(s) if b == 1])
    return result


def has_blue_k12(bits):
    # bit=1 means the edge is colored RED here (matches four-cycle search);
    # blue is the complement. A vertex has a blue K_{1,2} iff at least 2 of
    # its incident edges are blue, i.e. at most 1 of its incident edges is
    # red (since each vertex has degree 3 in K_4).
    for v in VERTICES:
        red_deg = sum(bits[i] for i in VERTEX_EDGES[v])
        blue_deg = 3 - red_deg
        if blue_deg >= 2:
            return True
    return False


counterexamples = [
    s for s in all_subsets if not has_red_c4(s) and not has_blue_k12(s)
]
print(f"\nColorings of K_4 avoiding both red C4 and blue K_1,2: "
      f"{len(counterexamples)} (expect 0, witnessing a(2) = 4)")
assert len(counterexamples) == 0


# ---------------------------------------------------------------------------
# 2. Build a Grover oracle that phase-flips exactly the 3 classically
#    identified 4-cycle bitstrings, then amplify with Grover diffusion.
# ---------------------------------------------------------------------------

def marked_state_phase_flip(qc, n_qubits, marked_int):
    """Apply an X-MCZ-X pattern that flips the phase of |marked_int>."""
    bits = [(marked_int >> i) & 1 for i in range(n_qubits)]
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)


def build_oracle(n_qubits, marked_ints):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked_ints:
        marked_state_phase_flip(qc, n_qubits, m)
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


n_qubits = N_EDGES  # 6
n_marked = len(four_cycle_ints)  # 3
n_total = 2 ** n_qubits  # 64

theta = math.asin(math.sqrt(n_marked / n_total))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

oracle = build_oracle(n_qubits, four_cycle_ints)
diffuser = build_diffuser(n_qubits)

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(n_qubits))
    qc.append(diffuser.to_gate(), range(n_qubits))
qc.measure(range(n_qubits), range(n_qubits))

print(f"\nGrover search: {n_qubits} qubits, {n_marked} marked states out of "
      f"{n_total}, {iterations} iteration(s)")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 8192
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()


def key_to_int(key):
    # Qiskit's classical-register string is big-endian (c[n-1]...c[0]);
    # reverse so index 0 corresponds to qubit 0 (edge 0), matching our
    # bit-i-selects-edge-i convention used above.
    return int(key[::-1], 2)


counts_by_int = {}
for key, c in counts.items():
    ki = key_to_int(key)
    counts_by_int[ki] = counts_by_int.get(ki, 0) + c

print("\nTop measurement outcomes (as integers, bit i = edge i selected):")
for i, c in sorted(counts_by_int.items(), key=lambda kv: -kv[1])[:8]:
    marker = " <- 4-cycle" if i in four_cycle_ints else ""
    print(f"  {i} ({i:06b}): {c}{marker}")

prob_marked = sum(counts_by_int.get(i, 0) for i in four_cycle_ints) / shots
print(f"\nFraction of shots landing on a classically-verified 4-cycle: "
      f"{prob_marked:.4f} (uniform baseline would be "
      f"{n_marked / n_total:.4f})")

measured_support = {i for i, c in counts_by_int.items() if c / shots > 0.01}
quantum_matches_classical = (
    measured_support == set(four_cycle_ints) and prob_marked > 0.9
)

print(f"\nClassical 4-cycle set: {sorted(four_cycle_ints)}")
print(f"Quantum measured support (>1% of shots): {sorted(measured_support)}")

if quantum_matches_classical:
    print("\nPASS: Grover search over K_4's 64 edge subsets converged "
          "exactly onto the 3 classically enumerated Hamiltonian 4-cycles "
          "(the 'red C4' pattern in the r(C4, K_{1,n}) = A006672 "
          "definition behind Erdos problem 552), and the companion "
          "classical enumeration confirms no K_4 coloring avoids both a "
          "red C4 and a blue K_1,2 -- consistent with a(2) = 4.")
else:
    print("\nFAIL: quantum search result does not match classical "
          "enumeration.")

assert quantum_matches_classical
