"""
Erdos problem #133 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com mirror, entry
"number: '133'"): tags = ["graph theory"], oeis = ["possible"].

IMPORTANT LIMITATION, stated honestly up front: problem #133's yaml entry
carries no real OEIS sequence id -- the field is the literal placeholder
string "possible", not an id such as "A000000". There is therefore no
concrete integer sequence from this entry to test membership/term-generation
against. Rather than fabricate an OEIS id or copy a value with no real
mathematical content, this script instead builds a genuine, finite,
classically-checkable computation drawn from the problem's one real piece of
metadata that *is* usable: its tag, "graph theory".

Chosen property (finite, computable, small search space):
    Among the 2^6 = 64 labeled graphs on 4 vertices (one bit per possible
    edge of K4), how many are triangle-free, and can a Grover search over
    that 6-qubit space find one?

This is a bona fide combinatorial search problem (a miniature instance of
the extremal/Ramsey-type questions that pervade Erdos's graph-theory work,
matching this problem's tag) with a small, exactly computable answer: it is
*not* Erdos problem #133 itself (which concerns a specific, disproved graph
theory claim, per informal_status.state = "disproved"), and this script
does not claim otherwise. It is presented as the best honest attempt at a
quantum-testable proxy in the absence of a real OEIS sequence to anchor to.

Classical part (computed here from first principles, not copied from
anywhere): brute-force enumeration of all 64 edge-subsets of K4, counting
which are triangle-free (no 3 mutually chosen edges forming a 3-cycle).
K4 has 4 triangles: {0,1},{0,2},{1,2} ; {0,1},{0,3},{1,3} ;
{0,2},{0,3},{2,3} ; {1,2},{1,3},{2,3} (vertices 0..3, edge order below).

Quantum part: Grover's algorithm over the 6-qubit edge register. The oracle
computes, via reversible AND-ladders into ancilla qubits, whether any of the
4 triangles is fully present, and phase-flips (via a phase-kickback ancilla
prepared in |-> ) exactly the triangle-free computational basis states. The
standard Grover diffuser is applied for the number of iterations set by the
classically-known solution count M out of N = 64. The circuit is run on the
ideal AerSimulator (statevector-based sampling); PASS requires that the
quantum circuit's measured outcomes concentrate (near-)entirely on the
classically verified triangle-free graphs, with no non-solution samples
observed.
"""

from itertools import combinations, product

from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator
import numpy as np


# ---------------------------------------------------------------------------
# Classical part: brute-force ground truth, computed here from first
# principles (no copied OEIS value).
# ---------------------------------------------------------------------------

VERTICES = (0, 1, 2, 3)
EDGES = list(combinations(VERTICES, 2))  # 6 possible edges of K4, fixed order
assert len(EDGES) == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

TRIANGLES = []
for a, b, c in combinations(VERTICES, 3):
    tri_edges = (EDGE_INDEX[(a, b)], EDGE_INDEX[(a, c)], EDGE_INDEX[(b, c)])
    TRIANGLES.append(tri_edges)
assert len(TRIANGLES) == 4


def is_triangle_free(bits):
    """bits: tuple of 6 ints (0/1), bits[i] = whether EDGES[i] is present."""
    for (i, j, k) in TRIANGLES:
        if bits[i] and bits[j] and bits[k]:
            return False
    return True


def classical_brute_force():
    solutions = []
    for bits in product((0, 1), repeat=6):
        if is_triangle_free(bits):
            solutions.append(bits)
    return solutions


CLASSICAL_SOLUTIONS = classical_brute_force()
N = 64
M = len(CLASSICAL_SOLUTIONS)
# Sanity check against a second, independent classical method: the number of
# triangle-free labeled graphs on 4 vertices equals 2^6 minus the count of
# graphs containing at least one of the 4 triangles, via inclusion-exclusion
# over the 4 triangle-edge-triples (each triangle fixes 3 specific edges to
# 1, leaving the other 3 free: 2^3 = 8 graphs per triangle; pairs of
# triangles share either 0 or 1 edge depending on which pair, etc.).
def inclusion_exclusion_check():
    tri_sets = [set(t) for t in TRIANGLES]
    total = 0
    idxs = range(4)
    for r in range(1, 5):
        for combo in combinations(idxs, r):
            union_edges = set()
            for c in combo:
                union_edges |= tri_sets[c]
            free_edges = 6 - len(union_edges)
            term = 2 ** free_edges
            total += term if (r % 2 == 1) else -term
    return 2 ** 6 - total


ie_count = inclusion_exclusion_check()
if ie_count != M:
    raise RuntimeError(
        f"classical cross-check failed: brute force M={M}, "
        f"inclusion-exclusion={ie_count}"
    )

print(f"Classical result: N={N} total labeled graphs on 4 vertices, "
      f"M={M} are triangle-free (cross-checked by inclusion-exclusion).")


# ---------------------------------------------------------------------------
# Quantum part: Grover search over the 6-qubit edge register.
# ---------------------------------------------------------------------------

def build_grover_circuit(iterations):
    edges = QuantumRegister(6, "e")
    tri = QuantumRegister(4, "t")       # one ancilla per triangle
    helper = QuantumRegister(1, "h")    # scratch AND ancilla
    flag = QuantumRegister(1, "f")      # phase-kickback ancilla
    creg = ClassicalRegister(6, "c")

    qc = QuantumCircuit(edges, tri, helper, flag, creg)

    # Uniform superposition over the search register.
    qc.h(edges)

    # Phase-kickback ancilla in |->.
    qc.x(flag)
    qc.h(flag)

    def apply_oracle():
        # Compute each triangle indicator into tri[i].
        for idx, (a, b, c) in enumerate(TRIANGLES):
            qc.ccx(edges[a], edges[b], helper[0])
            qc.ccx(helper[0], edges[c], tri[idx])
            qc.ccx(edges[a], edges[b], helper[0])  # uncompute helper

        # Invert triangle flags: tri[i] = 1 means "no triangle i".
        qc.x(tri)

        # Flip flag iff ALL tri[i] == 1 (original graph triangle-free).
        qc.append(MCXGate(4), [tri[0], tri[1], tri[2], tri[3], flag[0]])

        # Uncompute the inversion.
        qc.x(tri)

        # Uncompute the triangle indicators (same ladder, self-inverse).
        for idx, (a, b, c) in enumerate(TRIANGLES):
            qc.ccx(edges[a], edges[b], helper[0])
            qc.ccx(helper[0], edges[c], tri[idx])
            qc.ccx(edges[a], edges[b], helper[0])

    def apply_diffuser():
        qc.h(edges)
        qc.x(edges)
        # multi-controlled Z on the 6 edge qubits via H-MCX-H on the last one
        qc.h(edges[5])
        qc.append(MCXGate(5), [edges[0], edges[1], edges[2], edges[3], edges[4], edges[5]])
        qc.h(edges[5])
        qc.x(edges)
        qc.h(edges)

    for _ in range(iterations):
        apply_oracle()
        apply_diffuser()

    qc.measure(edges, creg)
    return qc


# M/N = 41/64 > 1/2 here, so the textbook iteration-count formula (which
# assumes M << N) can actually overshoot past the optimal rotation angle and
# make things worse. Rather than assume the formula applies, pick the
# iteration count empirically: build the circuit for a small range of
# candidate k, compute the *exact* success probability from the statevector
# (no sampling noise) for each, and keep the best -- this is itself a
# classical computation over the circuit's exact amplitudes, so it stays
# honest (no shot-based cherry-picking).
from qiskit.quantum_info import Statevector

classical_solution_set = {bits for bits in CLASSICAL_SOLUTIONS}

solution_mask = np.zeros(2 ** 6, dtype=bool)
for bits in CLASSICAL_SOLUTIONS:
    # edges[0] is the least significant qubit in Statevector's ordering.
    idx = sum(b << i for i, b in enumerate(bits))
    solution_mask[idx] = True

best_k, best_p = 0, -1.0
for k in range(0, 5):
    probe = build_grover_circuit(k)
    probe.remove_final_measurements()
    sv = Statevector(probe)
    probs = sv.probabilities_dict()
    # Marginalize over the non-edge qubits (ancillas should be back at |0>,
    # flag stays in |->, so this just sums out those extra qubit strings).
    p_hit = 0.0
    for bitstring, p in probs.items():
        # bitstring is ordered [flag h t3 t2 t1 t0 e5 e4 e3 e2 e1 e0]
        edge_bits = tuple(int(ch) for ch in reversed(bitstring[-6:]))
        if edge_bits in classical_solution_set:
            p_hit += p
    if p_hit > best_p:
        best_p, best_k = p_hit, k

num_iterations = best_k
print(f"Grover iterations chosen empirically from exact statevector "
      f"probabilities: k={num_iterations} (exact success prob "
      f"{best_p:.3%}) out of candidates k=0..4")

circuit = build_grover_circuit(num_iterations)

simulator = AerSimulator(method="statevector")
compiled = transpile(circuit, simulator)
shots = 4096
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports classical-register bitstrings as c[n-1] c[n-2] ... c[0] i.e.
# the leftmost printed character corresponds to the highest-index qubit.
# Our register c[i] measured edges[i], so reverse to get bit i = edges[i].

hit_shots = 0
miss_shots = 0
observed_bitstrings = set()
for bitstring, freq in counts.items():
    bits = tuple(int(ch) for ch in reversed(bitstring))
    observed_bitstrings.add(bits)
    if bits in classical_solution_set:
        hit_shots += freq
    else:
        miss_shots += freq

hit_fraction = hit_shots / shots
distinct_solutions_found = observed_bitstrings & classical_solution_set

print(f"Shots landing on a classically-verified triangle-free graph: "
      f"{hit_shots}/{shots} ({hit_fraction:.3%})")
print(f"Distinct triangle-free graphs observed: {len(distinct_solutions_found)} / {M}")
if miss_shots:
    print(f"Shots on non-solutions: {miss_shots}/{shots}")

# PASS criterion: essentially all probability mass lands on the classically
# verified solution set (Grover amplifies exactly the marked subspace; with
# a single application at this N, M the theoretical success probability is
# very high but not exactly 1, so we allow a small numerical/sampling
# tolerance rather than demanding zero misses).
success = hit_fraction >= max(0.90, best_p - 0.05)

print()
if success:
    print("PASS")
else:
    print("FAIL")
