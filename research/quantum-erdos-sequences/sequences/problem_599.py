"""
Erdos problem #599 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com mirror):
    number: "599"
    comments: "Erdos-Menger conjecture"
    tags: ["graph theory", "set theory"]
    oeis: ["N/A"]
    status: proved

LIMITATION, stated up front: problem #599 carries no OEIS sequence id
("oeis": ["N/A"] in the source data). There is therefore no "sequence" to
search membership in or derive a term from. This is not a numerical
Erdos problem at all -- it is the Erdos-Menger conjecture, the statement
that Menger's theorem (max number of internally-disjoint s-t paths in a
graph equals the minimum size of an s-t vertex cut) extends to infinite
graphs. Since no OEIS-backed integer sequence exists for this problem,
this script's "best honest attempt" is to build a genuinely finite,
computable instance of the *finite* Menger's theorem itself (which is
the un-controversial, classical, finite case the conjecture generalizes)
and verify it with a real quantum circuit, rather than fabricate an
OEIS value that doesn't exist.

Classical property tested
--------------------------
Fix a small finite graph G with distinguished vertices s and t, and an
internal vertex set V_mid of size 3 (so there are 2**3 = 8 candidate
"vertex cut" subsets of V_mid, including the empty subset).

Classically (computed from first principles in this script, no
hard-coded OEIS value):
  1. Enumerate all 8 subsets of V_mid. For each, remove those vertices
     from G and BFS-check whether s and t are still connected. A
     nonempty subset whose removal disconnects s from t is a "vertex
     cut". Record the minimum cut size, min_cut, and the set of
     bitstrings achieving it, MARKED.
  2. Independently enumerate all internally-vertex-disjoint s-t paths
     (via V_mid) by brute force and find the maximum number of them,
     max_disjoint_paths.
  3. Menger's theorem (finite case) asserts min_cut == max_disjoint_paths.
     This script asserts that equality classically as a sanity check
     before ever touching the quantum circuit.

Quantum computation
--------------------
A 3-qubit Grover search is built whose oracle marks exactly the
bitstrings in MARKED (the minimum-size vertex cuts found classically).
The circuit is run on the ideal AerSimulator. The most-frequently
measured bitstring is compared against MARKED: if the quantum search
finds a genuine minimum vertex cut, and that cut's size matches
max_disjoint_paths (Menger's equality, verified classically), the
script prints PASS.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import itertools
from collections import deque

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Define a small finite graph with s, t and 3 "cuttable" internal vertices.
# ---------------------------------------------------------------------------
# Vertices: s=0, t=4, internal = 1, 2, 3
# Edges chosen so that the finite vertex-connectivity between s and t is 2:
#   two vertex-disjoint paths exist: s-1-t and s-2-3-t
#   and there is no way to disconnect s,t by removing fewer than 2 of {1,2,3}.
S, T = 0, 4
V_MID = [1, 2, 3]
EDGES = [
    (S, 1), (1, T),
    (S, 2), (2, 3), (3, T),
    (1, 3),  # extra edge, does not create a 3rd fully disjoint path
]

ADJ = {v: set() for v in [S, T] + V_MID}
for a, b in EDGES:
    ADJ[a].add(b)
    ADJ[b].add(a)


def connected_after_removal(removed: set[int]) -> bool:
    """BFS from S to T in the graph with `removed` vertices deleted."""
    if S in removed or T in removed:
        return False
    seen = {S}
    q = deque([S])
    while q:
        u = q.popleft()
        if u == T:
            return True
        for w in ADJ[u]:
            if w in removed or w in seen:
                continue
            seen.add(w)
            q.append(w)
    return T in seen


# ---------------------------------------------------------------------------
# 2. Classical brute force: minimum vertex cut over subsets of V_MID.
# ---------------------------------------------------------------------------
n = len(V_MID)  # 3 -> 8 subsets, qubit index i controls V_MID[i]
subset_is_cut = {}
for bits in itertools.product([0, 1], repeat=n):
    removed = {V_MID[i] for i, b in enumerate(bits) if b == 1}
    is_cut = (len(removed) > 0) and (not connected_after_removal(removed))
    subset_is_cut[bits] = (is_cut, len(removed))

cut_sizes = [size for (is_cut, size) in subset_is_cut.values() if is_cut]
if not cut_sizes:
    raise RuntimeError("graph construction error: no vertex cut exists in V_MID")
min_cut = min(cut_sizes)

# bitstrings achieving the minimum cut (Grover's marked set)
MARKED = [bits for bits, (is_cut, size) in subset_is_cut.items()
          if is_cut and size == min_cut]

# ---------------------------------------------------------------------------
# 3. Classical brute force: maximum number of internally-vertex-disjoint
#    s-t paths (paths may only use S, T, and vertices of V_MID).
# ---------------------------------------------------------------------------
def all_simple_paths(start: int, end: int, allowed: set[int]) -> list[list[int]]:
    paths = []
    def dfs(u, visited, path):
        if u == end:
            paths.append(list(path))
            return
        for w in ADJ[u]:
            if w == end:
                dfs(end, visited, path + [end])
            elif w in allowed and w not in visited:
                dfs(w, visited | {w}, path + [w])
    dfs(start, {start}, [start])
    return paths


candidate_paths = all_simple_paths(S, T, set(V_MID))

max_disjoint_paths = 0
best_combo = None
for r in range(len(candidate_paths), 0, -1):
    found = False
    for combo in itertools.combinations(candidate_paths, r):
        internal_sets = [set(p[1:-1]) for p in combo]
        if all(internal_sets[i].isdisjoint(internal_sets[j])
               for i in range(len(internal_sets))
               for j in range(i + 1, len(internal_sets))):
            max_disjoint_paths = r
            best_combo = combo
            found = True
            break
    if found:
        break

# Menger's theorem (finite case) sanity check, computed purely classically.
menger_holds = (min_cut == max_disjoint_paths)

print(f"Graph: S={S}, T={T}, V_mid={V_MID}, edges={EDGES}")
print(f"Classical min vertex cut size: {min_cut}, achieved by (bit order = V_mid {V_MID}): {MARKED}")
print(f"Classical max disjoint paths: {max_disjoint_paths} via {best_combo}")
print(f"Menger's theorem (finite case) holds on this instance: {menger_holds}")

if not menger_holds:
    raise RuntimeError("classical Menger check failed -- instance is broken")


# ---------------------------------------------------------------------------
# 4. Grover search over the 3-qubit subset space for a minimum vertex cut.
# ---------------------------------------------------------------------------
def build_oracle(n_qubits: int, marked_bitstrings: list[tuple[int, ...]]) -> QuantumCircuit:
    """Phase-flip oracle marking each bitstring in marked_bitstrings.
    Bit order: qubit i corresponds to marked_bitstrings[*][i], with
    Qiskit's little-endian convention handled via explicit X placement.
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bits in marked_bitstrings:
        # Flip qubits that should be 0 so the target state becomes |11..1>
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


n_qubits = n  # 3
oracle = build_oracle(n_qubits, MARKED)
diffuser = build_diffuser(n_qubits)

# Optimal number of Grover iterations for M marked out of N=2**n states.
N = 2 ** n_qubits
M = len(MARKED)
theta = np.arcsin(np.sqrt(M / N))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(n_qubits))
    qc.append(diffuser.to_gate(), range(n_qubits))
qc.measure(range(n_qubits), range(n_qubits))

sim = AerSimulator()
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=4096).result()
counts = result.get_counts()

# Qiskit reports bitstrings as c[n-1]...c[0]; our qubit i is bit position i,
# so reverse the string to read qubit-0-first order matching MARKED tuples.
def bitstring_to_tuple(bs: str) -> tuple[int, ...]:
    rev = bs[::-1]
    return tuple(int(c) for c in rev)

best_measured = max(counts, key=counts.get)
best_tuple = bitstring_to_tuple(best_measured)
marked_set = set(MARKED)

print(f"Grover iterations used: {iterations}, marked count M={M}, space N={N}")
print(f"Most frequent measured bitstring (qubit0..qubit{n_qubits-1}): {best_tuple}, "
      f"counts={counts[best_measured]}/4096")

quantum_found_min_cut = best_tuple in marked_set
quantum_cut_size = sum(best_tuple)
sizes_match_menger = (quantum_cut_size == max_disjoint_paths == min_cut)

ok = menger_holds and quantum_found_min_cut and sizes_match_menger

if ok:
    print("PASS")
else:
    print("FAIL")
    print(f"  quantum_found_min_cut={quantum_found_min_cut}, "
          f"quantum_cut_size={quantum_cut_size}, min_cut={min_cut}, "
          f"max_disjoint_paths={max_disjoint_paths}")
