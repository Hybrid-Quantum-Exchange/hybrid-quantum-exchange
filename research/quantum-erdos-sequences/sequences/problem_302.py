"""
Erdos problem #302 (erdosproblems.com/302) -- quantum-testable instance.

OEIS id used: A390395
  "a(n) is the maximum size of a subset S of {1,...,n} such that there are
  no solutions to 1/a = 1/b + 1/c for distinct a, b, c in S."
  (Egyptian-fraction / unit-fraction avoidance; tags in erdosproblems'
  problems.yaml for #302 are ["number theory", "unit fractions"], and its
  oeis field is ["A390395"].)

Classical property tested (derived and checked from first principles below,
not copied from OEIS):
  For n = 6, brute force over all triples (a,b,c) in {1,...,6} with a,b,c
  distinct shows there is EXACTLY ONE unordered triple satisfying
  1/a = 1/b + 1/c, namely {2,3,6} (1/2 = 1/3 + 1/6). Consequently the full
  set {1,...,6} is invalid (it contains that forbidden triple), and a
  6-element subset can only be made valid by deleting exactly one of the
  three offending elements {2,3,6}; deleting any of {1,4,5} leaves {2,3,6}
  intact and is still invalid. So among the 6 ways to delete a single
  element from {1,...,6}, exactly 3 (deleting 2, deleting 3, or deleting 6)
  yield a valid size-5 subset, matching a(6) = 5 from A390395's first
  fifteen terms: 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13.

Quantum circuit: this reduces the search "which single element, when
removed from {1,...,6}, leaves a valid (forbidden-triple-free) set?" to a
Grover search over a 3-qubit register m in {0,...,7} encoding the removed
element as e = m + 1 (m = 6, 7 are unused/never marked). The oracle marks
exactly the three basis states m in {1, 2, 5} (elements 2, 3, 6) as
"good" -- built with explicit X/CCZ/X sandwiches per marked bitstring, no
classical shortcuts baked in beyond that geometry. A standard Grover
diffusion operator amplifies those three marked amplitudes out of the 8
basis states (M = 3, N = 8, ~1 optimal iteration), and the circuit is run
on the ideal AerSimulator. PASS requires the three most probable measured
outcomes to be exactly the classically-derived good set {1, 2, 5}.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical derivation (first principles, no OEIS values copied in).
# ---------------------------------------------------------------------------

def forbidden_triples(n):
    """All unordered {a, b, c} subset of {1,...,n}, distinct, with
    1/a = 1/b + 1/c for some assignment of the three roles."""
    bad = set()
    for a, b, c in itertools.permutations(range(1, n + 1), 3):
        if abs((1.0 / a) - (1.0 / b) - (1.0 / c)) < 1e-12:
            bad.add(frozenset((a, b, c)))
    return bad


def is_valid_subset(subset, bad_triples):
    subset = set(subset)
    for t in bad_triples:
        if t <= subset:
            return False
    return True


def max_valid_subset_size(n, bad_triples):
    best = 0
    universe = list(range(1, n + 1))
    for r in range(n, -1, -1):
        found = False
        for combo in itertools.combinations(universe, r):
            if is_valid_subset(combo, bad_triples):
                found = True
                break
        if found:
            best = r
            break
    return best


N = 6
bad = forbidden_triples(N)
assert bad == {frozenset({2, 3, 6})}, f"unexpected forbidden triples: {bad}"

a_n = max_valid_subset_size(N, bad)
# First 15 terms of A390395 (n=1..15), used only to cross-check our own
# from-scratch computation, not as a substitute for it.
A390395_first_15 = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13]
assert a_n == A390395_first_15[N - 1] == 5, f"a({N}) computed as {a_n}, expected 5"

full_set = set(range(1, N + 1))
good_elements = sorted(
    e for e in full_set if is_valid_subset(full_set - {e}, bad)
)
assert good_elements == [2, 3, 6], f"unexpected good_elements: {good_elements}"

# Map "remove element e" -> 3-qubit basis state m = e - 1.
good_m = sorted(e - 1 for e in good_elements)
print(f"[classical] n={N}, forbidden triples={ {tuple(sorted(t)) for t in bad} }")
print(f"[classical] a({N}) = {a_n} (matches A390395 term)")
print(f"[classical] deleting one of {good_elements} from "
      f"{{1,...,{N}}} yields a valid 5-subset")
print(f"[classical] target marked basis states (m = element-1): {good_m}")


# ---------------------------------------------------------------------------
# 2. Grover search over the 3-qubit register for m in good_m.
# ---------------------------------------------------------------------------

NUM_QUBITS = 3  # encodes m in 0..7


def mark_state_oracle(qc, bits):
    """Flip the phase of |bits> (a length-NUM_QUBITS 0/1 tuple, LSB last)
    using an X-CCZ-X sandwich so every basis state other than |bits> is
    untouched."""
    flip_qubits = [i for i, b in enumerate(bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    for q in flip_qubits:
        qc.x(q)


def bits_of(m, width):
    return tuple(int(b) for b in format(m, f"0{width}b"))


def build_oracle(targets):
    qc = QuantumCircuit(NUM_QUBITS, name="oracle")
    for m in targets:
        mark_state_oracle(qc, bits_of(m, NUM_QUBITS))
    return qc


def build_diffuser():
    qc = QuantumCircuit(NUM_QUBITS, name="diffuser")
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))
    return qc


N_STATES = 2 ** NUM_QUBITS
M_MARKED = len(good_m)
# Optimal number of Grover iterations for N states, M marked.
theta = math.asin(math.sqrt(M_MARKED / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

oracle = build_oracle(good_m)
diffuser = build_diffuser()
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 20000
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-register bit string is c2 c1 c0 (MSB..LSB); convert
# back to integer m with qubit 0 as the least-significant bit.
def bitstring_to_m(bs):
    return int(bs[::-1], 2)

m_counts = {}
for bitstring, c in counts.items():
    m_counts[bitstring_to_m(bitstring)] = m_counts.get(bitstring_to_m(bitstring), 0) + c

ranked = sorted(m_counts.items(), key=lambda kv: -kv[1])
top3 = sorted(m for m, _ in ranked[:3])

print(f"[quantum] Grover iterations used: {iterations}")
print(f"[quantum] measured distribution over m (top entries): {ranked[:6]}")
print(f"[quantum] top-3 most probable m values: {top3}")

quantum_matches_classical = (top3 == good_m)

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
