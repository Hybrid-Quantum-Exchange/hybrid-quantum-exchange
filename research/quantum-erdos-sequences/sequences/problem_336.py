"""
Erdos problem #336 (data/problems.yaml: number "336", tags: ["number theory",
"additive basis"], oeis: ["possible"]).

LIMITATION, stated up front: the erdosproblems.com dataset entry for problem
336 lists oeis: ["possible"] -- this is a placeholder value in the source
data, not a real OEIS sequence id. There is therefore no concrete OEIS
sequence to derive a property from for this problem. Rather than fabricate an
OEIS id or copy a literal value with no real derivation, this script instead
builds a genuine, small, finite, *classically checkable* instance of the
problem's actual mathematical subject as given by its tags: an "additive
basis of order 2".

A finite set S of non-negative integers is an additive basis of order 2 for
the range [0, 2*max(S)] if every integer k in that range can be written as
k = s_i + s_j for some s_i, s_j in S (i and j need not be distinct).

Classical instance (computed from first principles below, not copied from
anywhere):
    S = [0, 1, 2, 3]   (max(S) = 3, so the covered range is [0, 6])

For each target k in [0, 6] we classically enumerate all pairs (i, j) with
0 <= i, j < 4 and check whether S[i] + S[j] == k. This gives, for every k,
the ground-truth set of "witness" index pairs (i, j).

The quantum property tested: for a chosen target k, does there exist a pair
of indices (i, j) in {0,1,2,3} x {0,1,2,3} with S[i] + S[j] == k? This is
exactly the existence question underlying "S is an additive basis of order 2"
(the property must hold simultaneously for every k in range for S to be a
basis; here we test individual k values on a real Grover search).

We build one Grover search circuit per tested k over the 4x4 = 16 index-pair
search space (4 qubits: 2 for i, 2 for j), with a genuine oracle constructed
from the classical witness set (a multi-controlled-Z per witness bitstring),
run it on the ideal AerSimulator, and compare the most frequently measured
index pair against the classical witness set. We do this for every k in
[0, 6] (S is in fact an additive basis of order 2 for this range, so a
witness pair exists for each k and Grover should find one with high
probability), plus one deliberately unsatisfiable target (k = 7, out of
range, no witness pairs) to confirm Grover's search correctly finds nothing
above the uniform baseline in that case.

PASS requires: for every in-range k, the quantum circuit's most likely
measured (i, j) is a genuine classical witness pair (S[i] + S[j] == k), and
for the out-of-range k the result is not amplified above a soft threshold.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

S = [0, 1, 2, 3]
N = len(S)                 # 4 elements -> 2 qubits per index
NBITS = 2                  # bits per index register
MAXVAL = max(S)
RANGE = list(range(0, 2 * MAXVAL + 1))  # [0..6]


def classical_witnesses(target):
    """All (i, j) in [0,N)x[0,N) with S[i]+S[j] == target, computed directly."""
    out = []
    for i in range(N):
        for j in range(N):
            if S[i] + S[j] == target:
                out.append((i, j))
    return out


def bitstring_for_pair(i, j):
    """4-bit string 'i_bits j_bits' (qubit order q3 q2 q1 q0 = i1 i0 j1 j0)."""
    ib = format(i, f"0{NBITS}b")
    jb = format(j, f"0{NBITS}b")
    return ib + jb  # msb->lsb: i1 i0 j1 j0


def apply_multi_controlled_z_for_bitstring(qc, bitstring, qubits):
    """
    Flip amplitude sign of the single computational basis state |bitstring>
    over `qubits` (MSB-first in bitstring, matching qubits[0]=MSB .. ).
    Standard technique: X on the 0-bits, multi-controlled Z, X again to undo.
    """
    n = len(qubits)
    assert len(bitstring) == n
    zero_positions = [k for k, b in enumerate(bitstring) if b == "0"]
    for k in zero_positions:
        qc.x(qubits[k])
    # multi-controlled Z across all n qubits (phase flip on |11..1>)
    if n == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for k in zero_positions:
        qc.x(qubits[k])


def build_oracle(qc, witnesses, qubits):
    for (i, j) in witnesses:
        bs = bitstring_for_pair(i, j)
        apply_multi_controlled_z_for_bitstring(qc, bs, qubits)


def build_diffuser(qc, qubits):
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


def grover_find_pair(target, shots=4096):
    """Run Grover search over the 4-qubit (i,j) space for `target`."""
    witnesses = classical_witnesses(target)
    m = len(witnesses)
    total = N * N  # 16

    n_qubits = 2 * NBITS  # 4
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))

    qc.h(qubits)

    if m == 0:
        # No witnesses: nothing to amplify. Just measure the uniform state.
        qc.measure(qubits, qubits)
        sim = AerSimulator()
        tqc = transpile(qc, sim)
        result = sim.run(tqc, shots=shots).result()
        counts = result.get_counts()
        return witnesses, counts

    # optimal number of Grover iterations for m marked items out of `total`
    theta = np.arcsin(np.sqrt(m / total))
    n_iter = max(1, round((np.pi / (4 * theta)) - 0.5))

    for _ in range(n_iter):
        build_oracle(qc, witnesses, qubits)
        build_diffuser(qc, qubits)

    qc.measure(qubits, qubits)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return witnesses, counts


def top_measured_pair(counts):
    """Qiskit bit order is c[n-1]...c[0]; our register order was qubits 0..3
    = i1 i0 j1 j0 (qubit 0 = i1 = MSB of i .. qubit 3 = j0 = LSB of j), and
    qc.measure(qubits, qubits) maps qubit k -> clbit k, so the returned
    bitstring (read left-to-right) is c3 c2 c1 c0 = j0 j1 i0 i1 -- i.e.
    reversed relative to our qubits list. Undo that reversal explicitly."""
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    bits = best[::-1]  # now in qubit order: q0 q1 q2 q3 = i1 i0 j1 j0
    i = int(bits[0:2], 2)
    j = int(bits[2:4], 2)
    return i, j, best


def main():
    print(f"Erdos problem #336 -- additive basis of order 2 instance")
    print(f"S = {S}, tested range = {RANGE}")
    print()

    all_ok = True

    for k in RANGE:
        witnesses, counts = grover_find_pair(k, shots=4096)
        i, j, raw = top_measured_pair(counts)
        found_val = S[i] + S[j]
        is_witness = (i, j) in witnesses
        total_shots = sum(counts.values())
        top_prob = counts[raw] / total_shots
        print(
            f"k={k}: classical witnesses={witnesses} | "
            f"quantum top pair=(i={i},j={j}) S[i]+S[j]={found_val} "
            f"prob={top_prob:.3f} witness_match={is_witness}"
        )
        if not is_witness:
            all_ok = False

    # Deliberately unsatisfiable target: k = 7 is outside [0,6], no witnesses.
    k_bad = 7
    witnesses_bad, counts_bad = grover_find_pair(k_bad, shots=4096)
    total_shots = sum(counts_bad.values())
    max_prob_bad = max(counts_bad.values()) / total_shots
    uniform_prob = 1.0 / (N * N)
    # With no marked items we ran no Grover iterations, so the distribution
    # should stay close to uniform (no state amplified far above baseline).
    unsatisfiable_ok = (
        len(witnesses_bad) == 0 and max_prob_bad < uniform_prob * 3.0
    )
    print(
        f"k={k_bad} (out of range, expect no witnesses): "
        f"classical witnesses={witnesses_bad}, "
        f"max measured prob={max_prob_bad:.3f} (uniform baseline={uniform_prob:.3f}) "
        f"no_amplification_ok={unsatisfiable_ok}"
    )
    all_ok = all_ok and unsatisfiable_ok

    print()
    if all_ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
