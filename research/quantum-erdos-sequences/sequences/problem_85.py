"""
Erdos problem #85  (erdosproblems.com, Ramsey theory / graph theory)
OEIS: A006672  ("possible" match, per data/problems.yaml)

A006672(n) is the Ramsey-type number r(C4, K_{1,n}): the least m such that
every red/blue edge-colouring of the complete graph K_m contains either a
red 4-cycle (C4) or a blue star K_{1,n} (a vertex with n blue edges).
Known initial terms (OEIS): a(1)=4, a(2)=4, a(3)=6, a(4)=7, ...

Classical property tested here (derived and checked in this script, not
copied from OEIS):

    Take n = 1, so K_{1,1} is just a single blue edge, and m = 3
    (triangle K3). Does there exist a red/blue colouring of the 3 edges
    of K3 that has

        * no red C4   (automatically true: K3 only has 3 vertices, so it
                        cannot contain a 4-cycle at all), and
        * no blue edge at all (a single blue edge would already be a
                        blue K_{1,1})

    Since a(1) = 4 > 3, such an "avoiding" colouring of K3 MUST exist
    (m=3 is too small to force the Ramsey property r(C4,K_{1,1})=4): the
    all-red colouring of K3 has no C4 (too few vertices) and no blue
    edge, and it is the *unique* such colouring, since any other
    colouring has at least one blue edge by definition.

    Classical answer (brute force over the 8 possible edge-colourings of
    K3, computed below): exactly one colouring is "good": the all-red
    colouring (0, 0, 0), i.e. Hamming weight 0.

Quantum circuit: an exact Grover search over the 3-qubit space of
edge-colourings of K3 (2^3 = 8 basis states), whose oracle marks
precisely that one "good" colouring. The oracle is built as an explicit
diagonal {+1,-1} phase-flip unitary constructed directly from the
classically computed marked state (not hard-coded from any OEIS value),
and the standard Grover diffusion operator amplifies it. With N=8 and a
single marked item (M=1), one Grover iteration is close to optimal and
drives the success probability from 1/8 up to ~0.945 on the ideal
simulator. The script runs the circuit on AerSimulator and checks that
the search overwhelmingly (and correctly) returns the all-red colouring
000, matching the classically brute-forced unique answer.
"""

import itertools

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

N_EDGES = 3  # edges of the triangle K3


def is_good_colouring(bits):
    """bits: tuple of 0/1 of length 3, bit=1 means that edge is blue.

    A colouring is 'good' (avoids red C4 and blue K_{1,1}) iff:
      - K3 has no C4 at all (trivially true, so the red-C4 condition is
        vacuous for m=3), and
      - there is no blue edge at all (a single blue edge is already a
        blue K_{1,1}), i.e. all 3 edges are red.
    """
    return sum(bits) == 0


def classical_brute_force():
    all_colourings = list(itertools.product([0, 1], repeat=N_EDGES))
    good = [c for c in all_colourings if is_good_colouring(c)]
    return all_colourings, good


def build_grover_circuit(marked_indices, n_qubits):
    dim = 2 ** n_qubits

    # Phase oracle: diagonal unitary flipping the sign of marked basis states.
    diag = np.ones(dim, dtype=complex)
    for idx in marked_indices:
        diag[idx] = -1.0
    oracle_gate = Operator(np.diag(diag)).to_instruction()
    oracle_gate.label = "Oracle"

    # Diffusion operator: 2|s><s| - I, built from H^n, X^n, multi-controlled Z, X^n, H^n.
    diffusion = QuantumCircuit(n_qubits, name="Diffusion")
    diffusion.h(range(n_qubits))
    diffusion.x(range(n_qubits))
    diffusion.h(n_qubits - 1)
    diffusion.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffusion.h(n_qubits - 1)
    diffusion.x(range(n_qubits))
    diffusion.h(range(n_qubits))
    diffusion_gate = diffusion.to_instruction()

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    M = len(marked_indices)
    # Optimal number of Grover iterations for N=dim, M marked items.
    theta = np.arcsin(np.sqrt(M / dim))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    for _ in range(iterations):
        qc.append(oracle_gate, range(n_qubits))
        qc.append(diffusion_gate, range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


def bits_to_int(bits):
    # Qiskit little-endian: qubit 0 is the least-significant bit of the
    # measured bitstring integer, matching the classical tuple order
    # (bits[0] -> edge 0 -> qubit 0).
    val = 0
    for i, b in enumerate(bits):
        val |= (b << i)
    return val


def main():
    all_colourings, good = classical_brute_force()
    print(f"Classical brute force over {len(all_colourings)} colourings of K3's 3 edges:")
    print(f"  good (avoid red C4 and blue K_1,1) colourings: {good}")
    assert good == [(0, 0, 0)]
    marked_indices = sorted(bits_to_int(c) for c in good)
    print(f"  marked basis-state indices: {marked_indices}")

    qc, iterations = build_grover_circuit(marked_indices, N_EDGES)
    print(f"Grover circuit built with {iterations} iteration(s) for N={2**N_EDGES}, M={len(marked_indices)}.")

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    print("Measurement counts:", counts)

    # Grover amplification is probabilistic even on an ideal simulator: a
    # single shot occasionally lands off the marked state, but the marked
    # state's probability should be strongly amplified above the uniform
    # baseline of 1/8 = 0.125. Check that (a) the single most-frequent
    # outcome is exactly the classically-computed marked state, and (b) its
    # measured probability clears a generous threshold well above baseline.
    most_common_bitstring, most_common_count = max(counts.items(), key=lambda kv: kv[1])
    most_common_idx = int(most_common_bitstring, 2)
    marked_probability = most_common_count / shots

    correct_outcome = most_common_idx == marked_indices[0]
    strongly_amplified = marked_probability > 0.8  # baseline (no Grover) would be 0.125

    verified = correct_outcome and strongly_amplified

    print()
    print(f"Classical good (unique) index: {marked_indices[0]} (bitstring {format(marked_indices[0], f'0{N_EDGES}b')})")
    print(f"Quantum most-frequent index:   {most_common_idx} (bitstring {most_common_bitstring}), "
          f"probability {marked_probability:.4f} (uniform baseline 0.125)")
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
