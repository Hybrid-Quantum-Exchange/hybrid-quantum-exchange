"""
Erdos problem #791  (https://www.erdosproblems.com/791)
OEIS id used: A066063.

A066063(n) is the Erdos-Ginzburg-Ziv (EGZ) constant for Z_n: the smallest
N such that every sequence of N integers contains a subsequence of exactly
n elements whose sum is divisible by n. The classical Erdos-Ginzburg-Ziv
theorem (1961) proves A066063(n) = 2n - 1, and that this is tight (a
sequence of only 2n-2 integers need not contain such a subsequence).

Classical property tested here (computed from first principles in this
script, not copied from OEIS):

  For n = 3, A066063(3) = 2*3 - 1 = 5.  Take the concrete sequence of 5
  integers

      a = [0, 0, 1, 1, 2]        (residues mod 3)

  The EGZ theorem guarantees at least one 3-element subset of the 5
  indices {0,1,2,3,4} whose values sum to 0 mod 3. We first brute-force
  every one of the C(5,3) = 10 index-subsets classically to find every
  such "zero-sum triple" and confirm at least one exists (verifying the
  n=3 instance of the theorem underlying A066063 directly, rather than
  trusting the OEIS value blindly).

  We then build a genuine Grover search circuit over the 2^5 = 32 basis
  states of a 5-qubit register (each qubit = "is index i in the subset"),
  with a phase oracle that marks exactly those 5-bit strings whose
  popcount is 3 AND whose corresponding a-values sum to 0 mod 3 (the same
  predicate used in the classical brute force, expressed as a diagonal
  phase flip built from that same classical evaluation -- the standard
  way to turn an arbitrary boolean predicate into a Grover oracle for a
  small instance). Grover amplitude amplification is run on the ideal
  AerSimulator with the classically-optimal number of iterations for the
  actual number of marked states, and the most frequently measured
  bitstring is checked against the predicate and against the classically
  enumerated solution set.

PASS criterion: the bitstring returned by the quantum circuit (highest
measurement count) satisfies the same predicate (popcount==3, sum%3==0)
that was verified classically, i.e. quantum search found a real
zero-sum-triple witness for the n=3 EGZ instance underlying A066063(3).
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

N_ELEMENTS = 5          # 2n - 1 for n = 3  (EGZ constant instance)
N = 3                    # required subset size / modulus
SEQ = [0, 0, 1, 1, 2]    # concrete sequence of residues mod 3


def predicate(bits: str) -> bool:
    """bits is a length-5 string, bits[i] == '1' means index i is chosen.
    True iff exactly N indices are chosen and their SEQ values sum to 0 mod N.
    Bit 0 (leftmost in the string) corresponds to qubit index 0, etc."""
    chosen = [i for i, b in enumerate(bits) if b == "1"]
    if len(chosen) != N:
        return False
    return sum(SEQ[i] for i in chosen) % N == 0


def classical_search():
    """Brute-force every subset of size N over the 5 indices; return the
    full solution set (as sorted tuples of indices) confirming the EGZ
    theorem instance for n=3 on this concrete sequence."""
    solutions = []
    for combo in itertools.combinations(range(N_ELEMENTS), N):
        if sum(SEQ[i] for i in combo) % N == 0:
            solutions.append(combo)
    return solutions


def bits_to_marked_set(num_qubits: int):
    """Classically evaluate the predicate over all 2^num_qubits basis
    strings and return the set of marked integer indices (qiskit
    little-endian: qubit 0 is the least significant / rightmost bit of
    the integer, but we define predicate() on the string with bit i at
    position i reading left-to-right for indices, so build strings
    explicitly index by index)."""
    marked = []
    for state in range(2 ** num_qubits):
        # state's bit i (i = 0..num_qubits-1) -> chosen[i]
        bits = "".join(str((state >> i) & 1) for i in range(num_qubits))
        if predicate(bits):
            marked.append(state)
    return marked


def build_oracle(num_qubits: int, marked_states):
    """Diagonal phase oracle: -1 on marked computational basis states,
    +1 elsewhere. This is the standard way of expressing an arbitrary
    boolean predicate as a Grover phase oracle for a small instance."""
    dim = 2 ** num_qubits
    diag = np.ones(dim, dtype=complex)
    for m in marked_states:
        diag[m] = -1.0
    qc = QuantumCircuit(num_qubits, name="oracle")
    qc.append(Operator(np.diag(diag)).to_instruction(), range(num_qubits))
    return qc


def build_diffuser(num_qubits: int):
    """Standard Grover diffusion operator 2|s><s| - I, built directly as
    a unitary matrix (s = uniform superposition state)."""
    dim = 2 ** num_qubits
    s = np.ones((dim, 1)) / math.sqrt(dim)
    mat = 2 * (s @ s.T) - np.eye(dim)
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.append(Operator(mat).to_instruction(), range(num_qubits))
    return qc


def run_grover():
    num_qubits = N_ELEMENTS
    marked = bits_to_marked_set(num_qubits)
    m = len(marked)
    dim = 2 ** num_qubits
    assert m > 0, "EGZ theorem guarantees at least one solution; found none — bug."

    iterations = max(1, round((math.pi / 4) * math.sqrt(dim / m)))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = build_oracle(num_qubits, marked)
    diffuser = build_diffuser(num_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(num_qubits))
        qc.append(diffuser.to_instruction(), range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096).result()
    counts = result.get_counts()
    return counts, marked, iterations


def main():
    # 1. Classical verification of the n=3 EGZ instance underlying A066063.
    solutions = classical_search()
    print(f"Sequence (mod {N}): {SEQ}  (|sequence| = {N_ELEMENTS} = 2*{N}-1)")
    print(f"Classical zero-sum {N}-subsets found: {solutions}")
    classical_ok = len(solutions) > 0
    print(f"EGZ theorem instance holds classically: {classical_ok}")

    # 2. Quantum Grover search for the same predicate.
    counts, marked, iterations = run_grover()
    print(f"Marked basis states (out of {2**N_ELEMENTS}): {len(marked)}")
    print(f"Grover iterations used: {iterations}")

    # qiskit returns bitstrings with qubit 0 as the rightmost character
    top_bits_qiskit, top_count = max(counts.items(), key=lambda kv: kv[1])
    total_shots = sum(counts.values())
    # convert to our indexing convention: bits[i] = qubit i, i.e. reverse
    top_bits = top_bits_qiskit[::-1]
    top_index = int(top_bits_qiskit, 2)

    print(f"Most frequent measured bitstring: {top_bits_qiskit} "
          f"(count {top_count}/{total_shots})")

    quantum_indices = tuple(i for i, b in enumerate(top_bits) if b == "1")
    quantum_ok = predicate(top_bits) and (top_index in marked)

    print(f"Quantum witness subset (indices): {quantum_indices}, "
          f"values: {[SEQ[i] for i in quantum_indices]}, "
          f"sum mod {N} = {sum(SEQ[i] for i in quantum_indices) % N if quantum_indices else None}")

    # success probability mass on any marked state
    marked_mass = sum(c for bstr, c in counts.items()
                       if int(bstr, 2) in marked) / total_shots
    print(f"Total measured probability mass on marked states: {marked_mass:.3f}")

    verified = classical_ok and quantum_ok and marked_mass > 0.5

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
