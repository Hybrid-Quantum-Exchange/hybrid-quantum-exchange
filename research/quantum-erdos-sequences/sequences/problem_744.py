"""
Erdos problem #744 -- quantum-testable lane.

Source metadata (data/problems.yaml in the erdosproblems repo, block for
"number: '744'"): prize "no", status "disproved" (2025-08-31),
oeis: ["possible"], tags: ["graph theory", "chromatic number"].

LIMITATION (read before trusting "verified"): the metadata record for
problem 744 carries no real OEIS sequence id -- the single listed value is
the literal string "possible", not an A-number. There is therefore no
genuine OEIS sequence to build a circuit against for this problem. Per the
task's own fallback instructions, this script does not fabricate an OEIS
term; instead it builds a real, honestly-verified quantum circuit for the
one piece of mathematical content the metadata *does* give us: the problem's
tags, "graph theory" and "chromatic number".

Chosen finite, computable property
-----------------------------------
Proper 2-colorability of the path graph P3 (vertices v0 - v1 - v2, edges
(v0,v1) and (v1,v2)), i.e. an instance of the general graph-coloring /
chromatic-number decision problem the tags name. A coloring is an
assignment of one bit (of 2 possible colors) to each vertex; it is proper
iff every edge joins differently-colored vertices:

    proper(v0,v1,v2)  <=>  (v0 XOR v1 == 1)  AND  (v1 XOR v2 == 1)

The classical answer (brute force over all 2**3 = 8 assignments, computed
in this script from first principles, no OEIS lookup) is exactly two
proper colorings: (v0,v1,v2) = (1,0,1) and (0,1,0). This matches the known
fact that a path graph is bipartite (chromatic number 2) and has exactly
2 proper 2-colorings when the two color classes are forced to alternate
between two fixed color labels.

Quantum circuit
----------------
A Grover search over the 3-qubit assignment space (v0,v1,v2), with an
oracle built from CNOTs and CCX gates that marks exactly the two proper
colorings identified above, amplified by the standard Grover diffusion
operator. With N=8 states and M=2 marked states, one Grover iteration
(the optimal integer round(pi/4 * sqrt(N/M)) = round(pi/4*2) = 2, and
with M=2 out of N=8, floor(pi/4*sqrt(4))=1 iteration is near-optimal) is
run, and the resulting measurement distribution is checked to concentrate
its probability mass on the two classically-valid colorings.

This is run on the ideal AerSimulator (statevector-exact, no shot noise in
the amplitude comparison; sampling is only used for the final PASS/FAIL
measurement check).
"""

import itertools

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def classical_proper_colorings():
    """Brute-force every 2-coloring of the path graph v0-v1-v2 (P3)."""
    edges = [(0, 1), (1, 2)]
    valid = []
    for bits in itertools.product([0, 1], repeat=3):
        if all(bits[a] != bits[b] for a, b in edges):
            valid.append(bits)
    return valid


def build_oracle(qc, qubits, ancilla):
    """Mark states satisfying (v0 XOR v1) AND (v1 XOR v2) with a phase flip.

    qubits = [v0, v1, v2] data qubits; ancilla is a helper qubit prepared
    in the |-> state by the caller so a controlled-X on it applies a
    phase kickback (standard oracle-via-ancilla trick).
    """
    v0, v1, v2 = qubits
    # temp0 = v0 XOR v1 ; temp1 = v1 XOR v2  -- compute into two fresh
    # scratch qubits appended after the ancilla in the register.
    t0, t1 = ancilla[1], ancilla[2]

    qc.cx(v0, t0)
    qc.cx(v1, t0)
    qc.cx(v1, t1)
    qc.cx(v2, t1)

    # Flip the phase-kickback ancilla iff t0 AND t1 are both 1.
    qc.ccx(t0, t1, ancilla[0])

    # uncompute
    qc.cx(v1, t1)
    qc.cx(v2, t1)
    qc.cx(v0, t0)
    qc.cx(v1, t0)


def build_diffuser(qc, qubits):
    n = len(qubits)
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def make_registers():
    data = QuantumRegister(3, "v")
    scratch = QuantumRegister(3, "s")  # s[0] = phase-kickback ancilla, s[1:3] = temps
    return data, scratch


def run_grover():
    data, scratch = make_registers()
    qc = QuantumCircuit(data, scratch)

    # init data qubits in uniform superposition
    qc.h(data)

    # prepare phase-kickback ancilla in |->
    qc.x(scratch[0])
    qc.h(scratch[0])

    iterations = 1  # near-optimal for N=8, M=2 (see docstring)
    for _ in range(iterations):
        build_oracle(qc, list(data), list(scratch))
        build_diffuser(qc, list(data))

    # uncompute the ancilla back to |0> (not strictly required for
    # measurement of data qubits alone, but keeps the circuit clean)
    qc.h(scratch[0])
    qc.x(scratch[0])

    qc.measure_all()
    return qc, data


def main():
    valid = classical_proper_colorings()
    valid_set = set(valid)
    print(f"Classical brute-force proper 2-colorings of P3: {valid}")
    assert valid_set == {(1, 0, 1), (0, 1, 0)}, "classical computation disagrees with derivation"

    qc, data = run_grover()

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Aggregate measured probability mass landing on the 3 data qubits
    # (v register), ignoring the scratch/ancilla qubits which should be
    # back at 0 for every shot since they're uncomputed.
    # Qiskit's measure_all with multiple registers reports a combined
    # bitstring "s2 s1 s0 v2 v1 v0" (reverse register order, each
    # register itself reversed); recover the 3 v-bits robustly by using
    # the classical register mapping from qc instead of hardcoding widths.
    creg_bits = qc.clbits
    # Identify which classical bits correspond to the `data` (v) qubits.
    v_qubit_set = set(data)
    qubit_to_clbit = {}
    for instr in qc.data:
        if instr.operation.name == "measure":
            qubit_to_clbit[instr.qubits[0]] = instr.clbits[0]
    v_clbit_indices = sorted(
        creg_bits.index(qubit_to_clbit[q]) for q in data
    )

    mass_on_valid = 0
    mass_total = 0
    per_coloring_counts = {c: 0 for c in valid}
    for bitstring, n in counts.items():
        # Qiskit prints classical bits MSB-first (clbit[-1] ... clbit[0]).
        bits = bitstring.replace(" ", "")[::-1]  # now index i == clbit i
        v_bits = tuple(int(bits[i]) for i in v_clbit_indices)  # (v0,v1,v2)
        mass_total += n
        if v_bits in valid_set:
            mass_on_valid += n
            per_coloring_counts[v_bits] = per_coloring_counts.get(v_bits, 0) + n

    frac_valid = mass_on_valid / mass_total
    print(f"Measured shots: {mass_total}, on valid colorings: {mass_on_valid} "
          f"({frac_valid:.3f})")
    print(f"Per-valid-coloring counts: {per_coloring_counts}")

    # Also do an exact (shot-free) statevector check as the primary
    # verification: after the Grover circuit, amplitude on the 2 marked
    # basis states (with scratch qubits restored to |0>) should exceed
    # the uniform baseline of 2/8 = 0.25 by a clear margin.
    data, scratch = make_registers()
    qc_sv = QuantumCircuit(data, scratch)
    qc_sv.h(data)
    qc_sv.x(scratch[0])
    qc_sv.h(scratch[0])
    build_oracle(qc_sv, list(data), list(scratch))
    build_diffuser(qc_sv, list(data))
    qc_sv.h(scratch[0])
    qc_sv.x(scratch[0])

    sv = Statevector.from_instruction(qc_sv)
    probs = sv.probabilities_dict(qargs=[qc_sv.qubits.index(q) for q in data])
    exact_valid_prob = sum(
        p for bits, p in probs.items()
        if tuple(int(b) for b in bits[::-1]) in valid_set
    )
    print(f"Exact (statevector) probability mass on valid colorings: "
          f"{exact_valid_prob:.4f} (uniform baseline would be {2/8:.4f})")

    baseline = 2 / 8
    ok = (exact_valid_prob > baseline + 0.15) and (frac_valid > baseline + 0.10)

    if ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
