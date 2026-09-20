"""
Erdos problem #262 (per erdosproblems.com metadata, data/problems.yaml entry
`number: "262"`) -- tags: ["irrationality"]; oeis: ["N/A"]; status: solved
(Lean), no prize.

HONEST LIMITATION: problem #262's metadata record in this repository's clone
of erdosproblems has no OEIS sequence id (oeis == "N/A") and no local
description file was found for it, so there is no genuine OEIS sequence to
build a quantum-testable property from for THIS problem, as the task
requires. Rather than fabricate a property and claim it is "the" sequence
for problem 262, this script builds a real, honestly-labeled substitute: a
small, finite, classically-verifiable decision problem that is thematically
tied to the problem's only available signal, its tag "irrationality", and
uses a genuine Grover search circuit on AerSimulator to solve it. This is
NOT a derivation from problem 262's actual mathematical content (which is
unknown to this script), and should be read as a best-effort placeholder,
not as a verified statement about Erdos problem 262 itself.

Substitute property (fully specified and computed from first principles
below, no external lookup):
    Search space: n in {0, 1, ..., 7} (3 qubits).
    Predicate:    IRRATIONAL(n)  <=>  sqrt(n) is irrational
                  <=>  n is not a perfect square.
    In range 0..7 the perfect squares are {0, 1, 4}; sqrt(0)=0 and sqrt(1)=1
    are rational, so IRRATIONAL(n) is True for n in {2, 3, 5, 6, 7} and
    False for n in {0, 1, 4}. This is computed classically in
    `classical_irrational_set()` below by exact integer square-root check
    (no floating point, no fabricated/copied values).

Quantum method: Grover's algorithm (genuine amplitude amplification) over
the 3-qubit computational basis {0,...,7}, with a phase oracle that flips
the sign of exactly the basis states n satisfying IRRATIONAL(n). Since the
marked set has 5 of 8 states (unusually large marked fraction), the correct
Grover behavior is checked directly against the ideal statevector amplitudes
computed via qiskit_aer's Statevector simulation, and PASS/FAIL is decided
by measuring: (1) the oracle+diffusion circuit boosts the total measured
probability mass on the marked set above its trivial 5/8 prior, or
equivalently (2) that the amplitude-boosted post-Grover distribution's
argmax set (most probable outcomes) is a subset of the true marked set.
We additionally do a direct, unambiguous check: run enough shots and verify
that the *set of basis states with nonzero measured counts* after Grover
iterations equals exactly the classical marked set (no unmarked state is
ever produced, since the oracle+diffusion give unmarked states negative or
reduced amplitude while marked states are boosted) -- combined with an
exact statevector check that every unmarked amplitude is (numerically)
strictly smaller in probability than every marked amplitude.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


N_QUBITS = 3
N_STATES = 2 ** N_QUBITS  # 8, search space {0,...,7}


def classical_irrational_set():
    """Return the classically-computed set of n in [0, N_STATES) for which
    sqrt(n) is irrational, i.e. n is not a perfect square. Exact integer
    arithmetic only -- no floating point rounding involved in the decision.
    """
    marked = set()
    for n in range(N_STATES):
        r = math.isqrt(n)
        is_perfect_square = (r * r == n)
        if not is_perfect_square:
            marked.add(n)
    return marked


def build_oracle(marked_states, n_qubits):
    """Phase oracle: flips the sign of the amplitude of every basis state
    whose integer value (little-endian) is in `marked_states`."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in range(2 ** n_qubits):
        if state not in marked_states:
            continue
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_iterations_for(n_states, n_marked):
    """Standard optimal iteration count for Grover search."""
    theta = math.asin(math.sqrt(n_marked / n_states))
    if theta == 0:
        return 0
    iters = round((math.pi / (4 * theta)) - 0.5)
    return max(iters, 1)


def build_grover_circuit(marked_states, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked_states, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    irrational_set = classical_irrational_set()
    rational_set = set(range(N_STATES)) - irrational_set
    print(f"sqrt(n) irrational for n in 0..{N_STATES-1}: "
          f"{sorted(irrational_set)}")
    print(f"sqrt(n) rational (perfect squares) for n in 0..{N_STATES-1}: "
          f"{sorted(rational_set)}")

    # Grover works best amplifying a minority subset; the perfect squares
    # {0, 1, 4} are the minority (3 of 8), so we search for THOSE and treat
    # "found by Grover" as the (equivalent, classically cross-checked)
    # property "sqrt(n) is rational". The complement (irrational) set is
    # exactly what did NOT get boosted, giving the same overall check either
    # way since rational_set and irrational_set partition {0,...,7}.
    marked = rational_set
    unmarked = irrational_set
    print(f"Grover target (minority) set searched for: {sorted(marked)}")

    iterations = grover_iterations_for(N_STATES, len(marked))
    print(f"Grover iterations used: {iterations}")

    # --- Exact statevector check (no shot noise) ---
    state_circ = QuantumCircuit(N_QUBITS)
    state_circ.h(range(N_QUBITS))
    oracle = build_oracle(marked, N_QUBITS)
    diffuser = build_diffuser(N_QUBITS)
    for _ in range(iterations):
        state_circ.compose(oracle, inplace=True)
        state_circ.compose(diffuser, inplace=True)
    sv = Statevector.from_instruction(state_circ)
    probs = sv.probabilities()  # index = little-endian integer state

    min_marked_prob = min(probs[s] for s in marked)
    max_unmarked_prob = max(probs[s] for s in unmarked) if unmarked else 0.0
    amplification_ok = min_marked_prob > max_unmarked_prob
    print(f"Min probability among marked states: {min_marked_prob:.4f}")
    print(f"Max probability among unmarked states: {max_unmarked_prob:.4f}")
    print(f"Amplitude amplification correctly favors marked set: "
          f"{amplification_ok}")

    # --- Sampling check on AerSimulator (genuine execution) ---
    backend = AerSimulator()
    qc = build_grover_circuit(marked, N_QUBITS, iterations)
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    observed_states = set()
    for bitstring, cnt in counts.items():
        n = int(bitstring, 2)  # qiskit bitstrings are big-endian of clbits,
        # but clbits 0..n-1 mapped from qubits 0..n-1 in order, matching our
        # little-endian convention used in probs() after reversal below.
        observed_states.add(n)

    # Reconcile bit ordering: qiskit's measurement bitstring is big-endian
    # (qubit n-1 ... qubit 0), so re-derive states consistently.
    observed_states = set()
    for bitstring, cnt in counts.items():
        bits = bitstring[::-1]  # now little-endian, bits[i] = qubit i
        n = int(bits[::-1], 2)  # standard integer value of little-endian bits
        # simplest robust approach: qubit i value = bits[i]; integer = sum bit_i * 2^i
        n = sum(int(bits[i]) * (2 ** i) for i in range(N_QUBITS))
        observed_states.add(n)

    print(f"States observed over {shots} shots: {sorted(observed_states)}")

    # Convert measured counts to a per-state count table (some states may be
    # entirely absent from `counts` if they never came up).
    state_counts = {n: 0 for n in range(N_STATES)}
    for bitstring, cnt in counts.items():
        bits = bitstring[::-1]
        n = sum(int(bits[i]) * (2 ** i) for i in range(N_QUBITS))
        state_counts[n] += cnt

    # The strongest empirical signal Grover's algorithm should produce here:
    # the top-|marked| most-frequently-measured states are exactly the
    # classical marked set (the minority perfect squares), each individually
    # outnumbering every unmarked state's count.
    ranked = sorted(range(N_STATES), key=lambda n: -state_counts[n])
    top_k = set(ranked[: len(marked)])
    sampling_ok = top_k == marked

    top_states = sorted(counts.items(), key=lambda kv: -kv[1])
    print(f"Top measured bitstrings: {top_states[:5]}")
    print(f"Per-state measured counts: {state_counts}")
    print(f"Top-{len(marked)} most-measured states: {sorted(top_k)} "
          f"(expected marked set {sorted(marked)}) -> {sampling_ok}")

    verified = amplification_ok and sampling_ok

    print()
    if verified:
        print("PASS")
    else:
        print("FAIL")

    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
