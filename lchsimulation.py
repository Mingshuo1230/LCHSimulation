import numpy as np
import sympy as sp

from qiskit import QuantumCircuit, transpile
from qiskit_aer import StatevectorSimulator, UnitarySimulator

from typing import Dict, List, Tuple, Union, Optional
import cmath

import warnings

from diff_op import Operator


class LCHSimulation:

    def __init__(self, 
                 op: 'Operator'):
        
        self._op = op
        self._num_bits = op.num_bits
        self._op_dict = op.op_dict
    
    def hermitian_check(self, tol=1e-8):
        op_conjugate = self._op.op_conjugate()
        op_dict = self._op_dict
        op_dict_conjugate = op_conjugate._op_dict
        tmp_dict = {}

        for op_string, coeff in op_dict.items():
            coeff_conjugate = op_dict_conjugate[op_string]
            tmp_dict[op_string] = coeff-coeff_conjugate
        
        for op_string, coeff in tmp_dict.items():
            if np.abs(coeff) > tol:
                return False
        
        return True
    
    def simulation_circuit(self, dt: float):

        num_qubits = self._num_bits
        op_dict = self._op_dict
        self._dt = dt

        hermitian_check = self.hermitian_check()
        if not hermitian_check:
            warnings.warn('The operator is not Hermitian. The result may be incorrect.')

        qc = QuantumCircuit(num_qubits)

        for op_string, coeff in op_dict.items():
            # Listing qubits where term_op acts on non-trivially.
            qb_R = []
            qb_L = []
            qb_U = []
            qb_D = []

            for i, op_char in enumerate(reversed(op_string)):    # The "op_string" is "reversed" to fit the qiskit.
                if op_char == 'R':
                    qb_R.append(i)
                elif op_char == 'L':
                    qb_L.append(i)
                elif op_char == 'U':
                    qb_U.append(i)
                elif op_char == 'D':
                    qb_D.append(i)

            if len(qb_R) > 0:
                if len(qb_L) == 0 or qb_R[-1] > qb_L[-1]:

                    q_controls = []

                    # Rotating basis to the Bell basis (2402.18398)
                    for i in reversed(qb_R[:-1]):
                        qc.cx(qb_R[-1], i)
                        qc.x(i)
                        q_controls.append(qc.qubits[i])
                    for i in reversed(qb_L):
                        qc.cx(qb_R[-1], i)
                        q_controls.append(qc.qubits[i])
                    for i in reversed(qb_U):
                        qc.x(i)
                        q_controls.append(qc.qubits[i])
                    for i in reversed(qb_D):
                        q_controls.append(qc.qubits[i])

                    pha = cmath.phase(coeff)    # phase
                    mag = abs(coeff)            # magnitude

                    qc.p(pha, qb_R[-1])
                    qc.h(qb_R[-1])

                    qc.barrier()
                    
                    # rotation
                    if len(q_controls) > 0:
                        qc.mcrz(2*dt*mag, q_controls, qc.qubits[qb_R[-1]])
                    else:
                        qc.rz(2*dt*mag, qc.qubits[qb_R[-1]])

                    qc.barrier()

                    # uncomputation
                    qc.h(qb_R[-1])
                    qc.p(-pha, qb_R[-1])

                    for i in qb_U:
                        qc.x(i)
                    for i in qb_L:
                        qc.cx(qb_R[-1], i)
                    for i in qb_R[:-1]:
                        qc.x(i)
                        qc.cx(qb_R[-1], i)

                    qc.barrier()
                    qc.barrier()
        
        self._qc_h = qc
        return qc
    
    def state_simulation_evolve(self, T: float, state: QuantumCircuit):
        """
        Simulate the state vector of the evolution operator and compute the final state directly.
        Suitable for the large time step.
        """

        backend = StatevectorSimulator()
        num_qubits = self._num_bits
        dt = self._dt

        if state is None:
            raise ValueError('Initial state in QuantumCircuit is not provided.')
        state.save_statevector(label="i0")
        circ = self._qc_h
        w_list = []


        for i in range(int(T/dt)):
            state = state.compose(circ, inplace=False)
            state.save_statevector(label="i{}".format(i+1))
        result = backend.run(state).result().data(0)
        for i in range(int(T/dt)+1):
            w = result["i{}".format(i)].data[:2**num_qubits]
            # cancel the global phase
            w = np.exp(-1j*np.angle(w[np.argmax(np.abs(w))])) * w 
            w_list.append(w)

        w_list = np.array(w_list)
        return w_list
    
    def matrix_simulation_evolve(self, T: float, state: np.ndarray, sample_rate: int=10):
        """
        Simulate the matrix of the evolution operator and compute the final state in the vector form.
        Suitable for the small time step.
        """

        backend = UnitarySimulator()
        dt = self._dt
        # the number of time steps to sample the state vector due to the small time step
        dt_sample = sample_rate * dt

        if state is None:
            raise ValueError('Initial state in numpy array is not provided.')
        if state.shape != (2**self._num_bits,):
            raise ValueError('Initial state is not in the vector form.')
        
        circ = self._qc_h
        circ = transpile(circ, backend=backend, optimization_level=2, seed_transpiler=42)
        U_circ = backend.run(circ).result().get_unitary().data
        # cancel the global phase
        U_circ = np.exp(-1j*np.angle(U_circ[0, 0])) * U_circ 
        
        w_list = []
        tmp_state = state.copy()
        for i in range(int(T/dt)):
            
            if i % int(dt_sample / dt) == 0:
                w_list.append(tmp_state)
            
            tmp_state = U_circ @ tmp_state

        w_list.append(tmp_state)
        w_list = np.array(w_list)
        return w_list