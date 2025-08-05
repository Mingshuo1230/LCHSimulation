import numpy as np
import sympy as sp

from typing import Dict, List, Tuple, Union, Optional

import scipy
import cmath

# The Operator class is the base class for all difference operators and will be used to build the Hamiltonian
# The basic idea is to decompose the operator into a dictionary of operator strings and their coefficients

class Operator():
    
    def __init__(self, 
                num_bits: int,
                op_dict: dict[str, float] = {}):
    
        self._num_bits = num_bits
        self._op_dict = op_dict

    @property
    def num_bits(self):
        return self._num_bits
    
    @property
    def op_dict(self):
        return self._op_dict

    # fundamental shift operator: element to build other operators
    def shift_operator(self, direction: str='forward'):
        s10, s01, s00, s11, I = sp.symbols('L R U D I', commutative=False)

        if direction == 'forward':
            # matrix to build the shift operator
            Cx = sp.Matrix([
                [I, s01],
                [0, s10,]
            ])
        
        elif direction == 'backward':
            # forward shift operator is the conjugate of the backward one
            Cx = sp.Matrix([
                [I, s10],
                [0, s01],
            ])

        else:
            raise NotImplementedError()

        # extract [1,2]th element in the matrix
        C_left = sp.Matrix([1, 0]).T
        C_right = sp.Matrix([0, 1])

        operator = C_left * Cx**self._num_bits * C_right
        # extract the element from the matrix
        operator = sp.expand(operator[0])
        return operator
    
    # decompose the operator into a dictionary of operators and their coefficients
    def op_decompose(self, operator: sp.Expr)->Dict[str, float]:

        op_dict = {}
        # split into terms
        terms = operator.args if operator.is_Add else [operator]

        for term in terms:
            # initialize numeric coefficient
            numeric_coeff = 1.0
            # collect the “pure” symbolic factors
            sym_factors = []

            # get the raw Mul factors (or a singleton list)
            factors = term.args if term.is_Mul else [term]

            for f in factors:
                if f.is_number:
                    # pull _all_ numeric pieces into numeric_coeff
                    numeric_coeff *= float(f)
                elif f.is_Pow and isinstance(f.base, sp.Symbol):
                    # e.g. I**2 → two I’s
                    sym_factors.extend([f.base.name] * int(f.exp))
                elif isinstance(f, sp.Symbol):
                    # a lone symbol
                    sym_factors.append(f.name)
                else:
                    # raise error for unsupported factors
                    raise ValueError(f"Unsupported factor: {f}")

            # build your operator‐string key
            op_string = ''.join(sym_factors)

            # accumulate into the dict
            op_dict[op_string] = op_dict.get(op_string, 0.0) + numeric_coeff
        
        self._op_dict = op_dict
        return op_dict

    # convert the operator to a matrix
    def op2matrix(self)->np.ndarray:

        I = np.eye(2)

        s01 = np.array([[0, 1], [0, 0]])
        s10 = np.array([[0, 0], [1, 0]])

        s00 = np.array([[1, 0], [0, 0]])
        s11 = np.array([[0, 0], [0, 1]])

        matrix_dict = {
            'I': I,      # identity operator
            'R': s01,    # rasing ladder operator
            'L': s10,    # lowering ladder operator
            'U': s00,    # up projector
            'D': s11     # down projector
        }

        mat = np.zeros((2**self._num_bits, 2**self._num_bits), dtype=np.complex128)

        for op_string, coeff in self._op_dict.items():
            base = 1
            for op_char in op_string:
                base = np.kron(base, matrix_dict[op_char])
            mat += coeff * base

        self._matrix = mat
        return mat

    def tensor_product(self, other: 'Operator')->'Operator':

        # check if the operators are decomposed
        if self._op_dict == {} or other._op_dict == {}:
            raise ValueError('Operators are not decomposed yet. Please decompose the operators first.')
        
        else:
            op_dict = {}

            for op_string, coeff in self._op_dict.items():
                for op_string_other, coeff_other in other._op_dict.items():
                    op_dict[op_string + op_string_other] = coeff * coeff_other
            
            num_bits = self._num_bits + other._num_bits
            return Operator(num_bits, op_dict)

    def __add__(self, other: 'Operator')->'Operator':
        assert self._num_bits == other._num_bits, 'Operators to be added must have the same length'

        op_dict = {}
        
        all_op_strings = list(set(self._op_dict.keys()) | set(other._op_dict.keys()))

        for op_string in all_op_strings:
            coeff1 = self._op_dict.get(op_string, 0.0)
            coeff2 = other._op_dict.get(op_string, 0.0)
            op_dict[op_string] = coeff1 + coeff2
        
        return Operator(self._num_bits, op_dict)
    
    def __sub__(self, other: 'Operator')->'Operator':
        assert self._num_bits == other._num_bits, 'Operators to be subtracted must have the same length'
        
        op_dict = {}
        
        all_op_strings = list(set(self._op_dict.keys()) | set(other._op_dict.keys()))
        
        for op_string in all_op_strings:
            coeff1 = self._op_dict.get(op_string, 0.0)
            coeff2 = other._op_dict.get(op_string, 0.0)
            op_dict[op_string] = coeff1 - coeff2
        
        return Operator(self._num_bits, op_dict)
    
    def __mul__(self, coefficient: float)->'Operator':
        op_dict = {}
        for op_string, coeff in self._op_dict.items():
            op_dict[op_string] = coeff * coefficient
        return Operator(self._num_bits, op_dict)
    
    def __rmul__(self, coefficient: float)->'Operator':
        return self.__mul__(coefficient)

    @staticmethod
    def _multiply_symbols(a: str, b: str) -> str | None:
        if a == 'I':
            return b
        if b == 'I':
            return a

        rules = {
            ('R', 'L'): 'U',
            ('R', 'D'): 'R',
            ('L', 'R'): 'D',
            ('L', 'U'): 'L',
            ('D', 'L'): 'L',
            ('D', 'D'): 'D',
            ('U', 'R'): 'R',
            ('U', 'U'): 'U',
        }

        return rules.get((a, b))
    
    def __matmul__(self, other: 'Operator') -> 'Operator':
        assert self._num_bits == other._num_bits, 'Operators to be multiplied must have the same length'

        op_dict = {}

        for op_string, coeff in self._op_dict.items():
            for op_string_other, coeff_other in other._op_dict.items():
                tmp_op_string = ''
                skip = False

                for a, b in zip(op_string, op_string_other):
                    result = self._multiply_symbols(a, b)
                    if result is None:
                        skip = True
                        break
                    tmp_op_string += result

                if not skip:
                    op_dict[tmp_op_string] = op_dict.get(tmp_op_string, 0.0) + coeff * coeff_other

        return Operator(self._num_bits, op_dict)
    
    def simplify(self, tol=1e-8):
        op_dict = {}
        for op_string, coeff in self._op_dict.items():
            real = coeff.real
            imag = coeff.imag

            # Zero out near-zero values
            real = 0.0 if np.abs(real) < tol else real
            imag = 0.0 if np.abs(imag) < tol else imag

            # Reconstruct simplified coefficient
            if real == 0.0 and imag == 0.0:
                continue  # skip zero terms
            elif imag == 0.0:
                op_dict[op_string] = real
            elif real == 0.0:
                op_dict[op_string] = imag * 1j
            else:
                op_dict[op_string] = complex(real, imag)

        self._op_dict = op_dict
        return op_dict
    
    @staticmethod
    def _conjugate_symbols(a: str) -> str | None:

        rules = {
            'I': 'I',
            'R': 'L',
            'L': 'R',
            'U': 'U',
            'D': 'D',
        }

        return rules.get(a)
    
    def op_conjugate(self):
        op_dict = {}
        for op_string, coeff in self._op_dict.items():
            op_tmp = ''
            for op_char in op_string:
                op_char_conjugate = self._conjugate_symbols(op_char)
                if op_char_conjugate is not None:
                    op_tmp += op_char_conjugate
                else:
                    raise ValueError(f'Operator {op_string} is not valid.')
            op_dict[op_tmp] = coeff.conjugate()
        op_conjugate = Operator(self._num_bits, op_dict)
        return op_conjugate


class IdentityOperator(Operator):
    
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits, {'I'*num_qubits: 1.0})
        

class DifferentialOperator1D(Operator):

    def __init__(self, 
                num_bits: int,
                direction: str='forward',
                axis: str='0',
                h: float=1.0,
                bc: dict[str, tuple[str]]={}
                ):
        
        self._direction = direction
        self._h = h

        s10, s01, s00, s11, I = sp.symbols('L R U D I', commutative=False)

        if direction not in ['forward', 'backward']:
            raise NotImplementedError()

        self._num_bits = num_bits

        if direction == 'forward':
            operator = 1/h * (self.shift_operator(direction) - I**self._num_bits)
            # boundary conditions
            if bc[axis][1] == 'N':
                operator += 1/h * (s11**self._num_bits)
            if bc[axis][0] == 'P':
                operator += 1/h * (s10**self._num_bits)

        elif direction == 'backward':
            operator = 1/h * (I**self._num_bits - self.shift_operator(direction))
            # boundary conditions
            if bc[axis][0] == 'N':
                operator -= 1/h * (s00**self._num_bits)
            if bc[axis][0] == 'P':
                operator -= 1/h * (s01**self._num_bits)

        self.op_decompose(operator.expand())
        

class DifferentialOperator(Operator):
    
    def __init__(self, 
                 num_bits_x: int,
                 num_bits_y: Optional[int]=None,
                 direction: str='forward',
                 dim: int=1,
                 axis: str='0',
                 h: float=1.0,
                 bc: dict[str, tuple[str]]={}):  # bc_dict = { '0': ('D', 'N'), '1': ('P') }

        self._h = h

        if dim == 1:
            self._num_bits_x = num_bits_x
            self._num_bits = self._num_bits_x
            op = DifferentialOperator1D(self._num_bits_x, direction=direction, axis=axis, h=h, bc=bc)
            op_dict = op._op_dict
            
        elif dim == 2:
            self._num_bits_x = num_bits_x
            self._num_bits_y = num_bits_y if num_bits_y is not None else num_bits_x
            self._num_bits = self._num_bits_x + self._num_bits_y

            if axis == '0':
                op = DifferentialOperator1D(self._num_bits_x, direction=direction, axis=axis, h=h, bc=bc)
                op_dict = op._op_dict
                tmp_dict = {}
                for key, value in op_dict.items():
                    key = 'I'*self._num_bits_y +key
                    tmp_dict[key] = value
                op_dict = tmp_dict

            elif axis == '1':
                op = DifferentialOperator1D(self._num_bits_y, direction=direction, axis=axis, h=h, bc=bc)
                op_dict = op._op_dict
                tmp_dict = {}
                for key, value in op_dict.items():
                    key = key + 'I'*self._num_bits_x
                    tmp_dict[key] = value
                op_dict = tmp_dict
                
        else:
            raise NotImplementedError()
        
        self._op_dict = op_dict


class LaplacianOperator1D(Operator):

    def __init__(self, 
                num_bits: int,
                axis: str='0',
                h: float=1.0,
                bc: dict[str, tuple[str]]={}
                ):
        
        self._h = h

        s10, s01, s00, s11, I = sp.symbols('L R U D I', commutative=False)
        self._num_bits = num_bits

        operator = 1/(h**2) * (self.shift_operator('forward') + self.shift_operator('backward') - 2*I**self._num_bits)
        # boundary conditions
        if bc[axis][0] == 'N':
            operator += 1/(h**2) * (s00**self._num_bits)
        if bc[axis][1] == 'N':
            operator += 1/(h**2) * (s11**self._num_bits)
        if bc[axis][0] == 'P':
            operator += 1/(h**2) * (s10**self._num_bits + s01**self._num_bits)

        self.op_decompose(operator.expand())
        

class LaplacianOperator(Operator):
    
    def __init__(self, 
                 num_bits_x: int,
                 num_bits_y: Optional[int]=None,
                 dim: int=1,
                 axis: str='0',
                 h: float=1.0,
                 bc: dict[str, tuple[str]]={}):  # bc_dict = { '0': ('D', 'N'), '1': ('P') }

        self._h = h

        if dim == 1:
            self._num_bits_x = num_bits_x
            self._num_bits = self._num_bits_x
            op = LaplacianOperator1D(self._num_bits_x, axis=axis, h=h, bc=bc)
            op_dict = op._op_dict
            
        elif dim == 2:
            self._num_bits_x = num_bits_x
            self._num_bits_y = num_bits_y if num_bits_y is not None else num_bits_x
            self._num_bits = self._num_bits_x + self._num_bits_y

            if axis == '0':
                op = LaplacianOperator1D(self._num_bits_x, axis=axis, h=h, bc=bc)
                op_dict = op._op_dict
                tmp_dict = {}
                for key, value in op_dict.items():
                    key = 'I'*self._num_bits_y + key
                    tmp_dict[key] = value
                op_dict = tmp_dict

            elif axis == '1':
                op = LaplacianOperator1D(self._num_bits_y, axis=axis, h=h, bc=bc)
                op_dict = op._op_dict
                tmp_dict = {}
                for key, value in op_dict.items():
                    key = key + 'I'*self._num_bits_x
                    tmp_dict[key] = value
                op_dict = tmp_dict
                
        else:
            raise NotImplementedError()
        
        self._op_dict = op_dict
