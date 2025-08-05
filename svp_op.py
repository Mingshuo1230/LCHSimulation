from typing import Dict, List, Tuple, Union, Optional
from collections import defaultdict


# step 1: convert the coordinates to binary strings
def binary_coord(coords: list[tuple[int, int]], digits: int=2) -> list[str]:
    """
    Given a list of (x, y) coordinate pairs and a bit-width `digits`,
    return a list of concatenated binary strings by+bx, each of length 2*digits.
    """
    out = []
    max_val = 1 << digits  # 2**digits
    fmt = f'0{digits}b'
    for x, y in coords:
        if not (0 <= x < max_val and 0 <= y < max_val):
            raise ValueError(f"Coordinate {(x, y)} out of range for {digits} bits (must be 0 ≤ value < {max_val})")
        bx = format(x, fmt)
        by = format(y, fmt)
        out.append(by + bx)
    return out

# Step 2: Count number of ones in the binary string
def count_ones(binary):
    return binary.count('1')

# Step 3: Check if two terms differ by only one bit
def is_combinable(a, b):
    diff_count = 0
    for x, y in zip(a, b):
        if x != y:
            if x != '-' and y != '-':
                diff_count += 1
            else:
                return False  # Don't combine '-' positions
        if diff_count > 1:
            return False
    return diff_count == 1

# Step 4: Combine two terms into a generalized one with '-'
def combine_terms(a, b):
    return ''.join([x if x == y else '-' for x, y in zip(a, b)])

# Step 5: Main function to perform Quine–McCluskey grouping
def qma_grouping(minterms):
    grouped = defaultdict(set)
    for m in minterms:
        grouped[count_ones(m)].add(m)

    prime_implicants = set()
    used = set()
    next_round = defaultdict(set)

    while True:
        combined_pairs = set()
        next_round.clear()
        used.clear()
        keys = sorted(grouped.keys())
        has_combined = False

        for i in range(len(keys) - 1):
            group1 = grouped[keys[i]]
            group2 = grouped[keys[i + 1]]

            for a in group1:
                for b in group2:
                    if is_combinable(a, b):
                        combined = combine_terms(a, b)
                        next_round[count_ones(combined.replace('-', ''))].add(combined)
                        used.add(a)
                        used.add(b)
                        has_combined = True

        # Add all terms that were not combined to the prime implicants
        for group in grouped.values():
            for term in group:
                if term not in used:
                    prime_implicants.add(term)

        if not has_combined:
            break

        grouped = defaultdict(set)
        for combined_set in next_round.values():
            for term in combined_set:
                grouped[count_ones(term.replace('-', ''))].add(term)

    c_dict = {}
    for string in sorted(prime_implicants):
        c_dict[string] = 1.0

    return c_dict

# step 6: resolve the duplicate bits
def duplicate_resolver(bits1:str, bits2:str, coeff:float)->Tuple[str, str, float]:
    # modify the bits1 to resolve the duplicate
    wildcard1 = []     # wildcard bits in bits1
    wildcard2 = []     # wildcard bits in bits2
    other_diff = []    # bits that are different between bits1 and bits2 other than wildcard
    
    for i, (b1, b2) in enumerate(zip(bits1, bits2)):
        if b1 == '-' or b2 == '-':
            if b1 == '-':
                wildcard1.append(i)
            if b2 == '-':
                wildcard2.append(i)
        elif b1 != b2:
            other_diff.append(i)

    # check duplication
    # if there is no difference other than wildcard, check if there is any wildcard that is not in the other string
    if len(other_diff) == 0:
        # check if there is any wildcard in bits1 that is not in bits2
        for i in wildcard1:
            if i not in wildcard2:
                if bits2[i] == '0':
                    b_mod = '1'
                else:
                    b_mod = '0'
                bits1 = bits1[:i] + b_mod + bits1[i+1:]
                return bits1, bits2, coeff

        # when there is no wildcard in bits1, skip flipping and multiply by -1 to resolve the duplicate
        # '100' + '10-' -> -'100' + '10-'
        for i in wildcard2:
            if i not in wildcard1:
                return bits1, bits2, -1*coeff

        # when bits1 and bits2 are the same, return 0 to avoid the duplicate
        return bits1, bits2, 0

    return bits1, bits2, coeff

# step 7: convert the bitstrings to a dictionary of matrices for 2d structure
def bit2op(c_dict:Dict[str, float], embed_bits:List[int]=None)->Dict[str, float]:
    '''
    rewrite the bitstrings to a dictionary of matrices for 2d structure
    embed_bits: number of the whole qubits in 2d structure
    If the length of bits is not equal to embed_bits, padding the bits with 'I'
    '''
    # cache dictionary to avoid duplicate bits
    cache_dict = {}

    for bits, val in c_dict.items():
        # check duplicate bits
        coeff = float(val)
        for key in cache_dict.keys():
            bits, _, coeff = duplicate_resolver(bits, key, coeff)
            if coeff == 0:
                break

        if coeff:
            cache_dict[bits] = coeff

    # final dictionary of quantum operators and coefficients
    qop_dict = {}
    for bits, coeff in cache_dict.items():
        bits = bits.replace('-', 'I').replace('0', 'U').replace('1', 'D')

        # padding the bits with 'I' if embed_bits is not None
        if embed_bits:
            padding_dim = [embed_bits[0]-len(bits)//2, embed_bits[1]-len(bits)//2]
            bits = bits[:len(bits)//2] + 'I'*padding_dim[0] + bits[len(bits)//2:] + 'I'*padding_dim[1]
        qop_dict[bits] = float(coeff)

    return qop_dict