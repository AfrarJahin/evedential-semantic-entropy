

def plausibility(F, M): # compute Pl only on focal sets

    Pl = [0.0] * len(F)

    for focal_setA, it in zip(F, range(len(F))):
        # print("---A = "+ str(focal_setA))
        for focal_setB, massB in zip(F, M):
            # print("B = "+ str(focal_setB))

            # if A inter B not empty
            is_intersection_empty = (focal_setA & focal_setB) == 0
            if not is_intersection_empty:
                Pl[it] += massB
                # print("on ajout " + str(massB))
    
    return Pl

def belief(F, M): # compute bel only on focal sets

    Bel = [0.0] * len(F)

    for focal_setA, it in zip(F, range(len(F))):
        # print("---A = "+ str(focal_setA))
        for focal_setB, massB in zip(F, M):
            # print("B = "+ str(focal_setB))

            # if A in B
            is_included = (focal_setA & focal_setB) == focal_setB
            # print(is_included)
            if is_included:
                Bel[it] += massB
                # print("on ajout " + str(massB))
    
    return Bel


def calculate_pignistic_probability(F, M, frame_size):
    # Initialize pignistic probabilities for each element in the frame of discernment
    BetP = [0.0] * frame_size

    for focal_set, mass in zip(F, M):
        if mass > 0:
            # Convert the decimal index to a subset using binary representation
            binary_representation = bin(focal_set)[2:].zfill(frame_size)
            subset = {i for i, bit in enumerate(reversed(binary_representation)) if bit == '1'}
            subset_size = len(subset)

            # Distribute the mass equally among the elements in the subset
            if subset_size > 0:
                contribute = mass / subset_size
                for element in subset:
                    BetP[element] += contribute

    return BetP



def myBin2dec(bin_vect):
    bin_vect_rev = bin_vect[::-1]
    bin_str = ''.join([str(item) for item in bin_vect_rev])
    return int(bin_str,2)


def vector_to_set_expression(binary_vector):
    # Check if all elements are 1, returning the special case '\Theta'
    if all(x == 1 for x in binary_vector):
        return r'\Theta'
    
    # Generate the set expression
    elements = []
    for index, value in enumerate(binary_vector):
        if value == 1:
            elements.append(rf'\theta_{{{index + 1}}}')  # Use the index + 1 for 1-based index

    return '\{' + ', '.join(elements) + '\}'
    
