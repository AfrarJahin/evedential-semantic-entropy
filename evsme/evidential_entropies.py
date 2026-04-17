import math
from evsme.evidence_theory_utils import calculate_pignistic_probability
from evsme.evidence_theory_utils import plausibility
from evsme.evidence_theory_utils import belief



def dubois_prade_entropy(F, M, C_dec2C_bin):
    entropy = 0
    for foc, mass_val in zip(F, M):
        entropy = mass_val * math.log(sum(C_dec2C_bin[foc]), 2)
    return entropy


def jousselme_entropy(F, M, frame_size):
    # Calculate pignistic probabilities
    BetP = calculate_pignistic_probability(F, M, frame_size)
    
    # Compute the entropy using the pignistic probabilities
    entropy = 0
    for prob in BetP:
        if prob > 0:
            entropy += prob * math.log(1/prob, 2)

    return entropy



def nguyen_entropy(M):

    entropy = 0
    for mass_val in M:
        entropy += mass_val * math.log(1/mass_val, 2)

    return entropy



def yager_entropy(F, M):

    Pl = plausibility(F, M)

    entropy = 0
    for i in range(len(M)):
        entropy -= M[i] * math.log(Pl[i], 2)

    return entropy


def hohle_entropy(F, M):

    Bel = belief(F, M)

    entropy = 0
    for i in range(len(M)):
        entropy -= M[i] * math.log(Bel[i], 2)

    return entropy
