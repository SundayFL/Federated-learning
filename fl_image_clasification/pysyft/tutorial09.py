Q = 1234567891011
x = 25

import random

def encrypt(x):
    share_a = random.randint(-Q,Q)
    share_b = random.randint(-Q,Q)
    share_c = (x - share_a - share_b) % Q
    return (share_a, share_b,  share_c)