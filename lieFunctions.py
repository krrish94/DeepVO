"""
Helper functions for working with the Lie groups SO(3) and SE(3)
"""

import numpy as np


# Log map for SO(3). Returns a 3-vector of so(3) exponential coordinates
# Actually, this does vee(SO3_log(R)), also denoted Log(R) in a few conventions
# Not to be confused with log(R), which simply returns an element on the tangent space
def SO3_log(R):

	
