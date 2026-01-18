# Greatest common divisor of 1 or more numbers.
from functools import reduce


def gcd(*numbers):
    """
    Return the greatest common divisor of 1 or more integers
    Examples
    --------
    >>> gcd(5)
    5
    >>> gcd(30, 40)
    10
    >>> gcd(120, 40, 60)
    20
    """
    # Am I terrible for doing it this way?
    from math import gcd

    return reduce(gcd, numbers)


# Least common multiple is not in standard libraries? It's in gmpy, but this
# is simple enough:


def lcm(*numbers):
    """
    Return lowest common multiple of 1 or more integers.
    Examples
    --------
    >>> lcm(5)
    5
    >>> lcm(30, 40)
    120
    >>> lcm(120, 40, 60)
    120
    """

    def lcm(a, b):
        return (a * b) // gcd(a, b)

    return reduce(lcm, numbers, 1)
