from sage.all import ZZ, gcd, lcm, Mod, prime_range


def polynomial_not_perfect_powers_values(f, a, S=None, L=None, S_bound = 50, L_bound=1000):
    r"""
    Return ``True`` if the equation ``f(x) = a^n`` with ``n > 0`` has no
    integer solutions ``(x, n)``, and ``False`` if the method fails to prove
    the non-existence.

    This implements Algorithm 1 from [8]

    INPUT:

    - ``f`` -- a univariate polynomial in ``ZZ[x]``.
    - ``a`` -- a positive integer with ``a > 1``.
    - ``S`` -- (optional) a list of positive integers coprime to ``a``.
      If not given, the primes up to 100 which do not divide ``a`` are used.
    - ``L`` -- (optional) a list of integers ``l`` such that
      ``a^t == 1 (mod l)``, where ``t`` is computed from ``S``.  If not given,
      all integers ``2 <= l <= L_bound`` coprime to ``a`` satisfying the
      congruence are used.
    - ``S_bound`` -- (optional) the upper bound for the default choice of S if S is None (default: 50).
    - ``L_bound`` -- (optional) the upper bound for the default choice of
      ``L`` (default: 1000).

    OUTPUT:

    ``True`` if it is proved that there are no solutions, ``False`` otherwise
    (i.e. the algorithm fails to prove the non-existence).
    """
    try:
        f = f.change_ring(ZZ)
    except AttributeError:
        f = ZZ["x"](f)
    a = ZZ(a)
    if a <= 1:
        raise ValueError("a must be an integer greater than 1")

    # Step 1: the method requires the gcd of the coefficients of f to divide a;
    # otherwise it returns Fail.
    g = gcd(f.coefficients())
    if g != 0 and a % g != 0:
        return False

    # Step 2: choose a finite set S of integers coprime to a.
    if S is None:
        S = [p for p in prime_range(S_bound) if a % p != 0]

    # Steps 3-10: accumulate the least common multiple of the orders ts for
    # which no value of f is congruent to a power of a modulo s.
    t = ZZ(1)
    for s in S:
        s = ZZ(s)
        if s <= 1 or gcd(a, s) != 1:
            continue

        ts = Mod(a, s).multiplicative_order()
        powers = {ZZ(Mod(a, s) ** k) for k in range(1, int(ts))}
        values = {ZZ(Mod(f(i), s)) for i in range(1, int(s) + 1)}
        if not powers & values:
            t = lcm(t, ts)

    # Steps 11-16: look for an integer l with a^t == 1 (mod l) for which f is
    # never congruent to 1 modulo l.
    if L is None:
        L = [l for l in range(2, L_bound + 1) if gcd(a, l) == 1 and pow(int(a), int(t), l) == 1]

    for l in L:
        l = ZZ(l)
        if l <= 1:
            continue
        if all(Mod(f(u), l) != 1 for u in range(int(l))):
            return True
    return False
