r"""
Integer solutions of the equations

    y^2 - d^k = x^3   (minus case, Section 3.1.1)
    y^2 + d^k = x^3   (plus  case, Section 3.1.2)

for 2 <= d <= D_BOUND, d not a perfect power, k >= 1, gcd(x, d) = 1.
These are the n = 2 solutions of the Cubic Lebesgue-Nagell equation
x^3 +/- d^k = y^n of [Cazorla Garcia, Koutsianas, "Cubic Lebesgue-Nagell
equations"].

Primary method: writing k = 6*lambda + k0 (k0 = 0..5), a solution (x0, y0)
corresponds to the S-integral point (X, Y) = (x0/d^(2 lambda), y0/d^(3 lambda))
on the Mordell curve Y^2 = X^3 +/- d^{k0}, where S = {p : p | d}.

When the S-integral points cannot be computed (Mordell-Weil group not
available) we use the explicit methods of Sections 3.1.1/3.1.2, which differ
for k even and k odd:

  - k even (minus): factorization over Q of (y + d^{k/2})(y - d^{k/2}) = x^3
    gives the three cubic Thue-Mahler equations (3.6),(3.7),(3.8), solved with
    ThueMahlerSolver and (for the reducible one) the Section 2.3 method.
  - k odd  (minus): factorization over Q(sqrt(d)); unit-group descent leads to
    Thue-Mahler equations (3.11) (Section 2.3 for the reducible t = 0 case)
    and the special case d = 79 (Lemma 3.4).
  - k even (plus): over Q(i), Lemma 3.6 gives the reducible cubic
    d^{k/2} = b(3a^2 - b^2), solved with the Section 2.3 method.
  - k odd  (plus): over Q(sqrt(-d)), Lemmas 3.8 (h_K coprime to 3) and 3.9
    (the twelve exceptional fields with 3 | h_K) give Thue-Mahler equations.

For Thue-Mahler equations we use ThueMahlerSolver (Gherga-Siksek) and for
equations a*x + b*y = z^2 with x, y S-units we use SUnitsSumSquare (de Weger).
"""

from sage.all import (EllipticCurve, ZZ, QQ, sqrt, prod, gcd, lcm,
                      QuadraticField, PolynomialRing, next_prime, oo)

import sys
sys.path.append("/home/akoutsianas/Sage/DiophantineSolvers")
from thue_mahler_solver import ThueMahlerSolver
from sunits_sum_square import SUnitsSumSquare

D_BOUND = 100

# The twelve values of d for which Q(sqrt(-d)) has class number divisible by 3
# (Lemma 3.9).
_PLUS_ODD_CLASS_NUMBER_NON_COPRIME_TO_THREE = {23, 26, 29, 31, 38, 53, 59, 61, 83, 87, 89, 92}

# Development flag.  The Section 2.3 branches (SUnitsSumSquare) are expensive;
# while validating the pipeline we may skip them.
USE_SUNITS_SUM_SQUARE = False


# --------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------- #
def _dedup(sols):
    """Remove duplicate solution tuples and sort them."""
    seen = set()
    out = []
    for s in sols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return sorted(out, key=lambda t: (t[2], t[0], t[1], t[3]))


def _power_of_d_exponent(m, d):
    r"""Return the integer e >= 0 with m == d^e, or None if no such e exists.

    Uses Sage's :meth:`is_power_of` and :meth:`perfect_power`.
    """
    m = ZZ(m)
    if m <= 0:
        return None
    if m == 1:
        return 0
    if not m.is_power_of(d):
        return None
    b, e = m.perfect_power()
    if b != d:
        return None
    return e


def _cube_root(n):
    """Return the integer cube root of n if n is a perfect cube, else None.

    Uses Sage's :meth:`is_perfect_power` and :meth:`perfect_power`.
    """
    n = ZZ(n)
    if n in (-1, 0, 1):
        return n
    if not n.is_perfect_power():
        return None
    b, e = n.perfect_power()
    if e % 3 != 0:
        return None
    return b ** (e / 3)


def _to_basis(alpha, omega):
    r"""
    Express ``alpha`` in the integral basis {1, omega} of Q(r):
    returns (u, v) in Q with alpha = u + v*omega.
    """
    ap = alpha.polynomial().list()
    a = QQ(ap[0]) if len(ap) > 0 else QQ(0)
    b = QQ(ap[1]) if len(ap) > 1 else QQ(0)
    op = omega.polynomial().list()
    w0 = QQ(op[0]) if len(op) > 0 else QQ(0)
    w1 = QQ(op[1]) if len(op) > 1 else QQ(0)
    v = b / w1
    u = a - v * w0
    return u, v


def _poly_basis(poly, omega):
    r"""
    For a polynomial ``poly`` in Q(r)[a, b], return (h, g) such that
    poly == h(a, b) + g(a, b)*omega  with h, g in Q[a, b].
    """
    R = poly.parent()
    a, b = R.gens()
    h = R(0)
    g = R(0)
    for mon, cf in poly.dict().items():
        u, v = _to_basis(cf, omega)
        term = a ** mon[0] * b ** mon[1]
        h = h + QQ(u) * term
        g = g + QQ(v) * term
    return h, g


# --------------------------------------------------------------------- #
# primary method: S-integral points on Y^2 = X^3 + sign*d^{k0}
# --------------------------------------------------------------------- #
def _s_integral_points(d, sign):
    r"""
    Compute S-integral points on Y^2 = X^3 + sign*d^{k0} for k0 = 0..5 and
    recover the solutions of y^2 = x^3 + sign*d^k.

    sign = +1 : minus case (y^2 - d^k = x^3), sign = -1 : plus case.

    Returns (sols, problematic_k0s).
    """
    sols = []
    problematic = []
    S = ZZ(d).prime_factors()
    for k0 in range(6):
        E0 = E0min = EllipticCurve([0, sign * d ** k0])
        if not E0.is_minimal():
            E0min = E0.minimal_model()
        phi = E0min.isomorphism_to(E0)
        for p in phi.u.denominator().prime_factors():
            if p not in S:
                S.append(p)
        try:
            pts = E0min.S_integral_points(S=S)
        except Exception:
            # mwrank could not determine the Mordell-Weil basis: try to obtain
            # it from PARI's ellrank (with increasing effort) and retry.
            pts = None
            for effort in (0, 2, 4):
                try:
                    mw_base = E0min.gens(algorithm="pari", pari_effort=effort)
                except Exception:
                    continue
                try:
                    pts = E0min.S_integral_points(S=S, mw_base=mw_base)
                    break
                except Exception:
                    continue
            if pts is None:
                problematic.append(k0)
                print(f"S-integral points failed for k0={k0}, sign={sign}.")
                continue
        for pt in pts:
            sol = _recover_sintegral(pt, phi, d, k0, sign)
            if sol is not None:
                sols.append(sol)
    return sols, problematic


def _recover_sintegral(pt, phi, d, k0, sign):
    r"""
    Recover the primitive solution (x0, y0, d, k) from an S-integral point
    pt on the minimal model, mapped back to Y^2 = X^3 + sign*d^{k0}.
    Returns None if the point does not come from a primitive solution.
    """
    P0 = phi(pt)
    X, Y = P0[0], P0[1]
    e2 = _power_of_d_exponent(X.denominator(), d)
    e3 = _power_of_d_exponent(Y.denominator(), d)
    if e2 is None or e3 is None:
        return None
    if 2 * e3 != 3 * e2:          # denominators must be d^{2 lambda}, d^{3 lambda}
        return None
    if e2 % 2 != 0 or e3 % 3 != 0:
        return None
    lam = e2 / 2
    x0 = ZZ(X.numerator())
    y0 = ZZ(Y.numerator())
    k = 6 * lam + k0
    if k < 1:
        return None
    if gcd(x0, y0) != 1:
        return None
    if y0 ** 2 != x0 ** 3 + sign * d ** k:
        return None
    return (x0, y0, d, k)


# --------------------------------------------------------------------- #
# Section 2.3: reducible cubic forms via SUnitsSumSquare
# --------------------------------------------------------------------- #
def _reducible_cubic_thue_mahler(Q, c, d, S, rec):
    r"""
    Solve  X * Q(X, Y) = c*d^e  for coprime integers (X, Y), where Q is the
    quadratic form  Q(X, Y) = A X^2 + B X Y + C Y^2  with A, B, C in Z and
    the linear factor is always the first variable X.

    Section 2.3: for F(X, Y) = Q(X, Y) the triple (F, X^2, 2 C Y + B X)
    solves
        (4 C) x + (B^2 - 4 A C) y = z^2,   x = F,  y = X^2,  z = 2 C Y + B X.
    Writing m = gcd(4 C, B^2 - 4 A C) = m1 m2^2 (m1 squarefree) this becomes
        (4 C / m2^2) x + ((B^2 - 4 A C) / m2^2) y = ((2 C Y + B X) / m2)^2,
    a sum of two S-units being a square, solved with SUnitsSumSquare.

    The caller must arrange that the actual linear factor is the first
    variable (swapping the variables if necessary).

    For each solution (e, X, Y) the callback ``rec(e, X, Y)`` is invoked; its
    non-None return values are collected.
    """
    sols = []
    if not USE_SUNITS_SUM_SQUARE:
        return sols
    R = Q.parent()
    Xv, Yv = R.gens()
    A = ZZ(Q.monomial_coefficient(Xv ** 2))
    B = ZZ(Q.monomial_coefficient(Xv * Yv))
    C = ZZ(Q.monomial_coefficient(Yv ** 2))
    c = ZZ(c)

    # (2.8) -> (2.9): reduce by the square part of the gcd of the coefficients.
    a_eq = 4 * C
    b_eq = B ** 2 - 4 * A * C
    m = gcd(a_eq, b_eq)
    m1 = m.squarefree_part()
    m2sq = m / m1
    m2 = m2sq.sqrt()
    a = a_eq / (m1 * m2sq)
    b = b_eq / (m1 * m2sq)
    if a == 0 or b == 0 or C == 0:
        return sols

    # If c != 1 we enlarge S by the primes dividing c.
    Sext = sorted(set(ZZ(p) for p in S) | {p for p in c.prime_factors()})
    try:
        ss_sols = SUnitsSumSquare(a, b, Sext).solve()
    except Exception as e:
        print(f"SUnitsSumSquare failed for reducible cubic: {e}")
        return sols

    for (x0, y0, z0) in ss_sols:
        # x0 = F(X, Y),  y0 = X^2,  z0 = (2 C Y + B X) / m2.
        y0 = QQ(y0)
        if y0 <= 0 or not y0.is_square():
            continue
        T = y0.sqrt()
        for Xsgn in (1, -1):
            X = Xsgn * T
            for zsgn in (1, -1):
                Y = (m2 * zsgn * QQ(z0) - B * X) / (2 * C)
                if X not in ZZ or Y not in ZZ:
                    continue
                X, Y = ZZ(X), ZZ(Y)
                if gcd(X, Y) != 1:
                    continue
                F = A * X**2 + B * X * Y + C * Y**2
                if F != x0:
                    continue
                XF = X * F
                if XF <= 0 or XF % c != 0:
                    continue
                e = _power_of_d_exponent(XF / c, d)
                if e is None:
                    continue
                out = rec(e, X, Y)
                if out is not None:
                    sols.append(out)
    return sols


# --------------------------------------------------------------------- #
# MINUS case (Section 3.1.1):  y^2 - d^k = x^3
# --------------------------------------------------------------------- #
def case_n_2_minus(d):
    r"""Solutions (x, y, d, k) of y^2 - d^k = x^3 with gcd(x, d) = 1."""
    sols = []
    sols_s, problematic = _s_integral_points(d, +1)
    sols += sols_s
    even_k0s = sorted({k0 for k0 in problematic if k0 % 2 == 0})
    odd_k0s = sorted({k0 for k0 in problematic if k0 % 2 == 1})
    if even_k0s:
        sols += [s for s in _minus_even_case(d) if s[3] % 6 in even_k0s]
    if odd_k0s:
        sols += [s for s in _minus_odd_case(d) if s[3] % 6 in odd_k0s]
    return _dedup(sols)


def _minus_even_case(d):
    r"""
    k even, minus case: from (y + d^{k/2})(y - d^{k/2}) = x^3 we get the three
    systems (3.3)-(3.5), giving the cubic Thue-Mahler equations
        (3.7) d^{k/2} = x1^3 - 2 x2^3,
        (3.8) d^{k/2} = 2 x1^3 - x2^3
    and the reducible equation
        (3.6) 2 d^{k/2} = x1^3 - x2^3 = X(X^2 + 3XY + 3Y^2),  X = x1 - x2,
    solved by Section 2.3.
    """
    sols = []
    S = ZZ(d).prime_factors()

    for coeffs in ([1, 0, 0, -2], [2, 0, 0, -1]):
        try:
            tm_sols = ThueMahlerSolver(coeffs, S).solve()
        except Exception as e:
            print(f"Thue-Mahler failed for minus even, coeffs={coeffs}: {e}")
            continue
        for sol in tm_sols:
            x1, x2 = sol[0], sol[1]
            dk = prod(p ** e for p, e in sol[2].items())
            e = _power_of_d_exponent(dk, d)
            if e is None or e < 1:
                continue
            k = 2 * e
            x0 = 2 * x1 * x2
            if x0 == 0 or gcd(x0, d) != 1:
                continue
            y2 = x0 ** 3 + d ** k
            if y2.is_square():
                y0 = ZZ(sqrt(y2))
                sols.append((x0, y0, d, k))

    # (3.6): X(X^2 + 3XY + 3Y^2) = 2 d^e  (A = 1, B = 3, C = 3)
    R = PolynomialRing(QQ, ["X", "Y"])
    XX, YY = R.gens()
    Q = XX ** 2 + 3 * XX * YY + 3 * YY ** 2

    def rec(e, X, Y):
        x1 = X + Y
        x2 = Y
        x0 = x1 * x2
        if x0 == 0 or gcd(x0, d) != 1:
            return None
        k = 2 * e
        if k < 1:
            return None
        y2 = x0 ** 3 + d ** k
        if y2.is_square():
            y0 = ZZ(sqrt(y2))
            return (x0, y0, d, k)
        return None

    sols += _reducible_cubic_thue_mahler(Q, ZZ(2), d, S, rec)
    return sols


def _minus_odd_case(d):
    if d == 79:
        return _minus_odd_d_79()
    return _minus_odd_general(d)


def _minus_odd_general(d):
    r"""
    k odd, minus case, d != 79: K = Q(sqrt(d)).  By Lemma 3.2 there are
    coprime a, b and t = 0, 1, 2 with
        d^{(k-1)/2} = (u^t(a + b omega)^3 - ubar^t(a + b omega_bar)^3)/(2 sqrt(d)).
    For t = 0 the form is reducible and we use Section 2.3; for t = 1, 2 we
    use the Thue-Mahler solver.
    """
    sols = []
    S = ZZ(d).prime_factors()
    K = QuadraticField(d, "rd")
    rd = K.gen()
    conj = [phi for phi in K.automorphisms() if phi(rd) != rd][0]
    u = K.units()[0]
    bu = conj(u)
    omega = [om for om in K.ring_of_integers().basis() if om != 1][0]
    bomega = conj(omega)

    R = PolynomialRing(K, ["a", "b"])
    a, b = R.gens()

    for t in [0, 1, 2]:
        F = ((u ** t * (a + omega * b) ** 3
              - bu ** t * (a + bomega * b) ** 3) / (2 * rd)).change_ring(QQ)
        f = F.numerator()
        c = F.denominator()

        def rec(e, a0, b0, t=t):
            k = 2 * e + 1
            yK = (u ** t * (a0 + omega * b0) ** 3
                  + bu ** t * (a0 + bomega * b0) ** 3) / 2
            y0 = QQ(yK)
            if y0.denominator() != 1:
                return None
            y0 = ZZ(y0)
            xc = y0 ** 2 - d ** k
            x0 = _cube_root(xc)
            if x0 is None:
                return None
            return (x0, y0, d, k)

        if f.is_irreducible():
            try:
                tm_sols = ThueMahlerSolver(f, S, a=c).solve()
            except Exception as e:
                print(f"Thue-Mahler failed for minus odd, d={d}, t={t}: {e}")
                continue
            for sol in tm_sols:
                e = _tm_exponent(sol, d)
                if e is None:
                    continue
                out = rec(e, sol[0], sol[1])
                if out is not None:
                    sols.append(out)
        else:
            # t = 0: F(a, b) = b * f2(a, b) / d1  (Remark 3.3, d1 | 2d).  Swap
            # the variables so the linear factor is X = b, Y = a:
            # Q(X, Y) = f2(Y, X),  c = d1.
            Rq = F.parent()
            aq, bq = Rq.gens()
            if F % bq != 0:
                raise ValueError("reducible case: b does not divide F")
            q = Rq(F / bq)
            d1 = lcm([cf.denominator() for cf in q.coefficients()])
            f2 = d1 * q
            Q = f2(bq, aq)

            def rec23(e, X, Y):
                return rec(e, Y, X)
            sols += _reducible_cubic_thue_mahler(Q, d1, d, S, rec23)
    return sols


def _tm_exponent(sol, d):
    """Exponent e with dk = prod p^z == d^e, from a Thue-Mahler solution."""
    dk = prod(p ** e for p, e in sol[2].items())
    return _power_of_d_exponent(dk, d)


def _minus_odd_d_79():
    r"""
    k odd, minus case, d = 79: Lemma 3.4.  K = Q(sqrt(79)) has class number
    3; the equations involve u^t(a + b sqrt(79))^3 = h_t + g_t sqrt(79) and
        s = 0 : 79^e  = g_t,                          y = h_t,
        s = 1 : 3^3 79^e = 2 h_t - 17 g_t,            y = (-17 h_t + 158 g_t)/3^3,
        s = 2 : 3^6 79^e = -68 h_t + 605 g_t,         y = (605 h_t - 5372 g_t)/3^6,
    with gcd(a, b) = 1, 3 or 9 respectively.
    """
    d = ZZ(79)
    sols = []
    S = [79]
    K = QuadraticField(79, "rd")
    rd = K.gen()
    conj = [phi for phi in K.automorphisms() if phi(rd) != rd][0]
    u = K.units()[0]
    omega = [om for om in K.ring_of_integers().basis() if om != 1][0]

    R = PolynomialRing(K, ["a", "b"])
    aa, bb = R.gens()

    for t in [0, 1, 2]:
        cube = u ** t * (aa + omega * bb) ** 3
        h, g = _poly_basis(cube, omega)
        h = h.change_ring(QQ)
        g = g.change_ring(QQ)

        for mult, F, ynum, scales in (
            (1, g, lambda H, G: H, [1]),
            (27, 2 * h - 17 * g, lambda H, G: -17 * H + 158 * G, [1, 3]),
            (729, -68 * h + 605 * g, lambda H, G: 605 * H - 5372 * G, [1, 3, 9]),
        ):
            if len(F.factor()) != 1:
                # Reducible only for s = t = 0 (Remark 3.5):
                #     79^e = b (3 a^2 + 79 b^2).
                # Section 2.3 with the linear factor X = b, Y = a.
                Rq = F.parent()
                aq, bq = Rq.gens()
                if F % bq != 0:
                    raise ValueError("reducible case: b does not divide F")
                qp = F / bq
                Q = qp(bq, aq)

                def rec23(e, X, Y, t=t, h=h, g=g, mult=mult, ynum=ynum):
                    a0, b0 = Y, X
                    k = 2 * e + 1
                    ynum_v = ynum(h(a0, b0), g(a0, b0))
                    if ynum_v % mult != 0:
                        return None
                    y0 = ynum_v / mult
                    x0 = _cube_root(y0 ** 2 - d ** k)
                    if x0 is not None:
                        return (x0, y0, d, k)
                    return None

                sols += _reducible_cubic_thue_mahler(Q, ZZ(mult), d, S, rec23)
                continue
            for sc in scales:
                a_mult = mult / (sc ** 3)
                if a_mult == 0:
                    continue
                try:
                    tm_sols = ThueMahlerSolver(F, S, a=a_mult).solve()
                except Exception as e:
                    print(f"Thue-Mahler failed for d=79, t={t}: {e}")
                    continue
                for sol in tm_sols:
                    ap, bp = sol[0], sol[1]
                    a0, b0 = sc * ap, sc * bp
                    e = _tm_exponent(sol, d)
                    if e is None:
                        continue
                    k = 2 * e + 1
                    ynum_v = ynum(h(a0, b0), g(a0, b0))
                    if ynum_v % mult != 0:
                        continue
                    y0 = ynum_v / mult
                    xc = y0 ** 2 - d ** k
                    x0 = _cube_root(xc)
                    if x0 is not None:
                        sols.append((x0, y0, d, k))
    return sols


# --------------------------------------------------------------------- #
# PLUS case (Section 3.1.2):  y^2 + d^k = x^3
# --------------------------------------------------------------------- #
def case_n_2_plus(d):
    r"""Solutions (x, y, d, k) of y^2 + d^k = x^3 with gcd(x, d) = 1."""
    sols = []
    sols_s, problematic = _s_integral_points(d, -1)
    sols += sols_s
    even_k0s = sorted({k0 for k0 in problematic if k0 % 2 == 0})
    odd_k0s = sorted({k0 for k0 in problematic if k0 % 2 == 1})
    if even_k0s:
        sols += [s for s in _plus_even_case(d) if s[3] % 6 in even_k0s]
    if odd_k0s:
        sols += [s for s in _plus_odd_case(d) if s[3] % 6 in odd_k0s]
    return _dedup(sols)


def _plus_even_case(d):
    r"""
    k even, plus case (Lemma 3.7).  Let K = Q(i); we have
        (y + d^{k/2} i)(y - d^{k/2} i) = x^3
    and the two factors are coprime.  Hence (3.19)
        y + d^{k/2} i = (a + bi)^3
    for coprime integers a, b.  The units of Z[i] are cubes (indeed i = (-i)^3,
    -1 = (-1)^3), so they are absorbed by the signs/order of a and b; the
    single case above is therefore exhaustive.  With
        (a + bi)^3 = U(a,b) + V(a,b) i,
        U(a,b) = a(a^2 - 3b^2),   V(a,b) = b(3a^2 - b^2),
    we get
        d^{k/2} = V,  y = U,  x = a^2 + b^2.
    (The closed forms printed in Lemma 3.7, b(2b^2+3a^2) and a(-3b^2+2a^2),
    are a typo: they contradict y + d^{k/2} i = (a+bi)^3.)
    The cubic b(3a^2 - b^2) is reducible (linear times irreducible quadratic),
    so we use the Section 2.3 method.
    """
    sols = []
    S = ZZ(d).prime_factors()
    R = PolynomialRing(QQ, ["a", "b"])
    aa, bb = R.gens()

    # (a + bi)^3 = U(a,b) + V(a,b) i
    U = aa * (aa ** 2 - 3 * bb ** 2)     # a(a^2 - 3b^2)
    V = bb * (3 * aa ** 2 - bb ** 2)     # b(3a^2 - b^2)

    def rec(e, X, Y):
        a0, b0 = Y, X                    # X = b (linear factor), Y = a
        k = 2 * e
        if k < 1:
            return None
        x0 = a0 ** 2 + b0 ** 2
        if gcd(x0, d) != 1:
            return None
        y0 = U(a0, b0)
        if y0 ** 2 + d ** k != x0 ** 3:
            return None
        return (x0, y0, d, k)

    Q = (R(V / bb))(bb, aa)              # 3a^2 - b^2, linear factor = 1st var
    sols += _reducible_cubic_thue_mahler(Q, ZZ(1), d, S, rec)
    return sols


def _zeta3_in_K(K, d):
    r"""The primitive cube root of unity zeta_3 in K = Q(sqrt(-d)), if present."""
    m2 = ZZ(3) * d
    m = m2.isqrt()
    if m * m != m2:
        return None
    r = K.gen()
    sqrtm3 = -m * r / d          # sqrt(-3) in K
    return K((-1 + sqrtm3) / 2)


def _plus_odd_case(d):
    if d in _PLUS_ODD_CLASS_NUMBER_NON_COPRIME_TO_THREE:
        return _plus_odd_class_number_non_coprime_to_three(d)
    return _plus_odd_general(d)


def _plus_odd_general(d):
    r"""
    k odd, plus case, d not in _PLUS_ODD_CLASS_NUMBER_NON_COPRIME_TO_THREE: K = Q(sqrt(-d)), h_K coprime to 3.
    Lemma 3.8: coprime a, b and t (with zeta_3 if zeta_3 in K) satisfy
        d^{(k-1)/2} = (zeta_3^t (a + b omega)^3
                       - zeta_3^{-t}(a + b omega_bar)^3)/(2 sqrt(-d)),
        y          = (zeta_3^t (a + b omega)^3
                      + zeta_3^{-t}(a + b omega_bar)^3)/2.
    """
    sols = []
    S = ZZ(d).prime_factors()
    K = QuadraticField(-d, "r")
    r = K.gen()
    conj = [phi for phi in K.automorphisms() if phi(r) != r][0]
    omega = [om for om in K.ring_of_integers().basis() if om != 1][0]
    bomega = conj(omega)

    zeta3 = _zeta3_in_K(K, d)
    if zeta3 is not None:
        bzeta = conj(zeta3)
        ts = (0, 1, 2)
    else:
        zeta3, bzeta = K(1), K(1)
        ts = (0,)

    R = PolynomialRing(K, ["a", "b"])
    aa, bb = R.gens()

    for t in ts:
        zt = zeta3 ** t
        ztb = bzeta ** t
        F = ((zt * (aa + omega * bb) ** 3 - ztb * (aa + bomega * bb) ** 3)
             / (2 * r)).change_ring(QQ)

        def rec(e, a0, b0, zt=zt, ztb=ztb):
            k = 2 * e + 1
            yK = (zt * (a0 + omega * b0) ** 3
                  + ztb * (a0 + bomega * b0) ** 3) / 2
            y0 = QQ(yK)
            if y0.denominator() != 1:
                return None
            y0 = ZZ(y0)
            x0 = _cube_root(y0 ** 2 + d ** k)
            if x0 is None:
                return None
            return (x0, y0, d, k)

        nonconst = [pol for pol, _ in F.factor() if pol.degree() > 0]
        if len(nonconst) == 1:
            # irreducible (always t != 0): cubic Thue-Mahler equation.
            f = F.numerator()
            c = F.denominator()
            try:
                tm_sols = ThueMahlerSolver(f, S, a=c).solve()
            except Exception as e:
                print(f"Thue-Mahler failed for plus odd, d={d}, t={t}: {e}")
                continue
            for sol in tm_sols:
                e = _tm_exponent(sol, d)
                if e is None:
                    continue
                out = rec(e, sol[0], sol[1])
                if out is not None:
                    sols.append(out)
        else:
            # reducible (always t = 0): F(a, b) = b * f2(a, b) / d1.  Swap so
            # that the linear factor is the first variable and use Section 2.3.
            Rq = F.parent()
            aq, bq = Rq.gens()
            if F % bq != 0:
                raise ValueError("reducible case: b does not divide F")
            q = Rq(F / bq)
            d1 = lcm([cf.denominator() for cf in q.coefficients()])
            f2 = d1 * q
            Q = f2(bq, aq)

            def rec23(e, X, Y):
                return rec(e, Y, X)
            sols += _reducible_cubic_thue_mahler(Q, d1, d, S, rec23)
    return sols


def _plus_odd_class_number_non_coprime_to_three(d):
    r"""
    k odd, plus case, d in _PLUS_ODD_CLASS_NUMBER_NON_COPRIME_TO_THREE: K = Q(sqrt(-d)) has 3 | h_K (Lemma
    3.9).  Let P be a prime ideal generating the cyclic class group and write
    P^{h_K} = <a0 + b0 omega>.  For s = 0, 1, 2 set
        (a0 + b0 omega)^s = A_s + C_s omega,
        (a0 + b0 omega)^s sqrt(-d) = B_s + D_s omega.
    With (a + b omega)^3 = h(a, b) + g(a, b) omega we get the Thue-Mahler
    equations
        d^{(k-1)/2} = (A_s g - C_s h)/(A_s D_s - B_s C_s),
        y          = (D_s h - B_s g)/(A_s D_s - B_s C_s),
    with (a, b) | p^{s h_K/3}.
    """
    sols = []
    S = ZZ(d).prime_factors()
    K = QuadraticField(-d, "r")
    r = K.gen()
    conj = [phi for phi in K.automorphisms() if phi(r) != r][0]
    omega = [om for om in K.ring_of_integers().basis() if om != 1][0]
    R = PolynomialRing(K, ["a", "b"])
    aa, bb = R.gens()

    Cl = K.class_group()
    h = Cl.order()

    P = None
    p = ZZ(2)
    while P is None:
        for Pc in K.ideal(p).prime_factors():
            if not Cl(Pc).is_one():
                P = Pc
                break
        p = next_prime(p)
    pp = ZZ(P.smallest_integer())

    alpha = (P ** h).gens_reduced()[0]
    a0, c0 = _to_basis(alpha, omega)
    a0, c0 = ZZ(round(a0)), ZZ(round(c0))
    base = a0 + c0 * omega

    cube = (aa + omega * bb) ** 3
    hc, gc = _poly_basis(cube, omega)
    hc = hc.change_ring(QQ)
    gc = gc.change_ring(QQ)

    for s in range(3):
        prod_s = base ** s
        A_s, C_s = _to_basis(prod_s, omega)
        A_s, C_s = ZZ(round(A_s)), ZZ(round(C_s))
        p2 = prod_s * r
        B_s, D_s = _to_basis(p2, omega)
        B_s, D_s = ZZ(round(B_s)), ZZ(round(D_s))
        det = A_s * D_s - B_s * C_s
        if det == 0:
            continue

        F = A_s * gc - C_s * hc
        if not F.is_irreducible():
            continue
        det_abs = abs(det)
        if det < 0:
            F = -F

        max_m = ZZ(s * h / 3)
        for m in range(int(max_m) + 1):
            p3 = p ** (3 * m)
            if det_abs % p3 != 0:
                continue
            a_mult = det_abs / p3
            if a_mult == 0:
                continue
            try:
                tm_sols = ThueMahlerSolver(F, S, a=a_mult).solve()
            except Exception as e:
                print(f"Thue-Mahler failed for plus odd D0, d={d}, s={s}: {e}")
                continue
            for sol in tm_sols:
                ap, bp = sol[0], sol[1]
                a1, b1 = p ** m * ap, p ** m * bp
                e = _tm_exponent(sol, d)
                if e is None:
                    continue
                k = 2 * e + 1
                yK = (D_s * hc(a1, b1) - B_s * gc(a1, b1)) / det
                y0 = QQ(yK)
                if y0.denominator() != 1:
                    continue
                y0 = ZZ(y0)
                xc = y0 ** 2 + d ** k
                x0 = _cube_root(xc)
                if x0 is not None:
                    sols.append((x0, y0, d, k))
    return sols


# --------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------- #
def case_n_2(d):
    r"""All solutions (x, y, d, k) of y^2 +/- d^k = x^3, gcd(x, d) = 1."""
    return _dedup(case_n_2_minus(d) + case_n_2_plus(d))


def main():
    for d in range(2, D_BOUND + 1):
        if ZZ(d).is_perfect_power():
            continue
        minus = case_n_2_minus(d)
        plus = case_n_2_plus(d)
        print(f"d={d}: minus (y^2-d^k=x^3): {minus}")
        print(f"       plus  (y^2+d^k=x^3): {plus}")


if __name__ == "__main__":
    main()