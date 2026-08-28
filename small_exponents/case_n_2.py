from sage.all import (EllipticCurve, ZZ, EllipticCurve_from_cubic, polygen, QQ, sqrt, prod)

from config import D_BOUND


##### Integer solutions of the equation y^2 \pm d^k = x^3 ####

#### The case y^2 - d^k = x^3

def case_n_2_minus(d):
    sols = []
    sols_sign, problematic_k0s = _s_integral_points(d, -1)
    sols.append(sols_sign)

    sols_temp = []
    even_k0 = (0 in problematic_k0s) or (2 in problematic_k0s) or (4 in problematic_k0s)
    odd_k0 = (1 in problematic_k0s) or (3 in problematic_k0s) or (5 in problematic_k0s)
    if even_k0:
        sols_temp += _minus_irreducible_cubic_even_case(d)
    for k0 in problematic_k0s:
        if k0 % 2 == 0:
            sols_temp += _minus_reducible_cubic_even_case(d, k0)
        else:
            sols_temp += []
    for sol in sols_temp:
        if sol not in sols:
            sols.append(sol)
    return sols


def _s_integral_points(d, sign):
    sols = []
    problematic_k0s = []
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
            for pt in pts:
                P0 = phi(pt)
                x0 = P0[0].denominator()
                y0 = P0[1].denominator()
                dk = sign * (x0 ** 3 - y0 ** 2)
                if dk.is_perfect_power():
                    b, k = dk.perfect_power()
                    if b == d:
                        sols.append((x0, y0, d, k))
        except:
            problematic_k0s.append(k0)
            print(f"We couldn't compute the S-integral points of k0={k0} and sign={sign}.")

    return sols, problematic_k0s


def _minus_reducible_cubic_even_case(d, k0):
    sols = []
    E = EllipticCurve([0, 0, 18 * d**(ZZ(k0/2)), 0, -108 * d**k0])
    S = ZZ(d).prime_factors()
    if 2 not in S:
        S.append(2)
    Emin = E.minimal_model()
    iso = Emin.isomorphism_to(E)
    for p in iso.u.denominator().prime_factors():
        if p not in S:
            S.append(p)
    try:
        pts = Emin.S_integral_points(S=S)
    except:
        print(f"We couldn't compute the S-integral points of elliptic curve associate to the reducible cubic form "
              f"for the sign={-1} and d={d} and k0={k0}.")

    for pt in pts:
        pt0 = iso(pt)
        fr0 = pt0[1] / (3*pt0[0])
        denom0 = fr0.denominator()
        if not denom0.is_power_of(d):
            continue
        _, lam = denom0.perfect_power()
        x1 = fr0.numerator()
        x2 = x1**3 - 2*d**(3*lam + ZZ(k0/2))
        b, e = x2.perfect_power()
        if e % 3 == 0:
            x2 = b**(ZZ(e/3))
        k = 6*lam + k0
        x0 = x1 * x2
        y2 = x0**3 + d**k
        if y2.is_square():
            y0 = ZZ(sqrt(y2))
            sols.append((x0, y0, d, k))
    return sols


def _minus_irreducible_cubic_even_case(d):
    sols = []
    S = ZZ(d).prime_factors()
    f1 = [1, 0, 0, -2]
    f2 = [2, 0, 0, -1]
    tms1 = ThueMahlerSolver(f1, S).solve()
    tms2 = ThueMahlerSolver(f2, S).solve()

    for sol in tms1 + tms2:
        x1 = sol[0]
        x2 = sol[1]
        dk = prod([p^e for p,e in sol[2].items()])
        if dk.is_power_of(d):
            _, e = dk.perfect_power()
            k = 2 * e
            x0 = 2*x1*x2
            y02 = x0**3 + d**k
            if y02.is_square() and x0 != 0 and k > 0:
                y0 = ZZ(sqrt(y02))
                sols.append((x0, y0, d, k))
    return sols
