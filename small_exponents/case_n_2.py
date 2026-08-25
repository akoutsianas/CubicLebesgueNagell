from sage.all import (EllipticCurve, ZZ)

from config import D_BOUND


sols = []
for d in range(2, D_BOUND + 1):
    if not ZZ(d).is_perfect_power():
        print(f"d: {d}")
        S = ZZ(d).prime_factors()
        for k0 in range(6):
            E0 = E0min = EllipticCurve([0, -d^k0])
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
                    dk = x0^3 - y0^2
                    if dk.is_perfect_power():
                        b, k = dk.perfect_power()
                        if b == d:
                            sols.append((x0, y0, d, k))
            except:
                print(f"We couldn't compute the S-integral points of {E0} for d={d} and S={S}")

print(f"The solutions are {sols}.")