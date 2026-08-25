from sage.all import (EllipticCurve, ZZ, polygen)

from config import D_BOUND
from small_exponents.utils import polynomial_not_perfect_powers_values


##### Solutions for d^k = x^3 - 1 #####

sols = []

# No solutions using Algorithm 1 in [8]

x = polygen(ZZ, 'x')
f = x^3 - 1
failed_d_values = []
for d in range(2, D_BOUND + 1):
    if not ZZ(d).is_perfect_power():
        bol = polynomial_not_perfect_powers_values(f, d, S_bound=100)
        if not bol:
            print(f"Algorithm 1 from [8] failed for d={d}")
            failed_d_values.append(d)

# Solutions for k0 = 0

Edm1 = EllipticCurve([0, -1])
for pt in Edm1.integral_points():
    if pt[0] * pt[1] != 0:
        d, lam = ZZ(pt [1]).abs().perfect_power()
        if d in failed_d_values:
            sols.append((d, 2*lam, pt[0]))

# Solutions for k0 = 1

for d in failed_d_values:
    Emd = EllipticCurve([0, -d^3])
    try:
        for pt in Emd.integral_points():
            if pt[0] * pt[1] != 0 and pt[0] % d == 0:
                d, lam2 = ZZ(pt[1]).abs().perfect_power()
                if lam2 >= 2:
                    lam = lam2 - 2
                    x0 = ZZ(pt[0] / d)
                    sols.append((d, 2*lam + 1, x0))
    except:
        print(f"We are not able to compute integral points on elliptic curve {Emd} for d = {d} for the case "
              f"{d}^k = x^3 - 1")

print(f"The solutions for the equation d^k = x^3 - 1 are (d, k, x) = {sols}")



##### Solutions for d^k = x^3 + 1 #####

sols = []

# Solutions for k0 = 0
Ed1 = EllipticCurve([0, 1])

for pt in Ed1.integral_points():
    if pt[0] * pt[1] != 0:
        d, lam = ZZ(pt [1]).abs().perfect_power()
        sols.append((d, 2*lam, pt[0]))


# Solutions from integral points when k0 = 1
for d in range(2, D_BOUND + 1):
    if not ZZ(d).is_perfect_power():
        Ed = EllipticCurve([0, d^3])
        try:
            for pt in Ed.integral_points():
                if pt[0] * pt[1] != 0 and pt[0] % d == 0:
                    d, lam2 = ZZ(pt[1]).abs().perfect_power()
                    if lam2 >= 2:
                        lam = lam2 - 2
                        x0 = ZZ(pt[0] / d)
                        sols.append((d, 2*lam + 1, x0))
        except:
            print(f"We are not able to compute integral points on elliptic curve {Ed} for d = {d} for the case "
                  f"{d}^k = x^3 + 1")

print(f"The solutions for the equation d^k = x^3 + 1 are (d, k, x) = {sols}")



