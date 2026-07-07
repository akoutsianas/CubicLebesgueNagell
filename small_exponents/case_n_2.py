from sage.all import (EllipticCurve, ZZ)


for d in range(3, 101):
    if (d - 1)**(1/3) not in ZZ and (d + 1)**(1/3) not in ZZ:
        Ed = EllipticCurve([0, -d])
        pts = Ed.integral_points()
        pts = [P for P in pts if gcd(P[0], P[1]) == 1]
        print(f"d:{d}, Points: {pts}")