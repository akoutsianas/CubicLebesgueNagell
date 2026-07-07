from sage.all import (ZZ, gcd)


x = polygen(ZZ, 'x')
for d in range(3, 101):
    if (d - 1)**(1/3) not in ZZ and (d + 1)**(1/3) not in ZZ:
        sols = []
        for d1 in ZZ(d).divisors():
            f = 3*x**2 - 3*d1*x + d1**2 - d/d1
            g = 3*x**2 + 3*d1*x + d1**2 + d/d1
            rts = f.roots() + g.roots()
            for r0 in rts:
                x0 = r0[0]
                y0 = x0**3 - d
                y0 = y0**(1/3)
                if y0 in ZZ and gcd(x0, y0) == 1:
                    sols.append((x0, y0))
        print(f"d:{d}, sols: {sols}")