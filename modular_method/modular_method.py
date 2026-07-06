from sage.all import (ZZ, EllipticCurve, Newforms, prime_range, gcd, prod)

class CubicLebesgueNagellModularMethod:

    def __init__(self, d):
        self.d = d
        self.D = ZZ(-d) if d % 3 == 2 else ZZ(d)
        self._levels_info = self._frey_curves_info()
        self._small_exponents = []

    def _frey_curves_info(self):
        Nn = prod([p for p in self.D.prime_factors() if p != 3])
        if self.D % 3 == 0:
            if self.D.valuation(3) == 1:
                e3 = 4
            elif self.D.valuation(3) == 2:
                e3 = 3
            elif self.D.valuation(3) == 3:
                e3 = 0
            else:
                e3 = 1
            info = [
                {'Nn': 3**e3 * Nn, 'frey_curve': lambda x: EllipticCurve([3*x, 0, self.D, 0, 0])}
            ]
        else:
            info = [
                {'Nn': 3 * Nn, 'frey_curve': lambda x: EllipticCurve([3 * x, 0, x**3 - self.D, 0, 0])},
                {'Nn': 3**2 * Nn, 'frey_curve': lambda x: EllipticCurve([3 * x, 0, self.D, 0, 0])},
                {'Nn': 3**3 * Nn, 'frey_curve': lambda x: EllipticCurve([3 * x, 0, self.D, 0, 0])}
            ]
        return info

    def elimination_method_trace_of_frobenius(self, primes_bound=50):

        for level_info in self._levels_info:
            Nn = level_info['Nn']
            Ex = level_info['frey_curve']

            print(f"##### Elimination step for level {Nn}. #####")

            newforms = Newforms(Nn, names='a')

            for newf in newforms:
                Bnewf = []
                for p in prime_range(primes_bound):
                    if p != 3 and self.D % p != 0:
                        Bp = p
                        apnewf = newf[p]
                        for x0 in range(p):
                            if (x0 **3 - self.D) % p != 0:
                                apEx0 = p + 1 - Ex(x0).reduction(p).order()
                                Bp *= (apnewf - apEx0).norm()
                            else:
                                Bp *= (apnewf**2 - (p + 1)**2).norm()
                        if Bp != 0:
                            Bnewf.append(Bp)

                if len(Bnewf) != 0:
                    small_exp_newf = ZZ(gcd(Bnewf)).prime_factors()
                    for p in small_exp_newf:
                        if p not in self._small_exponents:
                            self._small_exponents.append(p)
                else:
                    print(f"We could not eliminate this newform!")

        print(f"Small exponents are {self._small_exponents}.")

