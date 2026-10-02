r"""
Cubic Lebesgue-Nagell equations -- the case :math:`n = 3` (Section 3.2).

We solve

    (3.27)      x^3 - d^k = y^3 ,        gcd(x, d) = 1 ,

for ``2 <= d <= D_BOUND`` with ``d`` not a perfect power.

Writing :math:`d^k = x^3 - y^3 = (x - y)(x^2 + xy + y^2)` and setting
:math:`X = x - y`, :math:`Y = y` (hence :math:`x = X + Y`) reduces the problem
to the reducible cubic binary form

    (3.25)      d^k = X (X^2 + 3 X Y + 3 Y^2) ,

where :math:`X` and :math:`X^2 + 3XY + 3Y^2` are :math:`S`-units with
:math:`S = \{p : p \mid d\}`.  By Section 2.3 it is enough to compute the
:math:`S`-integral points of the elliptic curve

    (3.26)      Y^2 + 9 d^i Y = X^3 - 27 d^{2i} ,        i = 0, 1, 2 ,

which is (2.11) with :math:`A = 1`, :math:`B = 3`, :math:`C = 3` and
:math:`D` the cube-free part of :math:`d^i`.  Every :math:`S`-integral point
gives a primitive pair :math:`(X, Y)`; we keep those for which
:math:`X (X^2 + 3XY + 3Y^2)` is exactly a power of :math:`d` and recover

    x = X + Y ,        y = Y .

The plus sign :math:`x^3 + d^k = y^3` is equivalent to the minus sign through
Remark 1.2 (:math:`(x, y) \mapsto (-x, -y)`), so it is not treated separately.

The script follows the structure of ``case_n_2.py``: failures of the
:math:`S`-integral-points computation are reported through
:func:`_report_failure`, and :func:`main` can spread the work over several CPUs
and reports both the wall-clock and the total CPU time.
"""

import contextlib
import io
import multiprocessing
import resource
import sys
import time

from sage.all import ZZ, QQ, EllipticCurve, PolynomialRing, gcd, prod

sys.path.append("/home/akoutsianas/Sage/DiophantineSolvers")


D_BOUND = 10


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


def _report_failure(reason, case=None, k=None, k0=None, info=""):
    r"""
    Print a uniform diagnostic whenever a sub-computation fails.

    Only the arguments that are genuinely available where the failure occurs
    are passed; the remaining ones stay ``None`` and are omitted from the
    message.

    INPUT:

    - ``reason`` -- the method that failed, ``"S-integral points"`` here.
    - ``case``   -- ``"minus"``/``"plus"`` (unused for ``n = 3``), or ``None``.
    - ``k``      -- ``"even"``, ``"odd"`` or ``None``.
    - ``k0``     -- a residue class, an iterable of residue classes, or
      ``None``.
    - ``info``   -- extra information (the equation or curve involved).

    The single printed line is what :func:`main` collects as a warning.
    """
    parts = []
    if case is not None:
        parts.append(f"case={case}")
    parts.append(f"reason={reason}")
    if k is not None:
        parts.append(f"k={k}")
    if k0 is not None:
        vals = sorted(k0) if isinstance(k0, (list, tuple, set, frozenset)) else [k0]
        parts.append("k0=" + ",".join(str(v) for v in vals))
    msg = "FAILED: " + ", ".join(parts)
    if info:
        msg += f" ({info})"
    print(msg)


def _power_of_d_exponent(m, d):
    r"""Return the integer e >= 0 with m == d^e, or None if no such e exists.

    Uses Sage's :meth:`is_power_of` and :meth:`perfect_power`.
    """
    m = ZZ(m)
    if m <= 0:
        return None
    if m == 1:
        return ZZ(0)
    if not m.is_power_of(d):
        return None
    b, e = m.perfect_power()
    if b != d:
        return None
    return ZZ(e)


# --------------------------------------------------------------------- #
# Section 2.3: reducible cubic forms via S-integral points (3.26)
# --------------------------------------------------------------------- #
def _reducible_cubic_thue_mahler(Q, c, d, S, rec):
    r"""
    Solve  X * Q(X, Y) = c*d^e  for coprime integers (X, Y), where Q is the
    quadratic form  Q(X, Y) = A X^2 + B X Y + C Y^2  with A, B, C in Z and
    the linear factor is always the first variable X.

    Section 2.3, eqs. (2.7)-(2.9).  Write the S-unit c d^e = D Z^3 with D
    cube-free (D depends only on e mod 3).  Then, for (X, Y, Z) with
    X*Q(X, Y) = D*Z^3, the point

        ( D C Z / X ,  C^2 D Y / X )

    is an S-integral point of the elliptic curve

        (2.9)   W^2 + B C D W = T^3 - A C^3 D^2
                ( W = C^2 D Y / X ,  T = D C Z / X ).

    In particular  Y / X = W / (C^2 D), which recovers the primitive (X, Y).

    For the ``n = 3`` case Q(X, Y) = X^2 + 3XY + 3Y^2 and c = 1, so (2.9) is
    the curve (3.26),  W^2 + 9 D W = T^3 - 27 D^2.

    The caller must arrange that the actual linear factor is the first
    variable (swapping the variables if necessary).

    For each solution (e, X, Y) the callback ``rec(e, X, Y)`` is invoked; its
    non-None return values are collected.

    This routine only knows ``d`` and the cube-free part ``D`` of ``c d^r``
    (for ``r = 0, 1, 2``); it has no notion of the parity of ``k`` or of the
    residue classes ``k0``, so its failure messages leave those unset.
    """
    sols = []
    R = Q.parent()
    Xv, Yv = R.gens()
    A = ZZ(Q.monomial_coefficient(Xv ** 2))
    B = ZZ(Q.monomial_coefficient(Xv * Yv))
    C = ZZ(Q.monomial_coefficient(Yv ** 2))
    c = ZZ(c)
    if A == 0 or C == 0 or c == 0:
        return sols

    # Only the primes of the S-unit c*d^e can occur in a denominator
    # (the point is (D C Z / X, C^2 D Y / X) with X | c*d^e).
    Sprime = sorted(set(ZZ(p) for p in S) | set(c.prime_factors()))

    found = set()
    for r in range(3):
        # cube-free part D of c * d^r  (so c*d^r = D * Z^3)
        N = c * d ** r
        sgn = 1 if N > 0 else -1
        D = sgn * prod(q ** (a % 3) for q, a in ZZ(abs(N)).factor())
        if D == 0:
            continue

        # elliptic curve (2.9) / (3.26)
        E = EllipticCurve([0, 0, B * C * D, 0, -A * C ** 3 * D ** 2])
        Emin = E if E.is_minimal() else E.minimal_model()
        phi = Emin.isomorphism_to(E)
        Spts = list(Sprime)
        for p in phi.u.denominator().prime_factors():
            if p not in Spts:
                Spts.append(p)

        pts = None
        try:
            pts = Emin.S_integral_points(S=Spts)
        except Exception:
            # mwrank could not determine the Mordell-Weil basis: try to
            # obtain it from PARI's ellrank (with increasing effort) and retry.
            for effort in (0, 2, 4):
                try:
                    mb = Emin.gens(algorithm="pari", pari_effort=effort)
                    pts = Emin.S_integral_points(S=Spts, mw_base=mb)
                    break
                except Exception:
                    continue
        if pts is None:
            _report_failure("S-integral points",
                            info=f"reducible cubic (3.26), d={d}, D={D}, r={r}")
            continue

        # Sage's S_integral_points may return only one point of a pair
        # {Q, -Q}; the S-integral points are closed under negation, so add
        # the negatives explicitly.
        for pt in list(pts) + [-Q for Q in pts]:
            P = phi(pt)
            W = QQ(P[1])
            ratio = W / (C ** 2 * D)              # = Y / X
            if ratio == 0:
                continue
            Y0 = ZZ(ratio.numerator())
            X0 = ZZ(ratio.denominator())
            for sg in (1, -1):
                X, Y = sg * X0, sg * Y0
                if gcd(X, Y) != 1:
                    continue
                XF = X * (A * X ** 2 + B * X * Y + C * Y ** 2)
                if XF % c != 0:
                    continue
                e = _power_of_d_exponent(XF // c, d)
                if e is None or e % 3 != r:
                    continue
                key = (X, Y, e)
                if key in found:
                    continue
                found.add(key)
                out = rec(e, X, Y)
                if out is not None:
                    sols.append(out)
    return sols


# --------------------------------------------------------------------- #
# the case n = 3 (Section 3.2):  x^3 - d^k = y^3
# --------------------------------------------------------------------- #
def case_n_3(d):
    r"""
    Return all solutions ``(x, y, d, k)`` of  ``x^3 - d^k = y^3`` with
    ``gcd(x, d) = 1`` and ``k >= 1``, obtained from the reducible cubic form
    (3.25) via the Section 2.3 method.
    """
    sols = []
    S = ZZ(d).prime_factors()

    # (3.25):  X * (X^2 + 3 X Y + 3 Y^2) = d^k     (A = 1, B = 3, C = 3, c = 1)
    R = PolynomialRing(QQ, ["X", "Y"])
    XX, YY = R.gens()
    Q = XX ** 2 + 3 * XX * YY + 3 * YY ** 2

    def rec(k, X, Y):
        if k < 1:                       # (1.1) assumes k >= 1
            return None
        x0 = X + Y                      # x = X + Y
        y0 = Y                          # y = Y
        if gcd(x0, d) != 1:             # gcd(x, d) = 1
            return None
        if x0 ** 3 - d ** k != y0 ** 3:  # safety check
            return None
        return (x0, y0, d, k)

    sols += _reducible_cubic_thue_mahler(Q, ZZ(1), d, S, rec)
    return _dedup(sols)


# --------------------------------------------------------------------- #
# timing helpers
# --------------------------------------------------------------------- #
def _cpu_seconds():
    r"""
    Total CPU seconds (user + system) used by this process and by the child
    processes it has reaped so far.
    """
    me = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime


def _self_cpu_seconds():
    r"""CPU seconds (user + system) used by this process alone."""
    me = resource.getrusage(resource.RUSAGE_SELF)
    return me.ru_utime + me.ru_stime


# --------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------- #
def _solve_d_worker(d):
    r"""
    Compute all solutions of ``x^3 - d^k = y^3`` for a single ``d`` and capture
    the diagnostic messages printed when some sub-computation fails.

    Returns ``(d, solutions, warnings, cpu_seconds)``, where ``cpu_seconds`` is
    the CPU time spent on this ``d`` (including child processes such as
    ``mwrank``).
    """
    cpu0 = _cpu_seconds()
    buf = io.StringIO()
    sols = []
    with contextlib.redirect_stdout(buf):
        try:
            sols = case_n_3(d)
        except Exception as exc:                   # pragma: no cover
            _report_failure("unexpected error", info=f"d={d}, " + repr(exc))
    warnings = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    return (d, sols, warnings, _cpu_seconds() - cpu0)


def main(ncpu=1):
    r"""
    Run the ``n = 3`` case for every ``2 <= d <= D_BOUND`` that is not a
    perfect power, using ``ncpu`` processes (``ncpu = None`` uses all cores).

    Results are printed as soon as each ``d`` finishes; at the end the values
    of ``d`` for which some sub-computation failed are listed, together with
    the wall-clock time and the total CPU time (summed over all processes).

    Since a single ``d`` may spawn ``mwrank`` subprocesses, every worker
    reports the CPU it used (itself plus its children) and ``main`` sums those
    figures, so the reported CPU time covers the whole computation.
    """

    if ncpu is None or ncpu < 1:
        ncpu = 1
    ds = [d for d in range(2, D_BOUND + 1) if not ZZ(d).is_perfect_power()]

    unsolved = []
    results = {}

    def report(d, sols, warnings):
        results[d] = (sols, warnings)
        for w in warnings:
            print(f"d={d}: WARNING: {w}")
        print(f"d={d}: x^3 - d^k = y^3: {sols}")
        if warnings:
            unsolved.append(d)
        sys.stdout.flush()

    cpu_worker = 0.0
    cpu_parent0 = _self_cpu_seconds()
    t0 = time.perf_counter()

    if ncpu > 1:
        # Python 3.14 changed the default start method on Linux to
        # "forkserver", which re-imports the main module in each child and
        # breaks when the code is run through ``sage -c`` (there is no proper
        # __main__ module).  Sage's parallel machinery relies on fork, so use
        # the fork context explicitly.
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=ncpu) as pool:
            for (d, sols, warnings, cpu) in pool.imap_unordered(_solve_d_worker, ds):
                cpu_worker += cpu
                report(d, sols, warnings)
        # The workers report their own CPU (including mwrank subprocesses);
        # add the parent's own time spent orchestrating them.
        cpu_total = cpu_worker + (_self_cpu_seconds() - cpu_parent0)
    else:
        for d in ds:
            d, sols, warnings, cpu = _solve_d_worker(d)
            cpu_worker += cpu
            report(d, sols, warnings)
        cpu_total = cpu_worker

    real = time.perf_counter() - t0

    print()
    print(f"Number of d values: {len(ds)}")
    print(f"Cases not fully solved: {sorted(unsolved)}")
    print(f"Processes used: {ncpu}")
    print(f"Real time: {real:.2f} s")
    print(f"CPU time:  {cpu_total:.2f} s")
    if real > 0:
        print(f"CPU/real:  {cpu_total / real:.2f}")
    return results


if __name__ == "__main__":
    _ncpu = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(ncpu=_ncpu)
