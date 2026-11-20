import contextlib

from cysignals.alarm import AlarmInterrupt, alarm, cancel_alarm
from sage.all import ZZ, QQ, gcd, lcm, Mod, prime_range


# --------------------------------------------------------------------- #
# time limits
# --------------------------------------------------------------------- #
class TimeLimitExceeded(BaseException):
    r"""
    Raised when a :func:`time_limit` block runs out of time.

    It derives from ``BaseException`` so that the broad ``except Exception``
    handlers used inside Sage do not swallow it.
    """


@contextlib.contextmanager
def time_limit(seconds):
    r"""
    Wall-clock limit for a block of code.

    It uses the ``alarm`` of ``cysignals``, which also interrupts long
    C-level computations (PARI, eclib, ...) running inside ``sig_on()``,
    not only Python code.  It must be used in the main thread of a process
    (this is where the ``multiprocessing`` workers run), and the blocks must
    not be nested.  ``seconds = None`` or ``seconds <= 0`` means no limit.
    """
    if seconds is None or seconds <= 0:
        yield
        return
    try:
        alarm(float(seconds))
        yield
    except AlarmInterrupt:
        raise TimeLimitExceeded() from None
    finally:
        cancel_alarm()


def format_duration(seconds):
    r"""Return ``seconds`` as a short human readable string, e.g. ``3 days``."""
    seconds = float(seconds)
    for unit, size in (("days", 86400), ("h", 3600), ("min", 60)):
        if seconds >= size:
            value = seconds / size
            return f"{value:.0f} {unit}" if value == int(value) else f"{value:.1f} {unit}"
    return f"{seconds:.0f} s"


# --------------------------------------------------------------------- #
# Mordell-Weil bases
# --------------------------------------------------------------------- #
class MordellWeilError(Exception):
    r"""Raised when :func:`mordell_weil_basis` cannot determine a basis."""


def _lseries_value_is_nonzero(E, derivative, multipliers=(2, 4, 8, 16)):
    r"""
    Return ``True`` if it is proved that ``L(E,1) != 0`` (``derivative =
    False``) or ``L'(E,1) != 0`` (``derivative = True``).

    We use :meth:`at1` / :meth:`deriv_at1` of Sage, which return an
    approximation together with a rigorous bound on the error, with
    ``k = m*sqrt(N)`` terms of the series for increasing ``m``; the value is
    nonzero as soon as the approximation is larger than the error.  The
    derivative is only meaningful when the root number is ``-1``.
    """
    L = E.lseries()
    f = L.deriv_at1 if derivative else L.at1
    sqrtN = ZZ(E.conductor()).isqrt() + 1
    for m in multipliers:
        value, err = f(k=m * sqrtN, prec=64)
        if abs(value) > 2 * err:
            return True
    return False


def _pari_point(E, P):
    r"""Convert a PARI point ``[x, y]`` into a point of the Sage curve ``E``."""
    return E([QQ(P[0]), QQ(P[1])])


def _saturated(E, points):
    r"""Return a full saturation of the independent points ``points`` of ``E``."""
    if not points:
        return []
    sat, _, _ = E.saturation(points)
    return list(sat)


def mordell_weil_basis(E, pari_efforts=(0, 1, 2, 4), pari_timeout=5 * 60,
                       lseries_timeout=10 * 60, heegner_timeout=30 * 60,
                       mwrank_timeout=60 * 60):
    r"""
    Return a basis of the free part of the Mordell-Weil group ``E(Q)``.

    The methods are tried in the following order, each one under its own
    time limit (in seconds, ``None`` meaning no limit):

    1. PARI's ``ellrank`` with increasing ``effort``.  If it finds as many
       independent points as its (2-descent) upper bound for the rank, these
       points, after saturation, form a basis.  If the upper bound is ``0``
       the rank is ``0``.
    2. Analytic methods, for curves whose rank is not determined by step 1
       (e.g. when ``Sha(E/Q)[2]`` is nontrivial or the generator has a large
       height):

       - root number ``+1`` and ``L(E,1) != 0``: the rank is ``0``
         (Kolyvagin);
       - root number ``-1`` and ``L'(E,1) != 0``: the rank is ``1``
         (Gross-Zagier-Kolyvagin) and a generator is obtained, after
         saturation, from the Heegner point computed by PARI's
         ``ellheegner``.

       The non-vanishing is proved with the error bounds of
       :meth:`at1` / :meth:`deriv_at1`.
    3. ``mwrank`` (``E.gens(proof=True)``).

    INPUT:

    - ``E`` -- an elliptic curve over `\QQ` given by an integral model.

    OUTPUT:

    A pair ``(basis, method)`` where ``basis`` is a list of points of ``E``
    (empty when the rank is ``0``) and ``method`` is a string describing how
    it was obtained.  A :class:`MordellWeilError` is raised if all the
    methods fail.
    """
    notes = []

    # 1. PARI ellrank
    pts = []
    upper = None
    for effort in pari_efforts:
        try:
            with time_limit(pari_timeout):
                _, r2, _, gens = E.pari_curve().ellrank(effort)
        except TimeLimitExceeded:
            notes.append(f"ellrank effort {effort}: timeout")
            continue
        except Exception as exc:
            notes.append(f"ellrank effort {effort}: {exc!r}")
            continue
        r2 = ZZ(r2)
        upper = r2 if upper is None else min(upper, r2)
        if len(gens) > len(pts):
            pts = [_pari_point(E, P) for P in gens]
        if len(pts) == upper:
            break
    if upper is not None and len(pts) == upper:
        return _saturated(E, pts), f"PARI ellrank (rank {upper})"
    notes.append(f"ellrank: {len(pts)} independent point(s), rank <= {upper}")

    # 2. analytic rank 0 or 1
    try:
        eps = E.root_number()
        if eps == 1 and not pts:
            with time_limit(lseries_timeout):
                rank0 = _lseries_value_is_nonzero(E, derivative=False)
            if rank0:
                return [], "analytic rank 0 (Kolyvagin)"
            notes.append("root number +1, L(E,1) != 0 not proved")
        elif eps == -1 and len(pts) <= 1:
            with time_limit(lseries_timeout):
                rank1 = _lseries_value_is_nonzero(E, derivative=True)
            if not rank1:
                notes.append("root number -1, L'(E,1) != 0 not proved")
            elif pts:
                return _saturated(E, pts), "analytic rank 1 (Gross-Zagier-Kolyvagin), point from PARI ellrank"
            else:
                with time_limit(heegner_timeout):
                    P = _pari_point(E, E.pari_curve().ellheegner())
                    if P.has_infinite_order():
                        return _saturated(E, [P]), "analytic rank 1 (Gross-Zagier-Kolyvagin), Heegner point"
                notes.append("ellheegner returned a torsion point")
    except TimeLimitExceeded:
        notes.append("analytic method: timeout")
    except Exception as exc:
        notes.append(f"analytic method: {exc!r}")

    # 3. mwrank
    try:
        with time_limit(mwrank_timeout):
            gens = E.gens(proof=True)
        return list(gens), f"mwrank (rank {len(gens)})"
    except TimeLimitExceeded:
        notes.append(f"mwrank: timeout after {format_duration(mwrank_timeout)}")
    except Exception as exc:
        notes.append(f"mwrank: {str(exc).splitlines()[0]}")

    raise MordellWeilError("; ".join(notes))


def s_integral_points(E, S, timeout=None):
    r"""
    Compute the S-integral points of ``E`` with a time limit.

    The Mordell-Weil basis is computed first with :func:`mordell_weil_basis`
    (which has its own time limits) and is then passed to Sage's
    :meth:`S_integral_points`, which runs for at most ``timeout`` seconds
    (``None`` means no limit).

    OUTPUT:

    A triple ``(points, status, info)``:

    - ``(points, None, method)`` on success, where ``method`` says how the
      Mordell-Weil basis was obtained;
    - ``(None, "basis", info)`` if the Mordell-Weil basis is not determined;
    - ``(None, "timeout", info)`` if :meth:`S_integral_points` runs out of
      time;
    - ``(None, "failure", info)`` if :meth:`S_integral_points` raises an
      error.
    """
    try:
        basis, method = mordell_weil_basis(E)
    except MordellWeilError as exc:
        return None, "basis", f"Mordell-Weil basis not determined: {exc}"
    try:
        with time_limit(timeout):
            pts = E.S_integral_points(S=list(S), mw_base=basis)
    except TimeLimitExceeded:
        return None, "timeout", (f"timeout after {format_duration(timeout)}, "
                                 f"Mordell-Weil basis from {method}")
    except Exception as exc:
        return None, "failure", f"failure: {exc!r}, Mordell-Weil basis from {method}"
    return pts, None, method


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
