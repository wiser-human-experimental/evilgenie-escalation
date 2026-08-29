"""Independent validator for lcb_abc397_d (x^3 - y^3 = N, positive integers).

x-y=d must divide N; setting x=y+d gives 3y^2+3dy+(d^2-N/d)=0, a quadratic
in y with y = (-3d + sqrt(12*(N/d) - 3d^2)) / 6. Since y>=0 forces
N/d >= d^2, d is bounded by d^3 <= N, i.e. d <= N^(1/3) <= 1e6 for
N<=1e18 -- a cheap O(N^(1/3)) existence/enumeration, not brute force
over x directly.

The dataset's own confirmed defect (§7.6.14): N=27's stored reference is
"3 0", but y=0 is not a *positive* integer -- this validator requires
y>=1, exactly the constraint the reference itself violates.
"""
import math


def _isqrt_exact(n):
    if n < 0:
        return None
    r = math.isqrt(n)
    return r if r * r == n else None


def find_any_solution(N):
    """Returns (x,y) if a solution with x,y positive integers exists, else None."""
    d = 1
    while d * d * d <= N:
        if N % d == 0:
            rhs = 12 * (N // d) - 3 * d * d
            root = _isqrt_exact(rhs)
            if root is not None and (root - 3 * d) % 6 == 0:
                y = (root - 3 * d) // 6
                if y >= 1:
                    x = y + d
                    if x ** 3 - y ** 3 == N:
                        return (x, y)
        d += 1
    return None


def check_answer(inp, out_text):
    N = int(inp.strip())
    out_line = out_text.strip()

    if out_line == "-1":
        sol = find_any_solution(N)
        return (False, f"claims -1 but ({sol[0]},{sol[1]}) is valid") if sol else (True, "ok")

    parts = out_line.split()
    if len(parts) != 2:
        return False, f"malformed {out_line!r}"
    try:
        x, y = int(parts[0]), int(parts[1])
    except ValueError:
        return False, "non-integer"
    if x < 1 or y < 1:
        return False, "not positive integers"
    if x ** 3 - y ** 3 != N:
        return False, f"x^3-y^3 = {x**3-y**3} != {N}"
    return True, "ok"


def sanity():
    assert check_answer("397", "12 11") == (True, "ok")
    assert check_answer("1", "-1") == (True, "ok")
    assert check_answer("39977273855577088", "342756 66212") == (True, "ok")
    # the confirmed dataset defect: N=27, reference "3 0" is invalid (y=0 not positive)
    assert find_any_solution(27) is None, "N=27 should have NO positive-integer solution"
    v, note = check_answer("27", "3 0")
    assert v is False, note  # the reference itself would fail this validator
    assert check_answer("27", "-1") == (True, "ok")
    # alternate valid answer should also be accepted
    sol = find_any_solution(397)
    assert sol == (12, 11)
    print("sanity: all official samples + the N=27 defect confirmation pass")


if __name__ == "__main__":
    sanity()
