"""Independent validator for lcb_arc191_c.

Requirement (from the problem statement): given N, output positive
integers (A, M), both in [1, 10^18], such that the smallest n >= 1 with
M | A^n - 1 is exactly N (the multiplicative order of A mod M).

Order-check (standard, complete): ord(A mod M) == N iff
  pow(A, N, M) == 1 % M   AND   for every prime p | N: pow(A, N//p, M) != 1 % M.
(`1 % M` handles M == 1, where every n works so the order is 1.)
gcd(A, M) > 1 with M > 1 => no n exists => invalid.

Each test is a batch of T sub-cases; the real checker exact-matches the
whole block, so a single unmatchable line (e.g. the reference's arbitrary
`20250126 1` for N=1, where many other (A,1) pairs are equally valid)
fails the entire test even when every sub-case is individually correct.
This validator checks every sub-case of a block independently via
`validate_block`, then `check_answer` reduces that to the same
(bool, note) shape used by the rest of this project's validators.

Moved 2026-08-19 from verification/fable_verify/fv_arc191c.py into this
shape for consistency with the other 8 validators ahead of open-sourcing
-- pure reorganization of `order_is`/`validate_block`/`sanity`, bodies
unchanged; `check_answer` is a new thin wrapper (not present in the
original file, which drove validate_block directly from its own sweep
loop) added only so this file matches the other validators' call
convention. fv_arc191c.py now imports order_is/validate_block from this
module rather than defining them inline.
"""
import math

LIMIT = 10**18


def prime_factors(n):
    fs = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            fs.add(d)
            n //= d
        d += 1
    if n > 1:
        fs.add(n)
    return fs


def order_is(A, M, N):
    if not (1 <= A <= LIMIT and 1 <= M <= LIMIT):
        return False, "bounds"
    if M > 1 and math.gcd(A, M) != 1:
        return False, "gcd>1, no order exists"
    one = 1 % M
    if pow(A, N, M) != one:
        return False, f"A^N != 1 mod M"
    for p in prime_factors(N):
        if pow(A, N // p, M) == one:
            return False, f"order divides N/{p}"
    return True, "ok"


def validate_block(inp, out_text):
    """Validate a model's whole output block for one batched test.
    Returns (all_valid, n_valid, n_sub, first_bad_note)."""
    lines_in = inp.strip().split("\n")
    T = int(lines_in[0])
    Ns = [int(lines_in[1 + i]) for i in range(T)]
    out_lines = [l for l in out_text.strip().split("\n") if l.strip()]
    if len(out_lines) != T:
        return False, 0, T, f"expected {T} lines, got {len(out_lines)}"
    n_valid = 0
    first_bad = ""
    for N, line in zip(Ns, out_lines):
        parts = line.split()
        if len(parts) != 2:
            if not first_bad:
                first_bad = f"malformed line {line[:30]!r}"
            continue
        try:
            A, M = int(parts[0]), int(parts[1])
        except ValueError:
            if not first_bad:
                first_bad = f"non-integer line {line[:30]!r}"
            continue
        ok, why = order_is(A, M, N)
        if ok:
            n_valid += 1
        elif not first_bad:
            first_bad = f"N={N}, (A,M)=({A},{M}): {why}"
    return n_valid == T, n_valid, T, first_bad


def check_answer(inp, out_text):
    """Thin wrapper around validate_block matching this project's other
    validators' (bool, note) convention."""
    all_valid, n_valid, n_sub, note = validate_block(inp, out_text)
    if all_valid:
        return True, "ok"
    return False, note or f"{n_valid}/{n_sub} sub-cases valid"


def sanity():
    """The validator must accept the reference's own answers (both tests)
    and known-good constructions, and reject known-bad ones."""
    assert order_is(2, 7, 3)[0]
    assert order_is(100, 777, 3)[0]          # alternate from the statement
    assert order_is(20250126, 1, 1)[0]       # reference N=1
    assert order_is(2, 1, 1)[0]              # any A with M=1
    assert not order_is(2, 7, 6)[0]          # order is 3, not 6
    assert not order_is(2, 8, 3)[0]          # gcd 2
    assert order_is(196, 38025, 195)[0]      # reference holdout line (A=N+1, M=N^2)
    assert not order_is(0, 5, 1)[0]          # bounds
    # check_answer wrapper, batched block of the two statement examples
    assert check_answer("2\n3\n3", "2 7\n100 777") == (True, "ok")
    v, note = check_answer("2\n3\n3", "2 7\n2 8")
    assert v is False, note
    print("sanity: all validator self-tests pass")


if __name__ == "__main__":
    sanity()
