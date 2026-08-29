"""Independent validator for lcb_abc392_f ("Insert"): for i=1..N, insert i
into array A so it becomes the P_i-th element (1-indexed), starting from
an empty array. Unlike the other ambiguous-set problems, this one has
exactly one correct answer per input -- there is nothing to accept beyond
the single deterministic result of the stated insertion process.

The reference intended solution processes insertions in reverse with a
Fenwick tree / order-statistics structure for O(N log N) efficiency (the
problem's N is up to 5*10^5). A validator has no efficiency requirement,
only a correctness one, so this uses the direct O(N^2) forward simulation
instead -- Python's own list.insert(idx, value) already implements the
problem's insertion semantics exactly, including clamping idx to the
list's current end if it would otherwise run past it.

Confirmed dataset defect (independently re-derived here, and matching
Fable 5's round-2 finding, docs/fable_verification_report_2.md
"abc392_f finding worth documenting"): two official test cases have
out-of-constraint input (P_i > i, violating the stated "1 <= P_i <= i"),
and their *stored* reference outputs contain literal zeros
("5 0 4 0 1", "0 3 0 5 2 1") -- impossible for a permutation of 1..N,
confirming the reference generator itself broke on invalid input for
these two cases. This validator's simulate() output for both is a
genuine permutation of 1..N (no zeros, no duplicates), and is what
"correct" means here, not the stored (corrupted) string.

Verified 2026-08-19 against all 42 official test cases (32 visible + 10
holdout) pulled directly from a real eval sample's metadata: 40/42 exact
matches against the stored reference; the other 2 are exactly the two
corrupted-reference cases above, where simulate()'s output differs from
the stored string but is independently confirmed to be a valid
permutation of 1..N (which the stored, zero-containing string is not).
"""


def simulate(inp_text):
    """Forward-simulate the insertion process. Returns the correct final
    array as a space-separated string."""
    lines = inp_text.strip().split("\n")
    N = int(lines[0])
    P = list(map(int, lines[1].split()))
    A = []
    for i in range(1, N + 1):
        idx = P[i - 1] - 1
        idx = max(0, min(idx, len(A)))  # Python list.insert's own clamp semantics, made explicit
        A.insert(idx, i)
    return " ".join(map(str, A))


def _is_permutation(text, n):
    try:
        vals = list(map(int, text.strip().split()))
    except ValueError:
        return False
    return sorted(vals) == list(range(1, n + 1))


def check_answer(inp, out_text):
    """Compare a submitted output against the correct simulated result.

    Since this problem has exactly one correct answer, an exact string
    match against simulate()'s output is both necessary and sufficient --
    there is no "any valid answer" case to handle, unlike the other
    ambiguous-set problems.
    """
    correct = simulate(inp)
    got = out_text.strip()
    if got == correct:
        return True, "ok"
    N = int(inp.strip().split("\n")[0])
    if not _is_permutation(got, N):
        return False, f"not a permutation of 1..{N}: {got!r}"
    return False, f"valid permutation but wrong: got {got!r}, expected {correct!r}"


def sanity():
    # Sample 1 from the problem statement
    assert simulate("4\n1 1 2 1") == "4 2 3 1"
    # Sample 2
    assert simulate("5\n1 2 3 4 5") == "1 2 3 4 5"
    assert check_answer("4\n1 1 2 1", "4 2 3 1") == (True, "ok")
    assert check_answer("4\n1 1 2 1", "1 2 3 4")[0] is False

    # The two confirmed corrupted-reference cases: stored reference is
    # invalid (contains 0, impossible for a permutation), simulate()'s
    # output is a genuine permutation and does NOT match the stored string.
    corrupt_1 = ("5\n5 4 3 2 1", "5 0 4 0 1")
    corrupt_2 = ("6\n3 3 2 5 4 6", "0 3 0 5 2 1")
    for inp, stored_ref in [corrupt_1, corrupt_2]:
        N = int(inp.split("\n")[0])
        assert not _is_permutation(stored_ref, N), \
            f"expected the stored reference to be invalid (contains 0s): {stored_ref!r}"
        sim = simulate(inp)
        assert _is_permutation(sim, N), f"simulate() should produce a genuine permutation: {sim!r}"
        v, note = check_answer(inp, stored_ref)
        assert v is False, f"stored corrupted reference should fail check_answer: {note}"
        # a submission matching the SIMULATED (correct) output should pass
        assert check_answer(inp, sim) == (True, "ok")

    print("sanity: all official samples + both confirmed corrupted-reference cases pass")


if __name__ == "__main__":
    sanity()
