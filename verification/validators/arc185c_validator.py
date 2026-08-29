"""Independent validator for lcb_arc185_c: given N, X, and array A, find
indices 1 <= i < j < k <= N with A_i + A_j + A_k == X, or report -1 if no
such triple exists.

Like arc190_a, existence-negation ("-1") can only be checked against what
the reference itself claims -- proving no triple exists in general would
require an independent O(N^2) or better search, which this validator does
not attempt (an accepted "-1" is only counted as adjudicated when the
reference also says "-1"; anything else is intentionally left
unadjudicated by check_answer's caller, not silently accepted). Any
triple the model *does* produce is fully independently verified against
the raw array and X, with no dependency on the reference.

Moved 2026-08-19 from verification/fable_verify/fv_p3.py (originally
`a185_validate`) into this shape for consistency with the other 8
validators ahead of open-sourcing -- pure reorganization, function body
unchanged. fv_p3.py now imports this module rather than defining the
function inline.
"""


def check_answer(inp, out_text, ref_text):
    lines = inp.strip().split("\n")
    N, X = map(int, lines[0].split())
    A = list(map(int, lines[1].split()))
    ref_first = ref_text.strip().split("\n")[0].strip()
    mline = out_text.strip().split("\n")[0].strip() if out_text.strip() else ""
    if mline == "-1":
        return (None, "claims -1") if ref_first == "-1" else (False, "-1 but reference found a triple")
    parts = mline.split()
    if len(parts) != 3:
        return False, f"malformed {mline[:30]!r}"
    try:
        i, j, k = map(int, parts)
    except ValueError:
        return False, "non-integer"
    if not (1 <= i < j < k <= N):
        return False, f"index constraint violated ({i},{j},{k})"
    ssum = A[i - 1] + A[j - 1] + A[k - 1]
    return (True, "ok") if ssum == X else (False, f"sum {ssum} != {X}")


def sanity():
    inp = "5 10\n1 2 3 4 5"
    # 2+3+5 = 10 -> indices (2,3,5)
    assert check_answer(inp, "2 3 5", "irrelevant") == (True, "ok")
    # out-of-order indices violate i<j<k even though the values would sum right
    v, note = check_answer(inp, "5 3 2", "irrelevant")
    assert v is False and "index constraint" in note, note
    # wrong sum
    v, note = check_answer(inp, "1 2 3", "irrelevant")
    assert v is False and "sum" in note, note
    # malformed
    v, note = check_answer(inp, "1 2", "irrelevant")
    assert v is False and "malformed" in note, note
    # model claims -1, reference agrees -> unadjudicated (None), not accepted
    v, note = check_answer("3 100\n1 2 3", "-1", "-1")
    assert v is None, note
    # model claims -1, but reference found a triple -> genuinely wrong
    v, note = check_answer(inp, "-1", "2 3 5")
    assert v is False and "reference found a triple" in note, note
    print("sanity: triple-sum and existence-claim checks pass")


if __name__ == "__main__":
    sanity()
