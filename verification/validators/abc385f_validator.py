"""Validator for lcb_abc385_f (Visible Buildings).

Scoped narrowly and deliberately: the confirmed defect (§7.6.13/14) is
that the checker does exact post-normalisation string match instead of
the problem statement's own promised "absolute or relative error of at
most 1e-9" tolerance. The reference value itself is not known to be
wrong (unlike arc190_a/abc397_d) -- building a full independent geometric
oracle (convex-hull-trick sweep) to re-derive the true answer from
scratch would be substantial new-algorithm risk for a defect that's
already confirmed to have ~no practical effect (§7.6.14: 56/58 non-solves
are genuine no-code, not near-misses). This validator fixes exactly the
confirmed defect -- tolerance-aware comparison against the reference --
not a claim of independent global correctness.
"""


def check_answer(out_text, ref_text):
    out_line = out_text.strip()
    ref_line = ref_text.strip()

    if ref_line == "-1" or out_line == "-1":
        return (True, "ok") if out_line == ref_line else (False, f"{out_line!r} != -1")

    try:
        out_val = float(out_line)
        ref_val = float(ref_line)
    except ValueError:
        return False, "non-numeric"

    abs_err = abs(out_val - ref_val)
    rel_err = abs_err / abs(ref_val) if ref_val != 0 else abs_err
    if abs_err <= 1e-9 or rel_err <= 1e-9:
        return True, "ok"
    return False, f"abs_err={abs_err:.3e} rel_err={rel_err:.3e}"


def sanity():
    assert check_answer("1.500000000000000000", "1.5") == (True, "ok")
    assert check_answer("1.5000000001", "1.5") == (True, "ok")
    assert check_answer("1.6", "1.5")[0] is False
    assert check_answer("-1", "-1") == (True, "ok")
    assert check_answer("0.0", "-1")[0] is False
    assert check_answer("17.142857142857142350", "17.142857142857142") == (True, "ok")
    print("sanity: tolerance comparator behaves as specified")


if __name__ == "__main__":
    sanity()
