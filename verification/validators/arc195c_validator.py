"""Independent validator for lcb_arc195_c (piece-placement cycle problem).

Feasibility: red pieces move orthogonally by 1 (flips checkerboard color
(r+c) mod 2 every move); blue pieces move diagonally by 1 (preserves it).
Walking the full R+B cycle, each piece is the "mover" for exactly one
outgoing edge (to the next piece in the cyclic order), contributing one
color flip if red, zero if blue. For the cycle to close (return to the
start color), the total flip count must be even -- and that count is
exactly R (every red piece flips, regardless of position). So R must be
even.

But R even is NOT sufficient (round-2 verification correction,
docs/fable_verification_report_2.md P1.1): a blue (diagonal) move always
changes r by +-1, so an all-blue cycle needs an even number of moves --
R = 0 with B odd is infeasible even though R is even. The true rule:

    a valid cyclic placement exists  <=>  R even AND NOT (R == 0 and B odd)

Verified against all 3 official samples, exhaustive backtracking for all
(R,B) with R+B <= 7, and the dataset's own reference output (visible
test 2, case (R=0,B=3) -> "No"). Sufficiency for R even >= 2 (any B) and
R = 0 with B even is confirmed constructively at every size tested.

For a claimed "Yes" placement, independently verify: R+B distinct
squares, piece-type counts match R and B, and every consecutive pair
(including wraparound) is a valid single-step move for the type of the
*earlier* piece in the pair (the "mover").
"""


def parse_case(line):
    R, B = map(int, line.split())
    return R, B


def feasible(R, B):
    return R % 2 == 0 and not (R == 0 and B % 2 == 1)


def check_answer(R, B, answer_lines):
    """answer_lines: list of lines making up this test case's output
    (either ['No'] or ['Yes', 'p1 r1 c1', ...]). Returns (verdict, note)."""
    if not answer_lines:
        return False, "empty"
    first = answer_lines[0].strip()

    if first == "No":
        return (True, "ok") if not feasible(R, B) else (False, "claims No but (R,B) is feasible")

    if first != "Yes":
        return False, f"malformed first line {first!r}"

    if not feasible(R, B):
        return False, "claims Yes but (R,B) is infeasible"

    n = R + B
    rows = answer_lines[1:1 + n]
    if len(rows) != n:
        return False, f"expected {n} placement lines, got {len(rows)}"

    pieces = []
    for row in rows:
        parts = row.split()
        if len(parts) != 3:
            return False, f"malformed placement line {row!r}"
        p, r, c = parts[0], int(parts[1]), int(parts[2])
        if p not in ("R", "B"):
            return False, f"bad piece type {p!r}"
        if not (1 <= r <= 10**9 and 1 <= c <= 10**9):
            return False, "coordinate out of bounds"
        pieces.append((p, r, c))

    n_red = sum(1 for p, _, _ in pieces if p == "R")
    n_blue = sum(1 for p, _, _ in pieces if p == "B")
    if n_red != R or n_blue != B:
        return False, f"piece-type counts wrong: R={n_red} (want {R}), B={n_blue} (want {B})"

    squares = [(r, c) for _, r, c in pieces]
    if len(set(squares)) != n:
        return False, "duplicate square"

    for i in range(n):
        p, r, c = pieces[i]
        nr, nc = pieces[(i + 1) % n][1], pieces[(i + 1) % n][2]
        dr, dc = nr - r, nc - c
        if p == "R":
            ok = (abs(dr), abs(dc)) in ((1, 0), (0, 1))
        else:
            ok = (abs(dr), abs(dc)) == (1, 1)
        if not ok:
            return False, f"invalid move at index {i}: {p} ({r},{c}) -> ({nr},{nc})"

    return True, "ok"


def check_full_output(inp, out_text):
    """inp: the whole stdin block (T followed by T 'R B' lines).
    out_text: the whole stdout block. Returns (verdict, note) for the
    WHOLE multi-case output (all cases must be individually valid)."""
    lines = inp.strip().split("\n")
    T = int(lines[0])
    cases = [parse_case(lines[1 + i]) for i in range(T)]

    out_lines = out_text.strip("\n").split("\n")
    idx = 0
    for ci, (R, B) in enumerate(cases):
        if idx >= len(out_lines):
            return False, f"case {ci}: ran out of output lines"
        if out_lines[idx].strip() == "No":
            verdict, note = check_answer(R, B, [out_lines[idx]])
            idx += 1
        elif out_lines[idx].strip() == "Yes":
            n = R + B
            block = out_lines[idx:idx + 1 + n]
            verdict, note = check_answer(R, B, block)
            idx += 1 + n
        else:
            return False, f"case {ci}: malformed line {out_lines[idx]!r}"
        if not verdict:
            return False, f"case {ci} (R={R},B={B}): {note}"
    return True, "ok"


def sanity():
    assert check_answer(2, 3, ["No"]) == (False, "claims No but (R,B) is feasible")
    assert check_answer(1, 1, ["No"]) == (True, "ok")
    assert check_answer(4, 0, ["No"])[0] is False
    # round-2 correction: R=0 with B odd is infeasible (all-blue cycles
    # need even length) -- present in the dataset (visible test 2) but in
    # no official sample, which is how the original rule slipped through
    assert check_answer(0, 3, ["No"]) == (True, "ok")
    assert check_answer(0, 3, ["Yes", "B 1 1", "B 2 2", "B 3 3"])[0] is False
    assert check_answer(0, 2, ["No"]) == (False, "claims No but (R,B) is feasible")
    assert check_answer(0, 2, ["Yes", "B 1 1", "B 2 2"]) == (True, "ok")
    # official sample 1 placement
    yes1 = ["Yes", "B 2 3", "R 3 2", "B 2 2", "B 3 3", "R 2 4"]
    assert check_answer(2, 3, yes1) == (True, "ok"), check_answer(2, 3, yes1)
    # official sample 3 placement
    yes3 = ["Yes", "R 1 1", "R 1 2", "R 2 2", "R 2 1"]
    assert check_answer(4, 0, yes3) == (True, "ok"), check_answer(4, 0, yes3)
    # tamper: swap two coords to break a move
    bad = ["Yes", "R 1 1", "R 1 3", "R 2 2", "R 2 1"]
    assert check_answer(4, 0, bad)[0] is False
    # full multi-case output
    full_out = "\n".join(yes1) + "\nNo\n" + "\n".join(yes3)
    assert check_full_output("3\n2 3\n1 1\n4 0", full_out) == (True, "ok")
    print("sanity: all official samples pass")


if __name__ == "__main__":
    sanity()
