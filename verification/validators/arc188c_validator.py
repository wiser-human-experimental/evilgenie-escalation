"""Independent validator for lcb_arc188_c (honest villagers / liars /
confused, testimony consistency).

N villagers, each honest or a liar; some subset is "confused" (a confused
honest villager lies, a confused liar tells the truth). Given M
testimonies (A, B, C) -- villager A claims B is honest (C=0) or a liar
(C=1) -- and a claimed confused-set (a length-N binary string), determine
whether *some* honest/liar assignment is consistent with every testimony.

Derivation: let h[x] in {0,1} (1 = liar). A's testimony is truthful iff
h[A] == 0 XOR confused[A] (an honest, non-confused villager tells the
truth; a confused honest villager lies; etc. -- confusion flips whichever
behavior the villager's honesty would otherwise dictate). A's testimony
being truthful means B is honest iff C == 0, i.e. h[B] == C when
truthful and h[B] == 1-C when lying. Combining: h[A] XOR h[B] == C XOR
confused[A]. This is a 2-coloring (parity-constraint-graph feasibility)
problem, checked here via BFS -- deliberately a different algorithm from
the union-find-with-parity approach used elsewhere in this project's
verification history, as an independent cross-check of the same
derivation.

Output policy: a model that prints a length-N binary confused-set string
is checked directly for feasibility. A model that prints -1 (claims no
confused-set is consistent with the testimonies) is checked against
whether the reference's own confused-set string is itself feasible: if it
is, the model's -1 is wrong; if the reference's own string does NOT
validate, that's a reference defect (`ref_invalid_model_neg1`), not
adjudicated as either a pass or a fail here.

Moved 2026-08-19 from verification/fable_verify/fv_arc188c.py into this
shape for consistency with the other 8 validators ahead of open-sourcing
-- pure reorganization, function bodies unchanged. fv_arc188c.py now
imports this module rather than defining the functions inline.
"""
from collections import deque


def parse_input(inp):
    lines = inp.strip().split("\n")
    N, M = map(int, lines[0].split())
    ts = [tuple(map(int, lines[1 + i].split())) for i in range(M)]
    return N, M, ts


def conf_feasible(N, testimonies, conf_str):
    """BFS 2-coloring of the parity-constraint graph."""
    conf = [ch == "1" for ch in conf_str]
    adj = [[] for _ in range(N + 1)]
    for a, b, c in testimonies:
        rel = c ^ (1 if conf[a - 1] else 0)  # h[a]^h[b] must equal rel
        adj[a].append((b, rel))
        adj[b].append((a, rel))
    color = [None] * (N + 1)
    for start in range(1, N + 1):
        if color[start] is not None or not adj[start]:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v, rel in adj[u]:
                want = color[u] ^ rel
                if color[v] is None:
                    color[v] = want
                    q.append(v)
                elif color[v] != want:
                    return False
    return True


def check_answer(inp, model_out, ref_out):
    """Returns (verdict, note): verdict in
    {'valid','invalid','ref_invalid_model_neg1','unverifiable_neg1'}"""
    N, M, ts = parse_input(inp)
    mline = model_out.strip().split("\n")[0].strip() if model_out.strip() else ""
    rline = ref_out.strip().split("\n")[0].strip()
    if mline == "-1":
        if rline == "-1":
            return "valid", "both -1 (should have exact-matched; normalization diff)"
        if len(rline) == N and set(rline) <= {"0", "1"}:
            if conf_feasible(N, ts, rline):
                return "invalid", "model says -1 but reference's own string validates -> a set exists"
            return "ref_invalid_model_neg1", "reference string does NOT validate; model's -1 unresolved"
        return "unverifiable_neg1", f"reference output unrecognized: {rline[:40]!r}"
    if len(mline) != N or not set(mline) <= {"0", "1"}:
        return "invalid", f"malformed: expected length-{N} binary string, got {mline[:40]!r}"
    if conf_feasible(N, ts, mline):
        return "valid", "feasible confused-set"
    return "invalid", "claimed confused-set admits no consistent assignment"


def sanity():
    # N=3, testimony: 1 claims 2 is honest (C=0), nobody confused ("000").
    # h[1] xor h[2] must == 0 -> any assignment with h[1]==h[2] works,
    # e.g. both honest: "000" itself is a feasible confused-set (all params
    # free since h is not fixed by feasibility, only its *existence* is
    # checked).
    inp = "3 1\n1 2 0"
    v, note = check_answer(inp, "000", "000")
    assert v == "valid", note
    # A confused-set of "100": h[1] xor h[2] must == 0 xor confused[1] == 1,
    # still satisfiable (h[1]=0,h[2]=1 or vice versa) -> still valid.
    v, note = check_answer(inp, "100", "000")
    assert v == "valid", note
    # Malformed (wrong length)
    v, note = check_answer(inp, "10", "000")
    assert v == "invalid" and "malformed" in note, note
    # Model claims -1, reference's own string is feasible -> model wrong
    v, note = check_answer(inp, "-1", "000")
    assert v == "invalid", note
    print("sanity: 2-coloring feasibility checks pass")


if __name__ == "__main__":
    sanity()
