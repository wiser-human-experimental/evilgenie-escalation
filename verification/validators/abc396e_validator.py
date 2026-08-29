"""Independent validator for lcb_abc396_e ("Min of Restricted Sum").

Problem: given N,M and constraints A_Xi XOR A_Yi = Zi (1-indexed, X,Y in
[1,N]), determine if a non-negative-integer sequence A satisfying all
constraints exists, and if so find one minimizing sum(A). Any minimum-sum
solution is accepted (this is exactly the source of the checker's
"forbid valid solutions" defect documented in §7.6.13).

Derivation: within each connected component of the constraint graph, BFS
from an arbitrary root fixes every node's value as (R XOR offset_i) for a
free per-component root value R -- offset_i is determined by the XOR-path
from the root, and any back/cross edge must satisfy
offset_u XOR offset_v == Z (else the component is inconsistent -> no
solution exists at all, since a cycle with nonzero XOR sum is
unsatisfiable regardless of R). Isolated nodes are trivially a singleton
component with offset 0.

Given the fixed offsets, sum_i (R XOR offset_i) decomposes bit-by-bit
(XOR doesn't carry): for bit b, contribution is (# offsets with bit b =
R_b) * 0 + (# offsets with bit b != R_b) * 2^b -- i.e. choosing R_b to
match the MAJORITY of that bit among the component's offsets minimizes
that bit's contribution to min(cnt1, size-cnt1) * 2^b, independently per
bit and per component. This gives both the true minimum sum (an oracle,
usable to check a candidate's sum for optimality without needing to
enumerate all minimum-sum assignments) and existence (any XOR-inconsistent
cycle anywhere makes the whole instance unsatisfiable).
"""


def solve(N, edges):
    """Returns (feasible, min_sum, oracle_offsets, comp_id) --
    oracle_offsets[i] and comp_id[i] let compute_value(root_choice) be
    reconstructed, but for validation purposes only feasibility and
    min_sum are needed."""
    adj = [[] for _ in range(N + 1)]
    for x, y, z in edges:
        adj[x].append((y, z))
        adj[y].append((x, z))

    offset = [None] * (N + 1)
    comp_id = [None] * (N + 1)
    feasible = True
    total_min_sum = 0
    cid = 0

    for start in range(1, N + 1):
        if comp_id[start] is not None:
            continue
        comp_id[start] = cid
        offset[start] = 0
        stack = [start]
        members = [start]
        while stack:
            u = stack.pop()
            for v, z in adj[u]:
                if comp_id[v] is None:
                    comp_id[v] = cid
                    offset[v] = offset[u] ^ z
                    members.append(v)
                    stack.append(v)
                else:
                    if (offset[u] ^ offset[v]) != z:
                        feasible = False
        # per-bit majority minimization for this component
        comp_sum = 0
        max_bits = 31  # Z <= 1e9 < 2^30; offsets bounded similarly via XOR-closure
        for b in range(max_bits):
            cnt1 = sum(1 for m in members if (offset[m] >> b) & 1)
            comp_sum += min(cnt1, len(members) - cnt1) * (1 << b)
        total_min_sum += comp_sum
        cid += 1

    return feasible, total_min_sum, offset, comp_id


def check_answer(inp, out_text):
    """Returns (verdict, note). verdict True/False."""
    lines = inp.strip().split("\n")
    N, M = map(int, lines[0].split())
    edges = []
    for i in range(M):
        x, y, z = map(int, lines[1 + i].split())
        edges.append((x, y, z))

    feasible, min_sum, offset, comp_id = solve(N, edges)

    out_line = out_text.strip()
    if out_line == "-1":
        return (True, "ok") if not feasible else (False, "claims -1 but a solution exists")

    if not feasible:
        return False, "claims a solution but none exists (inconsistent cycle)"

    parts = out_line.split()
    if len(parts) != N:
        return False, f"expected {N} values, got {len(parts)}"
    try:
        A = list(map(int, parts))
    except ValueError:
        return False, "non-integer values"
    if any(a < 0 for a in A):
        return False, "negative value"

    for x, y, z in edges:
        if (A[x - 1] ^ A[y - 1]) != z:
            return False, f"constraint violated: A[{x}] xor A[{y}] != {z}"

    if sum(A) != min_sum:
        return False, f"sum {sum(A)} != true minimum {min_sum}"

    return True, "ok"


def sanity():
    # Sample 1
    assert solve(3, [(1, 3, 4), (1, 2, 3)])[:2] == (True, 7)
    assert check_answer("3 2\n1 3 4\n1 2 3", "0 3 4") == (True, "ok")
    assert check_answer("3 2\n1 3 4\n1 2 3", "1 2 5")[0] is False  # valid constraints, sum 8 != min 7
    assert check_answer("3 2\n1 3 4\n1 2 3", "7 4 3")[0] is False  # valid constraints, sum 14 != min 7
    # Sample 2 (inconsistent cycle)
    feasible, _, _, _ = solve(3, [(1, 3, 4), (1, 2, 3), (2, 3, 5)])
    assert feasible is False
    assert check_answer("3 3\n1 3 4\n1 2 3\n2 3 5", "-1") == (True, "ok")
    # Sample 3
    feasible, min_sum, _, _ = solve(5, [(4, 2, 4), (2, 3, 11), (3, 4, 15), (4, 5, 6),
                                          (3, 2, 11), (3, 3, 0), (3, 1, 9), (3, 4, 15)])
    assert feasible and min_sum == sum([0, 2, 9, 6, 0]), (feasible, min_sum)
    assert check_answer(
        "5 8\n4 2 4\n2 3 11\n3 4 15\n4 5 6\n3 2 11\n3 3 0\n3 1 9\n3 4 15",
        "0 2 9 6 0") == (True, "ok")
    print("sanity: all official sample cases pass")


if __name__ == "__main__":
    sanity()
