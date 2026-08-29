#!/usr/bin/env python3
"""Cost comparison by model x condition -- REAL OpenRouter per-call cost
wherever it's available in the .eval logs, calibrated-estimate for the rest.

See cost_lib.py's module docstring for the full mechanism: real per-call
cost is only preserved in the logs for the first ~5 turns of each episode
plus the final judge call (an inspect_ai log-truncation default, not
recoverable). Every .eval FILE is calibrated using its own real-vs-
reconstructed ratio on those covered turns, then that factor is applied to
the reconstructed cost of the file's uncovered turns -- self-correcting per
file, which matters because at least one model (gpt-5.6-luna) shows a real,
stable, non-transient ~3.5-10x mismatch between what OpenRouter's live
/api/v1/models pricing advertises today and what was actually billed.

Usage: source .venv/bin/activate && python3 analyse_costs.py
"""
import json
from collections import defaultdict

import zipfile_zstd as zf_zstd

from viewer import MODEL_REGISTRY, COND_LABELS, COND_ORDER
import cost_lib

EXCLUDE_MODELS = {"kwaipilot-kat-coder-pro-v2.5"}  # excluded 2026-08-07, reportedly heavily fine-tuned


def main():
    pricing, live = cost_lib.fetch_live_pricing()
    print(f"Pricing source: {'live OpenRouter fetch' if live else '2026-07-30 fallback snapshot'}")
    ground_truth_usage = cost_lib.fetch_ground_truth_usage()

    # rows[model_key][cond_letter] = accumulated cell totals
    rows = defaultdict(lambda: defaultdict(lambda: {
        "agent": 0.0, "judge": 0.0, "real_confirmed": 0.0,
        "n": 0, "n_errored": 0, "files": 0,
    }))
    # For reporting which models needed real correction
    model_factors = defaultdict(list)  # model_key -> [(file, agent_factor, judge_factor), ...]

    for model_key, conds in MODEL_REGISTRY.items():
        if model_key in EXCLUDE_MODELS:
            continue
        for letter, info in conds.items():
            eval_dir = info["dir"]
            cell = rows[model_key][letter]
            for fp in sorted(eval_dir.glob("*.eval")):
                try:
                    with zf_zstd.ZipFile(fp) as z:
                        sample_names = [n for n in z.namelist() if n.startswith("samples/")]
                        samples = []
                        n_unreadable = 0
                        for sn in sample_names:
                            try:
                                samples.append(json.loads(z.read(sn)))
                            except Exception:
                                n_unreadable += 1
                except Exception as e:
                    print(f"  WARN: failed to read {fp}: {e!r}")
                    continue
                if n_unreadable:
                    print(f"  WARN: {fp}: {n_unreadable}/{len(sample_names)} sample entries corrupted, excluded.")

                breakdowns = [cost_lib.sample_usage_breakdown(s, pricing) for s in samples]
                calibrated, factors = cost_lib.calibrate_breakdowns(breakdowns)
                model_factors[model_key].append((fp.name, factors["agent_factor"], factors["judge_factor"]))

                n_errored = sum(1 for s in samples if s.get("error"))
                cell["n"] += len(samples)
                cell["n_errored"] += n_errored
                cell["files"] += 1
                for c in calibrated:
                    cell["agent"] += c["agent_cost"]
                    cell["judge"] += c["judge_cost"]
                    cell["real_confirmed"] += c["real_confirmed"]

    # ---- print table ----
    print("\nTOTAL cost (agent + judge), $ per model x condition -- REAL cost where "
          "available, calibrated-estimate for the rest\n"
          "(! = every sample in this cell errored -- network failure, not scored; "
          "~ = some samples errored)\n")
    col_w = 17
    print(f"{'model':<38}" + "".join(f"{c:>{col_w}}" for c in COND_ORDER) + f"{'TOTAL':>{col_w}}")
    grand_by_cond = defaultdict(float)
    grand_total = 0.0
    grand_real_confirmed = 0.0
    for model_key in sorted(rows.keys()):
        conds = rows[model_key]
        vals = []
        row_total = 0.0
        for c in COND_ORDER:
            cell = conds.get(c)
            if cell is None or cell["files"] == 0:
                vals.append("—")
                continue
            total = cell["agent"] + cell["judge"]
            all_errored = cell["n_errored"] == cell["n"] and cell["n"] > 0
            flag = "!" if all_errored else ("~" if cell["n_errored"] else "")
            vals.append(f"${total:,.2f}{flag}(n={cell['n']})")
            row_total += total
            grand_by_cond[c] += total
            grand_real_confirmed += cell["real_confirmed"]
        grand_total += row_total
        print(f"{model_key:<38}" + "".join(f"{v:>{col_w}}" for v in vals) + f"{'$'+format(row_total, ',.2f'):>{col_w}}")

    print(f"{'--- condition totals ---':<38}" +
          "".join(f"{'$'+format(grand_by_cond[c], ',.2f'):>{col_w}}" for c in COND_ORDER) +
          f"{'$'+format(grand_total, ',.2f'):>{col_w}}")

    print(f"\nOf ${grand_total:,.2f} total, ${grand_real_confirmed:,.2f} "
          f"({100*grand_real_confirmed/grand_total:.1f}%) is REAL OpenRouter-confirmed cost "
          f"(not estimated) -- the rest is calibrated per-file using each file's own real-vs-"
          f"reconstructed ratio on its covered turns.")

    # ---- calibration factor report: which models needed real correction ----
    print("\n\nCalibration factors by model (real / reconstructed-at-current-pricing, on covered "
          "turns) -- 1.00 means current OpenRouter pricing matches what was actually billed; "
          "far from 1.00 means it doesn't and the calibrated total corrects for it:\n")
    for model_key in sorted(model_factors.keys()):
        entries = model_factors[model_key]
        agent_factors = [f for _, f, _ in entries]
        avg = sum(agent_factors) / len(agent_factors)
        spread = max(agent_factors) - min(agent_factors)
        flag = "  <-- CORRECTED (>10% off)" if abs(avg - 1.0) > 0.10 else ""
        print(f"  {model_key:32s} avg_agent_factor={avg:.3f}  spread={spread:.3f} (n_files={len(entries)}){flag}")

    print(f"\n--- Ground truth cross-check ---")
    print(f"Cost-report grand total (roster + control runs): ${grand_total:,.2f}")
    if ground_truth_usage is not None:
        print(f"Real all-time usage on this project's OpenRouter key: ${ground_truth_usage:,.2f}")
        diff = grand_total - ground_truth_usage
        print(f"Difference: ${diff:,.2f} ({100*diff/ground_truth_usage:+.1f}%)")
    else:
        print("Real usage not fetched (no network / no API key).")

    # ---- dump machine-readable CSV ----
    import csv
    out_path = "docs/cost_by_model_condition.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "condition_label", "n_samples", "n_errored", "n_files",
                    "agent_cost_usd", "judge_cost_usd", "total_cost_usd", "cost_per_sample_usd",
                    "real_confirmed_usd", "real_confirmed_pct", "all_samples_errored"])
        for model_key in sorted(rows.keys()):
            for c in COND_ORDER:
                cell = rows[model_key].get(c)
                if cell is None or cell["files"] == 0:
                    continue
                total = cell["agent"] + cell["judge"]
                per = total / cell["n"] if cell["n"] else 0.0
                all_errored = cell["n_errored"] == cell["n"] and cell["n"] > 0
                real_pct = 100 * cell["real_confirmed"] / total if total else 0.0
                w.writerow([model_key, c, COND_LABELS[c], cell["n"], cell["n_errored"], cell["files"],
                            f"{cell['agent']:.4f}", f"{cell['judge']:.4f}", f"{total:.4f}", f"{per:.4f}",
                            f"{cell['real_confirmed']:.4f}", f"{real_pct:.1f}", all_errored])
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(repr(e), flush=True)
        raise
