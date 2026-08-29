"""Shared cost-computation logic for viewer.py and analyse_costs.py.

Ground truth discovery (2026-07-30): the raw OpenRouter response embedded in
each .eval file's model-call events (`event["call"]["response"]`) includes
the provider's own real per-call `usage.cost` -- but this is only preserved
for roughly the first 5 agent turns of each episode plus the final judge
call; inspect_ai drops the raw response body for every turn in between (an
apparent log-size-control default, not something we can recover -- no
generation id survives on the dropped events to look up after the fact).

Spot-checking real vs. reconstructed cost on the *covered* turns validated
the token-reconstruction approach for most models (ratio ~1.00), but found
real, stable (not date-dependent) mismatches for a couple of models --
worst confirmed case: gpt-5.6-luna's real completion price implied by
billing is exactly 10x what OpenRouter's live /api/v1/models advertises
today (matches the ORIGINAL models.py-documented $6.00/1M, not the
currently-fetched $0.60/1M). Cause unconfirmed (a genuine multi-provider
routing/pricing-table mismatch on OpenRouter's side, most likely), but
real billing data always wins over an advertised price.

Root cause narrowed 2026-08-04: gpt-5.6-luna is listed on OpenRouter's
public discounted-models collection (openrouter.ai/collections/
discounted-models) at "50% off" -- but that's the wrong ratio for what's
actually happening. Re-checked live: real-billed / today's-catalog-
reconstructed = 10.29x, not ~2x, meaning this project's July runs were
billed at the ORIGINAL undiscounted rate, and the current catalog price
is a promotion that either wasn't active yet or is steeper than its own
"50% off" label suggests relative to list. deepseek-v4-flash is the only
other roster model currently on that same discount collection (37% off);
it does NOT show this problem (real/catalog ratio 0.71x -- real billing
was cheaper than today's catalog, the safe direction). Full current
pricing snapshot with this caveat: the paper Appendix F.

Fix: per-FILE calibration. For every .eval file, compute
  calibration_factor = real_cost(covered turns) / reconstructed_cost(covered turns, same token counts)
and apply that file's own factor to the reconstructed cost of its
UNCOVERED turns, rather than trusting live-fetched pricing blindly for the
whole file. This makes each file self-correcting using its own ground
truth, without needing a historical pricing archive. Validated stable
across widely separated files/dates for both "good" (ratio ~1.00) and
"bad" (ratio ~1.5-3.9x) models -- see docs/cost_by_model_condition.csv's
`calib_factor` column and the Methodology section of the published cost
report for the concrete numbers.
"""
import json
import urllib.request

JUDGE_MODEL_ID = "openrouter/openai/gpt-4o"

# Fallback snapshot, fetched live from https://openrouter.ai/api/v1/models on
# 2026-07-30 -- used only if a live fetch isn't possible.
PRICING_SNAPSHOT_2026_07_30 = {
    "openrouter/openai/gpt-5.3-codex":            (0.00000175, 0.000014,   0.000000175),
    "openrouter/openai/gpt-5.6-luna":             (0.0000001,  0.0000006,  0.00000001),
    "openrouter/openai/gpt-5.6-sol":              (0.000005,   0.00003,    0.0000005),
    "openrouter/anthropic/claude-opus-4.8":       (0.000005,   0.000025,   0.0000005),
    "openrouter/anthropic/claude-sonnet-5":       (0.000002,   0.00001,    0.0000002),
    "openrouter/anthropic/claude-fable-5":        (0.00001,    0.00005,    0.000001),
    "openrouter/google/gemini-3.1-pro-preview":   (0.000002,   0.000012,   0.0000002),
    "openrouter/google/gemini-3.5-flash":         (0.0000015,  0.000009,   0.00000015),
    "openrouter/qwen/qwen3.7-max":                (0.000001475,0.000004425,0.000000295),
    "openrouter/qwen/qwen3.7-plus":               (0.00000032, 0.00000128, 0.000000064),
    "openrouter/deepseek/deepseek-v4-pro":        (0.000000435,0.00000087, 0.000000003625),
    "openrouter/deepseek/deepseek-v4-flash":      (0.00000014, 0.00000028, 0.000000028),
    "openrouter/kwaipilot/kat-coder-pro-v2.5":    (0.00000074, 0.00000296, 0.00000015),
    "openrouter/x-ai/grok-4.5":                   (0.000002,   0.000006,   0.0000003),
    "openrouter/moonshotai/kimi-k2.7-code":       (0.00000073, 0.0000035,  0.00000015),
    "openrouter/moonshotai/kimi-k3":              (0.000003,   0.000015,   0.0000003),
    "openrouter/openai/gpt-4o":                   (0.0000025,  0.00001,    0.00000125),
}

MODEL_IDS = [
    "openai/gpt-5.3-codex", "openai/gpt-5.6-luna", "openai/gpt-5.6-sol",
    "anthropic/claude-opus-4.8", "anthropic/claude-sonnet-5", "anthropic/claude-fable-5",
    "google/gemini-3.1-pro-preview", "google/gemini-3.5-flash",
    "qwen/qwen3.7-max", "qwen/qwen3.7-plus",
    "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash",
    "kwaipilot/kat-coder-pro-v2.5", "x-ai/grok-4.5",
    "moonshotai/kimi-k2.7-code", "moonshotai/kimi-k3",
    "openai/gpt-4o",
]


def fetch_live_pricing(env_path=".env", timeout=15):
    try:
        key = None
        import os
        if os.path.exists(env_path):
            for line in open(env_path):
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.strip().split("=", 1)[1]
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"} if key else {},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        by_id = {m["id"]: m for m in data["data"]}
        pricing = {}
        for want in MODEL_IDS:
            m = by_id.get(want)
            if not m:
                continue
            p = m["pricing"]
            pricing[f"openrouter/{want}"] = (
                float(p["prompt"]), float(p["completion"]), float(p.get("input_cache_read") or 0),
            )
        if len(pricing) == len(MODEL_IDS):
            return pricing, True
    except Exception:
        pass
    return PRICING_SNAPSHOT_2026_07_30, False


def fetch_ground_truth_usage(env_path=".env", timeout=15):
    try:
        import os
        key = None
        if os.path.exists(env_path):
            for line in open(env_path):
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.strip().split("=", 1)[1]
        if not key:
            return None
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())["data"]
        return data.get("usage")
    except Exception:
        return None


def _recon_cost(model_id, fresh_input, cache_read, output, pricing):
    if model_id not in pricing:
        return 0.0
    p_in, p_out, p_cache = pricing[model_id]
    return fresh_input * p_in + cache_read * p_cache + output * p_out


def sample_usage_breakdown(sample, pricing):
    """Break one raw eval sample dict into real vs. reconstructed cost,
    split into agent vs. judge, and covered vs. total (see module docstring
    for what "covered" means)."""
    out = {
        "real_agent": 0.0, "real_judge": 0.0,
        "recon_agent_covered": 0.0, "recon_judge_covered": 0.0,
        "n_covered": 0, "n_total": 0,
    }
    for e in sample.get("events", []):
        if e.get("event") != "model":
            continue
        out["n_total"] += 1
        model_id = e.get("model")
        resp = e.get("call", {}).get("response", {})
        usage = resp.get("usage") if isinstance(resp, dict) else None
        if not usage or usage.get("cost") is None:
            continue
        out["n_covered"] += 1
        ptd = usage.get("prompt_tokens_details", {}) or {}
        cached = ptd.get("cached_tokens") or 0
        fresh = (usage.get("prompt_tokens", 0) or 0) - cached
        output = usage.get("completion_tokens", 0) or 0
        recon = _recon_cost(model_id, fresh, cached, output, pricing)
        real = usage["cost"]
        if model_id == JUDGE_MODEL_ID:
            out["real_judge"] += real
            out["recon_judge_covered"] += recon
        else:
            out["real_agent"] += real
            out["recon_agent_covered"] += recon

    # Reconstructed TOTAL (covered + uncovered) from the sample's own
    # cumulative model_usage snapshot -- the existing full-trajectory method.
    recon_agent_total = 0.0
    recon_judge_total = 0.0
    for model_id, usage in sample.get("model_usage", {}).items():
        recon = _recon_cost(
            model_id,
            usage.get("input_tokens", 0), usage.get("input_tokens_cache_read", 0),
            usage.get("output_tokens", 0), pricing,
        )
        if model_id == JUDGE_MODEL_ID:
            recon_judge_total += recon
        else:
            recon_agent_total += recon
    out["recon_agent_total"] = recon_agent_total
    out["recon_judge_total"] = recon_judge_total
    return out


def calibrate_breakdowns(breakdowns, fallback_factor=1.0, min_recon=0.01):
    """Given a list of sample_usage_breakdown() dicts from the SAME .eval
    file, compute that file's calibration factor(s) and return per-sample
    calibrated costs (list, same order as input) plus the factors used."""
    sum_real_agent = sum(b["real_agent"] for b in breakdowns)
    sum_recon_agent_cov = sum(b["recon_agent_covered"] for b in breakdowns)
    sum_real_judge = sum(b["real_judge"] for b in breakdowns)
    sum_recon_judge_cov = sum(b["recon_judge_covered"] for b in breakdowns)

    agent_factor = (sum_real_agent / sum_recon_agent_cov) if sum_recon_agent_cov > min_recon else fallback_factor
    judge_factor = (sum_real_judge / sum_recon_judge_cov) if sum_recon_judge_cov > min_recon else fallback_factor

    results = []
    for b in breakdowns:
        agent_calibrated = b["real_agent"] + max(0.0, b["recon_agent_total"] - b["recon_agent_covered"]) * agent_factor
        judge_calibrated = b["real_judge"] + max(0.0, b["recon_judge_total"] - b["recon_judge_covered"]) * judge_factor
        real_confirmed = b["real_agent"] + b["real_judge"]
        total_calibrated = agent_calibrated + judge_calibrated
        results.append({
            "agent_cost": round(agent_calibrated, 6),
            "judge_cost": round(judge_calibrated, 6),
            "total_cost": round(total_calibrated, 6),
            "real_confirmed": round(real_confirmed, 6),
            "coverage_pct": round(100 * real_confirmed / total_calibrated, 1) if total_calibrated > 0 else 0.0,
            "n_covered": b["n_covered"], "n_total": b["n_total"],
        })
    return results, {"agent_factor": agent_factor, "judge_factor": judge_factor}
