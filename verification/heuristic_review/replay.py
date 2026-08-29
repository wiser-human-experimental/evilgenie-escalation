"""Sandboxed-replay reconstruction: actually execute each tool call in order
against a real temp directory, instead of regex-guessing file contents.

Two environment-fidelity fixes on top of the naive version:
1. The real sandbox cwd is an absolute path (observed: /home/agent/workspace,
   varies by run) baked into tool-call args and safe_bash/safe_python strings.
   Detected per-sample from a "cat -n` on <PREFIX>/problem.md" style tool
   result and stripped so paths resolve inside our temp dir instead of
   silently failing (`cd /home/agent/workspace` doesn't exist on this host).
2. Some models rely on text_editor str_replace against the pre-seeded
   starter solution.py stub without ever issuing `create` first. The stub is
   pre-populated in the temp dir before replay so those edits have something
   to apply to (mirrors the harness, which starts every sandbox with it).

Handles text_editor (any path), safe_bash / bash (shell, any file-writing
command including heredocs, cp, redirects), and safe_python / python
(executes the code, capturing filesystem side effects like open(...).write()).
Returns the final on-disk solution.py content, or None.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_CWD_RE = re.compile(r"on (\S+)/(?:problem\.md|solution\.py|test\.py):")


def _detect_cwd_prefix(messages):
    for m in messages:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            hit = _CWD_RE.search(m["content"])
            if hit:
                return hit.group(1)
    return None


def _strip_prefix(s, prefix):
    if not prefix or not isinstance(s, str):
        return s
    return s.replace(prefix + "/", "").replace(prefix, ".")


def replay_reconstruct(messages, timeout_per_call=8, starter_stub=None):
    prefix = _detect_cwd_prefix(messages)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        if starter_stub:
            (tdp / "solution.py").write_text(starter_stub)

        for m in messages:
            if m.get("role") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                args = tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        continue
                if not isinstance(args, dict):
                    continue

                try:
                    if fn == "text_editor":
                        args = {**args, "path": _strip_prefix(args.get("path", ""), prefix)}
                        _apply_text_editor(tdp, args)
                    elif fn in ("safe_bash", "bash"):
                        cmd = _strip_prefix(args.get("cmd") or args.get("command") or "", prefix)
                        if isinstance(cmd, str) and cmd.strip():
                            subprocess.run(cmd, shell=True, cwd=td,
                                           capture_output=True, timeout=timeout_per_call)
                    elif fn in ("safe_python", "python"):
                        code = _strip_prefix(args.get("code") or "", prefix)
                        if isinstance(code, str) and code.strip():
                            subprocess.run([sys.executable, "-c", code], cwd=td,
                                           capture_output=True, timeout=timeout_per_call)
                except Exception:
                    pass  # a single failed replay step shouldn't abort the whole reconstruction

        sol = tdp / "solution.py"
        if sol.exists():
            try:
                return sol.read_text()
            except Exception:
                return None
        return None


def _apply_text_editor(tdp, args):
    path = args.get("path", "")
    if not path:
        return
    p = tdp / Path(path).name if Path(path).is_absolute() else tdp / path
    p.parent.mkdir(parents=True, exist_ok=True)
    cmd = args.get("command")
    if cmd == "create":
        p.write_text(args.get("file_text") or "")
    elif cmd == "str_replace" and p.exists():
        content = p.read_text()
        old = args.get("old_str") or ""
        new = args.get("new_str") or ""
        if old and old in content:
            p.write_text(content.replace(old, new, 1))
    elif cmd == "insert" and p.exists():
        content = p.read_text()
        ln = max(0, int(args.get("insert_line") or 0))
        txt = args.get("insert_text") or ""
        lines = content.split("\n")
        ln = min(ln, len(lines))
        p.write_text("\n".join(lines[:ln] + [txt] + lines[ln:]))


def find_starter_stub(iter_problem_samples_fn, problem_id, marker):
    """Scan samples for a raw (unwrapped) cat of the pristine solution.py
    stub -- identified as the most frequently occurring exact content
    containing `marker` (the mode -- an untouched stub recurs identically
    across many episodes; edited/partial variants are each rare)."""
    from collections import Counter
    counts = Counter()
    for _key, _model, _cond, _epoch, s in iter_problem_samples_fn(problem_id):
        for m in s.get("messages", []):
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                c = m["content"]
                if marker in c and "Here's the result" not in c:
                    counts[c] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]
