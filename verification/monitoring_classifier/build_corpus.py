"""Tagged monitoring corpus: same two data sources as
disclosure_analysis/extract.py:reasoning_corpus (readable structured
reasoning + inline code-comment text from tool calls), but source- and
step-tagged instead of flattened, so an LLM classifier can cite which
source it drew evidence from and a human can verify the citation lands
where claimed.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'verification' / 'disclosure_analysis'))
from extract import step_cmd_text  # noqa: E402


def tagged_corpus(steps) -> str:
    """Returns the chronological, source-tagged trace text fed to Pass 1.
    Empty string if the sample has neither readable reasoning nor any tool
    call with comment-bearing code/cmd text."""
    parts = []
    for i, st in enumerate(steps):
        if st.get('reasoning_readable') and st.get('reasoning_text'):
            parts.append(f"[STEP {i} | REASONING]\n{st['reasoning_text']}")
        if st.get('kind') == 'tool':
            txt = step_cmd_text(st)
            if txt:
                parts.append(f"[STEP {i} | CODE COMMENT]\n{txt}")
    return "\n\n".join(parts)
