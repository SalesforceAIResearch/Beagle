"""Deterministic union of monet `bundled-skills.js` files (additive-skill merge).

The recombination of additive-skill parents is structurally a UNION of their
`BUNDLED_SKILLS` arrays. A naive line/diff concat corrupts the multi-line
template-literal `promptTemplate` strings (this broke the manual merge). This
module parses the array with a string/template-literal/brace-aware scanner and
unions skill objects by `name`, so the merged file is always valid JS.

Usage:
    merged = union_bundled_skills(base_text, *variant_texts)   # base keeps structure
"""
from __future__ import annotations
import re


def _find_array_span(text: str) -> tuple[int, int]:
    """Return (open_bracket_idx, close_bracket_idx) of the BUNDLED_SKILLS array."""
    m = re.search(r"const\s+BUNDLED_SKILLS\s*=\s*\[", text)
    if not m:
        raise ValueError("BUNDLED_SKILLS array not found")
    open_idx = text.index("[", m.start())
    depth = 0
    instr: str | None = None
    esc = False
    i = open_idx
    while i < len(text):
        c = text[i]
        if instr is not None:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == instr:
                instr = None
        else:
            if c in ("'", '"', "`"):
                instr = c
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return open_idx, i
        i += 1
    raise ValueError("unbalanced BUNDLED_SKILLS array")


def _split_top_level_objects(body: str) -> list[str]:
    """Split an array body into its top-level ``{...}`` object texts (string/
    template-literal aware; comments between objects are dropped — cosmetic)."""
    objs: list[str] = []
    depth = 0
    start: int | None = None
    instr: str | None = None
    esc = False
    i = 0
    while i < len(body):
        c = body[i]
        if instr is not None:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == instr:
                instr = None
        else:
            if c in ("'", '"', "`"):
                instr = c
            elif c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(body[start : i + 1])
                    start = None
        i += 1
    return objs


def _skill_name(obj_text: str) -> str | None:
    m = re.search(r"name:\s*'([^']+)'", obj_text) or re.search(r'name:\s*"([^"]+)"', obj_text)
    return m.group(1) if m else None


def union_bundled_skills(base_text: str, *variant_texts: str) -> tuple[str, list[str]]:
    """Union the skill objects of ``base`` + each variant by ``name``.

    ``base`` provides the file structure (imports, helpers, the array wrapper);
    the returned text is base with its BUNDLED_SKILLS array replaced by the
    deduped union (base objects first, then each variant's new-by-name objects).
    Returns (merged_text, added_skill_names).
    """
    bo, bc = _find_array_span(base_text)
    seen: dict[str, str] = {}
    order: list[str] = []
    for obj in _split_top_level_objects(base_text[bo + 1 : bc]):
        nm = _skill_name(obj)
        if nm and nm not in seen:
            seen[nm] = obj.strip()
            order.append(nm)
    base_names = set(order)
    added: list[str] = []
    for vt in variant_texts:
        vo, vc = _find_array_span(vt)
        for obj in _split_top_level_objects(vt[vo + 1 : vc]):
            nm = _skill_name(obj)
            if nm and nm not in seen:
                seen[nm] = obj.strip()
                order.append(nm)
                added.append(nm)
    merged_body = "\n  " + ",\n  ".join(seen[nm] for nm in order) + ",\n"
    merged = base_text[: bo + 1] + merged_body + base_text[bc:]
    return merged, added


if __name__ == "__main__":
    import sys
    base = open(sys.argv[1]).read()
    variants = [open(p).read() for p in sys.argv[2:]]
    merged, added = union_bundled_skills(base, *variants)
    sys.stderr.write(f"added {len(added)} skills: {added}\n")
    sys.stdout.write(merged)
