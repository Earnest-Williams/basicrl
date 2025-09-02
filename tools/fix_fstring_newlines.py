#!/usr/bin/env python3
"""
Fix newline splits inside f-string expressions:

    f"... {                # newline anywhere inside braces
        expr(
            arg
        ) ... } ..."

→

    f"... {expr(arg) ... } ..."

Runs on git-tracked .py files. Use:
  - --diff   to preview changes
  - --write  to apply changes
"""

from __future__ import annotations
import pathlib, subprocess, sys, difflib

PREFIX_CHARS = set("rRbBuUfF")


def git_tracked_py_files() -> list[pathlib.Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return [pathlib.Path(p) for p in out]


def fix_text(text: str) -> tuple[str, int]:
    i, n = 0, len(text)
    out = []
    changes = 0

    while i < n:
        ch = text[i]

        # Detect start of a quoted string (possible f-string)
        if ch in ("'", '"'):
            # Look back for valid prefix letters only
            j = i - 1
            while j >= 0 and text[j] in PREFIX_CHARS:
                j -= 1
            prefix = text[j + 1 : i]
            is_f = "f" in prefix.lower()
            is_raw = "r" in prefix.lower()

            q = ch
            triple = text.startswith(q * 3, i)
            # Emit prefix + opening quotes
            out.append(prefix)
            out.append(q * (3 if triple else 1))
            i += 3 if triple else 1

            if not is_f:
                # Not an f-string: just copy until closing quote
                if triple:
                    while i < n:
                        if text.startswith(q * 3, i):
                            out.append(q * 3)
                            i += 3
                            break
                        out.append(text[i])
                        i += 1
                else:
                    while i < n:
                        c = text[i]
                        out.append(c)
                        i += 1
                        if c == "\\" and not is_raw and i < n:
                            out.append(text[i])
                            i += 1
                        elif c == q:
                            break
                continue

            # f-string: scan while tracking braces depth
            brace_depth = 0
            # For triple, we close on q*3; for single, on q (not escaped unless raw)
            while i < n:
                # Close conditions (only when not inside an expression)
                if brace_depth == 0:
                    if triple and text.startswith(q * 3, i):
                        out.append(q * 3)
                        i += 3
                        break
                    if not triple:
                        c = text[i]
                        out.append(c)
                        i += 1
                        if c == "\\" and not is_raw and i < n:
                            out.append(text[i])
                            i += 1
                        elif c == q:
                            break
                        elif c == "{":
                            # expression start or literal '{{'
                            if i < n and text[i] == "{":
                                # literal '{{'
                                out.append("{")
                                i += 1
                            else:
                                brace_depth = 1
                        elif c == "}":
                            # literal '}}' handling
                            if i < n and text[i] == "}":
                                out.append("}")
                                i += 1
                        continue

                # Inside expression (brace_depth > 0)
                c = text[i]

                # Collapse newline + indentation inside { ... }
                if c == "\n":
                    # Replace newline and following indentation with a single space
                    out.append(" ")
                    i += 1
                    while i < n and text[i] in " \t":
                        i += 1
                    changes += 1
                    continue

                # Handle nested braces and escaped braces
                if c == "{":
                    # '{{' is literal brace, keep both and do not change depth
                    if i + 1 < n and text[i + 1] == "{":
                        out.append("{{")
                        i += 2
                    else:
                        out.append("{")
                        i += 1
                        brace_depth += 1
                    continue
                if c == "}":
                    # '}}' is literal, keep both
                    if i + 1 < n and text[i + 1] == "}":
                        out.append("}}")
                        i += 2
                    else:
                        out.append("}")
                        i += 1
                        brace_depth -= 1
                    continue

                # Regular char inside expression
                out.append(c)
                i += 1

            continue  # end f-string handling

        # Not a quote start: copy and advance
        out.append(ch)
        i += 1

    return "".join(out), changes


def main() -> int:
    write = "--write" in sys.argv
    show_diff = "--diff" in sys.argv or not write

    any_changed = False
    for p in git_tracked_py_files():
        src = p.read_text(encoding="utf-8", errors="surrogatepass")
        fixed, count = fix_text(src)
        if count > 0:
            any_changed = True
            if show_diff:
                diff = difflib.unified_diff(
                    src.splitlines(True),
                    fixed.splitlines(True),
                    fromfile=str(p),
                    tofile=str(p),
                    n=3,
                )
                sys.stdout.writelines(diff)
            if write:
                p.write_text(fixed, encoding="utf-8")
    return 0 if any_changed or write else 1


if __name__ == "__main__":
    raise SystemExit(main())
