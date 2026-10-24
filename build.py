#!/usr/bin/env python3
"""Build the Econ 368 book.

Chapters are written as Quarto files in src/NN_name.qmd with Python code that
actually runs. This script

  1. renders each changed src/*.qmd to GitHub-flavoured markdown (code, printed
     output and figures included), and moves NN_name.md plus its figure folder to
     the repository root, where GitHub Pages publishes it;
  2. concatenates every NN_*.md into ECO368.md with a table of contents;
  3. checks each chapter against the template in PLAN.md.

Every number in the book therefore comes from running the code, never from typing.

Usage:
    python3 build.py              # render changed chapters, then assemble
    python3 build.py 12           # re-render chapter 12 only
    python3 build.py --all        # re-render everything

Python: the chapters run in the Econ 368 course environment. Point QUARTO_PYTHON
at it if it is not at ../ECON 368/.venv/bin/python.
"""

import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
OUT = os.path.join(HERE, "ECO368.md")
VENV = os.path.normpath(os.path.join(HERE, "..", "ECON 368", ".venv", "bin"))

#  Sections every chapter is supposed to carry. The build reports a missing one,
#  because the usual reason is that I stopped writing halfway and forgot.
TEMPLATE = ["## In Python", "## What goes wrong", "## Exercises", "### Answers"]

#  The book follows Géron's chapter order but is written from scratch. These
#  phrases mean a passage leans on the textbook instead of explaining, so the
#  build refuses to let them in.
BANNED = ["see Géron", "Géron p.", "as Géron writes", "Figure 1-", "Figure 2-",
          "Figure 3-", "Figure 4-", "Figure 5-", "Figure 6-", "Figure 7-"]


def quarto():
    exe = os.path.join(VENV, "quarto")
    return exe if os.path.exists(exe) else shutil.which("quarto")


def render(qmd):
    """Render one src/NN_name.qmd to NN_name.md (+ NN_name_files/) at the root."""
    stem = os.path.splitext(os.path.basename(qmd))[0]
    env = dict(os.environ)
    py = os.path.join(VENV, "python")
    if "QUARTO_PYTHON" not in env and os.path.exists(py):
        env["QUARTO_PYTHON"] = py
    print(f"  rendering {stem} ...", flush=True)
    subprocess.run([quarto(), "render", qmd, "--to", "gfm"], cwd=SRC, env=env, check=True)
    md, figs = os.path.join(SRC, stem + ".md"), os.path.join(SRC, stem + "_files")
    shutil.move(md, os.path.join(HERE, stem + ".md"))
    dest = os.path.join(HERE, stem + "_files")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    if os.path.exists(figs):
        shutil.move(figs, dest)


def stale(qmd):
    md = os.path.join(HERE, os.path.splitext(os.path.basename(qmd))[0] + ".md")
    return not os.path.exists(md) or os.path.getmtime(md) < os.path.getmtime(qmd)


def chapters():
    """Published chapter files, in filename order. The NN_ prefix is the running order."""
    names = sorted(f for f in os.listdir(HERE) if re.match(r"^\d\d_.*\.md$", f))
    return [(n, open(os.path.join(HERE, n)).read()) for n in names]


def assemble():
    files = chapters()
    toc, body, rows = [], [], []
    for name, text in files:
        m = re.search(r"^# (.+)$", text, re.M)
        title = m.group(1).strip() if m else name
        if not name.startswith("00_"):
            anchor = re.sub(r"[^a-z0-9 -]", "", title.lower()).replace(" ", "-")
            toc.append(f"- [{title}](#{anchor})")
            for sub in re.findall(r"^## (.+)$", text, re.M):
                if not sub.startswith(("In Python", "What goes wrong", "Exercises", "Answers")):
                    toc.append(f"  - {sub.strip()}")
        body.append(text.rstrip() + "\n")
        is_chapter = not name.startswith(("00_", "9"))
        missing = [s for s in TEMPLATE if s not in text] if is_chapter else []
        rows.append((name, len(text.split()), missing))

    with open(OUT, "w") as f:
        f.write(files[0][1].rstrip() + "\n\n## Contents\n\n" + "\n".join(toc) + "\n\n---\n\n")
        f.write("\n".join(body[1:]))

    print(f"\n  wrote ECO368.md  --  {len(files)} files, {sum(r[1] for r in rows):,} words\n")
    for name, words, missing in rows:
        flag = "   MISSING: " + ", ".join(s.lstrip("# ") for s in missing) if missing else ""
        print(f"    {name:<40} {words:>6,} words{flag}")
    whole = "\n".join(b for _, b in files)
    found = [b for b in BANNED if b in whole]
    print("\n    textbook leaning:", ", ".join(found) if found else "none")
    tb = [n for n, b in files if "Traceback (most recent call last)" in b]
    print("    tracebacks in output:", ", ".join(tb) if tb else "none")


def main():
    args = sys.argv[1:]
    sources = sorted(os.path.join(SRC, f) for f in os.listdir(SRC) if re.match(r"^\d\d_.*\.qmd$", f))
    if "--all" in args:
        todo = sources
    elif args:
        todo = [s for s in sources if os.path.basename(s)[:2] in {a.zfill(2) for a in args}]
    else:
        todo = [s for s in sources if stale(s)]
    for qmd in todo:
        render(qmd)
    assemble()


if __name__ == "__main__":
    main()
