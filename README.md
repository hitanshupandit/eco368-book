# Data Science for Economists

**Teaching notes for Econ 368 — Bates College**

Hitanshu Pandit · Department of Economics

Read it at **https://hitanshupandit.github.io/eco368-book/**.

This is the course written down, in the order we teach it, with Python that runs. Recommended reading for Econ 368 is **Géron, *Hands-On Machine Learning with Scikit-Learn and PyTorch* (O'Reilly, 2025)**. This repository is not a replacement for it: it is the course's teaching notes, so students can read the material in the order it is taught, run every example, and work the exercises.

## How it is built

Chapters are written in `src/NN_name.qmd` (Quarto, with Python code). `build.py` runs the code and writes `NN_name.md` plus its figures at the top level, where GitHub Pages publishes them. Every number in the text is produced by the code.

    python3 build.py          # render changed chapters, then assemble ECO368.md
    python3 build.py 12       # re-render chapter 12
    python3 build.py --all    # re-render everything

The chapters run in the Econ 368 course environment (`../ECON 368/.venv`). See `PLAN.md` for the chapter template and the rules the book follows.

## Sources

Part III follows the order and level of Géron's book. **The writing, examples and code here are original**; Géron's text and figures are copyrighted and are not reproduced. Where a code snippet is adapted from his Apache-2.0 companion notebooks (github.com/ageron/handson-mlp), a comment says so.

## Licence

© 2026 Hitanshu Pandit. Text under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); code under the MIT license.
