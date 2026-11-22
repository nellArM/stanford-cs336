"""Compile the report:  uv run --no-project --with typst python writeup/compile.py"""
from pathlib import Path

import typst

here = Path(__file__).resolve().parent
typst.compile(str(here / "writeup.typ"), output=str(here / "writeup.pdf"), root=str(here.parent))
print("wrote", here / "writeup.pdf")
