#!/usr/bin/env python3
"""METHOD 2 -- RDKit-native depiction with condensed abbreviations.

A completely different approach from the interactive rigid-body viewer: let
RDKit / CoordGen lay out and draw the complex, with standard ABBREVIATIONS
(t-Bu, OMe, Ph ...) condensed by rdAbbreviations. Produces a static
publication image per ligand.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdAbbreviations, rdCoordGen  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402

from gen_panel import LIGANDS  # noqa: E402
from src import chem  # noqa: E402


def draw(key, name, smi, out):
    lig = chem.make_dtbm_segphos() if smi is None else Chem.MolFromSmiles(smi)
    cx, info = chem.build_complex(lig)
    abbr = rdAbbreviations.GetDefaultAbbreviations()
    try:
        cx = rdAbbreviations.CondenseMolAbbreviations(cx, abbr, maxCoverage=1.0)
    except Exception as exc:
        print("  abbrev skip:", exc)
    rdCoordGen.AddCoords(cx)
    d = rdMolDraw2D.MolDraw2DCairo(720, 620)
    o = d.drawOptions()
    o.addStereoAnnotation = False
    o.padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(d, cx, legend=name)
    d.FinishDrawing()
    Path(out).write_bytes(d.GetDrawingText())
    print("  wrote", out)


if __name__ == "__main__":
    for key, (name, smi) in LIGANDS.items():
        print("[m2]", key)
        draw(key, name, smi, f"/tmp/m2_{key}.png")
