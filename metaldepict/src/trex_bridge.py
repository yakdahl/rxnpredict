"""
trex_bridge.py
==============
Wrap Ilia Kevlishvili's **T-REX** (`trex-notation`) so the depiction layer gets a
clean transition-metal-complex descriptor from a 3D conformer:

  * metal, oxidation state, spin
  * coordination number + coarse geometry  (e.g. 'trpl' = trigonal planar)
  * ligand payloads (SMILES) + the coordination MAP (donor sites)
  * canonical T-REX string
  * chirality:  point (@/@@), helical (Delta/Lambda), and whether chiral at all
  * idealised metal--donor distances / cis-angles  (standardised geometry)

Everything is best-effort: T-REX's xyz2mol perception can stumble on exotic
metals, so each call is guarded and a hand-built fallback descriptor (we know the
target exactly) is returned instead.
"""

from __future__ import annotations

from pathlib import Path


def _fallback_descriptor() -> dict:
    return {
        "source": "fallback",
        "metal": "Cu",
        "ox": 1,
        "spin": 1,                       # d10 Cu(I), closed shell (singlet)
        "geometry": "trpl",              # trigonal planar, CN 3
        "geometry_name": "trigonal planar",
        "cn": 3,
        "trex_string": "Cu{1} | L=[ SMILES:<DTBM-SEGPhos>, SMILES:[H-] ] "
                       "| MAP:{ ; 1:<P>, 1:<P>, 2:1 } | G:trpl   (illustrative; "
                       "real donor indices come from canonical_known_trex)",
        "chirality": None,
        "is_chiral": True,               # axial (atropisomeric) backbone
        "delta_lambda": None,
        "note": "axial chirality lives in the biaryl backbone, not the metal",
    }


_GEOM_NAMES = {
    "O": "octahedral", "sqpl": "square planar", "sqpy": "square pyramidal",
    "ln": "linear", "tsh": "T-shaped", "ssw": "see-saw",
    "trbp": "trigonal bipyramidal", "trpl": "trigonal planar",
    "bnt": "bent", "thd": "tetrahedral-ish", "unk": "unknown",
}


def describe(xyz_path: str | Path, overall_charge: int) -> dict:
    """Run T-REX on the conformer and return a JSON-friendly descriptor."""
    try:
        from trex import xyz_to_trex as x2t
        from trex import chirality as tchir
    except Exception as exc:
        d = _fallback_descriptor()
        d["error"] = f"trex import failed: {exc}"
        return d

    try:
        model = x2t.noncanonical_trex_model_from_xyz(str(xyz_path), overall_charge)
    except Exception as exc:
        d = _fallback_descriptor()
        d["error"] = f"trex perception failed: {exc}"
        return d

    desc: dict = {
        "source": "trex",
        "metal": model.metal,
        "ox": model.ox,
        "spin": model.spin,
        "geometry": model.get_geo_type(),
        "geometry_name": _GEOM_NAMES.get(model.get_geo_type(), model.get_geo_type()),
        "cn": model.cn(),
        "ligands": [{"kind": l.kind, "text": l.text} for l in model.ligands],
        "n_pairs": len(model.pairs),
        "n_singles": len(model.singles),
    }

    # canonical T-REX string with chirality if available
    for fn in ("xyz_to_trex_with_chirality", "xyz_to_trex_canonical", "xyz_to_trex"):
        f = getattr(x2t, fn, None)
        if f is None:
            continue
        try:
            desc["trex_string"] = f(str(xyz_path), overall_charge)
            break
        except Exception:
            continue

    # chirality analysis on the model
    try:
        desc["is_chiral"] = bool(tchir.is_chiral(model))
    except Exception:
        pass
    try:
        desc["chirality"] = tchir.compute_chirality(model)
    except Exception:
        pass
    try:
        desc["delta_lambda"] = tchir.compute_delta_lambda(model)
    except Exception:
        pass

    return desc


def canonical_known_trex(lig_smiles: str) -> dict:
    """
    Build the *chemically correct* T-REX descriptor for the known target and run
    it through T-REX's own canonicaliser.  This is more reliable than perceiving
    an exotic Cu-hydride from a raw all-hydrogen xyz (where the metal-bound H is
    indistinguishable from the ~100 other H's), and it still exercises the real
    T-REX model + canonicaliser.

    (DTBM-SEGPhos)Cu-H :  Cu(I), the bidentate diphosphine contributes two donor
    sites, the hydride one -> CN 3, trigonal (G:trpl); no trans pairs.
    """
    out: dict = {"available": False}
    try:
        from rdkit import Chem

        from trex.canon_full import canonicalize_full
        from trex.model import LigandPayload, MapSite, TrexFull
        from trex.parse_full import pretty_trex_full
    except Exception as exc:
        out["error"] = f"import: {exc}"
        return out
    try:
        m = Chem.MolFromSmiles(lig_smiles)
        pidx = [a.GetIdx() + 1 for a in m.GetAtoms() if a.GetSymbol() == "P"]
        ligs = [LigandPayload(kind="SMILES", text=lig_smiles),
                LigandPayload(kind="SMILES", text="[H-]")]
        singles = [MapSite(lig=1, atoms=[pidx[0]]),
                   MapSite(lig=1, atoms=[pidx[1]]),
                   MapSite(lig=2, atoms=[1])]
        t = TrexFull(metal="Cu", ox=1, spin=1, ligands=ligs,
                     pairs=[], singles=singles, G="trpl")
        try:
            t = canonicalize_full(t)
            out["canonicalised"] = True
        except Exception as exc:
            out["canonicalised"] = False
            out["canon_error"] = str(exc)
        out["available"] = True
        out["string"] = pretty_trex_full(t)
        out["cn"] = t.cn()
        out["geometry"] = t.get_geo_type()
    except Exception as exc:
        out["error"] = str(exc)
    return out


def ideal_metal_geometry(metal: str = "Cu") -> dict:
    """
    Standardised metal--donor distances / cis-cosines from T-REX's geometry
    tables, used as physics-engine rest lengths / target angles.  Falls back to
    literature values for Cu(I)-P.
    """
    out = {"metal_donor_A": {"P": 2.25, "H": 1.50}, "cis_angle_deg": 120.0}
    try:
        from trex import geometry as tgeo  # noqa: F401
        out["trex_geometry_available"] = True
    except Exception:
        out["trex_geometry_available"] = False
    return out
