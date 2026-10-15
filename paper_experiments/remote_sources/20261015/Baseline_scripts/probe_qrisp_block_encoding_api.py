#!/usr/bin/env python3
"""Probe installed Qrisp block-encoding APIs for the TFIM_8 Kraus example."""

from __future__ import annotations

import inspect
import json
import pkgutil
import sys
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def safe_signature(obj: Any) -> str | None:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return None


def main() -> None:
    out_dir = ROOT / "Baseline_results" / "qrisp_block_encoding_api_probe_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}

    try:
        import qrisp

        result["qrisp_version"] = getattr(qrisp, "__version__", "unknown")
        result["qrisp_file"] = getattr(qrisp, "__file__", None)
        result["top_level_names_matching"] = [
            name for name in dir(qrisp) if "block" in name.lower() or "encoding" in name.lower() or "lcu" in name.lower()
        ]
        result["top_level_has_BlockEncoding"] = hasattr(qrisp, "BlockEncoding")
        result["top_level_BlockEncoding_signature"] = safe_signature(getattr(qrisp, "BlockEncoding", None))

        modules = []
        for mod in pkgutil.walk_packages(qrisp.__path__, qrisp.__name__ + "."):
            low = mod.name.lower()
            if "block" in low or "encoding" in low or "lcu" in low or "hamilton" in low or "operator" in low:
                modules.append(mod.name)
        result["candidate_modules"] = modules[:300]

        module_details = {}
        for mod_name in modules:
            try:
                module = __import__(mod_name, fromlist=["*"])
            except Exception:
                module_details[mod_name] = {"import_error": traceback.format_exc(limit=3)}
                continue
            names = [
                name for name in dir(module)
                if "Block" in name or "block" in name.lower() or "lcu" in name.lower()
                or "Hamilton" in name or "Operator" in name or name in {"X", "Y", "Z"}
            ]
            detail: dict[str, Any] = {"file": getattr(module, "__file__", None), "names": names[:100]}
            for name in names[:100]:
                obj = getattr(module, name, None)
                detail[f"{name}_repr"] = repr(obj)
                sig = safe_signature(obj)
                if sig is not None:
                    detail[f"{name}_signature"] = sig
                for attr in ("from_operator", "from_lcu", "from_foqcs_lcu", "from_foqcs_lcu_operator", "from_foqcs_lcu_prep", "apply", "compile", "transpile", "resources"):
                    member = getattr(obj, attr, None)
                    if member is not None:
                        detail[f"{name}.{attr}_signature"] = safe_signature(member)
            module_details[mod_name] = detail
        result["module_details"] = module_details
    except Exception:
        result["qrisp_import_error"] = traceback.format_exc()

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=repr) + "\n", encoding="utf-8")
    print(f"summary: {out_path}")


if __name__ == "__main__":
    main()
