#!/usr/bin/env python
"""Build a mesh asset bank from one prior × N prompts × M seeds.

Usage:
    python scripts/morphology/build_asset_bank.py \
        --prior random_shapes \
        --bank-name loco_v1 \
        --bank-root data/asset_banks \
        --prompts-file scripts/morphology/prompts/locomotion.txt \
        --n-per-prompt 10 --seed 0 \
        [--prior-kwargs k1=v1,k2=v2]

Outputs:
    <bank-root>/<bank-name>/manifest.json
    <bank-root>/<bank-name>/meshes/<asset_id>.glb        (one per asset)

The script is idempotent: re-running with the same args reuses previously
generated meshes (asset_id is a deterministic hash of inputs).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List

# Make `genedynamics` importable when running from repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from genedynamics.morphology import AssetBank
from genedynamics.morphology.priors import get_prior, list_priors, MissingDependencyError


log = logging.getLogger("build_asset_bank")


def _parse_kv(s: str) -> Dict[str, Any]:
    """Parse 'k1=v1,k2=v2' → dict, with int/float coercion."""
    if not s:
        return {}
    out: Dict[str, Any] = {}
    for chunk in s.split(","):
        if "=" not in chunk:
            raise ValueError(f"bad --prior-kwargs entry: {chunk!r}")
        k, v = chunk.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Try numeric coercion in order: int → float → string.
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def _load_prompts(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"prompts file not found: {path}")
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prior", required=True,
                    help=f"prior name; one of {list_priors()}")
    ap.add_argument("--bank-name", required=True,
                    help="folder name under --bank-root")
    ap.add_argument("--bank-root", default="data/asset_banks",
                    help="root directory for asset banks")
    ap.add_argument("--prompts-file", required=True,
                    help="text file with one prompt per line (# comments allowed)")
    ap.add_argument("--n-per-prompt", type=int, default=10,
                    help="number of meshes per prompt")
    ap.add_argument("--seed", type=int, default=0,
                    help="base RNG seed; per-asset seed = base + index")
    ap.add_argument("--prior-kwargs", default="",
                    help="comma-separated k=v passed to the prior constructor")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan + exit without sampling")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    prompts = _load_prompts(args.prompts_file)
    log.info("Loaded %d prompts from %s", len(prompts), args.prompts_file)

    bank_root = os.path.join(args.bank_root, args.bank_name)
    if args.dry_run:
        log.info("DRY RUN — would generate %d × %d = %d assets into %s",
                 len(prompts), args.n_per_prompt,
                 len(prompts) * args.n_per_prompt, bank_root)
        for p in prompts:
            log.info("  prompt: %s", p)
        return 0

    # Construct prior; surface missing-dep errors with actionable hints.
    try:
        prior_kwargs = _parse_kv(args.prior_kwargs)
        prior = get_prior(args.prior, **prior_kwargs)
    except MissingDependencyError as exc:
        log.error("prior %r unavailable: %s", args.prior, exc)
        log.error("install hint: %s", exc.install_hint)
        return 2
    except KeyError as exc:
        log.error("%s", exc)
        return 2

    # Bank: open existing, or create fresh.
    manifest_path = os.path.join(bank_root, "manifest.json")
    if os.path.isfile(manifest_path):
        bank = AssetBank.open(bank_root)
        log.info("Resuming bank %s (%d assets already present)",
                 bank_root, len(bank.manifest.assets))
        # Ensure prompts are registered (idempotent — bank.create handles dedupe).
        existing_prompt_texts = {p.text for p in bank.manifest.prompts}
        for p in prompts:
            if p not in existing_prompt_texts:
                from genedynamics.morphology.asset_bank import PromptEntry, prompt_id as _pid
                bank.manifest.prompts.append(PromptEntry(id=_pid(p), text=p))
    else:
        bank = AssetBank.create(
            root=bank_root,
            bank_name=args.bank_name,
            prior_name=prior.metadata.name,
            prior_version=prior.metadata.version,
            prompts=prompts,
        )
        log.info("Created bank %s", bank_root)

    # Generate.
    t0 = time.perf_counter()
    n_new = 0
    for prompt in prompts:
        for i in range(args.n_per_prompt):
            seed_i = args.seed + i
            try:
                meshes = prior.sample(prompt, n=1, seed=seed_i)
            except Exception:
                log.exception("prior.sample failed for prompt=%r seed=%d", prompt, seed_i)
                continue
            entry = bank.add_mesh(meshes[0], prompt=prompt, seed=seed_i, index=i)
            n_new += 1
            log.debug("  prompt=%s seed=%d → asset_id=%s", prompt, seed_i, entry.id)
        bank.save()  # checkpoint after each prompt batch
    wall = time.perf_counter() - t0

    log.info("Generated %d assets in %.1fs (%.2fs/asset)",
             n_new, wall, wall / max(n_new, 1))
    log.info("Bank manifest: %s", os.path.join(bank_root, "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
