"""Replay the single IceCube boundary-direction failure with the opt-in guard."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
import proposal as pp

from atmfcp.constants import R_EARTH
from atmfcp.scattering_transport import get_scattering_backend, propagate_ray, scattering_config
from atmfcp.transport_boundary import HandoffSphere


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert pp.atmfcp_scattering_boundary_guard == "stop-before-inconsistent-sphere-crossing-v1"
    assert hashlib.sha256(args.core.read_bytes()).hexdigest() == "7d64d12a6dc27a4942f91541acb50be58f1d12f0ccd5e0979bd27a87eb7fa9dc"
    core = json.loads(args.core.with_name("core.json").read_text())
    rays = dict(np.load(args.core))
    sphere = HandoffSphere(**core["sphere"])
    index, seed = 612, 1806823149
    config = scattering_config(0.1, 0.01, "ice", float(rays["energies_gev"].max()), sphere, scattering=True)
    backend = get_scattering_backend(config, args.cache)
    state = propagate_ray(backend, float(rays["energies_gev"][index]), 0.1,
                          rays["anchors_m"][index], rays["directions"][index], sphere, seed, 4 * R_EARTH)
    assert backend.last_termination_reason == "invalid_scattering"
    assert state.reason == 6
    assert state.travelled_m < 232801.30001057067
    report = dict(schema="proposal-sphere-crossing-guard-validation-v1", status="succeeded", exit_code=0,
                  hostname=socket.gethostname(), finished_utc=datetime.now(timezone.utc).isoformat(),
                  original_index=index, particle_seed=seed, raw_reason=backend.last_termination_reason,
                  terminal=state.__dict__, library_path=pp.__file__,
                  library_sha256=hashlib.sha256(Path(pp.__file__).read_bytes()).hexdigest(),
                  scattering_boundary_guard=pp.atmfcp_scattering_boundary_guard)
    for name, value in list(report["terminal"].items()):
        if isinstance(value, np.ndarray):
            report["terminal"][name] = value.tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
