"""Check native long-ray sphere entries and existing border conventions."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess

import proposal as pp


def coordinates(vector: object) -> list[float]:
    return [vector.x, vector.y, vector.z]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    library = Path(pp.__file__).resolve()
    if not library.is_relative_to(source):
        raise ValueError("Load the native library built from this isolated checkout.")
    assert pp.atmfcp_boundary_precision_patch == "sphere-distant-cross-product-v1"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    center = [4380871.312771655, 466409.9003062877, 4611760.791954992]
    radius = 12.582735956179723 * 100
    cases = [
        ("region0_index61", [5333605.822288933, 2475432.705616122, 2470580.063954948],
         [-0.30864456887165603, -0.650836666176702, 0.6936498857968808], 3086819.249873918),
        ("region2_index7", [4711445.11842791, 3603563.211363389, -2344248.5102373837],
         [-0.043280827159740555, -0.41073704309358644, 0.9107260023910078], 7637859.07822116),
    ]
    sphere = pp.geometry.Sphere(pp.Cartesian3D(0, 0, 0), radius)
    results = []
    for name, anchor, direction, analytic_entry_m in cases:
        position = [(value - origin) * 100 for value, origin in zip(anchor, center)]
        for scale in [1.0, 1.0 - 1e-12, 1.0 + 1e-12]:
            tangent = [value * scale for value in direction]
            entry, exit = sphere.distance_to_border(pp.Cartesian3D(*position), pp.Cartesian3D(*tangent))
            terminal = [p + entry * d for p, d in zip(position, tangent)]
            residual_m = (math.sqrt(sum(value * value for value in terminal)) - radius) / 100
            assert 0 < entry < exit
            assert abs(residual_m) < 1e-6, (name, scale, residual_m)
            assert abs(entry / 100 - analytic_entry_m / scale) < 1e-6
            assert sum(p * d for p, d in zip(terminal, tangent)) < 0
            results.append(dict(case=name, direction_scale=scale, entry_m=entry / 100,
                                radius_error_m=residual_m, terminal_cm=terminal))
    small = pp.geometry.Sphere(pp.Cartesian3D(0, 0, 0), 10)
    for name, position, direction, expected in [
        ("crossing", [-20, 0, 0], [1, 0, 0], (10, 30)),
        ("miss", [-20, 10.1, 0], [1, 0, 0], (-1, -1)),
        ("tangent", [-20, 10, 0], [1, 0, 0], (-1, -1)),
        ("inside", [0, 0, 0], [1, 0, 0], (10, -1)),
        ("boundary_inward", [-10, 0, 0], [1, 0, 0], (20, -1)),
        ("boundary_outward", [-10, 0, 0], [-1, 0, 0], (-1, -1)),
    ]:
        actual = small.distance_to_border(pp.Cartesian3D(*position), pp.Cartesian3D(*direction))
        assert tuple(actual) == expected, (name, actual, expected)
        results.append(dict(case=name, distance_cm=list(actual)))
    hollow = pp.geometry.Sphere(pp.Cartesian3D(0, 0, 0), 10, 5)
    for position, expected in [([-20, 0, 0], (10, 15)), ([-7, 0, 0], (2, -1))]:
        actual = hollow.distance_to_border(pp.Cartesian3D(*position), pp.Cartesian3D(1, 0, 0))
        assert tuple(actual) == expected, (actual, expected)
        results.append(dict(case="hollow", position_cm=position, distance_cm=list(actual)))
    report = dict(schema="proposal-sphere-precision-validation-v1", status="succeeded", exit_code=0,
                  hostname=socket.gethostname(), created_utc=datetime.now(timezone.utc).isoformat(),
                  source_revision=revision, library_path=str(library),
                  library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                  scattering_patch=pp.atmfcp_scattering_patch,
                  scattering_domain_guard=pp.atmfcp_scattering_domain_guard, cases=results)
    report["boundary_precision_patch"] = pp.atmfcp_boundary_precision_patch
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
