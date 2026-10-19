Finite-angle Coulomb transport
=============================

The native ``Propagator`` routes ``Moliere`` to mixed collision transport,
``CoulombDirect`` to individually sampled nuclear collisions, and
``MoliereInterpol`` to the same mixed physics with cached moment interpolation.
``MoliereLegacy`` retains the previous propagation calculation. The projected
``scatter``, ``scattering_angle`` and ``scattering_angle_2d`` methods remain
legacy interfaces, including on Moliere; they are not finite-angle transport.

This is an approximate physical model, not a reproduction of Geant4 or a full
Mott calculation. Its identity is ``WM-Dirac-McF-exponential-v1``. The kernel
uses the screened, reduced-mass, factorized McKinley–Feshbach formulation and
exponential nuclear form factor described in the Geant4 electron reference:
https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/elastic_scattering/elecnuc.html
The mixed treatment separates cross sections before sampling, following the
approach discussed in https://arxiv.org/abs/hep-ph/0405023 .

For total projectile energy E, mass m, charge q in units of positron charge,
target mass M and nuclear charge Z, define p² = E² − m²,
s = m² + M² + 2ME, p* = Mp/√s, β = p/E and x = sin²(θ*/2).
The CM domain is 0 ≤ x ≤ 1. Screening uses the Thomas–Fermi radius and includes
(qZα/β)². The differential cross section includes the Dirac factor 1 − β²x,
the charge-signed first McKinley–Feshbach correction, and the squared exponential
nuclear form factor. The recoil is T = 2p*²x/M. Lorentz transformation gives the
outgoing laboratory direction. Material moments sum atoms per gram multiplied
by per-atom integrals. No effective-Z substitution is used.

Applicability and approximations
--------------------------------

* Supported definitions are electrons, positrons, muons, and custom particles
  whose IDs are zero or at least 1000000000. Using such a custom ID explicitly
  asserts a pointlike spin-½ electric particle; ParticleDef has no spin field.
  Distinct custom definitions require distinct IDs in PROPOSAL's registry.
  This version accepts finite positive masses and finite nonzero charge. Each
  constituent must satisfy |q|Zα ≤ 0.6, an explicit model-domain cap rather than
  an accuracy guarantee. Charges above one are supported in sufficiently low-Z
  targets. Monopoles,
  scalar particles and other standard species are rejected.
* The kinetic-energy floor is 0.2 MeV. An initial unsupported energy or an
  unsupported species raises an exception. A transported particle reaching
  the floor returns ``applicability_limit`` when the requested minimum was
  lower. This is not proof that the particle stopped physically.
* Densities must be homogeneous within each sector. Material changes between
  sectors are supported. Nonhomogeneous density raises an explicit exception.
* Nuclear masses use the existing constituent-weighted proton/neutron masses;
  isotope-specific binding energies and nuclear magnetic structure are absent.
  The exponential form factor is a model, not measured isotope form-factor data.
* McKinley–Feshbach is a first Coulomb correction. High-Z and large-angle
  accuracy, especially for electrons and positrons, is not established by
  implementation tests. No full Mott coefficient tables are installed.
* Hard electron transfers use the existing Møller, Bhabha or Bethe–Bloch–Rossi
  rates and two-body deflection. JSON construction caps the ionization cut at
  1 keV, retaining the original fractional cut and multiplier. All loss utilities
  are constructed from that same revised partition. Manually constructed
  utilities must supply a sufficiently small cut: at most 0.005 of kinetic
  energy. Arbitrary cross-section wrappers beyond the standard multiplier
  have not been validated for automatic repartitioning.
* Restricted electron angles use exact free-electron kinematics between the
  material excitation energy and the cut. The unresolved excitation stopping
  uses θ² ≈ 2mₑT/p². Atomic binding angular distributions are not resolved.
  Electron recoil energy is already included in ionization and is not added
  a second time. ``Direct`` refers to nuclear elastic collisions; it still
  condenses unresolved electron excitations.
* Native competition uses local interaction/decay rates, limited fractional
  energy changes, and local stopping. This is a controlled-step approximation,
  not exact integration of energy-dependent hazards. The nuclear direct sampler
  itself samples the specified angular cross section without a small-angle
  projection. Direct-reference thin slabs keep energy changes small.
* Soft nuclear stopping and straggling use restricted first/second recoil
  moments. A positive gamma approximation matches those two energy moments;
  it does not reproduce the complete soft recoil distribution or its joint
  angular–energy distribution. Rotations use a small-step Gaussian tangent
  increment. A uniformly placed random hinge produces displacement–direction
  correlation. Convergence must be tested for each use regime.

Geometry and accounting
-----------------------

Physical segments advance along normalized tangents. Their lengths, grammage,
endpoint states and times are available through ``transport_result.segments``.
Nuclear and hard electron collision states are recorded separately from those
segments in ``transport_result.collisions`` (``electron`` identifies the latter).
``diagnostics`` counts collisions, electron collisions, soft steps and
applicability failures; it is separate from the legacy B diagnostics.

Mixed path length is bounded by half the distance to every supported surface.
Near surfaces the nuclear transport becomes individual collisions. Sphere,
box and cylinder surfaces are supported; unknown shapes conservatively use
individual collisions. A scale-aware forward tangent probe resolves sector
ownership without adding a fictitious displacement. Ownership is checked again
before each native step, including when density inversion landed just short of
a surface. This prevents missing a crossing through floating-point roundoff.

Legacy transverse proposals outside the unit direction domain now throw an
exception instead of being silently converted to a non-unit vector. A comparison
must retain those failed histories in its denominator.

Configuration and bindings
--------------------------

A sector or global scattering object can contain::

    "scattering": {
      "multiple_scattering": "Moliere",
      "coulomb": {
        "mode": "mixed",
        "model": "WM-Dirac-McF-exponential-v1",
        "separation": 0.00001,
        "max_angular_variance": 0.01,
        "max_fractional_loss": 0.01,
        "max_step": 100000
      }
    }

``separation`` is the CM x boundary, not a laboratory angle. ``max_step`` is
in cm; energy units throughout are MeV. ``mode`` also accepts ``direct``.
Angular multipliers are rejected for this backend; set the physical charge.
The Python ``MultipleScattering.coulomb`` and ``Scattering.coulomb`` properties
expose the model and its settings. The standalone ``scattering.Coulomb`` class
exposes differential cross sections (cm² per atom per x), material moments,
sampling, and fixed-angle recoil kinematics.

Native MoliereInterpol tables are generated in memory, keyed by energy and
restriction bounds. Their scheme is v1, linear interpolation on a grid spaced
by 0.02 in log kinetic energy. They never consume legacy Moliere angular tables.
Projected-angle legacy methods still use the old angular interpolants. Model
instances and their mutable caches should not be shared across concurrent threads.

Verification
------------

Build with the existing Conan-generated CMake toolchain. The focused pytest file
is ``tests/python/test_coulomb_transport.py`` and exercises the compiled native
kernel, transport and Python interface. Use the built module explicitly: a stale
root-level ``proposal*.so`` can otherwise shadow the new build. For example::

    PYTHONPATH="$PWD/build/src/pyPROPOSAL" python -P -m pytest tests/python/test_coulomb_transport.py
