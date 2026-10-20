"""Deterministic kernel and native path regressions, using the built binding."""
import math
import random
from functools import lru_cache
from typing import Any
from pathlib import Path

import numpy as np
import pytest

import proposal as pp


@lru_cache(maxsize=None)
def particle(name: str, mass: float = 100., charge: float = 0.01) -> Any:
    definitions = {"electron": pp.particle.EMinusDef, "positron": pp.particle.EPlusDef,
                   "mu-": pp.particle.MuMinusDef, "mu+": pp.particle.MuPlusDef}
    if name in definitions:
        return definitions[name]()
    builder = pp.particle.ParticleDefBuilder()
    builder.SetParticleDef(pp.particle.MuMinusDef())
    builder.SetName(f"Dirac_mCP_{mass}_{charge}")
    builder.SetMass(mass)
    builder.SetLow(mass)
    builder.SetCharge(charge)
    builder.SetLifetime(math.inf)
    builder.SetParticleType(1000000000+int(mass)*100+int(round(charge*1000)))
    return builder.build()


def medium(name: str) -> Any:
    constructors = {"ice": pp.medium.Ice, "rock": pp.medium.StandardRock,
                    "hydrogen": pp.medium.Hydrogen, "lead": pp.medium.Lead}
    if name in constructors:
        return constructors[name]()
    if name == "aluminium":
        return pp.medium.Medium("aluminium", 166., -4.2395, 0.08024, 3.6345,
            0.1708, 3.0127, 0.12, 2.699,
            [pp.component.Component("Al", 13., 26.9815385, 1.)])
    if name == "carbon":
        return pp.medium.Medium("carbon", 78., -2.9925, 0.2024, 3.0036,
            -0.0351, 2.486, 0.1, 2.265,
            [pp.component.Component("C", 6., 12.0107, 1.)])
    raise ValueError(name)


def utility(particle_def: Any, material: Any, model: str,
            separation: float = 1e-5, step_scale: float = 1., full: bool = False, max_step: float = 1e5) -> Any:
    cuts = pp.EnergyCutSettings(0.001, 1.)
    if full:
        cross = pp.crosssection.make_std_crosssection(particle_def, material, cuts, False)
    else:
        ion = pp.parametrization.ionization
        parametrization = (ion.BergerSeltzerMoller(cuts) if particle_def.particle_type == 11 else
                           ion.BergerSeltzerBhabha(cuts) if particle_def.particle_type == -11 else
                           ion.BetheBlochRossi(cuts))
        cross = [pp.crosssection.make_crosssection(parametrization, particle_def, material, cuts, False)]
    collection = pp.PropagationUtilityCollection()
    collection.displacement = pp.make_displacement(cross, False)
    collection.interaction = pp.make_interaction(collection.displacement, cross, False)
    collection.time = pp.make_time(cross, particle_def, False)
    scattering = pp.make_multiple_scattering(model, particle_def, material, cross, False)
    if scattering.coulomb is not None:
        settings = scattering.coulomb.settings
        settings.separation = separation
        settings.max_angular_variance = 0.01*step_scale
        settings.max_fractional_loss = 0.01*step_scale
        settings.max_step = max_step*step_scale
    collection.scattering = pp.scattering.Scattering(scattering)
    return pp.PropagationUtility(collection)


def slab(particle_def: Any, material: Any, model: str, thickness: float,
         separation: float = 1e-5, step_scale: float = 1.) -> Any:
    util = utility(particle_def, material, model, separation, step_scale, max_step=thickness/8.)
    foil = pp.geometry.Box(pp.Cartesian3D(0, 0, thickness/2), 1e12, 1e12, thickness)
    foil.hierarchy = 2
    world = pp.geometry.Sphere(pp.Cartesian3D(0, 0, 0), 1e15)
    world.hierarchy = 0
    density = pp.density_distribution.density_homogeneous(material.mass_density)
    return pp.Propagator(particle_def, [(foil, util, density), (world, util, density)])


def state(energy: float, z: float = 0.) -> Any:
    result = pp.particle.ParticleState()
    result.energy = energy
    result.position = pp.Cartesian3D(0, 0, z)
    result.direction = pp.Cartesian3D(0, 0, 1)
    return result


def xyz(vector: pp.Cartesian3D) -> np.ndarray:
    return np.asarray(vector.cartesian_coordinates)


def test_partition_and_charge() -> None:
    target = medium("ice")
    p = particle("mcp", 100., .01)
    c = pp.scattering.Coulomb(p, target)
    whole, soft, hard = [c.moments(1000., *bounds) for bounds in [(0., 1.), (0., 1e-5), (1e-5, 1.)]]
    for key in ("rate", "angular", "stopping", "straggling"):
        assert getattr(whole, key) == pytest.approx(getattr(soft, key)+getattr(hard, key), rel=3e-5)
    small = pp.scattering.Coulomb(particle("mcp", 100., .001), target)
    assert c.differential(1000., 1, .1)/small.differential(1000., 1, .1) == pytest.approx(100., rel=.005)


@pytest.mark.parametrize("name,mass", [("electron", 1.), ("positron", 1.), ("mu-", 1.), ("mcp", 1000.)])
def test_recoil_and_unit_direction(name: str, mass: float) -> None:
    p = particle(name, mass, .1)
    c = pp.scattering.Coulomb(p, medium("hydrogen"))
    s = state(p.mass+10.)
    for x in (0., 1e-6, .5, 1.):
        event = c.scatter(s, 0, x, 1.2)
        assert np.linalg.norm(xyz(event.after.direction)) == pytest.approx(1., abs=2e-15)
        assert s.energy-event.after.energy == pytest.approx(event.recoil, abs=2e-13)
        pi = math.sqrt((s.energy-p.mass)*(s.energy+p.mass))
        pf = math.sqrt(max(0., (event.after.energy-p.mass)*(event.after.energy+p.mass)))
        recoil_p2 = np.sum((pi*xyz(s.direction)-pf*xyz(event.after.direction))**2)
        # Hydrogen's target is represented by its constituent-weighted nucleon mass.
        target = medium("hydrogen").components[0]
        nuclear_mass = target.atomic_number*target.average_nucleon_weight
        assert recoil_p2 == pytest.approx(event.recoil*(event.recoil+2*nuclear_mass), rel=2e-9, abs=1e-9)


def test_sampling_cdf_and_constituents() -> None:
    p = particle("mu-")
    c = pp.scattering.Coulomb(p, medium("ice"))
    s = state(200.)
    low, split = 1e-5, 5e-5
    expected = c.moments(s.energy, low, split).rate/c.moments(s.energy, low, 1.).rate
    rng = random.Random(73612)
    samples = [c.sample(s, low, rng.random) for _ in range(3000)]
    observed = sum(event.x <= split for event in samples)/len(samples)
    assert abs(observed-expected) < 5*math.sqrt(expected*(1-expected)/len(samples))
    assert {event.component for event in samples} == {0, 1}


@pytest.mark.parametrize("model", ["CoulombDirect", "Moliere", "MoliereInterpol"])
def test_native_segments(model: str) -> None:
    p, mat = particle("mu-"), medium("ice")
    prop = slab(p, mat, model, .003)
    pp.RandomGenerator.get().set_seed(87)
    out = prop.propagate(state(1000.), 1., p.mass+.2, 1)
    result = prop.transport_result
    assert prop.last_termination_reason == "exit"
    assert result.segments
    assert result.mode == ("direct" if model == "CoulombDirect" else "mixed")
    assert result.table == (pp.scattering.Coulomb.table_identity if model == "MoliereInterpol" else "none")
    for segment in result.segments:
        delta = xyz(segment.after.position)-xyz(segment.before.position)
        assert np.linalg.norm(delta) == pytest.approx(segment.length, rel=1e-9, abs=1e-13)
        assert delta == pytest.approx(segment.length*xyz(segment.before.direction), rel=1e-9, abs=1e-13)
        assert segment.grammage == pytest.approx(segment.length*mat.mass_density, rel=1e-12)
        assert segment.after.time >= segment.before.time
    assert out.track_propagated_distances()[-1] == pytest.approx(sum(s.length for s in result.segments), rel=1e-10)


@pytest.mark.parametrize("shape", ["sphere", "box", "cylinder"])
def test_crossings(shape: str) -> None:
    p, mat = particle("mcp", 100., .001), medium("ice")
    util = utility(p, mat, "Moliere")
    center = pp.Cartesian3D(0,0,0)
    geometries = {"sphere": lambda: pp.geometry.Sphere(center, 1.),
                  "box": lambda: pp.geometry.Box(center, 2.,2.,2.),
                  "cylinder": lambda: pp.geometry.Cylinder(center, 2.,1.)}
    inner = geometries[shape](); inner.hierarchy = 2
    world = pp.geometry.Sphere(center, 1e10)
    density = pp.density_distribution.density_homogeneous(mat.mass_density)
    prop = pp.Propagator(p, [(inner,util,density),(world,util,density)])
    pp.RandomGenerator.get().set_seed(981)
    out = prop.propagate(state(1000.), 10., p.mass+.2, 1)
    assert prop.last_termination_reason == "exit"
    assert out.track_positions()[-1].z == pytest.approx(1., abs=1e-7)


def test_inward_earth_tangent() -> None:
    p, mat = particle("mcp", 100., .001), medium("ice")
    util = utility(p, mat, "Moliere")
    radius = 6.371e8
    earth = pp.geometry.Sphere(pp.Cartesian3D(0,0,0), radius); earth.hierarchy = 2
    world = pp.geometry.Sphere(pp.Cartesian3D(0,0,0), 1e12)
    density = pp.density_distribution.density_homogeneous(mat.mass_density)
    prop = pp.Propagator(p, [(earth,util,density),(world,util,density)])
    s = state(1000., radius+2e-7); s.direction = pp.Cartesian3D(0,0,-1)
    out = prop.propagate(s, 2., p.mass+.2, 1)
    assert prop.last_termination_reason == "max_distance"
    assert out.track_positions()[-1].z < radius-1.9


def test_backward_transport_and_legacy_domain_guard() -> None:
    p, mat = particle("electron"), medium("ice")
    c = pp.scattering.Coulomb(p,mat)
    s = c.scatter(state(10.), 1, 1., 0.).after
    assert s.direction.z < -.999
    util = utility(p,mat,"CoulombDirect")
    world = pp.geometry.Sphere(pp.Cartesian3D(0,0,0),1e10)
    prop = pp.Propagator(p,[(world,util,pp.density_distribution.density_homogeneous(mat.mass_density))])
    out = prop.propagate(s,1e-5)
    assert out.track_positions()[-1].z < 0.
    offset = pp.scattering.scattering_offset()
    offset.tx = 2.
    with pytest.raises(RuntimeError, match="domain"):
        pp.scattering.scatter_initial_direction(pp.Cartesian3D(0,0,1),offset)
    assert pp.scattering.MoliereLegacy(p,mat).coulomb is None


def test_applicability() -> None:
    with pytest.raises(ValueError):
        pp.scattering.Coulomb(pp.particle.StauMinusDef(),medium("ice"))
    with pytest.raises(ValueError):
        pp.scattering.Coulomb(particle("electron"),medium("ice")).moments(.6,0.,1.)


def test_independent_quadrature_and_interpolation() -> None:
    mat, p = medium("ice"), particle("mu-")
    c = pp.scattering.Coulomb(p,mat)
    x = np.r_[0.,np.geomspace(1e-16,1.,16000)]
    molecular_mass = sum(a.atomic_number*a.atoms_in_molecule for a in mat.components)
    rate = 0.
    for i, component in enumerate(mat.components):
        integral = np.trapezoid([c.differential(300.,i,float(v)) for v in x],x)
        rate += 6.02214076e23*component.atoms_in_molecule/molecular_mass*integral
    assert c.moments(300.,0.,1.).rate == pytest.approx(rate,rel=1e-5)
    fast = pp.scattering.MoliereInterpol(p,mat).coulomb
    for bounds in [(0.,1e-5),(1e-5,1.)]:
        exact, table = c.moments(300.,*bounds), fast.moments(300.,*bounds)
        for field in ("rate","angular","stopping","straggling"):
            assert getattr(table,field) == pytest.approx(getattr(exact,field),rel=.001)


@pytest.mark.parametrize("material,species,model,seed,optical", [
    ("rock","electron","Moliere",21731125,32),
    ("rock","mu-","CoulombDirect",21941121,32),
    ("lead","electron","Moliere",26021120,32),
])
def test_near_surface_handoff_seeds(material: str, species: str, model: str, seed: int, optical: int) -> None:
    p,mat=particle(species),medium(material)
    energy=p.mass+(10. if species=="electron" else 100.)
    rate=pp.scattering.Coulomb(p,mat).moments(energy,0.,1.).rate
    thickness=optical/rate/mat.mass_density
    prop=slab(p,mat,model,thickness)
    pp.RandomGenerator.get().set_seed(seed)
    track=prop.propagate(state(energy),1.,0.,1)
    assert prop.last_termination_reason=="exit"
    assert track.track_propagated_distances()[-1] < 2*thickness


def test_actual_high_b_legacy_domain_failure() -> None:
    legacy=pp.scattering.MoliereLegacy(particle("electron"),medium("lead"))
    offset=legacy.scatter(1.,10.,10.,[.99,.99,.99,.99])
    assert min(legacy.last_b)>10.
    assert offset.tx**2+offset.ty**2>1.
    with pytest.raises(RuntimeError,match="domain"):
        pp.scattering.scatter_initial_direction(pp.Cartesian3D(0,0,1),offset)


def test_json_configuration_and_partition(tmp_path: Path) -> None:
    import json
    config={"global":{"medium":"ice","do_interpolation":False,
        "cuts":{"e_cut":500.,"v_cut":1.,"cont_rand":False},
        "CrossSections":{"ioniz":{"parametrization":"BetheBlochRossi"}},
        "scattering":{"multiple_scattering":"Moliere","coulomb":{"mode":"direct",
        "model":"WM-Dirac-McF-exponential-v1"}}},
        "sectors":[{"geometries":[{"shape":"sphere","origin":[0,0,0],"outer_radius":1000.,"hierarchy":0}]}]}
    path=tmp_path/"transport.json"
    path.write_text(json.dumps(config))
    prop=pp.Propagator(particle("mu-"),str(path))
    pp.RandomGenerator.get().set_seed(72)
    prop.propagate(state(1000.),.01)
    assert prop.transport_result.diagnostics.collisions>0
    assert prop.transport_result.model=="WM-Dirac-McF-exponential-v1"
    config["global"]["scattering"]["coulomb"]["model"]="unknown"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match="identity"):
        pp.Propagator(particle("mu-"),str(path))


def test_material_change_grammage() -> None:
    p=particle("mcp",100.,.001)
    ice,lead=medium("ice"),medium("lead")
    ice_util,lead_util=utility(p,ice,"Moliere"),utility(p,lead,"Moliere")
    first=pp.geometry.Box(pp.Cartesian3D(0,0,.5),100.,100.,1.);first.hierarchy=2
    second=pp.geometry.Box(pp.Cartesian3D(0,0,1.5),100.,100.,1.);second.hierarchy=3
    world=pp.geometry.Sphere(pp.Cartesian3D(0,0,0),1000.)
    ice_density=pp.density_distribution.density_homogeneous(ice.mass_density)
    lead_density=pp.density_distribution.density_homogeneous(lead.mass_density)
    prop=pp.Propagator(p,[(first,ice_util,ice_density),(second,lead_util,lead_density),(world,ice_util,ice_density)])
    pp.RandomGenerator.get().set_seed(2013)
    prop.propagate(state(1000.),2.1)
    segments=prop.transport_result.segments
    densities=[s.grammage/s.length for s in segments]
    assert min(densities)==pytest.approx(ice.mass_density,rel=1e-12)
    assert max(densities)==pytest.approx(lead.mass_density,rel=1e-12)
    assert sum(s.grammage for s in segments)==pytest.approx(1.1*ice.mass_density+lead.mass_density,rel=.001)


def test_charge_above_one_with_explicit_coupling_domain() -> None:
    p=particle("mcp",100.,2.)
    kernel=pp.scattering.Coulomb(p,medium("hydrogen"))
    assert kernel.moments(1000.,0.,1.).rate>0.
    with pytest.raises(ValueError,match="applicability"):
        pp.scattering.Coulomb(p,medium("lead"))


def test_lower_applicability_endpoint() -> None:
    p,mat=particle("mcp",1000.,.01),medium("hydrogen")
    cuts=pp.EnergyCutSettings(.001,1.)
    cross=pp.crosssection.make_crosssection(pp.parametrization.ionization.BetheBlochRossi(cuts),p,mat,cuts,False)
    kernel=pp.scattering.Coulomb(p,mat)
    angular=kernel.electron_angular(p.mass+.21,cross)
    assert math.isfinite(angular) and angular>0.
    electron=particle("electron")
    table=pp.scattering.MoliereInterpol(electron,medium("ice")).coulomb
    assert table.moments(electron.mass+.2,0.,1.).rate>0.


def test_positron_annihilation_competes_with_collisions(tmp_path: Path) -> None:
    import json
    config={"global":{"medium":"ice","do_interpolation":False,
        "cuts":{"e_cut":.001,"v_cut":1.,"cont_rand":False},
        "CrossSections":{"ioniz":{"parametrization":"BergerSeltzerBhabha"},
                         "annihilation":{"parametrization":"Heitler","multiplier":1e8}},
        "scattering":{"multiple_scattering":"CoulombDirect"}},
        "sectors":[{"geometries":[{"shape":"sphere","origin":[0,0,0],"outer_radius":1000.,"hierarchy":0}]}]}
    path=tmp_path/"annihilation.json";path.write_text(json.dumps(config))
    prop=pp.Propagator(particle("positron"),str(path))
    pp.RandomGenerator.get().set_seed(72)
    prop.propagate(state(10.),10.)
    assert prop.last_termination_reason=="annihilation"


def stopping_policy(radius: float = 1., center: tuple[float, float, float] = (0., 0., 0.)) -> Any:
    policy = pp.StoppingPolicy()
    policy.radius = radius
    policy.center = pp.Cartesian3D(*center)
    return policy


def envelope_state(impact: float = 3., forward: float = 10.) -> Any:
    s = state(1e6, -forward)
    s.position = pp.Cartesian3D(impact, 0., -forward)
    return s


def test_core_uniform_and_heterogeneous_panels() -> None:
    moments = pp.CoreMoments()
    moments.add_panel(.02, 12., 6.)
    assert moments.a0 == pytest.approx(.24)
    assert moments.a1 == pytest.approx(12.*moments.a0/2.)
    assert moments.a2 == pytest.approx(12.**2*moments.a0/3.)
    assert moments.positive_semidefinite()
    split = pp.CoreMoments()
    split.add_panel(.02, 4., 10.)
    split.add_panel(.02, 8., 4.)
    assert (split.a0, split.a1, split.a2) == pytest.approx((moments.a0, moments.a1, moments.a2))
    heterogeneous = pp.CoreMoments()
    heterogeneous.add_panel(.02, 4., 10.)
    heterogeneous.add_panel(.04, 8., 4.)
    assert heterogeneous.a0 == pytest.approx(.4)
    assert heterogeneous.a1 == pytest.approx(.02*(12.**2-8.**2)/2.+.04*8.**2/2.)
    assert heterogeneous.a2 == pytest.approx(.02*(12.**3-8.**3)/3.+.04*8.**3/3.)
    assert heterogeneous.positive_semidefinite()
    for scale in (1e-100, 1., 1e100):
        bad = pp.CoreMoments()
        bad.a0, bad.a1, bad.a2 = scale, 2.*scale, scale
        assert not bad.positive_semidefinite()
        assert not pp.Propagator.evaluate_core_envelope(envelope_state(), stopping_policy(), bad).triggered
    invalid = pp.CoreMoments()
    invalid.add_panel(-1., 1., .5)
    assert not invalid.valid


@pytest.mark.parametrize('score,trigger', [(2.49, False), (2.5, False), (2.51, True)])
def test_core_score_threshold(score: float, trigger: bool) -> None:
    m = pp.CoreMoments()
    m.a0, m.a1, m.a2 = 1., 0., (2./score)**2
    result = pp.Propagator.evaluate_core_envelope(envelope_state(), stopping_policy(), m)
    assert result.score == pytest.approx(score)
    assert result.triggered == trigger


@pytest.mark.parametrize('distance,trigger', [(5.99, False), (6., False), (6.01, True)])
def test_core_distance_threshold(distance: float, trigger: bool) -> None:
    s = envelope_state(3., math.sqrt(distance**2-9.))
    assert pp.Propagator.evaluate_core_envelope(s, stopping_policy(), pp.CoreMoments()).triggered == trigger


@pytest.mark.parametrize('impact,forward,reason', [(1.,10.,'overlap'), (.5,10.,'overlap'), (10.,0.,'nonforward'), (3.,-10.,'nonforward')])
def test_core_geometry_fallbacks(impact: float, forward: float, reason: str) -> None:
    out = pp.Propagator.evaluate_core_envelope(envelope_state(impact,forward), stopping_policy(), pp.CoreMoments())
    assert not out.triggered
    assert out.fallback_counts[reason] == 1


def test_zero_variance_finite_encoding_and_units() -> None:
    result = pp.Propagator.evaluate_core_envelope(envelope_state(), stopping_policy(), pp.CoreMoments())
    assert result.triggered and result.zero_variance_excluded and result.score == 0.
    moments = pp.CoreMoments()
    moments.a0, moments.a1, moments.a2 = .1, .2, 1.
    reference = pp.Propagator.evaluate_core_envelope(envelope_state(), stopping_policy(), moments)
    scaled = pp.CoreMoments()
    scaled.a0, scaled.a1, scaled.a2 = moments.a0, moments.a1*100., moments.a2*10000.
    s = envelope_state(300., 1000.)
    converted = pp.Propagator.evaluate_core_envelope(s, stopping_policy(100.), scaled)
    assert converted.score == pytest.approx(reference.score)
    s.position = pp.Cartesian3D(1000., -300., 0.)
    s.direction = pp.Cartesian3D(-1., 0., 0.)
    rotated = pp.Propagator.evaluate_core_envelope(s, stopping_policy(100.), scaled)
    assert rotated.score == pytest.approx(reference.score)
    s.position = s.position+pp.Cartesian3D(100.,200.,300.)
    translated = pp.Propagator.evaluate_core_envelope(s, stopping_policy(100.,(100.,200.,300.)), scaled)
    assert translated.score == pytest.approx(reference.score)


def cutoff_backend(charge: float = .001, material: str = 'ice', earth_radius: float = 1e5) -> Any:
    p, mat = particle('mcp',100.,charge), medium(material)
    util = utility(p, mat, 'MoliereInterpol')
    earth = pp.geometry.Sphere(pp.Cartesian3D(0.,0.,0.),earth_radius)
    earth.hierarchy = 2
    target = pp.geometry.Sphere(pp.Cartesian3D(0.,0.,0.),1.)
    target.hierarchy = 3
    world = pp.geometry.Sphere(pp.Cartesian3D(0.,0.,0.),1e15)
    density = pp.density_distribution.density_homogeneous(mat.mass_density)
    return pp.Propagator(p,[(target,util,density),(earth,util,density),(world,util,density)])


def test_native_first_stop_trigger_reset_and_rng() -> None:
    prop = cutoff_backend()
    initial = envelope_state(20.,1000.)
    initial.direction = pp.Cartesian3D(0.,0.,2.)
    pp.RandomGenerator.get().set_seed(812)
    expected = pp.RandomGenerator.get().random_double()
    pp.RandomGenerator.get().set_seed(812)
    final = prop.propagate(initial,2000.,0.,1,3,True,stopping_policy()).final_state()
    result = prop.stopping_diagnostics
    assert prop.last_termination_reason == 'approximate_miss'
    assert result.triggered and result.predictions == 1 and result.completed_panels > 0
    assert result.transport.soft_steps == result.transport.collisions == 0
    assert xyz(final.position) == pytest.approx(xyz(result.trigger.position))
    assert xyz(final.direction) == pytest.approx([0.,0.,1.])
    assert xyz(final.direction) == pytest.approx(xyz(result.trigger.direction))
    assert final.energy == initial.energy and final.propagated_distance == 0.
    assert pp.RandomGenerator.get().random_double() == expected
    prop.propagate(state(1000.),0.,0.,1,3,True)
    reset = prop.stopping_diagnostics
    assert not reset.triggered and reset.predictions == reset.evaluations == 0


def test_native_priority_and_domain_fallbacks() -> None:
    prop = cutoff_backend()
    s = envelope_state(20.,1000.)
    assert prop.predict_core_moments(s,1000.,100.).fallback == 'remaining_path'
    s.energy = 100.2
    assert prop.predict_core_moments(s,1000.).fallback == 'energy_domain'
    prop.propagate(s,2000.,0.,1,3,True,stopping_policy())
    assert prop.last_termination_reason == 'applicability_limit'
    assert not prop.stopping_diagnostics.triggered
    s.energy = 1e6
    prop.propagate(s,0.,0.,1,3,True,stopping_policy())
    assert prop.last_termination_reason == 'max_distance'
    limited = cutoff_backend(earth_radius=10.)
    assert limited.predict_core_moments(envelope_state(3.,1.),100.,1e20,0.,1).fallback == 'earth_exit'
    # A long reference loses enough kinetic energy to exceed the fixed panel budget.
    low = envelope_state(20.,1000.)
    low.energy = 101.
    result = cutoff_backend(.1,'rock',1e15).predict_core_moments(low,1e12)
    assert not result.valid and result.fallback in {'panel_budget','energy_domain'}
    assert result.attempted_panels <= 256


def test_native_entry_priority_and_boundary_material() -> None:
    prop = cutoff_backend()
    pp.RandomGenerator.get().set_seed(147)
    out = prop.propagate(state(1e6,-2.),10.,0.,1,3,True,stopping_policy())
    assert prop.last_termination_reason == 'entry'
    assert not prop.stopping_diagnostics.triggered
    assert out.final_state().position.z == pytest.approx(-1.,abs=1e-7)
    p=particle('mcp',100.,.001)
    ice,rock=medium('ice'),medium('rock')
    ui,ur=utility(p,ice,'MoliereInterpol'),utility(p,rock,'MoliereInterpol')
    first=pp.geometry.Box(pp.Cartesian3D(0.,0.,.5),100.,100.,1.); first.hierarchy=2
    second=pp.geometry.Box(pp.Cartesian3D(0.,0.,1.5),100.,100.,1.); second.hierarchy=3
    world=pp.geometry.Sphere(pp.Cartesian3D(0.,0.,0.),1e6)
    di=pp.density_distribution.density_homogeneous(ice.mass_density)
    dr=pp.density_distribution.density_homogeneous(rock.mass_density)
    layered=pp.Propagator(p,[(first,ui,di),(second,ur,dr),(world,ui,di)])
    pred=layered.predict_core_moments(state(1e6),2.,10.,0.,1)
    assert pred.valid and pred.completed_panels >=16
    pp.RandomGenerator.get().set_seed(827)
    out=layered.propagate(state(1e6),1.5,0.,0,0,True,stopping_policy())
    assert layered.last_termination_reason == 'max_distance'
    densities=[x.grammage/x.length for x in layered.transport_result.segments]
    assert min(densities)==pytest.approx(ice.mass_density)
    assert max(densities)==pytest.approx(rock.mass_density)


@pytest.mark.parametrize('material',['ice','rock'])
@pytest.mark.parametrize('charge',[.001,.01,.1])
def test_study_interpolated_moments_and_projected_power(material: str, charge: float) -> None:
    p,mat=particle('mcp',100.,charge),medium(material)
    exact=pp.scattering.Coulomb(p,mat)
    fast=pp.scattering.MoliereInterpol(p,mat).coulomb
    for energy in (100.21,1000.,1e6):
        for bounds in ((0.,1e-5),(1e-5,1.)):
            a,b=exact.moments(energy,*bounds),fast.moments(energy,*bounds)
            for field in ('rate','angular','stopping','straggling'):
                assert getattr(a,field)==pytest.approx(getattr(b,field),rel=.002)
    cuts=pp.EnergyCutSettings(.001,1.)
    cross=pp.crosssection.make_crosssection(pp.parametrization.ionization.BetheBlochRossi(cuts),p,mat,cuts,False)
    power=mat.mass_density*(fast.moments(1e6,0.,1e-5).angular+fast.electron_angular(1e6,cross))/2.
    backend=cutoff_backend(charge,material)
    prediction=backend.predict_core_moments(envelope_state(),.001)
    assert prediction.valid
    assert prediction.a0==pytest.approx(power*.001,rel=1e-7)
    assert prediction.a1==pytest.approx(.001*prediction.a0/2.,rel=1e-7)
    assert prediction.a2==pytest.approx(.001**2*prediction.a0/3.,rel=1e-7)
    pp.RandomGenerator.get().set_seed(6021)
    backend.propagate(state(1e6),.001)
    result=backend.transport_result
    assert result.mode == 'mixed'
    assert result.table == pp.scattering.Coulomb.table_identity


def test_disabled_policy_compatibility_and_rejection() -> None:
    prop=cutoff_backend()
    pp.RandomGenerator.get().set_seed(493)
    first=prop.propagate(state(1000.),.01).final_state()
    pp.RandomGenerator.get().set_seed(493)
    second=prop.propagate(state(1000.),.01,stopping_policy=None).final_state()
    assert xyz(first.position)==pytest.approx(xyz(second.position),rel=0.,abs=0.)
    assert xyz(first.direction)==pytest.approx(xyz(second.direction),rel=0.,abs=0.)
    assert first.energy==second.energy
    invalid=stopping_policy()
    invalid.identity='unknown'
    with pytest.raises(ValueError,match='policy'):
        prop.propagate(state(1000.),.01,stopping_policy=invalid)


def test_particle_builder_owns_temporary_source_table() -> None:
    import gc

    builder = pp.particle.ParticleDefBuilder()
    builder.SetParticleDef(pp.particle.MuMinusDef())
    gc.collect()
    # Reuse native vector allocations after the source definition has died.
    replacements = [pp.particle.MuMinusDef() for _ in range(32)]
    built = builder.build()
    assert built == replacements[0]
    del builder, replacements
    gc.collect()
    assert built == pp.particle.MuMinusDef()
