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
