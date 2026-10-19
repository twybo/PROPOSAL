#include "PROPOSAL/Propagator.h"
#include "PROPOSAL/Constants.h"
#include "PROPOSAL/scattering/Scattering.h"
#include "PROPOSAL/crosssection/CrossSection.h"
#include "PROPOSAL/density_distr/density_distr.h"
#include "PROPOSAL/density_distr/density_homogeneous.h"
#include "PROPOSAL/geometry/Sphere.h"
#include "PROPOSAL/geometry/Box.h"
#include "PROPOSAL/geometry/Cylinder.h"
#include <cmath>

using namespace PROPOSAL;
namespace {
double surface_safety(const Geometry& geometry, const Cartesian3D& point)
{
    const auto p = point-geometry.GetPosition();
    if (const auto* s = dynamic_cast<const Sphere*>(&geometry)) {
        double d = std::abs(p.magnitude()-s->GetRadius());
        if (s->GetInnerRadius() > 0.) d = std::min(d,std::abs(p.magnitude()-s->GetInnerRadius()));
        return d;
    }
    if (const auto* b = dynamic_cast<const Box*>(&geometry))
        return std::min({std::abs(std::abs(p.GetX())-b->GetX()/2.),
            std::abs(std::abs(p.GetY())-b->GetY()/2.),std::abs(std::abs(p.GetZ())-b->GetZ()/2.)});
    if (const auto* c = dynamic_cast<const Cylinder*>(&geometry)) {
        double r = std::hypot(p.GetX(),p.GetY());
        double d = std::min(std::abs(r-c->GetRadius()),std::abs(std::abs(p.GetZ())-c->GetZ()/2.));
        if (c->GetInnerRadius() > 0.) d = std::min(d,std::abs(r-c->GetInnerRadius()));
        return d;
    }
    return 0.; // Unknown surfaces use individual collisions.
}
double open_uniform(const std::function<double()>& rnd)
{
    double u = rnd();
    if (!std::isfinite(u) || u < 0. || u >= 1.)
        throw std::runtime_error("Invalid transport random number");
    return std::max(u, std::numeric_limits<double>::min());
}
double gamma_sample(double shape, const std::function<double()>& rnd)
{
    if (shape < 1.) return gamma_sample(shape+1.,rnd)*std::pow(open_uniform(rnd),1./shape);
    const double d = shape-1./3., c = 1./std::sqrt(9.*d);
    for (int i=0; i<10000; ++i) {
        double z = std::sqrt(-2*std::log(open_uniform(rnd)))*std::cos(2*PI*open_uniform(rnd));
        double v = 1.+c*z;
        if (v <= 0.) continue;
        v = v*v*v;
        double u = open_uniform(rnd);
        if (std::log(u) < 0.5*z*z+d*(1.-v+std::log(v))) return d*v;
    }
    throw std::runtime_error("Soft recoil sampler exhausted");
}
}

int Propagator::AdvanceCoulomb(ParticleState& state, double target_energy,
    double final_distance, std::function<double()> rnd, Sector& sector,
    bool min_energy_step, double min_energy)
{
    // A density inversion can stop just short of a surface by less than its
    // coordinate tolerance. Resolve the forward tangent before using another
    // material step, including when the previous step was not tagged a crossing.
    if (std::get<GEOMETRY>(GetCurrentSector(state.position,state.direction))
        != std::get<GEOMETRY>(sector))
        return ReachedBorder;
    auto& utility = std::get<UTILITY>(sector);
    auto& density = std::get<DENSITY_DISTR>(sector);
    auto model = utility.collection.scattering->GetCoulomb();
    model->settings.Validate();
    try { model->ValidateEnergy(state.energy); }
    catch (...) { ++transport_result_.diagnostics.applicability_failures; throw; }
    const auto& settings = model->settings;
    transport_result_.model = Coulomb::ModelIdentity();
    const std::string mode = settings.mode == CoulombMode::Direct ? "direct" : "mixed";
    transport_result_.mode = transport_result_.mode.empty() || transport_result_.mode == mode ? mode : "multiple";
    transport_result_.table = settings.interpolate ? Coulomb::TableIdentity() : "none";
    const double remaining = final_distance-state.propagated_distance;
    if (remaining <= 0.) return ReachedMaxDistance;
    if (!dynamic_cast<const Density_homogeneous*>(density.get()))
        throw std::invalid_argument("Finite-angle v1 requires homogeneous density within each sector");
    const double rho = density->Evaluate(state.position);
    if (!(rho > 0.) || !std::isfinite(rho)) throw std::domain_error("Coulomb requires positive finite density");
    double safety = INF;
    for (const auto& s : sector_list) safety = std::min(safety,surface_safety(*std::get<GEOMETRY>(s),state.position));
    double separation = settings.mode == CoulombMode::Direct ? 0. : settings.separation;
    auto hard = model->Moments(state.energy,separation,1.);
    // A condensed path of length <= safety/2 cannot touch any surface, even
    // after its hinge. Switch to detailed transport rather than asymptotically
    // approaching an interface with ever shorter condensed steps.
    const auto restricted = model->Moments(state.energy,0.,separation);
    if (safety < 4./(rho*(hard.rate+restricted.rate))) {
        separation = 0.;
        hard = model->Moments(state.energy,0.,1.);
    }
    auto soft = model->Moments(state.energy,0.,separation);
    double stopping = 0., electron_angular = 0.;
    for (const auto& cross : utility.collection.interaction_calc->GetCrossSections()) {
        const double loss = cross->CalculatedEdx(state.energy);
        stopping += loss;
        if (cross->GetInteractionType() == InteractionType::Ioniz) {
            const auto cuts = cross->GetEnergyCutSettings();
            if (!cuts || cuts->GetCut(state.energy)*state.energy > 0.005*(state.energy-p_def.mass)*(1.+1e-12)) {
                ++transport_result_.diagnostics.applicability_failures;
                throw std::invalid_argument("Coulomb requires ionization transfer cut <= 0.005 times kinetic energy; JSON construction caps it at 1 keV");
            }
            // Small-transfer free-electron relation theta² = 2 m_e T / p².
            // Excitation stopping shares this approximation; no recoil energy
            // is added here because the ionization model already accounts for it.
            electron_angular += model->ElectronAngular(state.energy,*cross);
        }
    }
    const double angular = soft.angular+electron_angular;
    const double event_grammage = (state.energy-target_energy)/stopping;
    const double collision_grammage = -std::log(open_uniform(rnd))/hard.rate;
    double limit_grammage = settings.max_fractional_loss*(state.energy-p_def.mass)/(stopping+soft.stopping);
    if (angular > 0.) limit_grammage = std::min(limit_grammage,settings.max_angular_variance/angular);
    const double selected_grammage = std::min({event_grammage,collision_grammage,limit_grammage});
    double border = CalculateDistanceToBorder(state.position,state.direction,*std::get<GEOMETRY>(sector));
    if (!(border > 0.)) border = INF;
    double length = std::min({remaining,border,settings.max_step});
    if (separation > 0.) length = std::min(length,0.5*safety);
    double grammage = density->Calculate(state.position,state.direction,length);
    if (grammage > selected_grammage) {
        length = density->Correct(state.position,state.direction,selected_grammage,length);
        grammage = selected_grammage;
    }
    if (!(length > 0.) || !std::isfinite(length) || !std::isfinite(grammage))
        throw std::runtime_error("Invalid Coulomb propagation step");
    int result = ReachedCoulombStep;
    bool collision = grammage == collision_grammage;
    if (grammage == event_grammage) result = ReachedInteraction;
    if (length == border) { result = ReachedBorder; collision = false; }
    if (length == remaining) { result = ReachedMaxDistance; collision = false; }

    // Near a surface even the small electron deflection is applied at the
    // segment end, so it cannot cause an untracked material crossing.
    const bool hinge = separation > 0. && angular > 0.;
    const double fraction = hinge ? open_uniform(rnd) : 1.;
    const double theta = std::sqrt(-angular*grammage*std::log(open_uniform(rnd)));
    const double phi = 2*PI*open_uniform(rnd);
    double recoil = soft.stopping*grammage;
    if (soft.straggling > 0. && recoil > 0.) {
        const double variance = soft.straggling*grammage;
        recoil = variance/recoil*gamma_sample(recoil*recoil/variance,rnd);
    }
    const double initial_energy = state.energy;
    double accumulated = 0.;
    for (int part=0; part<(hinge ? 2 : 1); ++part) {
        const double distance = length*(part==0 ? fraction : 1.-fraction);
        const auto before = state;
        const double g = density->Calculate(state.position,state.direction,distance);
        accumulated += g;
        double energy = (initial_energy-stopping*accumulated);
        energy -= recoil*accumulated/grammage;
        energy = std::max(p_def.mass,energy);
        const double mean = (state.energy+energy)/2.;
        const double beta = std::sqrt((mean-p_def.mass)*(mean+p_def.mass))/mean;
        state.time += distance/(SPEED*beta);
        state.position += distance*state.direction;
        state.propagated_distance += distance;
        state.energy = energy;
        transport_result_.segments.push_back({before,state,distance,g});
        if (part == 0 && angular > 0.) state.direction = Coulomb::Rotate(state.direction,theta,phi);
    }
    if (angular > 0.) ++transport_result_.diagnostics.soft_steps;
    if (!(min_energy_step && result == ReachedInteraction))
        state.energy = std::max(p_def.mass, utility.EnergyRandomize(initial_energy,
            (initial_energy-stopping*accumulated),rnd,min_energy)-recoil);
    transport_result_.segments.back().after.energy = state.energy;
    if (collision && state.energy-p_def.mass >= 0.2) {
        auto event = model->Sample(state,separation,rnd);
        state = event.after;
        transport_result_.collisions.push_back(event);
        ++transport_result_.diagnostics.collisions;
    }
    if (!std::isfinite(state.energy) || !std::isfinite(state.position.magnitude())
        || std::abs(state.direction.magnitude()-1.) > 1e-12)
        throw std::runtime_error("Invalid finite-angle transport state");
    return result;
}
