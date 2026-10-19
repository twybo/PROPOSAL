#pragma once

#include "PROPOSAL/medium/Medium.h"
#include "PROPOSAL/particle/Particle.h"
#include "PROPOSAL/particle/ParticleDef.h"
#include <functional>
#include <map>
#include <tuple>

namespace PROPOSAL {
struct CrossSectionBase;
enum class CoulombMode { Direct, Mixed };

struct CoulombSettings {
    CoulombMode mode = CoulombMode::Mixed;
    double separation = 1e-5; // sin²(theta_CM/2)
    double max_angular_variance = 0.01;
    double max_fractional_loss = 0.01;
    double max_step = 1e5; // cm
    bool interpolate = false;
    void Validate() const;
};

struct CoulombMoments {
    double rate = 0.; // cm²/g
    double angular = 0.; // rad² cm²/g, laboratory polar angle
    double stopping = 0.; // MeV cm²/g
    double straggling = 0.; // MeV² cm²/g
};

struct CoulombCollision {
    ParticleState before, after;
    size_t component = 0;
    double recoil = 0.;
    double x = 0.;
    bool electron = false;
};

struct TransportSegment {
    ParticleState before, after;
    double length = 0., grammage = 0.;
};

struct CoulombDiagnostics {
    size_t collisions = 0, soft_steps = 0, applicability_failures = 0;
    size_t electron_collisions = 0;
};

struct CoulombTransportResult {
    std::vector<TransportSegment> segments;
    std::vector<CoulombCollision> collisions;
    CoulombDiagnostics diagnostics;
    std::string model, mode, table;
};

// Pointlike spin-1/2 electric projectiles; custom particle definitions assert
// this interpretation. Cross sections are per atom, material moments per gram.
class Coulomb {
public:
    Coulomb(const ParticleDef&, const Medium&, CoulombSettings = {});
    CoulombSettings settings;
    static std::string ModelIdentity() { return "WM-Dirac-McF-exponential-v1"; }
    static std::string TableIdentity() { return "WM-Dirac-McF-moments-logK-0p02-v1"; }
    void ValidateEnergy(double) const;
    double Differential(double energy, size_t component, double x) const;
    CoulombMoments Moments(double energy, double lower, double upper) const;
    double ElectronAngular(double energy, CrossSectionBase&) const;
    CoulombCollision Sample(const ParticleState&, double lower,
        const std::function<double()>&) const;
    CoulombCollision Scatter(const ParticleState&, size_t component,
        double x, double azimuth) const;
    static Cartesian3D Rotate(const Cartesian3D&, double theta, double azimuth);
    double GetMass() const { return particle_.mass; }

private:
    struct Kernel {
        double target_mass, pcm, beta, screening, prefactor, radius;
    };
    Kernel MakeKernel(double, size_t) const;
    double Correction(const Kernel&, double x, double z) const;
    CoulombMoments Integrate(double, size_t, double, double) const;
    ParticleDef particle_;
    Medium medium_;
    std::vector<Component> components_;
    std::vector<double> atoms_per_gram_;
    mutable std::map<std::tuple<double,double,double>, CoulombMoments> moment_table_;
    CoulombMoments ExactMoments(double, double, double) const;
};
} // namespace PROPOSAL
