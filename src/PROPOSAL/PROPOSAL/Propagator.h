#pragma once

#include "PROPOSAL/Secondaries.h"
#include "PROPOSAL/scattering/multiple_scattering/Moliere.h"
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <memory>

namespace PROPOSAL {
struct ParticleDef;

class Geometry;

using Sector = std::tuple<std::shared_ptr<const Geometry>, PropagationUtility,
    std::shared_ptr<const Density_distr>>;

struct CrossSectionBase;
}

namespace PROPOSAL {
struct StoppingPolicy {
    std::string identity = "handoff-core-envelope-v1";
    std::string predictor = "midpoint-256-v1";
    Cartesian3D center;
    double radius = 0.; // cm
    double sigma = 2.5, distance_factor = 6.;
    void Validate() const;
};

struct CoreMoments {
    double a0 = 0., a1 = 0., a2 = 0.;
    size_t attempted_panels = 0, completed_panels = 0, angular_evaluations = 0;
    bool valid = true, nuclear_nontrigger = false;
    std::string fallback;
    void AddPanel(double power, double length, double remaining);
    bool IsPositiveSemidefinite() const;
    bool ProvesNontrigger(double offset, double sigma) const;
};

struct StoppingDiagnostics {
    bool triggered = false, zero_variance_excluded = false;
    ParticleState trigger;
    double distance = 0., forward = 0., impact = 0., score = 0.;
    double a0 = 0., a1 = 0., a2 = 0.;
    size_t evaluations = 0, predictions = 0, fallbacks = 0, nuclear_nontrigger_bounds = 0;
    size_t attempted_panels = 0, completed_panels = 0;
    double predictor_cpu_seconds = 0.;
    std::map<std::string, size_t> fallback_counts;
    CoulombDiagnostics transport;
};

class Propagator {
public:
    Propagator(const ParticleDef& p_def, const nlohmann::json& config);
    Propagator(const ParticleDef& p_def, const std::string& config_file)
        : Propagator(p_def, ParseConfig(config_file))
    {
    }
    Propagator(const ParticleDef&, std::vector<Sector> sectors);

    Secondaries Propagate(const ParticleState& initial_particle,
        double max_distance = 1e20, double min_energy = 0.,
        unsigned int hierarchy_condition = 0,
        unsigned int stop_on_entry_hierarchy = 0,
        bool stop_on_invalid_scattering = false,
        std::shared_ptr<const StoppingPolicy> stopping_policy = nullptr,
        bool record_history = true);
    multiple_scattering::MoliereDiagnostics GetMoliereDiagnostics() const
    { return moliere_diagnostics_; }
    const std::string& GetLastTerminationReason() const
    { return last_termination_reason_; }
    const CoulombTransportResult& GetTransportResult() const { return transport_result_; }
    StoppingDiagnostics GetStoppingDiagnostics() const
    { auto result = stopping_diagnostics_; result.transport = transport_result_.diagnostics; return result; }
    CoreMoments PredictCoreMoments(const ParticleState&, double length,
        double max_distance = 1e20, double min_energy = 0., unsigned int hierarchy_condition = 0,
        bool defer_angular = true);
    static StoppingDiagnostics EvaluateCoreEnvelope(const ParticleState&,
        const StoppingPolicy&, const CoreMoments&);
    enum { GEOMETRY, UTILITY, DENSITY_DISTR };

private:
    CoreMoments PredictCoreMomentsImpl(const ParticleState&, double, double, double,
        unsigned int, bool, double, double);
    int AdvanceCoulomb(ParticleState&, double, double, std::function<double()>,
        Sector&, bool, double);
    Interaction::Loss DoStochasticInteraction(
        ParticleState&, PropagationUtility&, std::function<double()>);
    int AdvanceParticle(ParticleState& p_cond, const double E_f,
                        const double max_distance, std::function<double()> rnd,
                        Sector& current_sector, bool min_energy_step,
                        const double min_energy, unsigned int hierarchy_condition,
                        unsigned int stop_on_entry_hierarchy,
                        bool stop_on_invalid_scattering);
    double CalculateDistanceToBorder(const Vector3D& particle_position,
        const Vector3D& particle_direction, const Geometry& current_geometry);
    int maximize(const std::array<double, 3>& InteractionEnergies);
    int minimize(const std::array<double, 3>& AdvanceDistances);
    Sector GetCurrentSector(
        const Vector3D& particle_position, const Vector3D& particle_direction);
    // Global settings
    struct GlobalSettings {
        GlobalSettings();
        GlobalSettings(const nlohmann::json& config_global);
        nlohmann::json cross;
        nlohmann::json scattering;
        std::shared_ptr<EnergyCutSettings> cuts = nullptr;
        std::shared_ptr<Medium> medium = nullptr;
        bool do_exact_time;
        bool do_interpolation;
    };

    // Initializing methods
    static nlohmann::json ParseConfig(const std::string& config_file);
    void InitializeSectorFromJSON(
        const ParticleDef&, const nlohmann::json&, GlobalSettings);

    PropagationUtility::Collection CreateUtility(
        std::vector<std::shared_ptr<CrossSectionBase>> crosss,
        std::shared_ptr<Medium> medium, bool do_cont_rand, bool do_interpol,
        bool do_exact_time, nlohmann::json scatter);

    std::vector<std::shared_ptr<CrossSectionBase>> CreateCrossSectionList(
        const ParticleDef& p_def, const Medium& medium,
        std::shared_ptr<const EnergyCutSettings> cuts, bool interpolate,
        double density_correction, const nlohmann::json& config);

    ParticleDef p_def;
    enum Type : int {
        Decay = 0,
        MinimalE = 1,
        Stochastic = 2,
    };
    enum AdvancementType : int {
        InvalidStep = 0,
        ReachedInteraction = 1,
        ReachedMaxDistance = 2,
        ReachedBorder = 3,
        ReachedHierarchyEntry = 4,
        ReachedHierarchyExit = 5,
        ReachedCoulombStep = 6,
        InvalidScattering = 7
    };

    std::vector<Sector> sector_list;
    multiple_scattering::MoliereDiagnostics moliere_diagnostics_;
    std::string last_termination_reason_ = "unknown";
    CoulombTransportResult transport_result_;
    bool record_history_ = true;
    StoppingDiagnostics stopping_diagnostics_;
};

} // namespace PROPOSAL
