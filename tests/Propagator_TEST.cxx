#include "gtest/gtest.h"
#include "PROPOSAL/crosssection/ParticleDefaultCrossSectionList.h"
#include "PROPOSAL/Propagator.h"
#include "PROPOSAL/propagation_utility/TimeBuilder.h"
#include "PROPOSAL/propagation_utility/InteractionBuilder.h"
#include "PROPOSAL/propagation_utility/ContRandBuilder.h"
#include "PROPOSAL/propagation_utility/DecayBuilder.h"
#include "PROPOSAL/density_distr/density_homogeneous.h"
#include "PROPOSAL/scattering/ScatteringFactory.h"
#include "PROPOSAL/geometry/Sphere.h"
#include "PROPOSAL/particle/Particle.h"
#include "PROPOSAL/Constants.h"
#include "PROPOSAL/math/RandomGenerator.h"
#include <limits>

using namespace PROPOSAL;

TEST(Propagator, min_energy)
{
    auto p_def = MuMinusDef();
    auto medium = Ice();
    auto cuts = std::make_shared<EnergyCutSettings>(INF, 0.05, true);
    auto cross = GetStdCrossSections(p_def, medium, cuts, true);

    auto collection = PropagationUtility::Collection();
    collection.interaction_calc = make_interaction(cross, true);
    collection.displacement_calc = make_displacement(cross, true);
    collection.time_calc = make_time(cross, p_def, true);
    collection.cont_rand = make_contrand(cross, true);
    collection.scattering = make_scattering(MultipleScatteringType::Highland, {}, p_def, medium);

    auto prop_utility = PropagationUtility(collection);

    auto density_distr = std::make_shared<Density_homogeneous>(medium);
    auto world = std::make_shared<Sphere>(Cartesian3D(0, 0, 0), 1e20);

    auto sector = std::make_tuple(world, prop_utility, density_distr);
    std::vector<Sector> sec_vec = {sector};

    auto prop = Propagator(p_def, sec_vec);

    auto init_state = ParticleState();
    init_state.energy = 1e5;
    init_state.position = Cartesian3D(0, 0, 0);
    init_state.direction = Cartesian3D(0, 0, 1);

    double min_energy = 1e4;
    for (size_t i=0; i<1000; i++) {
        auto sec = prop.Propagate(init_state, 1e20, min_energy);
        double final_energy = sec.GetTrack().back().energy;
        // If we propagate stable particles to min_energy, no particles should
        // have an energy bigger than min_energy
        EXPECT_LE(final_energy, min_energy);

        // Particles with a stochastic loss as their last interaction are allowed
        // to have energies smaller than min_energy. Otherwise, they don't.
        if (final_energy < min_energy)
            EXPECT_NE(sec.GetTrackTypes().back(), InteractionType::ContinuousEnergyLoss);
        else
            EXPECT_EQ(sec.GetTrackTypes().back(), InteractionType::ContinuousEnergyLoss);
    }
}

TEST(Propagator, RestMassDecay)
{
    // make sure that all muons decay at the end of their propagation, even if they have reached their rest mass
    // (see issue https://github.com/tudo-astroparticlephysics/PROPOSAL/issues/323)
    auto p_def = MuMinusDef();
    auto medium = Ice();
    auto cuts = std::make_shared<EnergyCutSettings>(INF, 0.05, true);
    auto cross = GetStdCrossSections(p_def, medium, cuts, true);

    auto collection = PropagationUtility::Collection();
    collection.interaction_calc = make_interaction(cross, true);
    collection.displacement_calc = make_displacement(cross, true);
    collection.time_calc = make_time(cross, p_def, true);
    collection.cont_rand = make_contrand(cross, true);
    collection.scattering = make_scattering(MultipleScatteringType::Highland, {}, p_def, medium);
    collection.decay_calc = make_decay(cross, p_def, true);

    auto prop_utility = PropagationUtility(collection);

    auto density_distr = std::make_shared<Density_homogeneous>(medium);
    auto world = std::make_shared<Sphere>(Cartesian3D(0, 0, 0), 1e20);

    auto sector = std::make_tuple(world, prop_utility, density_distr);
    std::vector<Sector> sec_vec = {sector};

    auto prop = Propagator(p_def, sec_vec);

    auto init_state = ParticleState();
    init_state.energy = 150;
    init_state.position = Cartesian3D(0, 0, 0);
    init_state.direction = Cartesian3D(0, 0, 1);

    for (size_t i=0; i<1000; i++) {
        auto sec = prop.Propagate(init_state);
        EXPECT_FALSE(sec.GetDecayProducts().empty()); // check that decay products are produced
    }

}


namespace {
class FixedDomainScatter : public multiple_scattering::Parametrization {
    multiple_scattering::ScatteringOffset offset_;
public:
    explicit FixedDomainScatter(multiple_scattering::ScatteringOffset offset)
        : Parametrization(MMU), offset_(offset) {}
    std::unique_ptr<multiple_scattering::Parametrization> clone() const override
    { return std::make_unique<FixedDomainScatter>(*this); }
    bool compare(const multiple_scattering::Parametrization&) const override { return false; }
    void print(std::ostream&) const override {}
    multiple_scattering::ScatteringOffset CalculateRandomAngle(
        double, double, double, const std::array<double, 4>&) override
    { return offset_; }
    double CalculateScatteringAngle(double, double, double, double) override { return 0.; }
    double CalculateScatteringAngle2D(double, double, double, double, double) override { return 0.; }
};

class FixedDomainDeflection : public stochastic_deflection::Parametrization {
    bool invalid_;
public:
    explicit FixedDomainDeflection(bool invalid) : invalid_(invalid) {}
    std::unique_ptr<stochastic_deflection::Parametrization> clone() const override
    { return std::make_unique<FixedDomainDeflection>(*this); }
    size_t RequiredRandomNumbers() const noexcept override { return 1; }
    InteractionType GetInteractionType() const noexcept override { return InteractionType::Ioniz; }
    UnitSphericalVector CalculateStochasticDeflection(
        double, double, const std::vector<double>&, size_t) const override
    { return {0.1, invalid_ ? std::numeric_limits<double>::quiet_NaN() : 0.2}; }
};

class FixedDomainInteraction : public Interaction {
public:
    FixedDomainInteraction(std::shared_ptr<Displacement> displacement, const crosssection_list_t& cross)
        : Interaction(displacement, cross) {}
    double EnergyInteraction(double energy, double) override { return 0.99 * energy; }
    double EnergyIntegral(double, double) override { return 0.; }
    double MeanFreePath(double) override { return 1.; }
};

Propagator DomainGuardPropagator(multiple_scattering::ScatteringOffset offset,
                                bool discrete = false, bool invalid_discrete = false)
{
    auto particle = MuMinusDef();
    auto medium = Ice();
    auto cuts = std::make_shared<EnergyCutSettings>(INF, discrete ? 0.001 : 1., false);
    crosssection_list_t cross;
    cross.emplace_back(make_crosssection(crosssection::IonizBetheBlochRossi(*cuts),
                                        particle, medium, cuts, false));
    PropagationUtility::Collection collection;
    collection.displacement_calc = make_displacement(cross, false);
    if (discrete)
        collection.interaction_calc = std::make_shared<FixedDomainInteraction>(collection.displacement_calc, cross);
    else
        collection.interaction_calc = make_interaction(collection.displacement_calc, cross, false);
    collection.time_calc = std::make_shared<ApproximateTimeBuilder>();
    std::vector<std::unique_ptr<stochastic_deflection::Parametrization>> deflections;
    if (discrete)
        deflections.emplace_back(std::make_unique<FixedDomainDeflection>(invalid_discrete));
    collection.scattering = std::make_shared<Scattering>(
        std::make_unique<FixedDomainScatter>(offset), std::move(deflections));
    auto world = std::make_shared<Sphere>(Cartesian3D(0, 0, 0), 1e20);
    return Propagator(particle, {std::make_tuple(world, PropagationUtility(collection),
                                               std::make_shared<Density_homogeneous>(medium))});
}

ParticleState DomainGuardInitialState()
{
    ParticleState state;
    state.energy = 1000.;
    state.position = Cartesian3D(0, 0, 0);
    state.direction = Cartesian3D(0, 0, 1);
    return state;
}
}

TEST(ScatteringDomainGuard, InvalidMeanAndFinalRemainAtLastValidState)
{
    const auto initial = DomainGuardInitialState();
    for (const auto invalid : {1.2, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
        for (const auto mean : {false, true}) {
            multiple_scattering::ScatteringOffset offset;
            if (mean) offset.sx = invalid;
            else offset.tx = invalid;
            auto propagator = DomainGuardPropagator(offset);
            const auto track = propagator.Propagate(initial, 1., 0., 0, 0, true);
            ASSERT_EQ(propagator.GetLastTerminationReason(), "invalid_scattering");
            ASSERT_EQ(track.GetTrack().size(), 1u);
            EXPECT_EQ(track.GetTrack().back(), initial);
        }
    }
}

TEST(ScatteringDomainGuard, DisabledByDefault)
{
    multiple_scattering::ScatteringOffset offset;
    offset.tx = 1.2;
    auto propagator = DomainGuardPropagator(offset);
    EXPECT_THROW(propagator.Propagate(DomainGuardInitialState(), 1.),
                 multiple_scattering::ScatteringDomainError);
}

TEST(ScatteringDomainGuard, ValidHistoriesAndRandomSequenceUnchanged)
{
    multiple_scattering::ScatteringOffset offset;
    offset.sx = 0.01;
    offset.sy = 0.02;
    offset.tx = 0.03;
    offset.ty = 0.04;
    auto propagator = DomainGuardPropagator(offset);
    const auto initial = DomainGuardInitialState();
    RandomGenerator::Get().SetSeed(18729);
    const auto ordinary = propagator.Propagate(initial, 1.).GetTrack();
    const auto next_ordinary = RandomGenerator::Get().RandomDouble();
    RandomGenerator::Get().SetSeed(18729);
    const auto guarded = propagator.Propagate(initial, 1., 0., 0, 0, true).GetTrack();
    EXPECT_EQ(ordinary, guarded);
    EXPECT_EQ(next_ordinary, RandomGenerator::Get().RandomDouble());
}

TEST(ScatteringDomainGuard, InvalidStochasticProposalDoesNotCommitEnergyOrDirection)
{
    auto propagator = DomainGuardPropagator({}, true, true);
    const auto track = propagator.Propagate(DomainGuardInitialState(), 1e20, 900., 0, 0, true);
    ASSERT_EQ(propagator.GetLastTerminationReason(), "invalid_scattering");
    ASSERT_EQ(track.GetTrack().size(), 2u);
    const auto terminal = track.GetTrack().back();
    EXPECT_DOUBLE_EQ(terminal.energy, 990.);
    EXPECT_EQ(terminal.direction, DomainGuardInitialState().direction);
    EXPECT_GT(terminal.propagated_distance, 0.);
    EXPECT_EQ(track.GetTrackTypes().back(), InteractionType::ContinuousEnergyLoss);
}

TEST(ScatteringDomainGuard, ValidStochasticHistoryAndRandomSequenceUnchanged)
{
    auto propagator = DomainGuardPropagator({}, true, false);
    const auto initial = DomainGuardInitialState();
    RandomGenerator::Get().SetSeed(18729);
    const auto ordinary = propagator.Propagate(initial, 1e20, 900.).GetTrack();
    const auto next_ordinary = RandomGenerator::Get().RandomDouble();
    RandomGenerator::Get().SetSeed(18729);
    const auto guarded = propagator.Propagate(initial, 1e20, 900., 0, 0, true).GetTrack();
    EXPECT_EQ(ordinary, guarded);
    EXPECT_EQ(next_ordinary, RandomGenerator::Get().RandomDouble());
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
