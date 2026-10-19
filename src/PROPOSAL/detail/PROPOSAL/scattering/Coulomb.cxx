#include "PROPOSAL/scattering/Coulomb.h"
#include "PROPOSAL/Constants.h"
#include "PROPOSAL/crosssection/CrossSection.h"
#include "PROPOSAL/crosssection/CrossSectionMultiplier.h"
#include "PROPOSAL/crosssection/parametrization/Ionization.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace PROPOSAL;
namespace {
constexpr double hbarc = 1.973269804e-11; // MeV cm
constexpr double alpha = 1. / 137.035999084;
double square(double x) { return x * x; }
double uniform(const std::function<double()>& rnd)
{
    double u = rnd();
    if (!std::isfinite(u) || u < 0. || u >= 1.)
        throw std::invalid_argument("Coulomb random variate must be in [0,1)");
    return u;
}
}

void CoulombSettings::Validate() const
{
    if (!std::isfinite(separation) || separation <= 0. || separation >= 1.
        || !std::isfinite(max_angular_variance) || max_angular_variance <= 0.
        || max_angular_variance > 0.1 || !std::isfinite(max_fractional_loss)
        || max_fractional_loss <= 0. || max_fractional_loss > 0.1
        || !std::isfinite(max_step) || max_step <= 0.)
        throw std::invalid_argument("Invalid Coulomb transport settings");
}

Coulomb::Coulomb(const ParticleDef& p, const Medium& m, CoulombSettings s)
    : settings(s), particle_(p), medium_(m), components_(m.GetComponents())
{
    settings.Validate();
    const int type = std::abs(p.particle_type);
    if (!(p.mass > 0.) || !std::isfinite(p.mass) || !std::isfinite(p.charge)
        || p.charge == 0.
        || (type != 0 && type != 11 && type != 13 && type < 1000000000))
        throw std::invalid_argument("Coulomb supports e+/-, mu+/- and custom pointlike Dirac electric particles with finite nonzero charge");
    for (const auto& c : components_) {
        // Above this coupling the first Coulomb correction is not a supported
        // approximation. This cap also prevents negative positron kernels.
        if (std::abs(p.charge)*c.GetNucCharge()*alpha > 0.6)
            throw std::domain_error("McKinley-Feshbach applicability requires |charge| Z alpha <= 0.6 for every constituent");
        atoms_per_gram_.push_back(NA * c.GetAtomInMolecule() / m.GetSumNucleons());
    }
}

void Coulomb::ValidateEnergy(double energy) const
{
    if (!std::isfinite(energy) || energy - particle_.mass < 0.2*(1.-1e-10))
        throw std::domain_error("Coulomb model requires kinetic energy >= 0.2 MeV");
}

Coulomb::Kernel Coulomb::MakeKernel(double energy, size_t i) const
{
    ValidateEnergy(energy);
    const auto& c = components_.at(i);
    const double m = particle_.mass;
    const double target = c.GetAtomicNum() * c.GetAverageNucleonWeight();
    const double p = std::sqrt((energy - m) * (energy + m));
    const double pcm = p * target / std::sqrt(m*m + target*target + 2*target*energy);
    const double beta = p / energy;
    const double a = 0.88534 * 5.29177210903e-9 / std::cbrt(c.GetNucCharge());
    const double eta = alpha * particle_.charge * c.GetNucCharge() / beta;
    return {target, pcm, beta, square(hbarc / (2*pcm*a)) * (1.13 + 3.76*eta*eta),
        PI * square(particle_.charge*c.GetNucCharge()*alpha*hbarc/(pcm*beta)),
        1.27e-13 * std::pow(c.GetAtomicNum(), 0.27)};
}

double Coulomb::Correction(const Kernel& k, double x, double z) const
{
    const double recoil = 2*k.pcm*k.pcm*x/k.target_mass;
    const double q2 = recoil*(recoil + 2*k.target_mass);
    double spin = 1. - k.beta*k.beta*x;
    // McKinley-Feshbach is a first Coulomb correction, not a full Mott table.
    spin -= particle_.charge*z*alpha*PI*k.beta*std::sqrt(x)*(1.-std::sqrt(x));
    if (spin < 0.)
        throw std::domain_error("McKinley-Feshbach approximation became negative");
    return spin / std::pow(1. + q2*square(k.radius/hbarc)/12., 4);
}

double Coulomb::Differential(double energy, size_t i, double x) const
{
    if (!std::isfinite(x) || x < 0. || x > 1.)
        throw std::invalid_argument("Coulomb x must be in [0,1]");
    const auto k = MakeKernel(energy, i);
    return k.prefactor*Correction(k, x, components_[i].GetNucCharge()) / square(k.screening+x);
}

Cartesian3D Coulomb::Rotate(const Cartesian3D& direction, double theta, double phi)
{
    auto n = direction;
    if (!std::isfinite(n.magnitude()) || n.magnitude() == 0.
        || !std::isfinite(theta) || !std::isfinite(phi))
        throw std::invalid_argument("Invalid Coulomb direction or angle");
    n.normalize();
    Cartesian3D axis = std::abs(n.GetZ()) < 0.9 ? Cartesian3D(0,0,1) : Cartesian3D(1,0,0);
    auto u = vector_product(axis, n);
    u.normalize();
    const auto v = vector_product(n, u);
    auto result = std::cos(theta)*n + std::sin(theta)*(std::cos(phi)*u + std::sin(phi)*v);
    result.normalize();
    return result;
}

CoulombCollision Coulomb::Scatter(const ParticleState& state, size_t i,
    double x, double phi) const
{
    if (!std::isfinite(x) || x < 0. || x > 1.)
        throw std::invalid_argument("Coulomb x must be in [0,1]");
    const auto k = MakeKernel(state.energy, i);
    const double p = std::sqrt((state.energy-particle_.mass)*(state.energy+particle_.mass));
    const double invariant = std::sqrt(square(particle_.mass)+square(k.target_mass)+2*k.target_mass*state.energy);
    const double gamma = (state.energy+k.target_mass)/invariant;
    const double boost = p/(state.energy+k.target_mass);
    const double ecm = std::hypot(k.pcm, particle_.mass);
    const double parallel = gamma*(k.pcm*(1.-2*x)+boost*ecm);
    const double transverse = 2*k.pcm*std::sqrt(x*(1.-x));
    CoulombCollision collision;
    collision.before = state;
    collision.after = state;
    collision.component = i;
    collision.x = x;
    collision.recoil = 2*k.pcm*k.pcm*x/k.target_mass;
    collision.after.energy -= collision.recoil;
    collision.after.direction = Rotate(state.direction, std::atan2(transverse, parallel), phi);
    return collision;
}

CoulombMoments Coulomb::Integrate(double energy, size_t i, double lo, double hi) const
{
    CoulombMoments sum;
    if (hi == lo) return sum;
    const auto k = MakeKernel(energy, i);
    const double start = std::log1p(lo/k.screening);
    const double width = (std::log1p(hi/k.screening)-start)/256;
    const double p = std::sqrt((energy-particle_.mass)*(energy+particle_.mass));
    const double invariant = std::sqrt(square(particle_.mass)+square(k.target_mass)+2*k.target_mass*energy);
    const double gamma = (energy+k.target_mass)/invariant;
    const double boost = p/(energy+k.target_mass);
    for (int j=0; j<=256; ++j) {
        const double x = std::min(hi, k.screening*std::expm1(start+j*width));
        const double w = (j==0 || j==256 ? 1. : (j%2 ? 4. : 2.))*width/3;
        const double rate = w*k.prefactor*Correction(k,x,components_[i].GetNucCharge())/(k.screening+x);
        const double recoil = 2*k.pcm*k.pcm*x/k.target_mass;
        const double theta = std::atan2(2*k.pcm*std::sqrt(std::max(0.,x*(1-x))),
            gamma*(k.pcm*(1-2*x)+boost*std::hypot(k.pcm,particle_.mass)));
        sum.rate += rate;
        sum.angular += rate*theta*theta;
        sum.stopping += rate*recoil;
        sum.straggling += rate*recoil*recoil;
    }
    return sum;
}

CoulombMoments Coulomb::Moments(double energy, double lo, double hi) const
{
    if (!settings.interpolate) return ExactMoments(energy,lo,hi);
    ValidateEnergy(energy);
    // v1 tables: log kinetic energy, spacing 0.02; bounds are part of the key.
    // No legacy Moliere angular table is read by this transport path.
    const double index = std::max(0.,std::floor(std::log((energy-particle_.mass)/0.2)/0.02));
    const double low = particle_.mass+0.2*std::exp(0.02*index);
    const double high = particle_.mass+0.2*std::exp(0.02*(index+1.));
    auto lookup = [&](double e) {
        const auto key = std::make_tuple(e,lo,hi);
        auto it = moment_table_.find(key);
        if (it == moment_table_.end())
            it = moment_table_.emplace(key,ExactMoments(e,lo,hi)).first;
        return it->second;
    };
    const auto a = lookup(low), b = lookup(high);
    const double t = (energy-low)/(high-low);
    return {a.rate+t*(b.rate-a.rate),a.angular+t*(b.angular-a.angular),
        a.stopping+t*(b.stopping-a.stopping),a.straggling+t*(b.straggling-a.straggling)};
}

CoulombMoments Coulomb::ExactMoments(double energy, double lo, double hi) const
{
    ValidateEnergy(energy);
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo < 0. || hi > 1. || hi < lo)
        throw std::invalid_argument("Invalid Coulomb integration bounds");
    CoulombMoments sum;
    for (size_t i=0; i<components_.size(); ++i) {
        const auto m = Integrate(energy,i,lo,hi);
        sum.rate += atoms_per_gram_[i]*m.rate;
        sum.angular += atoms_per_gram_[i]*m.angular;
        sum.stopping += atoms_per_gram_[i]*m.stopping;
        sum.straggling += atoms_per_gram_[i]*m.straggling;
    }
    return sum;
}

double Coulomb::ElectronAngular(double energy, CrossSectionBase& cross) const
{
    ValidateEnergy(energy);
    const auto cuts = cross.GetEnergyCutSettings();
    if (!cuts) throw std::invalid_argument("Ionization partition is missing");
    std::unique_ptr<crosssection::Ionization> ion;
    const auto name = cross.GetParametrizationName();
    if (name == "BergerSeltzerMoller") ion.reset(new crosssection::IonizBergerSeltzerMoller(*cuts));
    else if (name == "BergerSeltzerBhabha") ion.reset(new crosssection::IonizBergerSeltzerBhabha(*cuts));
    else if (name == "BetheBlochRossi") ion.reset(new crosssection::IonizBetheBlochRossi(*cuts));
    else throw std::invalid_argument("Unsupported Coulomb ionization parametrization: "+name);
    const double p2 = (energy-particle_.mass)*(energy+particle_.mass);
    const double coefficient = 2*ME/p2;
    double result = coefficient*cross.CalculatedEdx(energy);
    const auto limits = ion->GetKinematicLimits(particle_,medium_,energy);
    const double lo = std::max(limits.v_min,1e-6*medium_.GetI()/energy);
    const double hi = std::min(cuts->GetCut(energy),limits.v_max);
    if (hi <= lo) return result;
    const double h = std::log(hi/lo)/64.;
    double correction = 0.;
    // Rossi's radiative correction has an integrable logarithmic singularity
    // at its upper kinematic endpoint. Never evaluate that endpoint itself.
    const bool endpoint = hi == limits.v_max;
    for (int j=0; j<(endpoint ? 64 : 65); ++j) {
        const double v = lo*std::exp(h*(j+(endpoint ? 0.5 : 0.))), transfer = energy*v;
        const double pf2 = (energy-transfer-particle_.mass)*(energy-transfer+particle_.mass);
        if (pf2 <= 0.) continue;
        const double cosine = (p2-(energy+ME)*transfer)/std::sqrt(p2*pf2);
        const double theta = std::acos(std::max(-1.,std::min(1.,cosine)));
        const double weight = endpoint ? h : (j==0 || j==64 ? 1. : (j%2 ? 4. : 2.))*h/3.;
        correction += weight*v*ion->DifferentialCrossSection(particle_,medium_,energy,v)
            *(theta*theta-coefficient*transfer);
    }
    // The unresolved atomic-excitation part retains the small-transfer relation.
    // Free-electron transfers between I and the actual cut use exact kinematics.
    const auto scaled = dynamic_cast<CrossSectionMultiplier*>(&cross);
    return std::max(0.,result+correction*(scaled ? scaled->GetMultiplier() : 1.));
}

CoulombCollision Coulomb::Sample(const ParticleState& state, double lo,
    const std::function<double()>& rnd) const
{
    if (!std::isfinite(lo) || lo < 0. || lo >= 1.)
        throw std::invalid_argument("Invalid Coulomb sampling bound");
    std::vector<double> rates;
    double total = 0.;
    for (size_t i=0; i<components_.size(); ++i) {
        total += atoms_per_gram_[i]*Integrate(state.energy,i,lo,1.).rate;
        rates.push_back(total);
    }
    const double choice = uniform(rnd)*total;
    size_t i = std::lower_bound(rates.begin(),rates.end(),choice)-rates.begin();
    i = std::min(i,components_.size()-1);
    const auto k = MakeKernel(state.energy,i);
    const double bound = 1.+std::abs(particle_.charge)*components_[i].GetNucCharge()*alpha*PI*k.beta/4.;
    for (size_t trial=0; trial<1000000; ++trial) {
        const double u = uniform(rnd);
        const double x = (k.screening+lo)*(k.screening+1.)/(k.screening+1.-u*(1.-lo))-k.screening;
        if (uniform(rnd)*bound <= Correction(k,x,components_[i].GetNucCharge()))
            return Scatter(state,i,std::max(lo,std::min(1.,x)),2*PI*uniform(rnd));
    }
    throw std::runtime_error("Coulomb rejection sampler exhausted");
}
