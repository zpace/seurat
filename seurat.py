import numpy as np

# for sampling
import emcee
import george
from george import kernels
import triangle

import misc_tools

def emline_model(params, inst, L, loc):
    '''
    model for a singlet, doublet, triplet, or really anything
        of a given species

    Arguments:
     - L: lambda array (length N)
     - inst: instrumental broadening (in L)
     - loc: wavelength location of each line's supposed center
     - params: {
         - a: amplitude of emission lines (length m)
         - v: velocity offset of line center
         - sigma: global wavelength spread (due to velocity broadening)
         }
    '''

    a, v, sigma = params

    c = 299792.458
    dloc = L / c * v
    sigma_L = L / c * sigma

    gs =  a * np.exp(
        -0.5 * (L[:,np.newaxis] - (loc + dloc))**2. / (sigma_L**2. + inst**2.))
    return gs.sum(axis = 1)

def emlines_model(params, inst, L, loc):
    '''
    '''
    emlines = np.array([emline_model(p, inst, L, loc) for p in params])
    if len(emlines) != 1:
        emlines = emlines.sum(axis=1)

    return emlines


'''
element of emlines:
    [L, R, dR]
    - L: wavelengths of each component of the species
    - R: line strength ratio of each component to the primary
        (if just 1 component, just use [1.])
    - dR: the factor by which R can change for each line component
        (in each direction)

'''

emlines = {
    'Balmer': [np.array([6564.60, 4682.71]),
               np.array([1., 0.25]),
               np.array([[.1, 3.]]),
               np.array([r'$H\alpha$', r'$H\beta$'])
               ],

    '[OIId]': [np.array([3727.09, 3729.88]),
               np.array([1., 0.35]),
               np.array([[0.5, 5.]]),
               np.array([r'[OII]3726', r'[OII]3729'])],

    '[OIIId]': [np.array([4364.435, 4960.295, 5008.240]),
                np.array([]),
                np.array([[]]),
                np.array([r'[OIII]4363', r'[OIII]4959', r'[OIII]5007'])],

    '[NIId]': [np.array([5756.19]),
               np.array([1.]),
               np.array([[]])],

    '[SIId]': [np.array([6718.294, 6732.673]),
               np.array([]),
               np.array([])],

    '[NeII]': [np.array([]),
               np.array([]),
               np.array([])],

    '[SIII]': [np.array([]),
               np.array([]),
               np.array([])]
}


def george_fit_emline(L, pp, stdev, specres, nwalkers = 32):
    '''
    use DFM's gaussian processes module to fit emission lines

    Arguments:
     - L: lambda array (length N)
     - pp: ppxf return object (the version that's written out to file)
            or just the version that's outputted, if you must
     - stdev: function that takes L and turns it into a dispersion in L
        (this is figuring out the instrumental response function at L)

    Approach:

    `params` elements:
     - 0: ln(a) [used in GP]
     - 1: ln(tau) [used in GP]
     - 2 to 2 + `nspecies`: wavelength locations of lines
     - 3 + `nspecies` to 3 + 2*`nspecies`: relative strengths of lines

    '''

    c = 299792.458

    if type(pp) == dict:
        data = (pp['lam'], pp['galaxy'] - pp['bestfit'], pp['noise'])
        sol = pp['sol']
    else:
        data = (pp.lam, pp.galaxy - pp.bestfit, pp.noise)
        sol = pp.sol

    species = ['Balmer']

    # where we think the lines are
    nspecies = len(species)
    lines_per_species = [len(s[0]) for s in species]
    locs0 = [emlines[s][0] for s in species]
    strengths0 = [emlines[s][1] for s in species]
    vs0 = [pp.sol[0] for s in species]

    initial = locs0 + strengths0

    nonprimary = np.concatenate([0,], np.cumsum(lines_per_species))
    nonprimary_b = np.array(
        [1 if n in nonprimary else 0 for n in species])
    ratio_lims = [emlines[s][2] for s in species]

    ndim = misc_tools.element_count(locs0) + \
           misc_tools.element_count(strengths0) + 2

    kw = {'inst': c/specres, 'SR': ratio_lims,
          'lines_per_species': lines_per_species, 'nspecies': nspecies}

    sampler = emcee.EnsembleSampler(nwalkers, ndim, emline_lnprob,
                                    args=data, kwargs=kw, threads=2)

    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, 1000)

    return sampler

def emline_lnlike(params, x, y, yerr):
    a, tau = np.exp(p[:2])
    gp = george.GP(a * kernels * Matern32Kernel(tau))
    gp.compute(x, yerr)
    return gp.lnlikelihood(y - emlines_model(p[2:], x))


def emline_lnprior(p, SR, nspecies, lines_per_species):
    lna = p[0]
    lntau = p[1]

    SR = np.concatenate(SR)

    # line ratio limits
    r_nom_l = SR[::2]
    r_nom_u = SR[1::2]

    # line ratios
    r = p[-nspecies:]

    if not -5 < lna < 5:
        return -np.inf
    if not -5 < lntau < 5:
        return -np.inf
    if not np.all(r_nom_l < r < r_nom_u):
        return -np.inf
    if not np.all(p[-2*nspecies]< 1000.):
        return -np.inf



def emline_lnprob(params, x, y, yerr, **kw):
    lp = emline_lprior(params, SR, nspecies, lines_per_species)
    if np.isfinite(lp):
        return lp + emline_lnlike(params, x, y, yerr, kw['inst'])
    else:
        return -np.inf
