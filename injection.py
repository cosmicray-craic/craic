from .particles import particles
from .transport import transport
from .accelerator import accelerator
part = particles()
tran = transport()
acel = accelerator()

def compute_pflux_impulsive_extended(Resc,
        N_0, Ep, d, a, dens, chi=0.05, ism=1, alpha=2.0):
    """
    Compute differential proton flux density for an impulsive injection from
    the surface of an extended source.
    The difference to the case of a point source is the normalisation.

    Parameters
    ----------
    Resc : `astropy.Quantity`
        Particle escape radius (=distance from source center). (~ pc)
    For the other parameters see compute_pflux_impulsive_point() method.

    Returns
    -------
    pflux : `astropy.Quantity`
        Differential flux density in ``GeV-1 cm-3``.
    """
    import numpy as np
    import astropy.units as u
    sqrt_pi = np.sqrt(np.pi)
    assert (np.size(Resc) == 1) or (np.shape(Resc) == np.shape(Ep))

    Rdiff = tran.R_diffusion(Ep, a, dens, chi=chi, ism=ism)

    # Normalisation factor (eq. 9 in Mitchell et al. 2020)
    term = sqrt_pi * Rdiff ** 2 + 2 * sqrt_pi * Resc ** 2
    f0 = (sqrt_pi * Rdiff ** 3) / (term * Rdiff + 4 * Resc * Rdiff ** 2)

    pflux = f0.decompose() * \
            compute_pflux_impulsive_point(N_0, Ep, d, a, dens=dens,
                chi=chi, ism=ism, alpha=alpha)

    # Return
    return pflux.to(u.GeV**-1 *u.cm**-3)


def compute_pflux_impulsive_point(N_0, Ep, d, a, dens, chi=0.05,
        ism=1, alpha=2.0):
    """
    Compute differential proton flux density for an impulsive injection from
    a point source as given by Aharonian & Atoyan (1996) (eq. 3).
    Equals & replaces former flux.flux_calc():
        def flux_calc(self, N_0, Ep, d, a, dens, chi=0.05, ism=0):

    Parameters
    ----------
    N_0 : `astropy.Quantity`
        Normalisation of particle distribution function normalising the
        total injected energy since the accelerator birth. (~ GeV)
    Ep : `astropy.Quantity`
        Particle energy. (~ GeV)
    d : `astropy.Quantity`
        Distance to particle injection location. (~ pc)
    a : `astropy.Quantity`
        Time since particle injection at source (can differ from its age). (~ yr)
    dens : `astropy.Quantity`
        Density of traversed ambient matter. (~ cm-3)
        (ism: 1 cm-3)
    chi : `float`
        Scalar suppression factor for diffusion coefficient, relating to
        the level of turbulence along the particles path.
        (ism: 1, clouds: 0.05)
    alpha : `float`
        Proton spectrum spectral index. (default: 2.0)

    Returns
    -------
    pflux : `astropy.Quantity`
        Differential flux density in ``GeV-1 cm-3``.
    """
    import numpy as np
    import astropy.units as u

    Rdiff = tran.R_diffusion(Ep, a, dens, chi=chi, ism=ism).to(u.cm)

    exp_frac = -(((alpha - 1) * a.to(u.s)) / (part.t_ppEK(dens, Ep))) \
               - ((d.to(u.cm)) / Rdiff) ** 2

    coeff = (N_0 * (Ep ** (-alpha))) / ((np.pi ** (3 / 2)) * (Rdiff ** 3))
    pflux = coeff * np.exp(exp_frac)

    # Return
    return pflux.to(u.GeV**-1 *u.cm**-3)


def compute_pflux_continuous_extended(Resc,
        N_0, Ep, d, a, dens, chi=0.05, ism=1, alpha=2.0):
    """
    Compute differential proton flux density for an continuous injection from
    the surface of an extended source.
    The difference to the case of a point source is the normalisation.

    Parameters
    ----------
    Resc : `astropy.Quantity`
        Particle escape radius (=distance from source center). (~ pc)
    For the other parameters see compute_pflux_continuous_point() method.

    Returns
    -------
    pflux : `astropy.Quantity`
        Differential flux density in ``GeV-1 cm-3``.
    """
    import numpy as np
    import astropy.units as u
    from scipy.special import erf, erfc

    raise NotImplementedError("f0 for cont. ext. case still to be revised!")
    ####### REVISION REQUIRED!! THIS NORMALISATION IS WRONG!!!
    #f0 = 2. * tran.Diffusion_Coefficient(Ep, dens, ism=1) * Ep**alpha
    #rratio = Resc / d
    #f0 = 4.*np.pi
    f0 = 1.
    #f0 /= erfc(rratio) * Resc**2. + erf(rratio) * 0.5 * d**(-2.) - np.exp(-rratio**2.) * Resc * d**(-1.) * np.pi**-0.5
    
    #pflux = f0.decompose() * \
    pflux = f0 *  compute_pflux_continuous_point(N_0, Ep, d, a, dens=dens,
                chi=chi, ism=ism, alpha=alpha)

    # Return
    return pflux


def compute_pflux_continuous_point(N_0, Ep, d, a, dens, chi=0.05,
        ism=1, alpha=2.0):
    """
    Compute differential proton flux density for a continuous injection from
    a point source as given by Aharonian & Atoyan (1996) (eq. 8).
    Equals & replaces former flux.flux_calc_cont():
        def flux_calc_cont(self, N_0, Ep, d, a, dens, chi=0.05, ism=0):

    Parameters
    ----------
    N_0 : `astropy.Quantity`
        Normalisation of particle distribution function normalising the
        energy injection rate (L_CR, which could be E_totCR/age). (~ GeV s-1)
    Ep : `astropy.Quantity`
        Particle energy. (~ GeV)
    d : `astropy.Quantity`
        Distance to particle injection location. (~ pc)
    a : `astropy.Quantity`
        Time since particle injection at source (can differ from its age). (~ yr)
    dens : `astropy.Quantity`
        Density of traversed ambient matter. (~ cm-3)
        (ism: 1 cm-3)
    chi : `float`
        Scalar suppression factor for diffusion coefficient, relating to
        the level of turbulence along the particles path.
        (ism: 1, clouds: 0.05)
    alpha : `float`
        Proton spectrum spectral index. (default: 2.0)

    Returns
    -------
    pflux : `astropy.Quantity`
        Differential flux density in ``GeV-1 cm-3``.
    """
    import numpy as np
    import astropy.units as u
    from scipy.special import erfc

    # Q_0 is injection rate, two possible ways:
    # (A) (normalisation computed for tot E_CR) / (time during which injection took place)
    Q_0 = N_0 / a.to(u.s)
    # (B) (normalisation computed for CR lum.)
    # Q_0 = N_0

    # Compute proton flux density according to
    # eq. 8 in Aharonian & Atoyan (1996) and/or eq. 21 in Atoyan (1995)
    rfrac = d.to(u.cm) / tran.R_diffusion(Ep, a, dens, chi=chi, ism=ism).to(u.cm)
    coeff = Q_0 * (Ep**-alpha) / (4.*np.pi * d.to(u.cm) \
            * tran.Diffusion_Coefficient(Ep, dens, chi=chi, ism=ism))
    pflux = coeff * erfc(rfrac)

    # Return
    return pflux.to(u.GeV**-1 *u.cm**-3)
