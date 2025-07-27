import astropy.units as u
import numpy as np
import astropy.constants as c
from scipy.interpolate import interp1d
import pandas as pd

from .particles import particles
from .transport import transport
from .accelerator import accelerator
part = particles()
tran = transport()
acel = accelerator()

@u.quantity_input(R_esc=u.pc, N_0=u.GeV, Ep=u.GeV, d=u.pc, a=u.yr, dens=u.cm**-3)
def compute_pflux_impulsive_extended(Resc,
        N_0, Ep, d, a, dens, chi=0.05, ism=1, alpha=2.0) -> u.GeV**-1 * u.cm**-3:
    """
    Compute differential proton flux density (:math:`\mathrm{GeV}^{-1}\,\mathrm{cm}^{-3}`) for an 
    impulsive injection from the surface of an extended source.
    The difference to the case of a point source is the normalisation.

    Parameters
    ----------
    Resc : :class:`~astropy.units.Quantity`
        Particle escape radius (=distance from source center). (pc)
    N_0 : :class:`~astropy.units.Quantity`
        Normalisation of particle distribution function normalising the
        energy injection rate (L_CR, which could be E_totCR/age). (GeV/s)
    Ep : :class:`~astropy.units.Quantity`
        Particle energy. (GeV)
    d : :class:`~astropy.units.Quantity`
        Distance to particle injection location. (pc)
    a : :class:`~astropy.units.Quantity`
        Time since particle injection at source (can differ from its age). (yr)
    dens : :class:`~astropy.units.Quantity`
        Density of traversed ambient matter. (:math:`\mathrm{cm}^{-3}`)
        Default value for ISM is 1 :math:`\mathrm{cm}^{-3}`.
    chi : float
        Scalar suppression factor for diffusion coefficient, relating to
        the level of turbulence along the particles path.
        (ism: 1, clouds: 0.05)
    alpha : float
        Proton spectrum spectral index. (default: 2.0)
    """
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

@u.quantity_input(N_0=u.GeV, Ep=u.GeV, d=u.pc, a=u.yr, dens=u.cm**-3)
def compute_pflux_impulsive_point(N_0, Ep, d, a, dens, chi=0.05,
        ism=1, alpha=2.0) -> u.GeV**-1 * u.cm**-3:
    """
    Compute differential proton flux density (:math:`\mathrm{GeV}^{-1}\,\mathrm{cm}^{-3}`) for an 
    impulsive injection from a point source as given by 
    `Aharonian & Atoyan (1996), A&A 309, 917 <https://ui.adsabs.harvard.edu/abs/1996A%26A...309..917A/abstract>`_ (eq. 3).
    

    Parameters
    ----------
    N_0 : :class:`~astropy.units.Quantity`
        Normalisation of particle distribution function normalising the
        energy injection rate (L_CR, which could be E_totCR/age). (GeV/s)
    Ep : :class:`~astropy.units.Quantity`
        Particle energy. (GeV)
    d : :class:`~astropy.units.Quantity`
        Distance to particle injection location. (pc)
    a : :class:`~astropy.units.Quantity`
        Time since particle injection at source (can differ from its age). (yr)
    dens : :class:`~astropy.units.Quantity`
        Density of traversed ambient matter. (:math:`\mathrm{cm}^{-3}`)
        Default value for ISM is 1 :math:`\mathrm{cm}^{-3}`.
    chi : float
        Scalar suppression factor for diffusion coefficient, relating to
        the level of turbulence along the particles path.
        (ism: 1, clouds: 0.05)
    alpha : float
        Proton spectrum spectral index. (default: 2.0)
    """

    Rdiff = tran.R_diffusion(Ep, a, dens, chi=chi, ism=ism).to(u.cm)

    exp_frac = -(((alpha - 1) * a.to(u.s)) / (part.t_ppEK(dens, Ep))) \
               - ((d.to(u.cm)) / Rdiff) ** 2

    coeff = (N_0 * (Ep ** (-alpha))) / ((np.pi ** (3 / 2)) * (Rdiff ** 3))
    pflux = coeff * np.exp(exp_frac)

    # Return
    return pflux.to(u.GeV**-1 *u.cm**-3)

# Equals & replaces former flux.flux_calc():
#     def flux_calc(self, N_0, Ep, d, a, dens, chi=0.05, ism=0):

@u.quantity_input(Resc=u.pc, N_0=u.GeV, Ep=u.GeV, d=u.pc, a=u.yr, dens=u.cm**-3)
def compute_pflux_continuous_extended(Resc,
        N_0, Ep, d, a, dens, chi=0.05, ism=1, alpha=2.0) -> u.GeV**-1 * u.cm**-3:
    """
    Compute differential proton flux density (:math:`\mathrm{GeV}^{-1}\,\mathrm{cm}^{-3}`) for an 
    continuous injection from the surface of an extended source.
    The difference to the case of a point source is the normalisation.

    Parameters
    ----------
    Resc : :class:`astropy.Quantity`
        Particle escape radius (=distance from source center). (pc)
    N_0 : :class:`~astropy.units.Quantity`
        Normalisation of particle distribution function normalising the
        energy injection rate (L_CR, which could be E_totCR/age). (GeV/s)
    Ep : :class:`~astropy.units.Quantity`
        Particle energy. (GeV)
    d : :class:`~astropy.units.Quantity`
        Distance to particle injection location. (pc)
    a : :class:`~astropy.units.Quantity`
        Time since particle injection at source (can differ from its age). (yr)
    dens : :class:`~astropy.units.Quantity`
        Density of traversed ambient matter. (:math:`\mathrm{cm}^{-3}`)
        Default value for ISM is 1 :math:`\mathrm{cm}^{-3}`.
    chi : float
        Scalar suppression factor for diffusion coefficient, relating to
        the level of turbulence along the particles path.
        (ism: 1, clouds: 0.05)
    alpha : float
        Proton spectrum spectral index. (default: 2.0)
    """
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

# Equals & replaces former flux.flux_calc_cont():
#     def flux_calc_cont(self, N_0, Ep, d, a, dens, chi=0.05, ism=0):

@u.quantity_input(N_0=u.GeV, Ep=u.GeV, d=u.pc, a=u.yr, dens=u.cm**-3)
def compute_pflux_continuous_point(N_0, Ep, d, a, dens, chi=0.05,
        ism=1, alpha=2.0) -> u.GeV**-1 * u.cm**-3:
    """
    Compute differential proton flux density (:math:`\mathrm{GeV}^{-1}\,\mathrm{cm}^{-3}`) for a continuous injection 
    from a point source as given by `Aharonian & Atoyan (1996), A&A 309, 917 <https://ui.adsabs.harvard.edu/abs/1996A%26A...309..917A/abstract>`_ (eq. 8).
    
    Parameters
    ----------
    N_0 : :class:`~astropy.units.Quantity`
        Normalisation of particle distribution function normalising the
        energy injection rate (L_CR, which could be E_totCR/age). (GeV/s)
    Ep : :class:`~astropy.units.Quantity`
        Particle energy. (GeV)
    d : :class:`~astropy.units.Quantity`
        Distance to particle injection location. (pc)
    a : :class:`~astropy.units.Quantity`
        Time since particle injection at source (can differ from its age). (yr)
    dens : :class:`~astropy.units.Quantity`
        Density of traversed ambient matter. (:math:`\mathrm{cm}^{-3}`)
        Default value for ISM is 1 :math:`\mathrm{cm}^{-3}`.
    chi : float
        Scalar suppression factor for diffusion coefficient, relating to
        the level of turbulence along the particles path.
        (ism: 1, clouds: 0.05)
    alpha : float
        Proton spectrum spectral index. (default: 2.0)
    """
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

# Define functions for Galactic CRs

@u.quantity_input(e=u.GeV)
def e2R(e, Z=1) -> u.GV:
    """
    Compute proton rigidity (``GV``) from particle energy.

    Parameters
    ==========
    e : :class:`~astropy.units.Quantity`
        Particle energy. (``GeV``)
    Z : float
        Charge number, 1 for a proton.
    """
    R = e / (Z * c.e.si)
    return R

@u.quantity_input(E=u.GeV)
def compute_fgal(E) -> u.GeV**(-1) * u.cm**(-3):
    """
    Returns differential proton density of galactic CRs (:math:`\mathrm{GeV}^{-1}\,\mathrm{cm}^{-3}`) 
    measured by `AMS Collab. (2015), PRL 114 171103 <https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.114.171103>`_.

    Parameters
    ----------
    E : :class:`~astropy.units.Quantity` or array-like
        Energy of Galactic cosmic rays. (GeV)
    """
    R = e2R(E)

    term = 1 + (R/(336*u.GV))**5.542
    phi = 0.4544 * (R/(45*u.GV))**-2.849 * term**0.024 * u.Unit("m-2 sr-1 s-1 GV-1")

    n = 4 * np.pi * u.sr / c.c * phi

    # rigidity -> energy (for a proton here)
    n = n / c.e.si

    # Return
    return n

def compute_fgal_dampe(E, E_tran=6.3*u.TeV):
    """
    Returns differential proton density of galactic CRs (:math:`\mathrm{GeV}^{-1}\,\mathrm{cm}^{-3}`) measured by 
    `DAMPE Collab (2019), SciAdv aax3793 <https://www.science.org/doi/10.1126/sciadv.aax3793>`_.
    
    Parameters
    ==========
    E : :class:`~astropy.units.Quantity` or array-like
        Energy values of Galactic cosmic rays (GeV).
    E_tran : :class:`~astropy.units.Quantity`
        Energy at which the low-energy and high-energy Smooth Broken Power-Law (SBPL) fits transition. (TeV)
    """
    
    def DAMPE_SBPL_low(E):
        """Smooth Broken Power Law fit for 100 GeV - 6.3 TeV. Returns differential CR density (GeV^-1 cm^-3).
        
        Parameter:
        ----------
        E: :class:`~astropy.units.Quantity`
            Energy values of Galactic cosmic rays (GeV)
        
        """
    
        E_0 = 1 * u.TeV
        E_b = 0.48*u.TeV
        flux = 7.58e-5 * (E/E_0)**(-2.772) * (1 + (E/E_b)**5.0)**(0.173/5.0) / (u.m**2 * u.sr * u.s * u.GeV)
        density = (flux *4*np.pi*u.sr/c.c).to('GeV^-1*cm^-3') 
        return density

    def DAMPE_SBPL_high(E):
        """Smooth Broken Power Law fit for 1 TeV - 100 TeV. Returns differential CR density (GeV^-1 cm^-3)
        
        Parameter:
        ----------
        E: :class:`~astropy.units.Quantity`
            Energy values of Galactic cosmic rays (GeV)
        """

        E_0 = 1.0 * u.TeV
        E_b = 13.6*u.TeV
        flux = 8.68e-5 * (E/E_0)**(-2.6) * (1 + (E/E_b)**5.0)**(-0.25/5.0) / (u.m**2 * u.sr * u.s * u.GeV)
        density = (flux *4*np.pi*u.sr/c.c).to('GeV^-1*cm^-3')
        return density
    
    
    E = E.to('GeV') # Confirm units of E

    result = np.zeros(E.shape) * (u.GeV**-1 * u.cm**-3)

    # Low-energy SBPL fit (100 GeV - 6.3 TeV)
    low_mask = E <= E_tran
    if np.any(low_mask):
        result[low_mask] = DAMPE_SBPL_low(E[low_mask])

    #High-energy SBPL fit (1-100 TeV)
    high_mask = E > E_tran
    if np.any(high_mask):
        result[high_mask] = DAMPE_SBPL_high(E[high_mask])

    return result