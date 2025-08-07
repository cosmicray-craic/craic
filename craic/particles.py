import numpy as np
import astropy.units as u
import astropy.constants as c

class particles:
    """Calculates the p-p cross-section, proton cooling time, 
    and the normalisation for proton spectrum."""

    # from Kafexhiu eq 1
    Epth = 0.2797 * u.GeV # GeV
    kappa = 0.45
    cspeed = c.c.to(u.cm / u.s)

    def __init__(self,alpha=2.,Ebudget=1e50*u.erg):
        self.alpha = alpha
        self.Ebudget = Ebudget

    @u.quantity_input(Ep=u.GeV)
    def sig_ppEK(self, Ep) -> u.cm**2:
        """
        Computes the total inelastic cross section of p-p collisions (:math:`\mathrm{cm}^{2}`) as given by
        `Kafexhiu et al. (2014), PhysRevD 90, 123014,  
        <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.90.123014>`_ (eq. 1).

        Parameters
        ----------
        Ep : :class:`~astropy.units.Quantity` 
            Proton energy. (GeV)
        """
        if np.any(Ep < 0.2797*u.GeV):
            raise ValueError(f"Energy of particles must be larger than the threshold 0.2797 GeV.")
        
        frac = Ep / self.Epth
        log_frac = np.log(frac)

        t1 = 0.96 * log_frac
        t2 = 0.18 * log_frac**2
        t3 = 1 - frac**-1.9
        sig_pp = (30.7 - t1 + t2) * (t3) ** 3 * u.mbarn

        sig_pp = np.clip(sig_pp, a_min=0, a_max=None)

        return sig_pp.to(u.cm**2)  # cm^2

    @u.quantity_input(dens=u.cm**-3, Ep=u.GeV)
    def t_ppEK(self, dens, Ep) -> u.s:
        """
        Computes the cooling time of protons (s).

        Parameters
        ----------
        dens : :class:`~astropy.units.Quantity`
            Number density. (:math:`\mathrm{cm}^{-3}`)
        Ep : :class:`~astropy.units.Quantity`
            Proton energy. (GeV)
        """
        t_ppEK = (dens * self.sig_ppEK(Ep)* self.cspeed * self.kappa) ** -1

        return t_ppEK

    def NormEbudget(self,Emin=10*u.GeV,Emax=3.*u.PeV):#1 PeV
        """
        Computes the normalisation of the proton flux. Normalises the integral
        of the energy flux to 1.

        Parameters
        ----------
        Emin : :class:`~astropy.units.Quantity`
            Minimum proton energy. (GeV)
        Emax : :class:`~astropy.units.Quantity`
            Maximum proton energy. (GeV)
        """

        if self.alpha == 2:
            N0 = (self.Ebudget.to(u.GeV) / (np.log(Emax / u.GeV) - np.log(Emin / u.GeV)))
        else:
            N0 = (self.Ebudget.to(u.GeV) * (2 - self.alpha) /
                  (Emax.to(u.GeV) ** (2 - self.alpha) - Emin.to(u.GeV) ** (2 - self.alpha)))

        return N0.to(u.GeV**(self.alpha-1)) # (GeV^(alpha-1))
