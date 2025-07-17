import numpy as np
import astropy.units as u
import astropy.constants as c

class particles:

    # from Kafexhiu eq 1
    Epth = 0.2797 * u.GeV # GeV
    kappa = 0.45
    cspeed = c.c.to(u.cm / u.s)

    def __init__(self,alpha=2.,Ebudget=1e50*u.erg):
        self.alpha = alpha
        self.Ebudget = Ebudget


    def sig_ppEK(self, Ep):
        """
        Compute the total inelastic cross section of p-p collisions as given by
        Kafexhiu et al. (2014) (eq. 1).

        Parameters
        ----------
        Ep : `astropy.Quantity`
            Proton energy.

        Returns
        -------
        sig_pp : `astropy.Quantity`
            Inelastic cross section. (cm2)
        """
        frac = Ep / self.Epth
        log_frac = np.log(frac)

        t1 = 0.96 * log_frac
        t2 = 0.18 * log_frac**2
        t3 = 1 - frac**-1.9
        sig_pp = (30.7 - t1 + t2) * (t3) ** 3 * u.mbarn

        sig_pp = np.clip(sig_pp, a_min=0, a_max=None)

        return sig_pp.to(u.cm**2)  # cm^2

    def t_ppEK(self, dens, Ep):
        """
        Compute the cooling time of protons.

        Parameters
        ----------
        dens : `astropy.Quantity`
            Matter density. (~ cm-3)
        Ep : `astropy.Quantity`
            Proton energy. (~ GeV)

        Returns
        -------
        t_ppEK : `astropy.Quantity`
            Cooling time. (s)
        """
        t_ppEK = (dens * self.sig_ppEK(Ep)* self.cspeed * self.kappa) ** -1

        return t_ppEK


    def NormEbudget(self,Emin=10*u.GeV,Emax=3.*u.PeV):#1 PeV
        """
        Compute the normalisation of the proton flux. Normalises the integral
        of the energy flux to 1.

        Parameters
        ----------
        Emin : `astropy.Quantity`
            Minimum proton energy. (~ GeV)
        Emax : `astropy.Quantity`
            Maximum proton energy. (~ GeV)

        Returns
        -------
        N0 : `astropy.Quantity`
            Proton flux normalisation. (= GeV^(alpha-1))
        """

        if self.alpha == 2:
            N0 = (self.Ebudget / (np.log(Emax / u.GeV) - np.log(Emin / u.GeV)))
        else:
            N0 = (self.Ebudget.to(u.GeV) * (2 - self.alpha) /
                  (Emax.to(u.GeV) ** (2 - self.alpha) - Emin.to(u.GeV) ** (2 - self.alpha)))

        return N0.to(u.GeV**(self.alpha-1))
