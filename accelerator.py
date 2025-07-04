import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.integrate import quad


class accelerator:

    Mej = (10.*u.Msun).to(u.gram)
    mp = c.m_p.to(u.gram)
    nh2ism = np.array([1.])*u.cm**(-3) # Density of Ambient Gas (cm^-3) ISM
    E51 = 1.  # Ejected Energy (normalised to E^{51} ergs)
    brkind = 3. #pulsar braking index
    eta = 1. #efficiency of conversion of energy into particles
    kappa = 100. #c.m_p/c.m_e #pair production multiplicity 

    def __init__(self, En=1e51*u.erg):
        # Sedov time is a fixed value (~1.6kyr), here in s
        self.t_sed = 0.495*(En/u.erg)**-0.5 * (self.Mej/u.gram)**(5/6)* ((self.mp/u.gram)*(self.nh2ism/(u.cm**(-3))))**(-1/3) *u.s

    def escape_time(self, Ep, typeIIb=True):             # input: GeV
        # To find how long it takes for cosmic rays to escape
        t_sed = self.t_sed.to(u.yr)

        # Hard-coded max momentum 1PeV, and shock index 2.5
        Pp = 1. * np.sqrt((Ep)**2 - (0.938*u.GeV)**2) #Ep is roughly equal Pp at these energies

        if typeIIb:
            t_esc = t_sed*((Pp/(1e6*u.GeV))**(-1/2.5))   #in yr type IIb core collapse (default)
        else:
            t_esc = (234.*u.yr) * ((Pp/(1e6*u.GeV))**(-1/2.5)) #type 1A in yr

        return t_esc

    def SNR_Radius(self, time):
        """
        Classical ST-stage solution for SNR radius as a function of its age.

        Parameters
        ----------
        time : `astropy.Quantity`
            SNR age.

        Returns
        -------
        radius : `astropy.Quantity`
            Radius of the SNR. (pc)
        """

        # (1) Truelove & McKee (1999):
        # mu_0 = 1.4 * c.m_p            # see text above eq. 4
        # rho_0 = self.nh2ism * mu_0    # see text above eq. 4

        # # eq. 52
        # radius = (2.026 * (self.E51*1e51*u.erg) / rho_0)**0.2 * time**0.4

        # (2) Reynolds (2008), which is based on (1):
        mu_1 = 1.4 # for neutral gas
        n    = self.nh2ism

        # eq. 8 (exact factor found by comparing with Truelove)
        radius = 0.31456744*u.pc \
                    * self.E51**0.2 \
                    * (n/self.nh2ism)**-0.2 \
                    * (mu_1/1.4)**-0.2 \
                    * (time/u.yr)**0.4

        # Return
        return radius.to(u.pc)


    def SNR_age(self, size):
        """
        Estimate SNR age based on its radius.

        Parameters
        ----------
        size : `astropy.Quantity`
            SNR radius.

        Returns
        -------
        age : `astropy.Quantity`
            Age of the SNR. (yr)
        """

        n = self.nh2ism
        mu_1 = 1.4 # for neutral gas

        # Simply invert eq. used in method SNR_Radius()
        age = (size /u.pc \
                / 0.31456744 \
                / (self.E51**0.2) \
                / ((n/self.nh2ism)**-0.2) \
                / (mu_1/1.4)**-0.2 \
                )**2.5 *u.yr

        # Return
        return age.to(u.yr)

    
