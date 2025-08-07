import numpy as np
import astropy.units as u
import astropy.constants as c
from .particles import particles

class transport:
    """Calculates magnetic field strength, diffusion coefficient and diffusion radius
    in the ISM and in a molecular cloud."""

    chiism = 1.
    cr_delta = 0.5
    part = particles()

    def __init__(self,D0=3*10**(26) *u.cm**2 /u.s ):
        self.D_0 = D0
        #Galactic Diffusion Coefficient (cm^2 s^-1) at 1 GeV
        #FAST = 3x10^27 SLOW = 3x10^26

    # Magnetic Field Strength
    @u.quantity_input(dens=u.cm**-3)
    def B_mag(self, dens) -> u.uG:
        r"""Returns the magnetic field strength of the cloud (:math:`\mathrm{\mu G}`)
        based on `Crutcher et al. 2010, ApJ 725 466 
        <https://iopscience.iop.org/article/10.1088/0004-637X/725/1/466>`_ (eq. 21).

        Parameter
        ----------
        dens: :class:`~astropy.units.Quantity`
            Number density of the molecular cloud (:math:`\mathrm{cm}^{-3}`)
        """
        B = np.where(dens > 300*u.cm**-3,
                        10*u.uG * (dens/(300*u.cm**-3))**0.65,
                        10*u.uG)

        return B  # micro G

    @u.quantity_input(Ep=u.GeV, dens=u.cm**-3)
    def Diffusion_Coefficient(self, Ep, dens, chi=0.05, ism=0) -> u.cm**2/u.s:  # input: GeV
        r"""Returns the diffusion coefficient of the ISM or the cloud (:math:`\mathrm{cm}^{2} \mathrm{/s}`)
        based on number density of the medium.
        
        Parameters
        ----------
        Ep : :class:`~astropy.units.Quantity` or array-like
            Energy of particles (GeV)
        dens: :class:`~astropy.units.Quantity`
            Number density (:math:`\mathrm{cm}^{-3}`)
        """

        # chi = 1, no suppression in the ISM = larger diffusion coefficient
        if ism:
            d_coeff = self.chiism * self.D_0 * ((Ep / (1.*u.GeV)) / (3. / 3.)) ** 0.5  # make this 3 micro gaus
        else:
            d_coeff = chi * self.D_0 * (((Ep / (1.*u.GeV))) / (self.B_mag(dens) / (3*u.microgauss))) ** 0.5  # cm^2s^-1

        return d_coeff

    @u.quantity_input(Ep=u.GeV, a=u.s, dens=u.cm**-3)
    def R_diffusion(self, Ep, a, dens, chi=0.05, ism=0) -> u.cm: 
        r"""Returns how far the accelerated particles can 
        propagate by diffusion in the ISM or in the molecular cloud (cm).
        
        Parameters
        ----------
        Ep : :class:`~astropy.units.Quantity`
            Energy of particles (GeV)
        a : :class:`~astropy.units.Quantity`
            Time of particle propagation in the medium. (seconds)
        dens: :class:`~astropy.units.Quantity`
            Number density (:math:`\mathrm{cm}^{-3}`)
        """


        R_dif = 2 * np.sqrt(self.Diffusion_Coefficient(Ep, dens, chi=chi, ism=ism) * a.to(u.s))  # no frac contribution
        
        return R_dif  

    # def t_diffusion(self, Rdiff, Ep, dens, chi=0.05, ism=0):

    #     D = self.Diffusion_Coefficient(Ep, dens, chi, ism)
    #     t_diff = Rdiff.to(u.cm)**2 / (4. * D)

    #     return t_diff

    # def vel_advection(self, rad):

    #     Rts = 0.1*u.pc
    #     Rc = 50.**0.5 * Rts

    #     va = (c.c/3.)*((rad - Rts) / Rts)**-2.

    #     mask = rad > Rc

    #     va[mask] = (c.c/3.)*((Rc - Rts) / Rts)**-2.

    #     va[va < 0.] = 0.
    #     va[va > (c.c/3.)] = c.c/3.

    #     return va.to(u.m/u.s)
