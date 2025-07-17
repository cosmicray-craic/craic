import numpy as np
import astropy.units as u
import astropy.constants as c
from particles import particles

class transport:

    chiism = 1.
    cr_delta = 0.5
    part = particles()

    def __init__(self,D0=3*10**(26) *u.cm**2 /u.s ):
        self.D_0 = D0
        #Galactic Diffusion Coefficient (cm^2 s^-1) at 1 GeV
        #FAST = 3x10^27 SLOW = 3x10^26

    # Magnetic Field Strength
    def B_mag(self, dens):

        B = np.where(dens > 300*u.cm**-3,
                        10*u.uG * (dens/(300*u.cm**-3))**0.65,
                        10*u.uG)

        return B  # micro G

    def Diffusion_Coefficient(self, Ep, dens, chi=0.05, ism=0):  # input: GeV

        # chi = 1, no suppression in the ISM = larger diffusion coefficient
        if ism:
            d_coeff = self.chiism * self.D_0 * ((Ep / (1.*u.GeV)) / (3. / 3.)) ** 0.5  # make this 3 micro gaus
        else:
            d_coeff = chi * self.D_0 * (((Ep / (1.*u.GeV))) / (self.B_mag(dens) / (3*u.microgauss))) ** 0.5  # cm^2s^-1

        return d_coeff

    def R_diffusion(self, Ep, a, dens, chi=0.05, ism=0):  # input: seconds


        R_dif = 2 * np.sqrt(self.Diffusion_Coefficient(Ep, dens, chi, ism) * a.to(u.s))  # no frac contribution


        
        return R_dif  # cm

    def t_diffusion(self, Rdiff, Ep, dens, chi=0.05, ism=0):

        D = self.Diffusion_Coefficient(Ep, dens, chi, ism)
        t_diff = Rdiff.to(u.cm)**2 / (4. * D)

        return t_diff

    def vel_advection(self, rad):

        Rts = 0.1*u.pc
        Rc = 50.**0.5 * Rts

        va = (c.c/3.)*((rad - Rts) / Rts)**-2.

        mask = rad > Rc

        va[mask] = (c.c/3.)*((Rc - Rts) / Rts)**-2.

        va[va < 0.] = 0.
        va[va > (c.c/3.)] = c.c/3.

        return va.to(u.m/u.s)
