import numpy as np
import astropy.units as u
from .particles import particles
from .transport import transport
from .accelerator import accelerator
from .flux import flux 
from . import injection

part = particles()
tran = transport()
acce = accelerator()
fl = flux()

#Define a convenient function
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx

def cosine_blend(E, pflux_low, pflux_high, E1, E2):
        """
        Blends two spectra pflux_low and pflux_high over energy E using a cosine window.
        The transition is defined between E1 and E2.
        
        Parameters:
        ----------
        E : array_like
            Energy of particles (~ TeV)
        F_low : array_like
            Spectrum from low-energy regime
        F_high : array_like
            Spectrum from high-energy regime
        E1 : float
            Start of blending region (lower boundary) (~ TeV)
        E2 : float
            End of blending region (upper boundary) (~ TeV)
        
        Returns:
        -------
        pflux_total : array_like
            Blended spectrum 
        """
        alpha=1
        # Initialise
        w = np.ones_like(E)

        # Define transition region mask
        blend_mask = (E >= E1) & (E <= E2)
        w[E > E2] = 0  # Fully high-energy component after E2
        w[blend_mask] = 0.5 * (1 + np.cos(np.pi * ((E[blend_mask] - E1) / (E2 - E1))**alpha))

        # Blend the spectra
        pflux_total = w * pflux_low + (1 - w) * pflux_high
        return pflux_total

def sigmoid_blend(E, pflux_low, pflux_high, E1, E2):
    """
    Blends two spectra pflux_low and pflux_high over energy E using a sigmoid window.
    The transition is defined between E1 and E2.

    Parameters:
    ----------
    E : array_like
        Energy of particles (~ TeV)
    pflux_low : array_like
        Spectrum from low-energy regime
    pflux_high : array_like
        Spectrum from high-energy regime
    E1 : float
        Start of blending region (~ TeV)
    E2 : float
        End of blending region (~ TeV)

    Returns:
    -------
    pflux_total : array_like
        Blended spectrum 
    """
    E = np.asarray(E)
    # Transition midpoint and sharpness scale
    Ec = 0.5 * (E1 + E2)
    delta = (E2 - E1) / 10  # Smaller = sharper transition

    # Sigmoid window: high at low E, low at high E
    w = 1 / (1 + np.exp((E - Ec) / delta))

    # Blend the spectra
    pflux_total = w * pflux_low + (1 - w) * pflux_high
    return pflux_total



class SNR_Cloud_Flux:
    """
    Class to compute gamma-ray flux from a molecular cloud given the properties of a nearby SNR.
    
    This class calculates gamma-ray and neutrino fluxes
    from cosmic ray interactions in molecular clouds illuminated by nearby supernova remnants.
    """
    
    def __init__(self, chi=0.05, distance_SNR=2000*u.pc, radius_MC=10*u.pc, Eg_lo=1.0*u.GeV, Eg_hi=3e3*u.TeV, accel_type='Impulsive', 
                 snr_typeII=True, F_gal=False, palpha=2.0, D_fast=True, flag_low=True):
        """
        Initialize the SNR Cloud Flux model by setting the properties of the SNR and the cloud.
        
        Parameters
        ----------
        chi : float
            Scalar suppression factor for diffusion coefficient in clouds, relating
            to the level of turbulence along the particles path.
            (ism: 1, clouds: 0.05)
        distance_SNR : astropy.Quantity
            Distance from earth to the SNR. (~ pc)
        radius_MC: astropy.Quantity
            Radius of the molecular cloud (~ pc)
        Eg_lo: astropy.Quantity
            Minimum energy of the produced gamma rays.
        Eg_hi: astropy.Quantity
            Maximum energy of the produced gamma rays.
        accel_type : str
            Type of acceleration. Possible values are:
            ``Impulsive`` or ``Continuous``
        snr_typeII : boolean
            Flag to indicate the SNR scenario considered
        F_gal : boolean or string
            Flag to include the galactic CR flux contribution. Can switch between 
            two Galactic CR flux inputs "AMS-O2" and "DAMPE".
        palpha : float
            Proton spectrum index
        D_fast : boolean
            Diffusion scenario (fast or slow normalisation)
        flag_low : boolean
            Flag to include low energy correction
        """
        # Set up initial energy arrays
        # N = 200 
        N=200
        # Eg_lo = 10. *u.GeV
        # Eg_lo = 1. * u.GeV
        # Eg_hi = 3e3 *u.TeV # 1 PeV

        # Set up gamma-ray evaluation energies
        Eg_edges = np.logspace(start = np.log10(Eg_lo.to(u.TeV).value),
                            stop  = np.log10(Eg_hi.to(u.TeV).value),
                            num   = N+1) *u.TeV
        self.Egs  = np.sqrt(Eg_edges[:-1] * Eg_edges[1:]) # log bin centers
        self.dEgs = np.diff(Eg_edges)

        # Define proton energies
        # Ep_edges = np.logspace(0, 6.48, 1001) *u.GeV #1 PeV
        Ep_edges = np.logspace(1, 6.48, 1001) *u.GeV #1 PeV
        self.Eps = np.sqrt(Ep_edges[:-1] * Ep_edges[1:]) # log bin centers
        self.dEps = np.diff(Ep_edges)

        #Define low energy part of proton energies 
        Epl_edges = np.logspace(0, 3.24, 1001) *u.GeV
        self.Epls = np.sqrt(Epl_edges[:-1] * Epl_edges[1:]) 
        self.dEpls = np.diff(Epl_edges)

        self.m_pion = 135. * u.MeV
        self.m_proton = 938. *u.MeV
        #Emin for integration (Kelner eq 78)
        self.Emin_g = self.Egs + self.m_pion **2. / (4. * self.Egs) 

        self.chi = chi
        self.distance_SNR = distance_SNR
        self.radius_MC = radius_MC
        # self.E_min = E_min
        # self.E_max = E_max
        self.accel_type = accel_type
        self.snr_typeII = snr_typeII
        self.F_gal = F_gal
        self.palpha = palpha
        self.D_fast = D_fast
        self.flag_low = flag_low

        # Set up initial parameters
        self._setup_parameters()
    
    def _setup_parameters(self):
        """Set up proton flux normalisation and diffusion coefficients."""
        # Suppress numpy warnings
        np.seterr(all="ignore")
        
        # Get proton flux normalisation, computed for total CR energy
        self.N0 = part.NormEbudget()
        
        if not self.palpha == 2.0:
            part.alpha = self.palpha
            
        if self.D_fast:
            tran.D_0 = 3*10**(27) * u.cm**2 / u.s
    
    @u.quantity_input(nh2=u.cm**-3, dist=u.pc, age=u.yr)
    def _compute_travel_parameters(self, nh2, dist, age) -> tuple[u.cm, u.pc, u.yr, u.pc]:
        """
        Compute proton travel parameters and cloud penetration depth.
        
        Parameters
        ----------
        nh2 : astropy.Quantity
            Matter density in the MC. (~ cm-3)
        dist : astropy.Quantity
            Distance from SNR center to MC center. (~ pc)
        age : astropy.Quantity
            SNR age. (~ yr)

        Returns
        -------
        tuple
            (cloud_depth, dism, ismtime, Resc)
        """
        # Compute Escape time
        tesc = acce.escape_time(Ep=self.Eps, typeIIb=self.snr_typeII)
        
        # Compute size of SNR at escape time
        Resc = acce.SNR_Radius(time=tesc).to(u.pc)
        
        timediff = age - tesc.to(u.yr)
        self.timediff = timediff
        distancediff = dist - Resc
        
        # Here, use two step approach with accelerator situated away from cloud
        # First reach cloud
        d2cloud = distancediff - self.radius_MC
        # clip: if negative, cloud and SNR overlap.
        d2cloud = d2cloud.clip(min=0.)
        
        # Compute travel time from Resc to cloud
        traveltime = d2cloud ** 2 / tran.Diffusion_Coefficient(self.Eps, acce.nh2ism, ism=1) / 4.
        
        ismtime = np.minimum(timediff, traveltime)
        ismtime = ismtime.clip(min=0)
        
        dism = d2cloud  # should be equivalent
        
        # Compute time spent in cloud & cloud penetration depth
        cloudtime = timediff - ismtime
        cloudtime = cloudtime.clip(min=0)
        cloud_depth = tran.R_diffusion(self.Eps, cloudtime, nh2, chi=self.chi, ism=0)
        
        return cloud_depth, dism, ismtime, Resc
    
    @u.quantity_input(nh2=u.cm**-3, dism=u.pc, time_diff=u.yr, ismtime=u.yr, Resc=u.pc)
    def _compute_proton_flux(self, nh2, dism, time_diff, ismtime, Resc) -> tuple[u.GeV**-1 * u.cm**-3, u.GeV**-1 * u.cm**-3]:
        """
        Compute energy distribution of protons at cloud entrance location.
        
        Parameters
        ----------
        nh2 : astropy.Quantity
            Number density of molecular cloud (~ cm^-3)
        dism : astropy.Quantity
            Distance in ISM (~ pc)
        timediff : astropy.Quantity
            Time difference (~ yr)
        ismtime : astropy.Quantity
            Time in ISM (~ yr)
        Resc : astropy.Quantity
            Escape radius (~ pc)
            
        Returns
        -------
        tuple
            (pflux_tmp, pfluxlow_tmp)
        """
        if self.accel_type == 'Impulsive':
            if np.isclose(sum(dism).value, 0):
                pflux_tmp = injection.compute_pflux_impulsive_extended(
                    Resc, self.N0, self.Eps, dism, time_diff, nh2, chi=self.chi, ism=1, alpha=part.alpha)
                pfluxlow_tmp = injection.compute_pflux_impulsive_extended(
                    Resc, self.N0, self.Epls, dism, time_diff, nh2, chi=self.chi, ism=1, alpha=part.alpha)
            else:
                pflux_tmp = injection.compute_pflux_impulsive_extended(
                    Resc, self.N0, self.Eps, dism, time_diff, acce.nh2ism, chi=self.chi, ism=1, alpha=part.alpha)
                pfluxlow_tmp = injection.compute_pflux_impulsive_extended(
                    Resc, self.N0, self.Epls, dism, time_diff, acce.nh2ism, chi=self.chi, ism=1, alpha=part.alpha)
        
        elif self.accel_type == 'Continuous':
            pflux_tmp = injection.compute_pflux_continuous_extended(
                Resc, self.N0, self.Eps, dism, ismtime, acce.nh2ism, chi=self.chi, ism=1, alpha=part.alpha)
            pfluxlow_tmp = injection.compute_pflux_continuous_extended(
                Resc, self.N0, self.Epls, dism, ismtime, acce.nh2ism, chi=self.chi, ism=1, alpha=part.alpha)
        
        else:
            raise ValueError(f"Bad acceleration type: {self.accel_type}, choose between Impulsive or Continuous")
        
        return pflux_tmp, pfluxlow_tmp
    
    @u.quantity_input(pflux_tmp=u.GeV**-1*u.cm**-3)
    def _add_galactic_flux(self, pflux_tmp) -> u.GeV**-1*u.cm**-3:
        """
        Add galactic cosmic ray flux contribution if requested.
        
        Parameters
        ----------
        pflux_tmp : array
            Proton flux array
            
        Returns
        -------
        array (GeV^-1 cm^-3)
            Modified proton flux with galactic contribution
        """
        if self.F_gal is not False:
            fgal_AMSO2 = injection.compute_fgal(self.Eps)
            fgal_DAMPE = injection.compute_fgal_dampe(self.Eps)
            
            if self.F_gal == "AMS-O2":
                gal_ratio = np.nanmean(pflux_tmp / fgal_AMSO2)
                if gal_ratio < 1.:
                    print("Mean ratio pflux / fgal:", gal_ratio)
                pflux_tmp += fgal_AMSO2
            
            elif self.F_gal == "DAMPE":
                gal_ratio = np.nanmean(pflux_tmp / fgal_DAMPE)
                if gal_ratio < 1.:
                    print("Mean ratio pflux / fgal:", gal_ratio)
                pflux_tmp += fgal_DAMPE
            
            else:
                raise ValueError('Choose a valid galactic flux model: "AMS-O2" or "DAMPE"')
        
        return pflux_tmp
    
    @u.quantity_input(nh2=u.cm**-3)
    def _compute_gamma_ray_flux(self, nh2, pflux, pfluxlow_tmp) -> 1 / (u.TeV*u.s*u.cm**2):
        """
        Compute gamma-ray flux from proton-proton interactions.
        
        Parameters
        ----------
        nh2 : astropy.Quantity
            Matter density in the MC (~ cm^-3)
        pflux : array
            Proton flux array (~ GeV^-1 cm^-3)
        pfluxlow_tmp : array
            Low energy proton flux (~ GeV^-1 cm^-3)
            
        Returns
        -------
        astropy.Quantity
            Total gamma-ray flux (~ TeV^-1 cm^-2 s^-1)
        """
        # Compute kernel function for gamma-ray flux
        F_gamma = fl.compute_gamma_kernel(self.Egs, self.Eps)
        
        # Get cross section (cm2) for given proton energy
        sig = part.sig_ppEK(Ep=self.Eps)
        
        # Compute gamma-ray flux
        intg = np.sum((sig[:,None] * pflux[:,None] * F_gamma) \
                      * self.dEps[:,None] / self.Eps[:,None], axis=0)
        
        phi = part.cspeed * nh2 * intg / (4. * np.pi * self.distance_SNR**2)
        phi = phi.to(1./(u.cm**2 * u.TeV * u.s))
        
        total_phig = np.zeros(len(self.Egs)) * (1./(u.TeV*u.s*u.cm**2))
        
        # Compute low energy part if requested
        if self.flag_low:
            total_phig = self._compute_low_energy_gamma(phi, pfluxlow_tmp)
        else:
            total_phig = phi
        
        return total_phig
    
    @u.quantity_input(nh2=u.cm**-3)
    def _compute_low_energy_gamma(self, phi, pfluxlow_tmp) -> 1 / (u.TeV*u.s*u.cm**2):
        """
        Compute 1 - 100 GeV part of the gamma-ray spectrum.
        
        Parameters
        ----------
        phi : array
            High energy gamma-ray flux (~ GeV^-1 cm^-3)
        pfluxlow_tmp : array
            Low energy proton flux (~ GeV^-1 cm^-3)
            
        Returns
        -------
        array
            Total gamma-ray flux including low energy part (~ TeV^-1 cm^-2 s^-1)
        """
        n_t = 1.  # Change to fix normalisation (1. as default)
        kappa = 1.
        K_pi = kappa / n_t
        E_pi = (self.Epls - self.m_proton) * K_pi
        dE_pi = self.dEpls
        
        # Compute q_pi
        q_pi = (n_t / K_pi) * part.cspeed * part.sig_ppEK(Ep=self.Epls) * pfluxlow_tmp
        
        phig_low = np.zeros(len(self.Egs)) * (1./(u.TeV*u.s*u.cm**2))
        
        # Integral eq 78 of Kelner from Emin for all Eg
        for ig in range(len(self.Egs)):
            mask = E_pi > self.Emin_g[ig]
            phi_low = 2. * np.nansum(q_pi[mask] * dE_pi[mask] / np.sqrt(E_pi[mask]**2. - self.m_pion**2), axis=0)
            phi_low /= (4. * np.pi * self.distance_SNR**2)
            phig_low[ig] = phi_low
        
        # Establish normalisation factor
        idxn = find_nearest(self.Egs, 0.1)  # at 200 GeV (smoother than 100 GeV)
        phig_norm = phi[idxn] / phig_low[idxn]
        
        Emask = self.Egs < self.Egs[idxn]
        
        total_phig = np.zeros(len(self.Egs)) * (1./(u.TeV*u.s*u.cm**2))
        total_phig[~Emask] = phi[~Emask]
        total_phig[Emask] = phig_norm * phig_low[Emask]

        pflux_total = sigmoid_blend(self.Egs.value, phig_norm * phig_low.value, phi.value, 0.05, 0.15) / (u.TeV * u.cm**2 * u.s) # TeV

        return pflux_total
    
    @u.quantity_input(nh2=u.cm**-3)
    def _compute_neutrino_flux(self, nh2, pflux):
        """
        Compute neutrino flux from proton-proton interactions.
        
        Parameters
        ----------
        nh2 : astropy.Quantity
            Matter density in the MC
        pflux : array
            Proton flux array
            
        Returns
        -------
        tuple
            [phi_nu, phi_nue_osc, phi_numu_osc, phi_nutau_osc] (~ TeV^-1 cm^-2 s^-1)
        """
        # Compute neutrino kernel function and neutrino flux
        F_nu_1, F_nu_2 = fl.compute_neutrino_kernel(self.Egs, self.Eps)
        
        # Get cross section
        sig = part.sig_ppEK(Ep=self.Eps)
        
        # Total (for muon neutrinos, electron neutrinos are roughly F_nu_2 only)
        F_nu_mu = F_nu_1 + F_nu_2
        
        intn_mu = np.nansum((sig[:,None] * pflux[:,None] * F_nu_mu) \
                          * self.dEps[:,None] / self.Eps[:,None], axis=0)
        
        phi_numu = part.cspeed * nh2 * intn_mu / (4. * np.pi * self.distance_SNR**2)
        phi_numu = phi_numu.to(1./(u.cm**2 * u.TeV * u.s))
        
        F_nue = F_nu_2
        intn_e = np.nansum((sig[:,None] * pflux[:,None] * F_nue) \
                          * self.dEps[:,None] / self.Eps[:,None], axis=0)
        
        phi_nue = part.cspeed * nh2 * intn_e / (4. * np.pi * self.distance_SNR**2)
        phi_nue = phi_nue.to(1./(u.cm**2 * u.TeV * u.s))
        
        # Total neutrino flux
        phi_nu = phi_numu + phi_nue
        
        # Single neutrino flavour flux with oscillation probabilities
        phi_nue_osc = fl.Pee * phi_nue + fl.Pemu * phi_numu
        phi_numu_osc = fl.Pemu * phi_nue + fl.Pmumu * phi_numu
        phi_nutau_osc = fl.Petau * phi_nue + fl.Pmutau * phi_numu
        
        return phi_nu, phi_nue_osc, phi_numu_osc, phi_nutau_osc
    
    @u.quantity_input(nh2=u.cm**-3, dist=u.pc, age=u.yr)
    def compute_flux(self, nh2, dist, age) -> tuple:
        """
        Compute the gamma-ray and neutrino flux from a molecular cloud
        irradiated by a nearby SNR.
        
        Parameters
        ----------
        nh2 : :class:`~astropy.units.Quantity`
            Matter density in the MC. (:math:`\mathrm{cm}^{-3}`)
        dist : :class:`~astropy.units.Quantity`
            Distance from SNR center to MC center. (pc)
        age : :class:`~astropy.units.Quantity`
            SNR age. (yr)
            
        Returns
        -------
        tuple
            (Egs, total_phig, phi_nu, phi_nue_osc, phi_numu_osc, phi_nutau_osc)
            
            Egs : :class:`~astropy.units.Quantity`
                Gamma-ray (or neutrino, i.e. secondary) energies. (TeV)
            total_phig : :class:`~astropy.units.Quantity`
                Gamma-ray flux. (:math:`\mathrm{TeV}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}`)
            phi_nu : :class:`~astropy.units.Quantity`
                Neutrino flux. (:math:`\mathrm{TeV}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}`)
            phi_nue_osc : :class:`~astropy.units.Quantity`
                Electron Neutrino flux. (:math:`\mathrm{TeV}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}`)
            phi_numu_osc : :class:`~astropy.units.Quantity`
                Muon Neutrino flux. (:math:`\mathrm{TeV}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}`)
            phi_nutau_osc : :class:`~astropy.units.Quantity`
                Tau Neutrino flux. (:math:`\mathrm{TeV}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}`)
        """
        # Compute proton travel parameters & cloud penetration depth
        cloud_depth, dism, ismtime, Resc = self._compute_travel_parameters(nh2, dist, age)
        
        # Compute energy distribution of protons at cloud entrance location
        pflux_tmp, pfluxlow_tmp = self._compute_proton_flux(nh2, dism, self.timediff, ismtime, Resc)
        
        # Include flux from galactic CR sea if requested
        pflux_tmp = self._add_galactic_flux(pflux_tmp)
        
        # Compute total proton flux interacting with the cloud (cell-based)
        pflux = fl.cloud_cell_flux(self.radius_MC, cloud_depth, pflux_tmp) 

        # To be sure, set NaN fluxes to zero
        pflux = np.nan_to_num(pflux, nan=0)
        
        # Compute gamma-ray flux
        total_phig = self._compute_gamma_ray_flux(nh2, pflux, pfluxlow_tmp)
        
        # Compute neutrino flux
        phi_nu, phi_nue_osc, phi_numu_osc, phi_nutau_osc = self._compute_neutrino_flux(nh2, pflux)
        
        return self.Egs, total_phig, phi_nu, phi_nue_osc, phi_numu_osc, phi_nutau_osc
    
    # def set_parameters(self, **kwargs):
    #     """
    #     Update calculator parameters if needed.
        
    #     Parameters
    #     ----------
    #     **kwargs : dict
    #         Parameter name-value pairs to update
    #     """
    #     for key, value in kwargs.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)
    #         else:
    #             raise ValueError(f"Unknown parameter: {key}")
        
    #     # Re-setup parameters if needed
    #     if any(key in kwargs for key in ['palpha', 'D_fast']):
    #         self._setup_parameters()


# Example usage:
# SNRcloud = SNRCloudFlux(
#     chi=0.05,
#     distance_SNR=1000*u.pc,
#     major_MC=10*u.pc,
#     minor_MC=10*u.pc,
#     accel_type='Impulsive',
#     snr_typeII=True,
#     F_gal="AMS-O2",
#     palpha=2.0,
#     D_fast=True,
#     flag_low=True
# )

# Egs, phi_gamma, phi_nu, phi_nue_osc, phi_numu_osc, phi_nutau_osc = SNRcloud.compute_flux(
#     nh2=1e3*u.cm**-3,
#     dist=50*u.pc,
#     age=1e4*u.yr
# )