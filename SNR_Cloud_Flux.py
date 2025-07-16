import numpy as np
import astropy.units as u
from particles import particles
from transport import transport
from accelerator import accelerator
from flux import flux
from craic import injection
from injection import compute_fgal
#



# Define initialisation parameters for gamma-ray scan energies
N = 200
Eg_lo = 10. *u.GeV
Eg_hi = 3e3 *u.TeV #1 PeV

# Set up gamma-ray evaluation energies
Eg_edges = np.logspace(start = np.log10(Eg_lo.to(u.TeV).value),
                       stop  = np.log10(Eg_hi.to(u.TeV).value),
                       num   = N+1) *u.TeV
Egs  = np.sqrt(Eg_edges[:-1] * Eg_edges[1:]) # log bin centers
dEgs = np.diff(Eg_edges)

# Define proton energies
Ep_edges = np.logspace(0, 6.48, 1001) *u.GeV #1 PeV
Eps = np.sqrt(Ep_edges[:-1] * Ep_edges[1:]) # log bin centers
dEps = np.diff(Ep_edges)

#Define low energy part of proton energies 
Epl_edges = np.logspace(0, 3.24, 1001) *u.GeV
Epls = np.sqrt(Epl_edges[:-1] * Epl_edges[1:]) 
dEpls = np.diff(Epl_edges)

m_pion = 135. * u.MeV
m_proton = 938. *u.MeV
#Emin for integration (Kelner eq 78)
Emin_g = Egs + m_pion **2. / (4. * Egs) 

part = particles()
tran = transport()
acce = accelerator()
fl = flux()

#Convenience function
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx


def SNR_Cloud_Flux(nh2, dist, age, chi=0.05,
                   distance_SNR=1000*u.pc, radius_MC=10*u.pc,
                   accel_type='Impulsive', snr_typeII=True, F_gal=False,
                   palpha=2.0, D_fast=True, flag_low=True):

    """
    Compute the gamma-ray flux from a MC given a close-by SNR.

    Parameters
    ----------
    nh2 : `astropy.Quantity`
        Matter density in the MC. (~ cm-3)
    dist : `astropy.Quantity`
        Distance from SNR center to MC center. (~ pc)
    age : `astropy.Quantity`
        SNR age. (~ yr)
    chi : `float`
        Scalar suppression factor for diffusion coefficient in clouds, relating
        to the level of turbulence along the particles path.
        (ism: 1, clouds: 0.05)
    distance_SNR : `astropy.Quantity`
        Distance from earth to the SNR. (~ pc)
    radius_MC : `astropy.Quantity`
        Radius of the MC. (~ pc)
    accel_type : `str`
        Type of acceleration. Possible values are:
        ``Impulsive`` or ``Continuous``
    snr_typeII : boolean
        Flag to indicate the SNR scenario considered
    F_gal : boolean
        Flag to include the galactic CR flux contribution
    palpha : `float'
        Proton spectrum index
    D_fast : boolean
        Diffusion scenario (fast or slow normalisation)
    flag_low : boolean
        Flag to include low energy correction

    Returns
    -------
    Egs : `astropy.Quantity`
        Gamma-ray (or neutrino, i.e. secondary) energies. (TeV)
    phi : `astropy.Quantity`
        Gamma-ray flux. (TeV-1 cm-2 s-1)
    phi_nu : `astropy.Quantity`
        Neutrino flux.  (TeV-1 cm-2 s-1 ) 
    phi_nue_osc : `astropy.Quantity`
        Electron Neutrino flux.  (TeV-1 cm-2 s-1 ) 
    phi_numu_osc : `astropy.Quantity`
        Muon Neutrino flux.  (TeV-1 cm-2 s-1 ) 
    phi_nutau_osc : `astropy.Quantity`
        Tau Neutrino flux.  (TeV-1 cm-2 s-1 ) 
    """
    # Suppress numpy warnings
    np.seterr(all="ignore")

    # Get proton flux normalisation, computed for total CR energy
    N0 = part.NormEbudget()
    if not palpha == 2.0:
        part.alpha = palpha
    if D_fast:
        tran.D_0 = 3*10**(27) * u.cm**2 / u.s
    
    ########################################################################
    #
    #  Compute proton travel parameters & cloud penetration depth
    #
    ########################################################################
    
    # Compute Escape time
    tesc = acce.escape_time(Ep=Eps,typeIIb=snr_typeII)

    # Compute size of SNR at escape time
    Resc = acce.SNR_Radius(time=tesc).to(u.pc)

    
    timediff = age - tesc.to(u.yr)
    distancediff = dist - Resc


    # Here, use two step approach with accelerator situated away from cloud
    # First reach cloud
    d2cloud = distancediff - radius_MC
    # clip: if negative, cloud and SNR overlap.
    d2cloud = d2cloud.clip(min=0.)

    # Compute travel time from Resc to cloud
    traveltime = d2cloud ** 2 / tran.Diffusion_Coefficient(Eps, acce.nh2ism, ism=1) /4.

    ismtime = np.minimum(timediff, traveltime)
    ismtime = ismtime.clip(min=0)

    dism = d2cloud  # should be equivalent
    
    # Compute time spent in cloud & cloud penetration depth
    cloudtime = timediff - ismtime
    cloudtime = cloudtime.clip(min=0)
    cloud_depth = tran.R_diffusion(Eps, cloudtime, nh2, chi=chi, ism=0)  # do not ignore the time
    
    ########################################################################
    #
    #  Compute energy distribution of protons at cloud entrance location
    #  (given in GeV-1 cm-3)
    #
    ########################################################################

    if accel_type == 'Impulsive':
        if not sum(dism):
            pflux_tmp = injection.compute_pflux_impulsive_extended(
                Resc, N0, Eps, dism, timediff, nh2, chi=chi, ism=1, alpha=part.alpha)
            pfluxlow_tmp = injection.compute_pflux_impulsive_extended(
                Resc, N0, Epls, dism, timediff, nh2, chi=chi, ism=1, alpha=part.alpha)
        else:
            pflux_tmp = injection.compute_pflux_impulsive_extended(
            Resc, N0, Eps, dism, timediff, acce.nh2ism, chi=chi, ism=1, alpha=part.alpha)
            pfluxlow_tmp = injection.compute_pflux_impulsive_extended(
            Resc, N0, Epls, dism, timediff, acce.nh2ism, chi=chi, ism=1, alpha=part.alpha)
        
    elif accel_type == 'Continuous':
        pflux_tmp = injection.compute_pflux_continuous_extended(
            Resc, N0, Eps, dism, ismtime, acce.nh2ism, chi=chi, ism=1, alpha=part.alpha)
        pfluxlow_tmp = injection.compute_pflux_continuous_extended(
            Resc, N0, Epls, dism, ismtime, acce.nh2ism, chi=chi, ism=1, alpha=part.alpha)

    else:
        print("Bad acceleration type:",accel_type,
              "choose between Impulsive or Continuous")
        
    ########################################################################
    #
    #  Compute total proton flux exposing the cloud
    #  (given in GeV-1)
    #
    ########################################################################

    # Include flux from galactic CR sea if requested
    fgal = compute_fgal(Eps)

    if F_gal:
        gal_ratio = np.nanmean(pflux_tmp/fgal)
        if gal_ratio < 1.:
            print("Mean ratio pflux / fgal:",gal_ratio)
        pflux_tmp += fgal
        #pflux_tmp = 1000*fgal #testing for reviewer
        
    # Compute total proton flux interacting with the cloud (cell-based)
    pflux = fl.cloud_cell_flux(radius_MC, cloud_depth, pflux_tmp)
    
    # To be sure, set NaN fluxes to zero
    pflux = np.nan_to_num(pflux, nan=0)

    #print("Cloud pflux",Eps,pflux,pflux_tmp) 


    ########################################################################
    #
    #  Compute kernel function for given Ep (for gamma-ray flux equation)
    #  (given as scalar)
    #  The analytical description used here is given by Kelner et al. 2006
    #
    ########################################################################

    F_gamma = fl.compute_gamma_kernel(Egs, Eps)

    ########################################################################
    #
    #  Compute the total gamma-ray flux emitted from the cloud for the
    #  current proton energy.
    #  (given in cm2 GeV-1, since factors c*n come in later)
    #
    #  see eq. 13 in Mitchell et al. 2021
    #
    ########################################################################

    # Get cross section (cm2) for given proton energy
    sig = part.sig_ppEK(Ep=Eps)

    # Compute gamma-ray flux
    intg = np.sum((sig[:,None] * pflux[:,None] * F_gamma) \
                  * dEps[:,None] / Eps[:,None], axis=0)

    phi = part.cspeed * nh2 * intg / (4. * np.pi * distance_SNR**2)
    phi = phi.to(1./(u.cm**2 * u.TeV * u.s))

    total_phig = np.zeros(len(Egs))*(1./(u.TeV*u.s*u.cm**2))
    
    ##########################################################################
    # 
    # Compute 1 - 100 GeV part of the gamma-ray spectrum 
    #    
    ##########################################################################

    if flag_low:
        # Compute 1 - 100 GeV part
        n_t = 1. # Change to fix normalisation (1. as default)
        kappa = 1.
        K_pi = kappa / n_t
        E_pi = (Epls - m_proton)*K_pi #Check units, define at start of function
        dE_pi = dEpls

        #Compute q_pi
        q_pi = (n_t / K_pi) * part.cspeed * part.sig_ppEK(Ep=Epls) * pfluxlow_tmp

        phig_low = np.zeros(len(Egs))*(1./(u.TeV*u.s*u.cm**2))

        #Integral eq 78 of Kelner from Emin for all Eg
        for ig in range(len(Egs)):
            mask = E_pi > Emin_g[ig]
            phi_low = 2. * np.nansum(q_pi[mask] * dE_pi[mask] / np.sqrt(E_pi[mask]**2. - m_pion**2), axis=0)
            phi_low /= (4. * np.pi * distance_SNR**2)
            phig_low[ig] = phi_low

        #Establish normalisation factor
        idxn = find_nearest(Egs,0.1) #at 200 GeV (smoother than 100 GeV)
        phig_norm = phi[idxn] / phig_low[idxn]

        Emask = Egs < Egs[idxn]
    
        total_phig[~Emask] = phi[~Emask]
        total_phig[Emask] = phig_norm * phig_low[Emask]

        #End of low energy part of computation
    else:
        total_phig = phi
        

    # Compute neutrino kernel function and neutrino flux
    F_nu_1, F_nu_2 = fl.compute_neutrino_kernel(Egs, Eps)
    
    #Total (for muon neutrinos, electron neutrinos are roughly F_nu_2 only)
    F_nu_mu = F_nu_1 + F_nu_2 
    
    intn_mu = np.nansum((sig[:,None] * pflux[:,None] * F_nu_mu) \
                  * dEps[:,None] / Eps[:,None], axis=0)

    phi_numu = part.cspeed * nh2 * intn_mu / (4. * np.pi * distance_SNR**2)
    phi_numu = phi_numu.to(1./(u.cm**2 * u.TeV * u.s))

    F_nue = F_nu_2
    intn_e =  np.nansum((sig[:,None] * pflux[:,None] * F_nue) \
                  * dEps[:,None] / Eps[:,None], axis=0)

    phi_nue = part.cspeed * nh2 * intn_e / (4. * np.pi * distance_SNR**2)
    phi_nue = phi_nue.to(1./(u.cm**2 * u.TeV * u.s))

    # Total neutrino flux:
    phi_nu = phi_numu + phi_nue
    
    ########################################################################
    #
    # Single neutrino flavour flux 
    # Evaluate taking oscillation probabilities into account
    #
    # ( see table 5 in appendix of Mascaretti JCAP 2019 )
    #
    ########################################################################

    phi_nue_osc = fl.Pee * phi_nue + fl.Pemu * phi_numu
    phi_numu_osc = fl.Pemu * phi_nue + fl.Pmumu * phi_numu
    phi_nutau_osc = fl.Petau * phi_nue + fl.Pmutau * phi_numu
    
    #Return:
    #Secondary energy (axis), gamma-ray flux, neutrino flux: total, e, mu, tau
    return Egs, total_phig, phi_nu, phi_nue_osc, phi_numu_osc, phi_nutau_osc 


