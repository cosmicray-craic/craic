Quickstart Tutorial
===================

The first step is to import the relevant packages and functions from craic.

.. code-block:: python

   import numpy as np
   import astropy.units as u
   import matplotlib.pyplot as plt
   from craic.particles import particles as pa
   from craic.transport import transport as tr
   from craic.accelerator import accelerator as ac
   from craic.flux import flux as fl
   from craic import injection as inj

The figure parameters can be set here (optional):

.. code-block:: python

    SMALL_SIZE = 11
    MEDIUM_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

The imported classes are initiated with default values, they can be altered if needed:

.. code-block:: python

    acel = ac() #default explosion energy 1e51 erg
    part = pa() #default power-law index alpha = 2, default energy budget 1e50 erg
    tran = tr() #default diffusion coefficient 3x10^27 cm^2/s at 1 GeV, default chi = 0.05
    fluxfunc = fl() #default index alpha = 2

To run through the model with particle acceleration at source, propagation through the ISM and interaction with target cloud to produce gamma-ray and neutrino flux, use the function SNR_Cloud_Flux.
The model follows Kelner et al (2006).

Input parameters:

* nh2 = target material density (:math:`\mathrm{cm}^{-3}`)
* dist = separation distance accelerator to target cloud (pc)
* age = SNR age (yr)

Optional input parameters:

* chi = diffusion suppression factor (:py:class:`float`)
* distance_SNR = distance of SNR from Earth (pc)
* radius_MC = radius of the cloud (pc)
* accel_type = acceleration type (:py:class:`str`: "Impulsive" or "Continuous")
* snr_typeII = a flag for type II or type IA SNRs (:py:class:`bool`)
* F_gal = flag whether or not to include which galactic CR spectrum: 
  F_gal=False (default), "AMS-O2", "DAMPE"
* palpha = power law index of the proton spectrum (:py:class:`float`)
* D_fast = flag fast or slow diffusion in the ISM (changes the normalisation) (:py:class:`bool``)
* flag_low = include low energy correction ~GeV range (:py:class:`bool``)

.. code-block:: python

    SNRcloud = SNR_Cloud_Flux(Eg_lo=1*u.GeV, Eg_hi=3*u.PeV, F_gal="DAMPE") # Change optional input parameters here if needed
    Eg, phi, phi_nu, phi_nue, phi_numu, phi_nutau = SNRcloud.compute_flux(
        nh2=100*u.cm**-3,
        dist=50*u.pc,
        age=2e4*u.yr
    )


Plotting the resultant spectra:

.. code-block:: python

    plt.plot(Eg, Eg*Eg*(phi),label=r'$\gamma$')
    plt.plot(Eg, Eg*Eg*(phi_nu),label=r'$\nu$')
    plt.plot(Eg, Eg*Eg*(phi_nue),label=r'$\nu_e$')
    plt.plot(Eg, Eg*Eg*(phi_numu),label=r'$\nu_\mu$')
    plt.plot(Eg, Eg*Eg*(phi_nutau),label=r'$\nu_\tau$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-14,1e-12)
    plt.xlim(0.01,100.)
    plt.xlabel(r"$E_{\gamma}$ (TeV)")
    plt.ylabel(r"$E^2$dN/dE (TeV cm$^{-2}$ s$^{-1}$)")
    plt.legend()
    None

.. image:: /_static/quickstart_tutorial.png