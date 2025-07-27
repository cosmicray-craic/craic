Modules & Functions
===================

Here are listed some useful functions from the modules *accelerator*, *particles*, *transport*, *flux*, and *injection*. 
They can be used to calculate the properties and CR flux accelerated by the supernova remnant, the transport in the ISM and in the molecular cloud, and the CR interactions within the cloud.

accelerator
-----------

This module returns the properties of the supernova remnant, which accelerates particles to high energies.

.. automodule:: craic.accelerator
   :members: escape_time, SNR_Radius, SNR_age
   :inherited-members:

injection
---------

This module computes the proton flux from point or extended sources in the impulsive or continuous injection cases given a set of source properties, based on Aharonian and Atoyan (1996). 

.. automodule:: craic.injection
   :members: compute_fgal, comptue_fgal_dampe, compute_pflux_impulsive_point, compute_pflux_impulsive_extended, compute_pflux_continuous_point, compute_pflux_continuous_extended
   :inherited-members:

transport
---------

This modules describes CR transport by calculating magnetic field strength, diffusion coefficient and diffusion radius in the ISM and in a molecular cloud.

.. automodule:: craic.transport
    :members: B_mag, Diffusion_Coefficient, R_diffusion
    :inherited-members:

particles
---------

This module handles particle interactions by calculating inelastic p-p interaction cross-section, proton cooling time, and the normalisation of proton spectrum.

.. automodule:: craic.particles
    :members: sigma_ppEK, t_ppEK, NormEbudget
    :inherited-members:

flux
----

This module computes the proton flux and kernal functions required to calculate the gamma-ray and neutrino emission from the molecular cloud."""

.. automodule:: craic.flux
    :members: cloud_cell_flux, compute_gamma_kernel, compute_neutrino_kernel
    :inherited-members:

SNR_Cloud_Flux
--------------

This module contains the essential functions to configure the properties of the supernova remnant and the molecular cloud, and compute the gamma-ray and neutrino spectra.

.. automodule:: craic.SNR_Cloud_Flux
    :members: 
    :special-members: __init__

