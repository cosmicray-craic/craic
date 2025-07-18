import numpy as np
import astropy.units as u
from scipy.special import erfc
from IPython import embed

class flux:
    """Return proton flux, gamma-ray and neutrino 
    kernal function from the molecular cloud."""
    ncbins = 30

    # Neutrino Oscillation Probabilities over long distances
    # From Mascaretti et al. 
    Pee = 0.56
    Pemu = 0.25
    Petau = 0.19
    Pmumu = 0.37
    Pmutau = 0.381
    Ptautau = 0.43
    
    @u.quantity_input(Rc=u.pc, Dc=u.pc)
    def cloud_cell_flux(self, Rc, Dc, flux):
        """
        Compute the total proton influx at the cloud.

        Parameters
        ----------
        Rc : `astropy.Quantity`
            Cloud radius (~ pc)
        Dc : `astropy.Quantity`
            Proton penetration depth in cloud (~ pc)
        flux : `astropy.Quantity`
            Proton flux at cloud entrance location

        Returns
        -------
        tot_ccf : `astropy.Quantity`
            Total cloud-cell-flux (~ GeV-1)
        """

        # Initialise return value to desired units
        tot_ccf = np.zeros(len(Dc)) * u.GeV**-1

        # Mask those penetration depths, that fully traverse the cloud
        vmask = Dc >= 2 * Rc

        #TEST flux reduction by escaped particles?
        norm_v = np.where(Dc > 2*Rc, 2*Rc/Dc, 1.0)
        
        # Compute cloud volume
        vol_cloud = (4 / 3.) * np.pi * Rc ** 3

        # Easy case: fully traversed cloud
        tot_ccf[vmask] = flux[vmask] * vol_cloud * norm_v[vmask]
        
        # If cloud fully traversed, we're done.
        if np.all(vmask):
            return tot_ccf #if without cells, always exit here

        # assume sphere
        x = np.linspace(-Rc, Rc, self.ncbins)
        y = np.linspace(-Rc, Rc, self.ncbins)
        X, Y = np.meshgrid(x, y)
        cx = np.ravel(X)
        cy = np.ravel(Y)

        cell_rad = np.sqrt(cx ** 2 + cy ** 2)
        cell_rad = cell_rad.clip(max=Rc)
        circ_x = -np.sqrt(Rc ** 2 - cy ** 2)

        # calculate line of sight depth
        ch = Rc - cell_rad
        slice_a = np.sqrt(ch * (2 * Rc - ch))
        cell_theta2 = np.arctan(cell_rad / Rc)
        cell_depth = 2 * slice_a * np.cos(cell_theta2)

        Area = (2 * Rc / self.ncbins) * (2 * Rc / self.ncbins)  # pc^2

        # Normalisation factor
        norm = vol_cloud / (sum(cell_depth) * Area)

        # re-writing the condition (circ_x + Dc > cx)
        cdiff = circ_x - cx
        ti, tj = np.meshgrid(cdiff, Dc[~vmask])

        tij = ti + tj
        maskt = tij > 0.


        tot_cell_depth = [sum(cell_depth[maskt[it]].value) for it in range(len(Dc[~vmask]))]

        tot_ccf[~vmask] = flux[~vmask] * Area * tot_cell_depth*u.pc
        tot_ccf[~vmask] *= norm.value

        return tot_ccf

    @u.quantity_input(Eg=u.TeV, Ep=u.GeV)
    def compute_gamma_kernel(self, Eg, Ep):
        """
        Compute the gamma-ray kernel function according to Kelner et al. 2006:
        Total spectrum of gamma-rays from pi0 & eta meson decay channels.

        Parameters
        ----------
        Eg : `astropy.Quantity`
            Gamma energy (~ TeV)
        Ep : `astropy.Quantity`
            Proton energy (~ GeV)

        Returns
        -------
        F_gamma : `numpyp.ndarray`
            Gamma kernel function with indexing [((proton,)gamma)]
        """
        Ep_ndim_in = np.ndim(Ep)
        Eg_ndim_in = np.ndim(Eg)
        Ep = np.atleast_1d(Ep)
        Eg = np.atleast_1d(Eg)

        # Pre-compute variables
        L = np.log(Ep.to(u.TeV).value)[:,None]
        x = (Eg/Ep[:,None]).decompose()
        L2 = L**2
        log_x = np.log(x)

        # Parameters of gamma-ray spectrum kernel function
        # (eqs. 59-61 in Kelner et al. 2006)
        B     =      1.30  + 0.14  * L + 0.011 * L2
        betag = 1 / (1.79  + 0.11  * L + 0.008 * L2)
        k     = 1 / (0.801 + 0.049 * L + 0.014 * L2)

        xbeta = x**betag

        # Components of gamma-ray spectrum kernel function
        part_1 = B * log_x / x
        part_2 = ((1 - xbeta) / (1 + k * xbeta * (1 - xbeta))) ** 4
        part_3 = ((1 / log_x) \
                  - ((4 * betag * xbeta) / (1 - xbeta)) \
                  - ((4 * k * betag * xbeta * (1 - 2 * xbeta)) \
                      / (1 + k * xbeta * (1 - xbeta))
                    )
                 )

        # The gamma-ray spectrum kernel function (eq. 58 in Kelner et al. 2006)
        F_gamma = part_1 * part_2 * part_3

        # Just to be sure, null unphysical (i.e. where Eg > Ep) factors
        F_gamma[x>=1] = 0.

        if (len(L.T[0])==1) and (L < -1):
            F_gamma[:] = 0.
        elif len(L.T[0]) > 1 :
            F_gamma[L.T[0]<-1] = 0.
            
        # Squeeze array to fit input shapes
        if (Ep_ndim_in==0) & (Eg_ndim_in==0): F_gamma=F_gamma[0,0]
        elif (Ep_ndim_in==0):                 F_gamma=F_gamma[0,:]
        elif (Eg_ndim_in==0):                 F_gamma=F_gamma[:,0]

        # Return [((proton,)gamma)]
        return F_gamma

    @u.quantity_input(En=u.TeV, Ep=u.GeV)
    def compute_neutrino_kernel(self, En, Ep):
        """
        Compute the neutrino kernel function according to Kelner et al. 2006:
        Total spectrum of neutrinos from charged pion & muon decay channels.

        Parameters
        ----------
        En : `astropy.Quantity`
            neutrino energy (~ TeV)
        Ep : `astropy.Quantity`
            Proton energy (~ GeV)

        Returns
        -------
        F_nu_1 : `numpyp.ndarray`
            Neutrino kernel function with indexing [((proton,)neutrino)] from muons

        F_nu_2 : `numpyp.ndarray`
            Neutrino kernel function with indexing [((proton,)neutrino)] from pions
        """
        Ep_ndim_in = np.ndim(Ep)
        En_ndim_in = np.ndim(En)
        Ep = np.atleast_1d(Ep)
        En = np.atleast_1d(En)

        # Pre-compute variables
        L = np.log(Ep.to(u.TeV).value)[:,None]
        x = (En / Ep[:,None]).decompose()

        # Note that for following parameters, read Kelners paper for explanation
        # PARAMETERS FOR MUONIC NEUTRINOS FROM DECAY OF CHARGED PIONS #

        Be = 1 / (69.5 + 2.65 * L + 0.3 * L ** 2)
        betae = 1 / ((0.201 + 0.062 * L + 0.00042 * L ** 2) ** (1 / 4))
        ke = (0.279 + 0.141 * L + 0.0172 * L ** 2) / (0.3 + (2.3 + L) ** 2)

        # PARAMETERS FOR MUONIC NEUTRINOS FROM DECAY OF MUONS #

        By = 1.75 + 0.204 * L + 0.010 * L ** 2
        betay = 1 / (1.67 + 0.111 * L + 0.0038 * L ** 2)
        ky = 1.07 - 0.086 * L + 0.002 * L ** 2
        y = x / 0.427

        ybeta = y ** betay

        # CALCULATION OF NEUTRINOS #

        # SPECTRUM BY DIRECT DECAYS OF CHARGED PIONS #

        part_1_nu2 = Be
        part_2_nu2 = ((1 + ke * ((np.log(x)) ** 2)) ** 3) / (x * (1 + 0.3 / (x ** betae)))
        part_3_nu2 = (-np.log(x)) ** 5

        
        F_nu_2 = part_1_nu2 * part_2_nu2 * part_3_nu2

        
        # SPECTRUM BY DECAY OF MUONS #

        part_1_nu1 = By * ((np.log(y)) / (y))
        part_2_nu1 = ((1 - ybeta) / (1 + ky * ybeta * (1 - ybeta))) ** 4
        part_3_nu1 = (1 / np.log(y) - (4 * betay * ybeta) / (1 - ybeta)
                      - (4 * ky * betay * ybeta * (1 - 2 * ybeta)) / (1 + ky * ybeta * (1 - ybeta)))

        F_nu_1 = part_1_nu1 * part_2_nu1 * part_3_nu1


        # Just to be sure, remove unphysical factors

        F_nu_1[x>=0.427] = 0.
        #remove low energies
        if (len(L.T[0])==1) and (L < -1):
            F_nu_2[:] = 0.
            F_nu_1[:] = 0.
        elif len(L.T[0]) > 1 :
            F_nu_2[L.T[0]<-1] = 0.
            F_nu_1[L.T[0]<-1] = 0.

            
        F_nu_2[F_nu_2 < 0.] = 0.
        
        # Squeeze array to fit input shapes (muon decay)
        if (Ep_ndim_in==0) & (En_ndim_in==0): F_nu_1=F_nu_1[0,0]
        elif (Ep_ndim_in==0):                 F_nu_1=F_nu_1[0,:]
        elif (En_ndim_in==0):                 F_nu_1=F_nu_1[:,0]

        # Squeeze array to fit input shapes (pion decay)
        if (Ep_ndim_in==0) & (En_ndim_in==0): F_nu_2=F_nu_2[0,0]
        elif (Ep_ndim_in==0):                 F_nu_2=F_nu_2[0,:]
        elif (En_ndim_in==0):                 F_nu_2=F_nu_2[:,0]

        # Return [((proton,)neutrino)]
        # Pion decay (F_nu_1) then muon decay (F_nu_2)
        return F_nu_1 , F_nu_2

    
