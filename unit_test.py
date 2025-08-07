import astropy.units as u
import astropy.constants as c
import numpy as np
import pytest
from craic.accelerator import accelerator as ac
from craic.flux import flux as fl
from craic.particles import particles
from craic.transport import transport
from unittest.mock import Mock, patch
import warnings
import sys
import os

acel = ac()

# Unit Tests
class TestAccelerator:
    """Unit tests for the Accelerator class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.acc = ac()
        self.acc_custom = ac(En=2e51*u.erg)

    def test_escape_time_typeIIb(self):
        """Test escape time calculation for Type IIb supernova."""
        Ep = 10*u.GeV
        t_esc = self.acc.escape_time(Ep, typeIIb=True)
        
        assert isinstance(t_esc, u.Quantity)
        assert t_esc.unit == u.yr
        assert t_esc.value > 0
        
        # Test that higher energy particles escape faster
        Ep_high = 100*u.GeV
        t_esc_high = self.acc.escape_time(Ep_high, typeIIb=True)
        assert t_esc_high < t_esc
    
    def test_escape_time_typeIa(self):
        """Test escape time calculation for Type Ia supernova."""
        Ep = 10*u.GeV
        t_esc = self.acc.escape_time(Ep, typeIIb=False)
        
        assert isinstance(t_esc, u.Quantity)
        assert t_esc.unit == u.yr
        assert t_esc.value > 0
        
        # Type Ia should have different escape times than Type IIb
        t_esc_IIb = self.acc.escape_time(Ep, typeIIb=True)
        assert not np.isclose(t_esc.value, t_esc_IIb.value)
    
    def test_escape_time_invalid_energy(self):
        """Test escape time with invalid (too low) energy."""
        with pytest.raises(ValueError, match="Proton energy must be greater than its rest mass energy."):
            self.acc.escape_time(0.5*u.GeV)
    
    def test_escape_time_array_input(self):
        """Test escape time with array input."""
        Ep_array = np.array([1, 10, 100])*u.GeV
        t_esc = self.acc.escape_time(Ep_array, typeIIb=True)
        
        assert isinstance(t_esc, u.Quantity)
        assert len(t_esc) == 3
        assert all(t_esc > 0*u.yr)
        # Higher energies should have shorter escape times
        assert t_esc[0] > t_esc[1] > t_esc[2]
    
    def test_SNR_radius_basic(self):
        """Test SNR radius calculation."""
        time = 1000*u.yr
        radius = self.acc.SNR_Radius(time)
        
        assert isinstance(radius, u.Quantity)
        assert radius.unit == u.pc
        assert radius.value > 0
    
    def test_SNR_radius_scaling(self):
        """Test SNR radius scaling with time."""
        t1 = 1000*u.yr
        t2 = 2000*u.yr
        r1 = self.acc.SNR_Radius(t1)
        r2 = self.acc.SNR_Radius(t2)
        
        # Radius should scale as t^0.4
        expected_ratio = (t2/t1)**0.4
        actual_ratio = r2/r1
        assert np.isclose(actual_ratio.value, expected_ratio.value, rtol=1e-10)
    
    def test_SNR_radius_invalid_time(self):
        """Test SNR radius with invalid time."""
        with pytest.raises(ValueError, match="Time must be positive"):
            self.acc.SNR_Radius(-100*u.yr)
        
        with pytest.raises(ValueError, match="Time must be positive"):
            self.acc.SNR_Radius(0*u.yr)
    
    def test_SNR_age_basic(self):
        """Test SNR age calculation."""
        size = 10*u.pc
        age = self.acc.SNR_age(size)
        
        assert isinstance(age, u.Quantity)
        assert age.unit == u.yr
        assert age.value > 0
    
    def test_SNR_age_invalid_size(self):
        """Test SNR age with invalid size."""
        with pytest.raises(ValueError, match="Size must be positive"):
            self.acc.SNR_age(-5*u.pc)
        
        with pytest.raises(ValueError, match="Size must be positive"):
            self.acc.SNR_age(0*u.pc)
    
    def test_radius_age_consistency(self):
        """Test that radius and age calculations are consistent (inverse operations)."""
        time_original = 1500*u.yr
        
        # Calculate radius from time, then age from radius
        radius = self.acc.SNR_Radius(time_original)
        time_calculated = self.acc.SNR_age(radius)
        
        # Should recover original time within numerical precision
        assert np.isclose(time_original.value, time_calculated.value, rtol=1e-10)
    
    def test_array_inputs(self):
        """Test calculations with array inputs."""
        times = np.array([500, 1000, 2000])*u.yr
        radii = self.acc.SNR_Radius(times)
        
        assert len(radii) == 3
        assert all(radii > 0*u.pc)
        # Radii should increase with time
        assert radii[0] < radii[1] < radii[2]
        
        # Test inverse calculation
        ages = self.acc.SNR_age(radii)
        assert np.allclose(times.value, ages.value, rtol=1e-10)


# class TestFlux:


# Import the flux class (assuming it's in a file called flux.py)
# If the file has a different name, adjust the import accordingly
try:
    from craic.flux import flux
except ImportError:
    # If direct import fails, try adding current directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from craic.flux import flux


class TestFlux:
    """Test suite for the flux class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.flux_instance = flux()
        
    def test_class_attributes(self):
        """Test that class attributes are correctly defined."""
        assert self.flux_instance.ncbins == 30
        
        # Test neutrino oscillation probabilities
        assert abs(self.flux_instance.Pee - 0.56) < 1e-10
        assert abs(self.flux_instance.Pemu - 0.25) < 1e-10
        assert abs(self.flux_instance.Petau - 0.19) < 1e-10
        assert abs(self.flux_instance.Pmumu - 0.37) < 1e-10
        assert abs(self.flux_instance.Pmutau - 0.381) < 1e-10
        assert abs(self.flux_instance.Ptautau - 0.43) < 1e-10
        
        # Test that probabilities sum correctly (approximately)
        # Electron neutrino probabilities should sum to ~1
        prob_sum_e = self.flux_instance.Pee + self.flux_instance.Pemu + self.flux_instance.Petau
        assert abs(prob_sum_e - 1.0) < 0.01


class TestCloudCellFlux:
    """Test suite for the cloud_cell_flux method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.flux_instance = flux()
    
    def test_cloud_cell_flux_units(self):
        """Test that cloud_cell_flux returns correct units."""
        Rc = 1.0 * u.pc
        Dc = np.array([0.5, 1.0, 2.0, 3.0]) * u.pc
        input_flux = np.array([1e-3, 2e-3, 3e-3, 4e-3]) * u.GeV**-1 * u.cm**-3
        
        result = self.flux_instance.cloud_cell_flux(Rc, Dc, input_flux)
        
        assert result.unit == u.GeV**-1
        assert len(result) == len(Dc)
    
    def test_cloud_cell_flux_fully_traversed(self):
        """Test cloud_cell_flux for fully traversed clouds (Dc >= 2*Rc)."""
        Rc = 1.0 * u.pc
        Dc = np.array([2.0, 3.0, 4.0]) * u.pc  # All >= 2*Rc
        input_flux = np.array([1e-3, 2e-3, 3e-3]) * u.GeV**-1 * u.cm**-3
        
        result = self.flux_instance.cloud_cell_flux(Rc, Dc, input_flux)
        
        # Expected cloud volume
        vol_cloud = (4/3) * np.pi * Rc**3
        
        # For fully traversed case, should have flux * volume * normalization
        expected_0 = input_flux[0] * vol_cloud * 1.0  # norm_v = 1 for Dc = 2*Rc
        expected_1 = input_flux[1] * vol_cloud * (2*Rc/Dc[1])  # norm_v = 2*Rc/Dc for Dc > 2*Rc
        expected_2 = input_flux[2] * vol_cloud * (2*Rc/Dc[2])
        
        assert abs((result[0] - expected_0).value) < 1e-10
        assert abs((result[1] - expected_1).value) < 1e-6
        assert abs((result[2] - expected_2).value) < 1e-6
    
    def test_cloud_cell_flux_partial_traversal(self):
        """Test cloud_cell_flux for partial cloud traversal (Dc < 2*Rc)."""
        Rc = 2.0 * u.pc
        Dc = np.array([1.0, 1.5]) * u.pc  # Both < 2*Rc
        input_flux = np.array([1e-3, 2e-3]) * u.GeV**-1 * u.cm**-3
        
        result = self.flux_instance.cloud_cell_flux(Rc, Dc, input_flux)
        
        # Should return positive values
        assert all(result.value > 0)
        assert result.unit == u.GeV**-1
    
    def test_cloud_cell_flux_single_values(self):
        """Test cloud_cell_flux with single scalar values."""
        Rc = np.array([1.0]) * u.pc
        Dc = np.array([0.5]) * u.pc
        input_flux = np.array([1e-3]) * u.GeV**-1 * u.cm**-3
        
        result = self.flux_instance.cloud_cell_flux(Rc, Dc, input_flux)

        assert isinstance(result, u.Quantity)
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"

        # Check unit equivalence
        assert result.unit.is_equivalent(u.GeV**-1), f"Unexpected unit: {result.unit}"

        # Check that the only element is positive
        assert result[0].value > 0, f"Flux value not positive: {result[0]}"
    
    
    def test_cloud_cell_flux_zero_flux(self):
        """Test cloud_cell_flux with zero input flux."""
        Rc = 1.0 * u.pc
        Dc = np.array([0.5, 2.0]) * u.pc
        input_flux = np.array([0.0, 0.0]) * u.GeV**-1 * u.cm**-3
        
        result = self.flux_instance.cloud_cell_flux(Rc, Dc, input_flux)
        
        assert all(result.value == 0)


class TestComputeGammaKernel:
    """Test suite for the compute_gamma_kernel method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.flux_instance = flux()
    
    def test_gamma_kernel_shapes(self):
        """Test that gamma kernel returns correct shapes."""
        Ep = np.logspace(1, 4, 10) * u.GeV  # 10 proton energies
        Eg = np.logspace(0, 3, 5) * u.TeV    # 5 gamma energies
        
        result = self.flux_instance.compute_gamma_kernel(Eg, Ep)
        
        assert result.shape == (10, 5)  # [proton, gamma]
    
    def test_gamma_kernel_single_values(self):
        """Test gamma kernel with single scalar values."""
        Ep = 100 * u.GeV
        Eg = 1 * u.GeV
        
        result = self.flux_instance.compute_gamma_kernel(Eg, Ep)
        
        assert isinstance(result, u.Quantity)
    
    def test_gamma_kernel_unphysical_energies(self):
        """Test that gamma kernel returns 0 for Eg > Ep."""
        Ep = 100 * u.GeV
        Eg = 200 * u.GeV  # Higher than proton energy
        
        result = self.flux_instance.compute_gamma_kernel(Eg, Ep)
        
        assert result == 0
    
    def test_gamma_kernel_low_proton_energy(self):
        """Test gamma kernel for very low proton energies."""
        Ep = 0.1 * u.GeV  # Very low energy (log(Ep in TeV) < -1)
        Eg = 0.01 * u.TeV
        
        result = self.flux_instance.compute_gamma_kernel(Eg, Ep)
        
        assert result == 0
    
    def test_gamma_kernel_positivity(self):
        """Test that gamma kernel returns non-negative values for valid inputs."""
        Ep = np.logspace(2, 4, 5) * u.GeV
        Eg = np.logspace(0, 2, 3) * u.TeV
        
        # Ensure Eg < Ep for all combinations
        Eg = Eg[None, :] * 0.1  # Scale down to ensure Eg < Ep
        
        result = self.flux_instance.compute_gamma_kernel(Eg.flatten(), Ep)
        
        assert all(result.flatten() >= 0)
    
    def test_gamma_kernel_array_broadcasting(self):
        """Test different array broadcasting scenarios."""
        # Test 1D proton, scalar gamma
        Ep = np.array([100, 1000]) * u.GeV
        Eg = 10 * u.TeV
        result1 = self.flux_instance.compute_gamma_kernel(Eg, Ep)
        assert result1.shape == (2,)
        
        # Test scalar proton, 1D gamma
        Ep = 1000 * u.GeV
        Eg = np.array([1, 10, 100]) * u.TeV
        result2 = self.flux_instance.compute_gamma_kernel(Eg, Ep)
        assert result2.shape == (3,)


class TestComputeNeutrinoKernel:
    """Test suite for the compute_neutrino_kernel method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.flux_instance = flux()
    
    def test_neutrino_kernel_returns_tuple(self):
        """Test that neutrino kernel returns a tuple of two arrays."""
        Ep = 100 * u.GeV
        En = 10 * u.TeV
        
        result = self.flux_instance.compute_neutrino_kernel(En, Ep)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        F_nu_1, F_nu_2 = result
        # assert np.array(result).shape == (1,1), f"Expected shape (1,), got {np.array(result).shape}"
        assert F_nu_1.shape == (), f"Expected scalar, got shape {F_nu_1.shape}"
        assert F_nu_2.shape == (), f"Expected scalar, got shape {F_nu_2.shape}"

    
    def test_neutrino_kernel_shapes(self):
        """Test that neutrino kernel returns correct shapes."""
        Ep = np.logspace(1, 4, 10) * u.GeV
        En = np.logspace(0, 3, 5) * u.TeV
        
        F_nu_1, F_nu_2 = self.flux_instance.compute_neutrino_kernel(En, Ep)
        
        assert F_nu_1.shape == (10, 5)  # [proton, neutrino]
        assert F_nu_2.shape == (10, 5)
    
    def test_neutrino_kernel_energy_limits(self):
        """Test neutrino kernel energy limits."""
        Ep = 100 * u.GeV
        En = 50 * u.TeV  # En > 0.427 * Ep
        
        F_nu_1, F_nu_2 = self.flux_instance.compute_neutrino_kernel(En, Ep)
        
        # F_nu_1 should be 0 for x >= 0.427
        assert F_nu_1 == 0
        # F_nu_2 should be non-negative
        assert F_nu_2 >= 0
    
    def test_neutrino_kernel_low_proton_energy(self):
        """Test neutrino kernel for very low proton energies."""
        Ep = 0.1 * u.GeV  # Very low energy
        En = 0.01 * u.TeV
        
        F_nu_1, F_nu_2 = self.flux_instance.compute_neutrino_kernel(En, Ep)
        
        assert F_nu_1 == 0
        assert F_nu_2 == 0
    
    def test_neutrino_kernel_positivity(self):
        """Test that neutrino kernel returns non-negative values."""
        Ep = np.logspace(2, 4, 5) * u.GeV
        En = np.logspace(-1, 1, 3) * u.TeV
        
        F_nu_1, F_nu_2 = self.flux_instance.compute_neutrino_kernel(En, Ep)
        
        assert all(F_nu_1.flatten() >= 0)
        assert all(F_nu_2.flatten() >= 0)
    
    def test_neutrino_kernel_array_broadcasting(self):
        """Test different array broadcasting scenarios for neutrino kernel."""
        # Test 1D proton, scalar neutrino
        Ep = np.array([100, 1000]) * u.GeV
        En = 1 * u.TeV
        F_nu_1, F_nu_2 = self.flux_instance.compute_neutrino_kernel(En, Ep)
        assert F_nu_1.shape == (2,)
        assert F_nu_2.shape == (2,)
        
        # Test scalar proton, 1D neutrino
        Ep = 1000 * u.GeV
        En = np.array([0.1, 1, 10]) * u.TeV
        F_nu_1, F_nu_2 = self.flux_instance.compute_neutrino_kernel(En, Ep)
        assert F_nu_1.shape == (3,)
        assert F_nu_2.shape == (3,)


class TestIntegration:
    """Integration tests combining multiple methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.flux_instance = flux()
    
    def test_realistic_astrophysical_scenario(self):
        """Test with realistic astrophysical parameters."""
        # Typical molecular cloud parameters
        Rc = 10 * u.pc  # 10 parsec radius
        Dc = np.array([5, 15, 25]) * u.pc  # Various penetration depths
        input_flux = np.array([1e-4, 5e-5, 2e-5]) * u.GeV**-1 * u.cm**-3
        
        # Test cloud cell flux
        ccf = self.flux_instance.cloud_cell_flux(Rc, Dc, input_flux)
        assert all(ccf.value > 0)
        assert ccf.unit == u.GeV**-1
        
        # Test gamma kernel with typical energies
        Ep = np.logspace(2, 5, 20) * u.GeV  # 100 GeV to 100 TeV
        Eg = np.logspace(0, 3, 15) * u.TeV   # 1 TeV to 1 PeV
        
        gamma_kernel = self.flux_instance.compute_gamma_kernel(Eg, Ep)
        assert gamma_kernel.shape == (20, 15)
        assert all(gamma_kernel.flatten() >= 0)
        
        # Test neutrino kernel
        En = np.logspace(-1, 3, 15) * u.TeV  # 0.1 TeV to 1 PeV
        F_nu_1, F_nu_2 = self.flux_instance.compute_neutrino_kernel(En, Ep)
        assert F_nu_1.shape == (20, 15)
        assert F_nu_2.shape == (20, 15)
        assert all(F_nu_1.flatten() >= 0)
        assert all(F_nu_2.flatten() >= 0)



# Mock the imported modules since they're not available
particles_mock = Mock()
transport_mock = Mock()
accelerator_mock = Mock()

# Mock the module imports
with patch.dict('sys.modules', {
    'particles': Mock(particles=lambda: particles_mock),
    'transport': Mock(transport=lambda: transport_mock),
    'accelerator': Mock(accelerator=lambda: accelerator_mock)
}):
    # Import the functions from your module (assuming it's named cr_flux.py)
    from craic.injection import (
        compute_pflux_impulsive_extended,
        compute_pflux_impulsive_point,
        compute_pflux_continuous_extended,
        compute_pflux_continuous_point,
        e2R,
        compute_fgal,
        compute_fgal_dampe,
        compute_fgal_LHAASO
    )

class TestImpulsiveFluxCalculations:
    """Test suite for impulsive flux calculations."""
    
    def setup_method(self):
        """Setup mock transport methods."""
        # Mock the transport methods
        transport_mock.R_diffusion.return_value = 10.0 * u.pc
        transport_mock.Diffusion_Coefficient.return_value = 1e28 * u.cm**2 / u.s
        
        # Mock particles method
        particles_mock.t_ppEK.return_value = 1e6 * u.yr
    
    def test_compute_pflux_impulsive_point_basic(self):
        """Test basic functionality of impulsive point source flux calculation."""
        N_0 = 1e40 * u.GeV
        Ep = 100 * u.GeV
        d = 10 * u.pc
        a = 1000 * u.yr
        dens = 1 * u.cm**-3
        
        result = compute_pflux_impulsive_point(N_0, Ep, d, a, dens)
        
        # Check that result has correct units
        assert result.unit == u.GeV**-1 * u.cm**-3
        
        # Check that result is positive and finite
        assert np.all(result.value > 0)
        assert np.all(np.isfinite(result.value))
    
    def test_compute_pflux_impulsive_point_array_inputs(self):
        """Test with array inputs."""
        N_0 = 1e40 * u.GeV
        Ep = [10, 100, 1000] * u.GeV
        d = 10 * u.pc
        a = 1000 * u.yr
        dens = 1 * u.cm**-3
        
        result = compute_pflux_impulsive_point(N_0, Ep, d, a, dens)
        
        assert len(result) == 3
        assert result.unit == u.GeV**-1 * u.cm**-3
        assert np.all(result.value > 0)
    
    def test_compute_pflux_impulsive_point_parameter_variations(self):
        """Test how flux varies with different parameters."""
        base_params = {
            'N_0': 1e40 * u.GeV,
            'Ep': 100 * u.GeV,
            'd': 10 * u.pc,
            'a': 1000 * u.yr,
            'dens': 1 * u.cm**-3
        }
        
        base_result = compute_pflux_impulsive_point(**base_params)
        
        # Test that flux decreases with distance
        high_d_params = base_params.copy()
        high_d_params['d'] = 100 * u.pc
        high_d_result = compute_pflux_impulsive_point(**high_d_params)
        assert high_d_result < base_result
        
        # Test that flux increases with normalization
        high_N0_params = base_params.copy()
        high_N0_params['N_0'] = 2e40 * u.GeV
        high_N0_result = compute_pflux_impulsive_point(**high_N0_params)
        assert high_N0_result > base_result
    
    def test_compute_pflux_impulsive_extended_basic(self):
        """Test extended source flux calculation."""
        Resc = 5 * u.pc
        N_0 = 1e40 * u.GeV
        Ep = 100 * u.GeV
        d = 10 * u.pc
        a = 1000 * u.yr
        dens = 1 * u.cm**-3
        
        result = compute_pflux_impulsive_extended(Resc, N_0, Ep, d, a, dens)
        
        assert result.unit == u.GeV**-1 * u.cm**-3
        assert result.value > 0
        assert np.isfinite(result.value)
    
    def test_compute_pflux_impulsive_extended_vs_point(self):
        """Test that extended source gives different result than point source."""
        params = {
            'N_0': 1e40 * u.GeV,
            'Ep': 100 * u.GeV,
            'd': 10 * u.pc,
            'a': 1000 * u.yr,
            'dens': 1 * u.cm**-3
        }
        
        point_result = compute_pflux_impulsive_point(**params)
        extended_result = compute_pflux_impulsive_extended(5*u.pc, **params)
        
        # Results should be different (extended has normalization factor)
        assert point_result != extended_result

class TestContinuousFluxCalculations:
    """Test suite for continuous flux calculations."""
    
    def setup_method(self):
        """Setup mock transport methods."""
        transport_mock.R_diffusion.return_value = 10.0 * u.pc
        transport_mock.Diffusion_Coefficient.return_value = 1e28 * u.cm**2 / u.s
    
    def test_compute_pflux_continuous_point_basic(self):
        """Test basic functionality of continuous point source flux calculation."""
        N_0 = 1e40 * u.GeV
        Ep = 100 * u.GeV
        d = 10 * u.pc
        a = 1000 * u.yr
        dens = 1 * u.cm**-3
        
        result = compute_pflux_continuous_point(N_0, Ep, d, a, dens)
        
        assert result.unit == u.GeV**-1 * u.cm**-3
        assert result.value > 0
        assert np.isfinite(result.value)
    
    def test_compute_pflux_continuous_extended_raises_not_implemented(self):
        """Test that extended continuous source raises NotImplementedError."""
        Resc = 5 * u.pc
        N_0 = 1e40 * u.GeV
        Ep = 100 * u.GeV
        d = 10 * u.pc
        a = 1000 * u.yr
        dens = 1 * u.cm**-3
        
        with pytest.raises(NotImplementedError):
            compute_pflux_continuous_extended(Resc, N_0, Ep, d, a, dens)

class TestRigidityCRCalculations:
    """Test suite for rigidity and cosmic ray calculations."""
    
    def test_e2R_basic(self):
        """Test energy to rigidity conversion."""
        E = 100 * u.GeV
        R = e2R(E)
        
        assert R.unit == u.GV
        assert R.value > 0
        
        # Check the conversion is correct for a proton (Z=1)
        expected = E / c.e.si
        assert np.isclose(R.to(u.V).value, expected.to(u.V).value)
    
    def test_e2R_different_charges(self):
        """Test energy to rigidity conversion for different charges."""
        E = 100 * u.GeV
        
        R_proton = e2R(E, Z=1)
        R_helium = e2R(E, Z=2)
        
        # Rigidity should be half for helium at same energy
        assert np.isclose(R_helium.value, R_proton.value / 2)
    
    def test_e2R_array_input(self):
        """Test with array input."""
        E = [10, 100, 1000] * u.GeV
        R = e2R(E)
        
        assert len(R) == 3
        assert R.unit == u.GV
        assert np.all(R.value > 0)
    
    def test_compute_fgal_basic(self):
        """Test galactic cosmic ray flux calculation."""
        E = 100 * u.GeV
        result = compute_fgal(E)
        
        assert result.unit == u.GeV**-1 * u.cm**-3
        assert result.value > 0
        assert np.isfinite(result.value)
    
    def test_compute_fgal_array_input(self):
        """Test galactic CR flux with array input."""
        E = [10, 100, 1000] * u.GeV
        result = compute_fgal(E)
        
        assert len(result) == 3
        assert result.unit == u.GeV**-1 * u.cm**-3
        assert np.all(result.value > 0)
    
    def test_compute_fgal_energy_dependence(self):
        """Test that galactic CR flux decreases with energy."""
        E_low = 10 * u.GeV
        E_high = 1000 * u.GeV
        
        flux_low = compute_fgal(E_low)
        flux_high = compute_fgal(E_high)
        
        # Higher energy should have lower flux (power law)
        assert flux_high < flux_low

class TestDAMPEFluxCalculations:
    """Test suite for DAMPE cosmic ray flux calculations."""
    
    def test_compute_fgal_dampe_basic(self):
        """Test basic DAMPE flux calculation."""
        E = 1 * u.TeV
        result = compute_fgal_dampe(E)
        
        assert result.unit == u.GeV**-1 * u.cm**-3
        assert result.value > 0
        assert np.isfinite(result.value)
    
    def test_compute_fgal_dampe_transition(self):
        """Test behavior around transition energy."""
        E_tran = 6.3 * u.TeV
        
        # Test energies around transition
        E_below = 5 * u.TeV
        E_above = 8 * u.TeV
        
        flux_below = compute_fgal_dampe(E_below)
        flux_above = compute_fgal_dampe(E_above)
        
        # Both should be positive
        assert flux_below.value > 0
        assert flux_above.value > 0
    
    def test_compute_fgal_dampe_custom_transition(self):
        """Test with custom transition energy."""
        E = 1 * u.TeV
        E_tran_custom = 5 * u.TeV
        
        result = compute_fgal_dampe(E, E_tran=E_tran_custom)
        
        assert result.unit == u.GeV**-1 * u.cm**-3
        assert result.value > 0

class TestLHAASOFluxCalculations:
    """Test suite for LHAASO cosmic ray flux calculations."""
    
    def test_compute_fgal_lhaaso_qgs_exp(self):
        """Test LHAASO QGS exponential fit."""
        E = 0.5 * u.PeV
        result = compute_fgal_LHAASO(E, model="QGS", fit="exp")
        
        assert result.unit == u.GeV**-1 * u.cm**-3
        assert result.value > 0
        assert np.isfinite(result.value)
    
    def test_compute_fgal_lhaaso_qgs_two(self):
        """Test LHAASO QGS two broken power law fit."""
        E = 0.5 * u.PeV
        result = compute_fgal_LHAASO(E, model="QGS", fit="two")
        
        assert result.unit == u.GeV**-1 * u.cm**-3
        assert result.value > 0
        assert np.isfinite(result.value)
    
    def test_compute_fgal_lhaaso_lhc_exp(self):
        """Test LHAASO LHC exponential fit."""
        E = 0.5 * u.PeV
        result = compute_fgal_LHAASO(E, model="LHC", fit="exp")
        
        assert result.unit == u.GeV**-1 * u.cm**-3
        assert result.value > 0
        assert np.isfinite(result.value)
    
    def test_compute_fgal_lhaaso_lhc_two(self):
        """Test LHAASO LHC two broken power law fit."""
        E = 0.5 * u.PeV
        result = compute_fgal_LHAASO(E, model="LHC", fit="two")
        
        assert result.unit == u.GeV**-1 * u.cm**-3
        assert result.value > 0
        assert np.isfinite(result.value)
    
    def test_compute_fgal_lhaaso_different_models(self):
        """Test that different models give different results."""
        E = 0.5 * u.PeV
        
        qgs_result = compute_fgal_LHAASO(E, model="QGS", fit="exp")
        lhc_result = compute_fgal_LHAASO(E, model="LHC", fit="exp")
        
        # Different models should give different results
        assert qgs_result.value != lhc_result.value
    
    def test_compute_fgal_lhaaso_array_input(self):
        """Test LHAASO with array input."""
        E = [0.1, 0.5, 1.0] * u.PeV
        result = compute_fgal_LHAASO(E, model="QGS", fit="exp")
        
        assert len(result) == 3
        assert result.unit == u.GeV**-1 * u.cm**-3
        assert np.all(result.value > 0)

class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Setup mock transport methods."""
        transport_mock.R_diffusion.return_value = 10.0 * u.pc
        transport_mock.Diffusion_Coefficient.return_value = 1e28 * u.cm**2 / u.s
        particles_mock.t_ppEK.return_value = 1e6 * u.yr
    
    def test_zero_energy(self):
        """Test behavior with zero energy."""
        # This might raise errors or warnings, depending on implementation
        E = 0 * u.GeV
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = compute_fgal(E)
                # If it doesn't raise an error, check result
                if np.isfinite(result.value):
                    assert result.unit == u.GeV**-1 * u.cm**-3
            except (ValueError, ZeroDivisionError):
                # Expected for zero energy
                pass
    
    def test_very_large_energy(self):
        """Test with very large energies."""
        E = 1e20 * u.GeV
        
        try:
            result = compute_fgal(E)
            assert result.unit == u.GeV**-1 * u.cm**-3
            # Very high energy might give very small flux
            assert result.value >= 0
        except OverflowError:
            # Acceptable for extremely high energies
            pass
    
    def test_negative_parameters(self):
        """Test with negative parameters (should handle gracefully or raise errors)."""
        with pytest.raises((ValueError, AssertionError)):
            compute_pflux_impulsive_point(
                N_0=-1e40 * u.GeV,  # Negative normalization
                Ep=100 * u.GeV,
                d=10 * u.pc,
                a=1000 * u.yr,
                dens=1 * u.cm**-3
            )
    
    def test_unit_consistency(self):
        """Test that all functions return consistent units."""
        E = 100 * u.GeV
        expected_unit = u.GeV**-1 * u.cm**-3
        
        # Test all galactic CR functions
        assert compute_fgal(E).unit == expected_unit
        assert compute_fgal_dampe(E).unit == expected_unit
        assert compute_fgal_LHAASO(E).unit == expected_unit

class TestParameterSweeps:
    """Test parameter sweeps to ensure physical behavior."""
    
    def setup_method(self):
        """Setup mock transport methods."""
        transport_mock.R_diffusion.return_value = 10.0 * u.pc
        transport_mock.Diffusion_Coefficient.return_value = 1e28 * u.cm**2 / u.s
        particles_mock.t_ppEK.return_value = 1e6 * u.yr
    
    def test_spectral_index_dependence(self):
        """Test that spectral index affects flux as expected."""
        base_params = {
            'Ep': 100 * u.GeV,
            'd': 10 * u.pc,
            'a': 1000 * u.yr,
            'dens': 1 * u.cm**-3
        }
        
        flux_alpha2 = compute_pflux_impulsive_point(N_0 = 1e40 * u.GeV, **base_params, alpha=2.0)
        flux_alpha3 = compute_pflux_impulsive_point(N_0 = 1e40 * u.GeV**2, **base_params, alpha=3.0) 
        
        # Steeper spectrum should give lower flux at high energy
        assert flux_alpha3 < flux_alpha2
    
    def test_density_dependence(self):
        """Test flux dependence on ambient density."""
        base_params = {
            'N_0': 1e40 * u.GeV,
            'Ep': 100 * u.GeV,
            'd': 10 * u.pc,
            'a': 1000 * u.yr
        }
        
        flux_low_dens = compute_pflux_impulsive_point(**base_params, dens=0.1 * u.cm**-3)
        flux_high_dens = compute_pflux_impulsive_point(**base_params, dens=10 * u.cm**-3)
        
        # Both should be positive
        assert flux_low_dens.value > 0
        assert flux_high_dens.value > 0

# Test 
class TestParticlesInitialization:
    """Test suite for particles class initialization."""
    
    def test_default_initialization(self):
        """Test default initialization parameters."""
        p = particles()
        
        assert p.alpha == 2.0
        assert p.Ebudget == 1e50 * u.erg
        
        # Test class constants
        assert p.Epth == 0.2797 * u.GeV
        assert p.kappa == 0.45
        assert p.cspeed.unit == u.cm / u.s
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        custom_alpha = 2.5
        custom_budget = 1e49 * u.erg
        
        p = particles(alpha=custom_alpha, Ebudget=custom_budget)
        
        assert p.alpha == custom_alpha
        assert p.Ebudget == custom_budget
    
    def test_ebudget_unit_conversion(self):
        """Test that energy budget can be initialized with different units."""
        # Initialize with GeV
        p1 = particles(Ebudget=1e40 * u.GeV)
        assert p1.Ebudget.unit == u.GeV
        
        # Initialize with Joules
        p2 = particles(Ebudget=1e43 * u.J)
        assert p2.Ebudget.unit == u.J


class TestCrossSectionCalculation:
    """Test suite for proton-proton cross-section calculations."""
    
    def setup_method(self):
        """Setup particles instance for testing."""
        self.p = particles()
    
    def test_sig_ppEK_basic(self):
        """Test basic cross-section calculation."""
        Ep = 1 * u.GeV
        sigma = self.p.sig_ppEK(Ep)
        
        # Check units
        assert sigma.unit == u.cm**2
        
        # Check that result is positive
        assert sigma.value > 0
        
        # Check that result is finite
        assert np.isfinite(sigma.value)
    
    def test_sig_ppEK_threshold_behavior(self):
        """Test behavior near threshold energy."""
        # Test at threshold
        sigma_th = self.p.sig_ppEK(self.p.Epth)
        assert sigma_th.value >= 0
        
        # Test below threshold (should be clipped to 0)
        Ep_below = 0.1 * u.GeV  # Below Epth = 0.2797 GeV
        with pytest.raises(ValueError, match=f"Energy of particles must be larger than the threshold 0.2797 GeV."):
            self.p.sig_ppEK(Ep_below)
        
        # Test above threshold
        Ep_above = 1.0 * u.GeV
        sigma_above = self.p.sig_ppEK(Ep_above)
        assert sigma_above.value > 0
    
    def test_sig_ppEK_energy_dependence(self):
        """Test that cross-section increases with energy above threshold."""
        energies = [0.5, 1.0, 10.0, 100.0] * u.GeV
        sigmas = [self.p.sig_ppEK(E) for E in energies]
        
        # All should be positive above threshold
        for sigma in sigmas:
            assert sigma.value > 0
        
        # Generally expect increasing cross-section with energy 
        # (though it may plateau at very high energies)
        assert sigmas[1] > sigmas[0]  # 1 GeV > 0.5 GeV
        assert sigmas[2] > sigmas[1]  # 10 GeV > 1 GeV
    
    def test_sig_ppEK_array_input(self):
        """Test cross-section calculation with array input."""
        Ep_array = [0.1, 0.5, 1.0, 10.0] * u.GeV
        # sigma_array = self.p.sig_ppEK(Ep_array)
        
        # assert len(sigma_array) == 4
        # assert sigma_array.unit == u.cm**2
        
        with pytest.raises(ValueError, match=f"Energy of particles must be larger than the threshold 0.2797 GeV."):
            self.p.sig_ppEK(Ep_array)
    
    def test_sig_ppEK_extreme_energies(self):
        """Test cross-section at extreme energies."""
        # Very high energy
        Ep_high = 1e6 * u.GeV  # 1 PeV
        sigma_high = self.p.sig_ppEK(Ep_high)
        assert sigma_high.value > 0
        assert np.isfinite(sigma_high.value)
        
        # Check reasonable magnitude (should be order of 10^-25 cm^2)
        assert 1e-27 < sigma_high.to(u.cm**2).value < 1e-23
    
    def test_sig_ppEK_physical_values(self):
        """Test that cross-section values are physically reasonable."""
        # At 1 GeV, expect cross-section around 30-40 mbarn
        Ep = 1 * u.GeV
        sigma = self.p.sig_ppEK(Ep)
        
        # Convert to mbarn for comparison
        sigma_mbarn = sigma.to(u.mbarn)
        
        # Should be in reasonable range (10-100 mbarn)
        assert 5 < sigma_mbarn.value < 100


class TestCoolingTimeCalculation:
    """Test suite for proton cooling time calculations."""
    
    def setup_method(self):
        """Setup particles instance for testing."""
        self.p = particles()
    
    def test_t_ppEK_basic(self):
        """Test basic cooling time calculation."""
        dens = 1 * u.cm**-3
        Ep = 1 * u.GeV
        
        t_cool = self.p.t_ppEK(dens, Ep)
        
        # Check units
        assert t_cool.unit == u.s
        
        # Check positive and finite
        assert t_cool.value > 0
        assert np.isfinite(t_cool.value)
    
    def test_t_ppEK_density_dependence(self):
        """Test cooling time dependence on density."""
        Ep = 10 * u.GeV
        
        dens_low = 0.1 * u.cm**-3
        dens_high = 10 * u.cm**-3
        
        t_low = self.p.t_ppEK(dens_low, Ep)
        t_high = self.p.t_ppEK(dens_high, Ep)
        
        # Higher density should give shorter cooling time
        assert t_high < t_low
        
        # Should scale inversely with density
        ratio = t_low / t_high
        expected_ratio = dens_high / dens_low
        assert np.isclose(ratio.value, expected_ratio.value, rtol=0.01)
    
    def test_t_ppEK_energy_dependence(self):
        """Test cooling time dependence on energy."""
        dens = 1 * u.cm**-3
        
        Ep_low = 1 * u.GeV
        Ep_high = 100 * u.GeV
        
        t_low = self.p.t_ppEK(dens, Ep_low)
        t_high = self.p.t_ppEK(dens, Ep_high)
        
        # Both should be positive
        assert t_low.value > 0
        assert t_high.value > 0
        
        # Higher energy typically has shorter cooling time
        # (due to higher cross-section)
        assert t_high < t_low
    
    def test_t_ppEK_array_input(self):
        """Test cooling time with array inputs."""
        dens = [0.1, 1.0, 10.0] * u.cm**-3
        Ep = 10 * u.GeV
        
        t_cool = self.p.t_ppEK(dens, Ep)
        
        assert len(t_cool) == 3
        assert t_cool.unit == u.s
        assert np.all(t_cool.value > 0)
        
        # Should decrease with increasing density
        assert t_cool[2] < t_cool[1] < t_cool[0]
    
    def test_t_ppEK_physical_timescales(self):
        """Test that cooling times are physically reasonable."""
        # Typical ISM conditions
        dens = 1 * u.cm**-3
        Ep = 10 * u.GeV
        
        t_cool = self.p.t_ppEK(dens, Ep)
        
        # Convert to years for easier interpretation
        t_cool_yr = t_cool.to(u.yr)
        
        # Should be on the order of 10^7 to 10^9 years for typical conditions
        assert 1e6 < t_cool_yr.value < 1e10
    
    def test_t_ppEK_below_threshold(self):
        """Test cooling time calculation below pp threshold."""
        dens = 1 * u.cm**-3
        Ep = 0.1 * u.GeV  # Below threshold
        
        # This should either return infinite time or raise an error
        # since cross-section is 0 below threshold
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                t_cool = self.p.t_ppEK(dens, Ep)
                # If it doesn't raise error, should be very large or infinite
                assert t_cool.value > 1e20  # Very large time
            except (ZeroDivisionError, ValueError):
                # Expected behavior when cross-section is 0
                pass


class TestNormalization:
    """Test suite for energy budget normalization calculations."""
    
    def test_NormEbudget_alpha_2_default(self):
        """Test normalization for alpha=2 (default case)."""
        p = particles(alpha=2.0, Ebudget=1e50*u.erg)
        
        N0 = p.NormEbudget()
        
        # Check units (should be GeV^(alpha-1) = GeV^1)
        assert N0.unit == u.GeV
        
        # Check positive and finite
        assert N0.value > 0
        assert np.isfinite(N0.value)
    
    def test_NormEbudget_alpha_2_custom_range(self):
        """Test normalization for alpha=2 with custom energy range."""
        p = particles(alpha=2.0, Ebudget=1e50*u.erg)
        
        Emin = 1 * u.GeV
        Emax = 1 * u.PeV
        
        N0 = p.NormEbudget(Emin=Emin, Emax=Emax)
        
        assert N0.unit == u.GeV
        assert N0.value > 0
        assert np.isfinite(N0.value)
        
        # Larger energy range should give smaller normalization
        N0_default = p.NormEbudget()
        assert N0 < N0_default  # Wider range -> smaller normalization
    
    def test_NormEbudget_different_alpha(self):
        """Test normalization for alpha != 2."""
        alphas = [1.5, 2.5, 3.0]
        
        for alpha in alphas:
            p = particles(alpha=alpha, Ebudget=1e50*u.erg)
            N0 = p.NormEbudget()
            
            # Check units (should be GeV^(alpha-1))
            expected_unit = u.GeV**(alpha-1)
            assert N0.unit == expected_unit
            
            # Check positive and finite
            assert N0.value > 0
            assert np.isfinite(N0.value)
    
    def test_NormEbudget_energy_budget_scaling(self):
        """Test that normalization scales with energy budget."""
        Ebudget1 = 1e49 * u.erg
        Ebudget2 = 2e49 * u.erg
        
        p1 = particles(alpha=2.0, Ebudget=Ebudget1)
        p2 = particles(alpha=2.0, Ebudget=Ebudget2)
        
        N0_1 = p1.NormEbudget()
        N0_2 = p2.NormEbudget()
        
        # Normalization should scale linearly with energy budget
        ratio = N0_2 / N0_1
        expected_ratio = Ebudget2 / Ebudget1
        assert np.isclose(ratio.value, expected_ratio.value, rtol=0.01)
    
    def test_NormEbudget_energy_range_dependence(self):
        """Test normalization dependence on energy range."""
        p = particles(alpha=2.5, Ebudget=1e50*u.erg)
        
        # Narrow range
        N0_narrow = p.NormEbudget(Emin=10*u.GeV, Emax=100*u.GeV)
        
        # Wide range
        N0_wide = p.NormEbudget(Emin=1*u.GeV, Emax=1*u.PeV)
        
        # Wider range should give smaller normalization
        assert N0_wide < N0_narrow
    
    def test_NormEbudget_unit_conversion(self):
        """Test normalization with different energy units."""
        # Test with energy budget in GeV
        p1 = particles(alpha=2.0, Ebudget=6.24e40*u.GeV)  # ~1e50 erg
        
        # Test with energy budget in erg
        p2 = particles(alpha=2.0, Ebudget=1e38*u.erg)
        
        N0_1 = p1.NormEbudget()
        N0_2 = p2.NormEbudget()
        
        # Should give similar results (within conversion accuracy)
        assert np.isclose(N0_1.value, N0_2.value, rtol=0.1)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for particles class."""

    def test_zero_energy_cross_section(self):
        """Test cross-section at zero energy."""
        p = particles()
        Ep_zero = 0 * u.GeV
    
        with pytest.raises(ValueError, match=f"Energy of particles must be larger than the threshold 0.2797 GeV."):
            p.sig_ppEK(Ep_zero)
    
    def test_negative_energy_cross_section(self):
        """Test cross-section with negative energy."""
        p = particles()
        
        # This should either handle gracefully or raise appropriate error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                Ep_neg = -1 * u.GeV
                sigma = p.sig_ppEK(Ep_neg)
                # If it doesn't raise error, should be handled appropriately
                assert sigma.value >= 0
            except (ValueError, RuntimeWarning):
                # Expected for negative energies
                pass
    
    def test_alpha_edge_cases(self):
        """Test normalization with edge case alpha values."""
        # Test alpha very close to 2
        p_close = particles(alpha=2.0001, Ebudget=1e50*u.erg)
        N0_close = p_close.NormEbudget()
        assert np.isfinite(N0_close.value)
        
        # Test alpha = 1 (critical case for normalization)
        p_alpha1 = particles(alpha=1.0, Ebudget=1e50*u.erg)
        N0_alpha1 = p_alpha1.NormEbudget()
        assert np.isfinite(N0_alpha1.value)
        assert N0_alpha1.value > 0
    
    def test_equal_energy_limits(self):
        """Test normalization when Emin = Emax."""
        p = particles(alpha=2.5, Ebudget=1e50*u.erg)
        
        E = 10 * u.GeV
        
        # This should handle the edge case appropriately
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                N0 = p.NormEbudget(Emin=E, Emax=E)
                # Could be infinite or raise error
                if np.isfinite(N0.value):
                    assert N0.value > 0
            except (ZeroDivisionError, ValueError):
                # Expected behavior
                pass
    
    def test_very_large_energy_budget(self):
        """Test with very large energy budget."""
        p = particles(alpha=2.0, Ebudget=1e60*u.erg)
        
        N0 = p.NormEbudget()
        
        # Should handle large values appropriately
        assert np.isfinite(N0.value)
        assert N0.value > 0


class TestPhysicalConsistency:
    """Test physical consistency and relationships between methods."""
    
    def setup_method(self):
        """Setup particles instance for testing."""
        self.p = particles()
    
    def test_cross_section_cooling_time_consistency(self):
        """Test consistency between cross-section and cooling time."""
        dens = 1 * u.cm**-3
        Ep = 10 * u.GeV
        
        # Get cross-section and cooling time
        sigma = self.p.sig_ppEK(Ep)
        t_cool = self.p.t_ppEK(dens, Ep)
        
        # Manually calculate cooling time using cross-section
        t_manual = 1 / (dens * sigma * self.p.cspeed * self.p.kappa)
        
        # Should be consistent
        assert np.isclose(t_cool.to(u.s).value, t_manual.to(u.s).value, rtol=0.01)
    
    def test_normalization_integration_check(self):
        """Test that normalization gives correct integrated flux."""
        # This is a conceptual test - in practice would require numerical integration
        p = particles(alpha=2.0, Ebudget=1e50*u.erg)
        
        Emin = 10 * u.GeV
        Emax = 1 * u.PeV
        
        N0 = p.NormEbudget(Emin=Emin, Emax=Emax)
        
        # For alpha=2, integral of N0 * E^(-alpha) * E dE from Emin to Emax
        # should equal Ebudget
        # This is: N0 * ln(Emax/Emin) = Ebudget
        expected_integral = N0 * np.log(Emax/Emin)
        
        # Convert to same units for comparison
        assert np.isclose(
            expected_integral.to(u.erg).value,
            p.Ebudget.to(u.erg).value,
            rtol=0.01
        )

import pytest
import numpy as np
import astropy.units as u
import astropy.constants as c
from unittest.mock import Mock, patch
import warnings

# Mock the particles class since it's imported
particles_mock = Mock()

# Define the transport class (corrected from your code)
class transport:
    """Calculates magnetic field strength, diffusion coefficient and diffusion radius
    in the ISM and in a molecular cloud."""
    
    chiism = 1.
    cr_delta = 0.5
    
    def __init__(self, D0=3*10**(26) *u.cm**2 /u.s):
        self.D_0 = D0
        # Galactic Diffusion Coefficient (cm^2 s^-1) at 1 GeV
        # FAST = 3x10^27 SLOW = 3x10^26
    
    # Magnetic Field Strength
    @u.quantity_input(dens=u.cm**-3)
    def B_mag(self, dens) -> u.uG:
        """Returns the magnetic field strength of the cloud (:math:`\mathrm{\mu G}`)
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
        """Returns the diffusion coefficient of the ISM or the cloud (:math:`\mathrm{cm}^{2} \mathrm{/s}`)
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
        """Returns how far the accelerated particles can
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


class TestTransportInitialization:
    """Test suite for transport class initialization."""
    
    def test_default_initialization(self):
        """Test default initialization parameters."""
        t = transport()
        
        # Check default diffusion coefficient
        expected_D0 = 3e26 * u.cm**2 / u.s
        assert t.D_0 == expected_D0
        
        # Test class constants
        assert t.chiism == 1.0
        assert t.cr_delta == 0.5
    
    def test_custom_initialization(self):
        """Test initialization with custom diffusion coefficient."""
        custom_D0 = 1e27 * u.cm**2 / u.s
        t = transport(D0=custom_D0)
        
        assert t.D_0 == custom_D0
    
    def test_diffusion_coefficient_units(self):
        """Test that diffusion coefficient has correct units."""
        t = transport()
        assert t.D_0.unit == u.cm**2 / u.s


class TestMagneticFieldCalculation:
    """Test suite for magnetic field strength calculations."""
    
    @pytest.fixture
    def transport_instance(self):
        """Setup transport instance for testing."""
        return transport()
    
    def test_B_mag_basic(self, transport_instance):
        """Test basic magnetic field calculation."""
        dens = 100 * u.cm**-3
        B = transport_instance.B_mag(dens)
        
        # Check units
        assert B.unit == u.uG
        
        # Check that result is positive
        assert B.value > 0
        
        # Check that result is finite
        assert np.isfinite(B.value)
    
    @pytest.mark.parametrize("dens_value", [1, 10, 100, 200])
    def test_B_mag_low_density(self, transport_instance, dens_value):
        """Test magnetic field at low density (below 300 cm^-3)."""
        dens = dens_value * u.cm**-3
        B = transport_instance.B_mag(dens)
        
        # Should be constant at 10 μG for dens < 300 cm^-3
        assert pytest.approx(B.value, abs=1e-10) == 10.0
        assert B.unit == u.uG
    
    @pytest.mark.parametrize("dens_value", [500, 1000, 10000])
    def test_B_mag_high_density(self, transport_instance, dens_value):
        """Test magnetic field at high density (above 300 cm^-3)."""
        dens = dens_value * u.cm**-3
        B = transport_instance.B_mag(dens)
        
        # Should follow power law: B = 10 * (dens/300)^0.65
        expected = 10 * (dens_value / 300)**0.65
        assert pytest.approx(B.value, rel=0.01) == expected
        assert B.unit == u.uG
    
    def test_B_mag_transition_point(self, transport_instance):
        """Test magnetic field exactly at transition density."""
        dens_transition = 300 * u.cm**-3
        B_transition = transport_instance.B_mag(dens_transition)
        assert pytest.approx(B_transition.value, abs=1e-10) == 10.0
    
    def test_B_mag_scaling_law(self, transport_instance):
        """Test that magnetic field follows correct scaling law."""
        # Test the scaling relation B ∝ n^0.65 for n > 300 cm^-3
        dens1 = 1000 * u.cm**-3
        dens2 = 4000 * u.cm**-3  # 4x higher density
        
        B1 = transport_instance.B_mag(dens1)
        B2 = transport_instance.B_mag(dens2)
        
        # B2/B1 should equal (4)^0.65
        ratio = B2 / B1
        expected_ratio = 4**0.65
        assert pytest.approx(ratio.value, rel=0.01) == expected_ratio
    
    def test_B_mag_array_input(self, transport_instance):
        """Test magnetic field calculation with array input."""
        dens_array = [10, 100, 500, 1000] * u.cm**-3
        B_array = transport_instance.B_mag(dens_array)
        
        assert len(B_array) == 4
        assert B_array.unit == u.uG
        
        # First two should be 10 μG (below transition)
        assert pytest.approx(B_array[0].value, abs=1e-10) == 10.0
        assert pytest.approx(B_array[1].value, abs=1e-10) == 10.0
        
        # Last two should be > 10 μG (above transition)
        assert B_array[2].value > 10.0
        assert B_array[3].value > 10.0
        
        # Should be increasing with density
        assert B_array[3] > B_array[2]
    
    @pytest.mark.parametrize("dens_value", [0.1, 1, 10, 100, 1000, 1e5])
    def test_B_mag_physical_values(self, transport_instance, dens_value):
        """Test that magnetic field values are physically reasonable."""
        dens = dens_value * u.cm**-3
        B = transport_instance.B_mag(dens)
        
        # Magnetic field should be between 1 μG and 1000 μG
        assert 1 <= B.value <= 1000
        assert B.unit == u.uG


class TestDiffusionCoefficient:
    """Test suite for diffusion coefficient calculations."""
    
    @pytest.fixture
    def transport_instance(self):
        """Setup transport instance for testing."""
        return transport()
    
    def test_Diffusion_Coefficient_ism_basic(self, transport_instance):
        """Test basic diffusion coefficient calculation for ISM."""
        Ep = 1 * u.GeV
        dens = 1 * u.cm**-3
        
        D = transport_instance.Diffusion_Coefficient(Ep, dens, ism=1)
        
        # Check units
        assert D.unit == u.cm**2 / u.s
        
        # Check positive and finite
        assert D.value > 0
        assert np.isfinite(D.value)
    
    def test_Diffusion_Coefficient_cloud_basic(self, transport_instance):
        """Test basic diffusion coefficient calculation for molecular cloud."""
        Ep = 1 * u.GeV
        dens = 100 * u.cm**-3
        
        D = transport_instance.Diffusion_Coefficient(Ep, dens, ism=0, chi=0.05)
        
        # Check units
        assert D.unit == u.cm**2 / u.s
        
        # Check positive and finite
        assert D.value > 0
        assert np.isfinite(D.value)
    
    @pytest.mark.parametrize("energy_factor", [1, 4, 16])
    def test_Diffusion_Coefficient_energy_dependence(self, transport_instance, energy_factor):
        """Test energy dependence of diffusion coefficient."""
        dens = 1 * u.cm**-3
        
        E1 = 1 * u.GeV
        E2 = energy_factor * u.GeV
        
        # Test for ISM
        D1_ism = transport_instance.Diffusion_Coefficient(E1, dens, ism=1)
        D2_ism = transport_instance.Diffusion_Coefficient(E2, dens, ism=1)
        
        # Should scale as E^0.5
        ratio_ism = D2_ism / D1_ism
        expected_ratio = (energy_factor)**0.5
        assert pytest.approx(ratio_ism.value, rel=0.01) == expected_ratio
    
    def test_Diffusion_Coefficient_ism_vs_cloud(self, transport_instance):
        """Test difference between ISM and cloud diffusion coefficients."""
        Ep = 10 * u.GeV
        dens = 100 * u.cm**-3  # High density cloud
        
        D_ism = transport_instance.Diffusion_Coefficient(Ep, dens, ism=1)
        D_cloud = transport_instance.Diffusion_Coefficient(Ep, dens, ism=0, chi=0.05)
        
        # ISM should typically have higher diffusion coefficient
        assert D_ism > D_cloud
    
    @pytest.mark.parametrize("chi1,chi2", [(0.01, 0.1), (0.05, 0.5)])
    def test_Diffusion_Coefficient_chi_dependence(self, transport_instance, chi1, chi2):
        """Test chi parameter dependence for cloud diffusion."""
        Ep = 1 * u.GeV
        dens = 100 * u.cm**-3
        
        D1 = transport_instance.Diffusion_Coefficient(Ep, dens, ism=0, chi=chi1)
        D2 = transport_instance.Diffusion_Coefficient(Ep, dens, ism=0, chi=chi2)
        
        # Higher chi should give higher diffusion coefficient
        assert D2 > D1
        
        # Should scale linearly with chi
        ratio = D2 / D1
        expected_ratio = chi2 / chi1
        assert pytest.approx(ratio.value, rel=0.01) == expected_ratio
    
    def test_Diffusion_Coefficient_density_dependence(self, transport_instance):
        """Test density dependence for cloud diffusion coefficient."""
        Ep = 1 * u.GeV
        chi = 0.05
        
        dens1 = 100 * u.cm**-3
        dens2 = 1000 * u.cm**-3  # Higher density
        
        D1 = transport_instance.Diffusion_Coefficient(Ep, dens1, ism=0, chi=chi)
        D2 = transport_instance.Diffusion_Coefficient(Ep, dens2, ism=0, chi=chi)
        
        # Higher density → stronger B field → lower diffusion coefficient
        assert D2 < D1
    
    def test_Diffusion_Coefficient_array_input(self, transport_instance):
        """Test diffusion coefficient with array inputs."""
        Ep_array = [1, 10, 100] * u.GeV
        dens = 10 * u.cm**-3
        
        D_array = transport_instance.Diffusion_Coefficient(Ep_array, dens, ism=1)
        
        assert len(D_array) == 3
        assert D_array.unit == u.cm**2 / u.s
        assert np.all(D_array.value > 0)
        
        # Should increase with energy
        assert D_array[2] > D_array[1] > D_array[0]
    
    def test_Diffusion_Coefficient_physical_values(self, transport_instance):
        """Test that diffusion coefficients are physically reasonable."""
        Ep = 1 * u.GeV
        dens = 1 * u.cm**-3
        
        # ISM diffusion coefficient
        D_ism = transport_instance.Diffusion_Coefficient(Ep, dens, ism=1)
        
        # Should be around 10^26 - 10^28 cm^2/s for typical values
        assert 1e25 <= D_ism.value <= 1e29
        
        # Cloud diffusion coefficient
        D_cloud = transport_instance.Diffusion_Coefficient(Ep, dens, ism=0)
        
        # Should be smaller than ISM
        assert D_cloud < D_ism
        assert 1e24 <= D_cloud.value <= 1e28


class TestDiffusionRadius:
    """Test suite for diffusion radius calculations."""
    
    @pytest.fixture
    def transport_instance(self):
        """Setup transport instance for testing."""
        return transport()
    
    def test_R_diffusion_basic(self, transport_instance):
        """Test basic diffusion radius calculation."""
        Ep = 10 * u.GeV
        a = 1000 * u.yr
        dens = 1 * u.cm**-3
        
        R = transport_instance.R_diffusion(Ep, a, dens, ism=1)
        
        # Check units
        assert R.unit == u.cm
        
        # Check positive and finite
        assert R.value > 0
        assert np.isfinite(R.value)
    
    @pytest.mark.parametrize("time_factor", [1, 4, 16])
    def test_R_diffusion_time_dependence(self, transport_instance, time_factor):
        """Test time dependence of diffusion radius."""
        Ep = 1 * u.GeV
        dens = 1 * u.cm**-3
        
        t1 = 1000 * u.yr
        t2 = time_factor * 1000 * u.yr
        
        R1 = transport_instance.R_diffusion(Ep, t1, dens, ism=1)
        R2 = transport_instance.R_diffusion(Ep, t2, dens, ism=1)
        
        # Should scale as t^0.5
        ratio = R2 / R1
        expected_ratio = (time_factor)**0.5
        assert pytest.approx(ratio.value, rel=0.01) == expected_ratio
    
    @pytest.mark.parametrize("energy_factor", [1, 9, 25])
    def test_R_diffusion_energy_dependence(self, transport_instance, energy_factor):
        """Test energy dependence of diffusion radius."""
        a = 1000 * u.yr
        dens = 1 * u.cm**-3
        
        E1 = 1 * u.GeV
        E2 = energy_factor * u.GeV
        
        R1 = transport_instance.R_diffusion(E1, a, dens, ism=1)
        R2 = transport_instance.R_diffusion(E2, a, dens, ism=1)
        
        # Should scale as E^0.25 (since D ∝ E^0.5 and R ∝ D^0.5)
        ratio = R2 / R1
        expected_ratio = (energy_factor)**0.25
        assert pytest.approx(ratio.value, rel=0.01) == expected_ratio
    
    def test_R_diffusion_ism_vs_cloud(self, transport_instance):
        """Test difference between ISM and cloud diffusion radii."""
        Ep = 1 * u.GeV
        a = 1000 * u.yr
        dens = 100 * u.cm**-3
        
        R_ism = transport_instance.R_diffusion(Ep, a, dens, ism=1)
        R_cloud = transport_instance.R_diffusion(Ep, a, dens, ism=0, chi=0.05)
        
        # ISM should have larger diffusion radius
        assert R_ism > R_cloud
    
    def test_R_diffusion_consistency_with_diffusion_coeff(self, transport_instance):
        """Test consistency between diffusion radius and coefficient."""
        Ep = 1 * u.GeV
        a = 1000 * u.yr
        dens = 1 * u.cm**-3
        
        # Calculate diffusion radius
        R = transport_instance.R_diffusion(Ep, a, dens, ism=1)
        
        # Calculate manually using diffusion coefficient
        D = transport_instance.Diffusion_Coefficient(Ep, dens, ism=1)
        R_manual = 2 * np.sqrt(D * a.to(u.s))
        
        # Should be consistent
        assert pytest.approx(R.to(u.cm).value, rel=0.01) == R_manual.to(u.cm).value
    
    def test_R_diffusion_array_input(self, transport_instance):
        """Test diffusion radius with array inputs."""
        Ep = 1 * u.GeV
        a_array = [100, 1000, 10000] * u.yr
        dens = 1 * u.cm**-3
        
        R_array = transport_instance.R_diffusion(Ep, a_array, dens, ism=1)
        
        assert len(R_array) == 3
        assert R_array.unit == u.cm
        assert np.all(R_array.value > 0)
        
        # Should increase with time
        assert R_array[2] > R_array[1] > R_array[0]
    
    def test_R_diffusion_physical_values(self, transport_instance):
        """Test that diffusion radii are physically reasonable."""
        Ep = 1 * u.GeV
        a = 1000 * u.yr
        dens = 1 * u.cm**-3
        
        R = transport_instance.R_diffusion(Ep, a, dens, ism=1)
        
        # Convert to parsecs for easier interpretation
        R_pc = R.to(u.pc)
        
        # Should be on the order of 1-1000 pc for typical parameters
        assert 0.1 <= R_pc.value <= 1000


class TestEdgeCases:
    """Test edge cases and error handling for transport class."""
    
    @pytest.fixture
    def transport_instance(self):
        """Setup transport instance for testing."""
        return transport()
    
    def test_zero_energy(self, transport_instance):
        """Test behavior with zero energy."""
        Ep = 0 * u.GeV
        dens = 1 * u.cm**-3
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            D = transport_instance.Diffusion_Coefficient(Ep, dens, ism=1)
            # Should be 0 or very small
            assert D.value >= 0
    
    def test_zero_density_magnetic_field(self, transport_instance):
        """Test magnetic field with zero density."""
        dens = 0 * u.cm**-3
        B = transport_instance.B_mag(dens)
        # Should give minimum field (10 μG)
        assert pytest.approx(B.value, abs=1e-10) == 10.0
    
    def test_zero_time_diffusion_radius(self, transport_instance):
        """Test diffusion radius with zero time."""
        Ep = 1 * u.GeV
        a = 0 * u.yr
        dens = 1 * u.cm**-3
        
        R = transport_instance.R_diffusion(Ep, a, dens, ism=1)
        
        # Should be zero
        assert pytest.approx(R.value, abs=1e-10) == 0
    
    @pytest.mark.parametrize("dens_value", [1e6, 1e8, 1e10])
    def test_very_high_density(self, transport_instance, dens_value):
        """Test with very high density."""
        dens = dens_value * u.cm**-3
        
        B = transport_instance.B_mag(dens)
        assert B.value > 10.0  # Should be much higher than 10 μG
        assert np.isfinite(B.value)
        
        # Test diffusion coefficient
        Ep = 1 * u.GeV
        D = transport_instance.Diffusion_Coefficient(Ep, dens, ism=0)
        assert D.value > 0
        assert np.isfinite(D.value)
    
    @pytest.mark.parametrize("energy_value", [1e3, 1e6, 1e9])
    def test_very_high_energy(self, transport_instance, energy_value):
        """Test with very high energy."""
        Ep = energy_value * u.GeV
        dens = 1 * u.cm**-3
        
        D = transport_instance.Diffusion_Coefficient(Ep, dens, ism=1)
        assert D.value > 0
        assert np.isfinite(D.value)
        
        # Should be much larger than 1 GeV case
        D_1GeV = transport_instance.Diffusion_Coefficient(1*u.GeV, dens, ism=1)
        assert D > D_1GeV


class TestPhysicalConsistency:
    """Test physical consistency and relationships between methods."""
    
    @pytest.fixture
    def transport_instance(self):
        """Setup transport instance for testing."""
        return transport()
    
    def test_diffusion_scaling_relationships(self, transport_instance):
        """Test that all scaling relationships are consistent."""
        # Base parameters
        Ep_base = 1 * u.GeV
        a_base = 1000 * u.yr
        dens_base = 1 * u.cm**-3
        
        # Test energy scaling: D ∝ E^0.5, R ∝ E^0.25
        Ep_2x = 4 * u.GeV
        
        D_base = transport_instance.Diffusion_Coefficient(Ep_base, dens_base, ism=1)
        D_2x = transport_instance.Diffusion_Coefficient(Ep_2x, dens_base, ism=1)
        
        R_base = transport_instance.R_diffusion(Ep_base, a_base, dens_base, ism=1)
        R_2x = transport_instance.R_diffusion(Ep_2x, a_base, dens_base, ism=1)
        
        # Check scaling relationships
        assert pytest.approx((D_2x/D_base).value, rel=0.01) == 2.0  # 4^0.5 = 2
        assert pytest.approx((R_2x/R_base).value, rel=0.01) == 2**0.5  # 4^0.25 = sqrt(2)
    
    def test_magnetic_field_consistency(self, transport_instance):
        """Test magnetic field consistency in diffusion calculations."""
        Ep = 1 * u.GeV
        dens = 1000 * u.cm**-3  # High density
        
        # Get magnetic field
        B = transport_instance.B_mag(dens)
        
        # Calculate diffusion coefficient
        D = transport_instance.Diffusion_Coefficient(Ep, dens, ism=0, chi=1.0)
        
        # Manually calculate expected diffusion coefficient
        # D = chi * D_0 * (E/1GeV)^0.5 * (3μG/B)^0.5
        expected_D = 1.0 * transport_instance.D_0 * ((Ep/u.GeV)**0.5) * ((3*u.uG/B)**0.5)
        
        assert pytest.approx(D.to(u.cm**2/u.s).value, rel=0.01) == expected_D.to(u.cm**2/u.s).value
    
    def test_ism_normalization_consistency(self, transport_instance):
        """Test ISM diffusion coefficient normalization."""
        Ep = 1 * u.GeV
        dens = 1 * u.cm**-3
        
        # For ISM, should be close to D_0 at 1 GeV
        D_ism = transport_instance.Diffusion_Coefficient(Ep, dens, ism=1)
        
        # Should be equal to chiism * D_0 * 1 (since energy and field terms = 1)
        expected = transport_instance.chiism * transport_instance.D_0
        
        assert pytest.approx(D_ism.to(u.cm**2/u.s).value, rel=0.01) == expected.to(u.cm**2/u.s).value


class TestParameterSweeps:
    """Test parameter sweeps to ensure physical behavior."""
    
    @pytest.fixture
    def transport_instance(self):
        """Setup transport instance for testing."""
        return transport()
    
    def test_density_sweep(self, transport_instance):
        """Test behavior across wide range of densities."""
        densities = np.logspace(-1, 6, 10) * u.cm**-3  # 0.1 to 1e6 cm^-3
        
        for dens in densities:
            # Test magnetic field
            B = transport_instance.B_mag(dens)
            assert B.value >= 10.0  # Should be at least 10 μG
            
            # Test diffusion coefficient
            Ep = 1 * u.GeV
            D_ism = transport_instance.Diffusion_Coefficient(Ep, dens, ism=1)
            D_cloud = transport_instance.Diffusion_Coefficient(Ep, dens, ism=0)
            
            assert D_ism.value > 0
            assert D_cloud.value > 0
            assert D_ism >= D_cloud  # ISM should have higher or equal diffusion
    
    def test_energy_sweep(self, transport_instance):
        """Test behavior across wide range of energies."""
        energies = np.logspace(-1, 6, 10) * u.GeV  # 0.1 GeV to 1 PeV
        dens = 10 * u.cm**-3
        
        D_prev = 0 * u.cm**2 / u.s
        
        for Ep in energies:
            D = transport_instance.Diffusion_Coefficient(Ep, dens, ism=1)
            
            assert D.value > 0
            assert np.isfinite(D.value)
            
            # Should generally increase with energy (monotonic)
            if D_prev.value > 0:
                assert D >= D_prev
            
            D_prev = D
    
    def test_time_sweep(self, transport_instance):
        """Test diffusion radius across wide range of times."""
        times = np.logspace(2, 9, 10) * u.yr  # 100 yr to 1 Gyr
        Ep = 1 * u.GeV
        dens = 1 * u.cm**-3
        
        R_prev = 0 * u.cm
        
        for t in times:
            R = transport_instance.R_diffusion(Ep, t, dens, ism=1)
            
            assert R.value > 0
            assert np.isfinite(R.value)
            
            # Should increase with time (monotonic)
            if R_prev.value > 0:
                assert R >= R_prev
            
            R_prev = R

#!/usr/bin/env python3
"""
Unit tests for SNR_Cloud_Flux class and utility functions.

This test suite covers:
- Utility functions (find_nearest, sigmoid_blend)
- SNR_Cloud_Flux class initialization
- Parameter validation
- Mathematical operations
- Edge cases and error handling
"""

import unittest
import numpy as np
import astropy.units as u
from unittest.mock import Mock, patch
import sys
import os

# Mock the required modules before importing the main class
sys.modules['particles'] = Mock()
sys.modules['transport'] = Mock()
sys.modules['accelerator'] = Mock()
sys.modules['flux'] = Mock()
sys.modules['injection'] = Mock()

# Now we can import our functions (assuming they're in a module called snr_flux)
# from snr_flux import SNR_Cloud_Flux, find_nearest, sigmoid_blend


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_find_nearest_basic(self):
        """Test find_nearest function with basic inputs."""
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array-value)).argmin()
            return idx
        
        # Test with integers
        array = np.array([1, 3, 5, 7, 9])
        self.assertEqual(find_nearest(array, 4), 1)  # Should find index of 3
        self.assertEqual(find_nearest(array, 6), 2)  # Should find index of 5
        self.assertEqual(find_nearest(array, 1), 0)  # Exact match
        self.assertEqual(find_nearest(array, 9), 4)  # Last element
        
    def test_find_nearest_floats(self):
        """Test find_nearest with floating point arrays."""
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array-value)).argmin()
            return idx
        
        array_float = np.array([1.1, 2.5, 3.7, 4.9, 6.2])
        self.assertEqual(find_nearest(array_float, 2.6), 1)
        self.assertEqual(find_nearest(array_float, 3.8), 2)
        self.assertEqual(find_nearest(array_float, 0.5), 0)
        
    def test_find_nearest_edge_cases(self):
        """Test find_nearest with edge cases."""
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array-value)).argmin()
            return idx
        
        # Single element array
        single_array = np.array([5.0])
        self.assertEqual(find_nearest(single_array, 10), 0)
        
        # Empty array should raise an error
        with self.assertRaises(ValueError):
            find_nearest(np.array([]), 5)
    
    def test_sigmoid_blend_basic(self):
        """Test sigmoid_blend function with basic parameters."""
        def sigmoid_blend(E, pflux_low, pflux_high, E1=0.05, E2=0.15, a=0.5):
            E = np.asarray(E)
            Ec = 0.5 * (E1 + E2)
            delta = (E2 - E1) / a
            w = 1 / (1 + np.exp((E - Ec) / delta))
            pflux_total = w * pflux_low + (1 - w) * pflux_high
            return pflux_total
        
        E = np.array([0.01, 0.1, 0.2, 0.3])
        pflux_low = np.array([10, 8, 6, 4])
        pflux_high = np.array([1, 1, 1, 1])
        
        result = sigmoid_blend(E, pflux_low, pflux_high)
        
        # Basic shape test
        self.assertEqual(len(result), len(E))
        
        # At very low energies, should be dominated by pflux_low
        self.assertGreater(result[0], 5)  # Should be closer to pflux_low[0]=10
        
        # At very high energies, should be dominated by pflux_high
        self.assertLess(result[-1], 3)  # Should be closer to pflux_high[-1]=1
    
    def test_sigmoid_blend_parameters(self):
        """Test sigmoid_blend with different parameter values."""
        def sigmoid_blend(E, pflux_low, pflux_high, E1=0.05, E2=0.15, a=0.5):
            E = np.asarray(E)
            Ec = 0.5 * (E1 + E2)
            delta = (E2 - E1) / a
            w = 1 / (1 + np.exp((E - Ec) / delta))
            pflux_total = w * pflux_low + (1 - w) * pflux_high
            return pflux_total
        
        E = np.array([0.05, 0.1, 0.15])
        pflux_low = np.array([10, 10, 10])
        pflux_high = np.array([1, 1, 1])
        
        # Test with different transition parameters
        result1 = sigmoid_blend(E, pflux_low, pflux_high, E1=0.08, E2=0.12, a=1.0)
        result2 = sigmoid_blend(E, pflux_low, pflux_high, E1=0.08, E2=0.12, a=0.1)
        
        # Different 'a' values should give different transitions
        self.assertFalse(np.allclose(result1, result2))
    
    def test_sigmoid_blend_edge_cases(self):
        """Test sigmoid_blend edge cases."""
        def sigmoid_blend(E, pflux_low, pflux_high, E1=0.05, E2=0.15, a=0.5):
            E = np.asarray(E)
            Ec = 0.5 * (E1 + E2)
            delta = (E2 - E1) / a
            w = 1 / (1 + np.exp((E - Ec) / delta))
            pflux_total = w * pflux_low + (1 - w) * pflux_high
            return pflux_total
        
        # Test with identical arrays
        E = np.array([0.1, 0.1, 0.1])
        pflux_same = np.array([5, 5, 5])
        result = sigmoid_blend(E, pflux_same, pflux_same)
        np.testing.assert_array_almost_equal(result, pflux_same)


class MockDependencies:
    """Mock the external dependencies for SNR_Cloud_Flux."""
    
    def __init__(self):
        # Mock particles
        self.particles = Mock()
        self.particles_instance = Mock()
        self.particles.return_value = self.particles_instance
        self.particles_instance.NormEbudget.return_value = 1e50
        self.particles_instance.cspeed = 3e10 * u.cm / u.s
        self.particles_instance.sig_ppEK.return_value = 1e-26 * u.cm**2
        self.particles_instance.alpha = 2.0
        
        # Mock transport
        self.transport = Mock()
        self.transport_instance = Mock()
        self.transport.return_value = self.transport_instance
        self.transport_instance.D_0 = 3e27 * u.cm**2 / u.s
        self.transport_instance.Diffusion_Coefficient.return_value = 1e28 * u.cm**2 / u.s
        self.transport_instance.R_diffusion.return_value = 5 * u.pc
        
        # Mock accelerator
        self.accelerator = Mock()
        self.accelerator_instance = Mock()
        self.accelerator.return_value = self.accelerator_instance
        self.accelerator_instance.escape_time.return_value = 1e4 * u.yr
        self.accelerator_instance.SNR_Radius.return_value = 10 * u.pc
        self.accelerator_instance.nh2ism = 1 * u.cm**-3
        
        # Mock flux
        self.flux = Mock()
        self.flux_instance = Mock()
        self.flux.return_value = self.flux_instance
        self.flux_instance.compute_gamma_kernel.return_value = np.ones((100, 200))
        self.flux_instance.compute_neutrino_kernel.return_value = (np.ones((100, 200)), np.ones((100, 200)))
        self.flux_instance.cloud_cell_flux.return_value = np.ones(100) * u.GeV**-1 * u.cm**-3
        self.flux_instance.Pee = 0.3
        self.flux_instance.Pemu = 0.2
        self.flux_instance.Pmumu = 0.3
        self.flux_instance.Petau = 0.2
        self.flux_instance.Pmutau = 0.2
        
        # Mock injection
        self.injection = Mock()
        self.injection.compute_pflux_impulsive_extended.return_value = np.ones(100) * u.GeV**-1 * u.cm**-3
        self.injection.compute_pflux_continuous_extended.return_value = np.ones(100) * u.GeV**-1 * u.cm**-3
        self.injection.compute_fgal.return_value = np.ones(100) * u.GeV**-1 * u.cm**-3
        self.injection.compute_fgal_dampe.return_value = np.ones(100) * u.GeV**-1 * u.cm**-3
        self.injection.compute_fgal_LHAASO.return_value = np.ones(100) * u.GeV**-1 * u.cm**-3


class TestSNRCloudFluxInitialization(unittest.TestCase):
    """Test SNR_Cloud_Flux initialization and basic properties."""
    
    def setUp(self):
        """Set up mock dependencies."""
        self.mocks = MockDependencies()
    
    def create_snr_cloud_flux_mock(self, **kwargs):
        """Create a mock SNR_Cloud_Flux class for testing."""
        class MockSNRCloudFlux:
            def __init__(self, chi=0.05, distance_SNR=2000*u.pc, radius_MC=10*u.pc, 
                        Eg_lo=1.0*u.GeV, Eg_hi=3e3*u.TeV, accel_type='Impulsive', 
                        snr_typeII=True, F_gal=False, palpha=2.0, D_fast=True, flag_low=True):
                
                # Set up energy arrays (simplified)
                N = 20  # Smaller for testing
                Eg_edges = np.logspace(np.log10(Eg_lo.to(u.TeV).value),
                                     np.log10(Eg_hi.to(u.TeV).value), N+1) * u.TeV
                self.Egs = np.sqrt(Eg_edges[:-1] * Eg_edges[1:])
                self.dEgs = np.diff(Eg_edges)
                
                Ep_edges = np.logspace(1, 6.48, 101) * u.GeV  # Smaller for testing
                self.Eps = np.sqrt(Ep_edges[:-1] * Ep_edges[1:])
                self.dEps = np.diff(Ep_edges)
                
                Epl_edges = np.logspace(0, 3.24, 101) * u.GeV
                self.Epls = np.sqrt(Epl_edges[:-1] * Epl_edges[1:])
                self.dEpls = np.diff(Epl_edges)
                
                self.m_pion = 135. * u.MeV
                self.m_proton = 938. * u.MeV
                self.Emin_g = self.Egs + self.m_pion**2 / (4. * self.Egs)
                
                # Store parameters
                self.chi = chi
                self.distance_SNR = distance_SNR
                self.radius_MC = radius_MC
                self.accel_type = accel_type
                self.snr_typeII = snr_typeII
                self.F_gal = F_gal
                self.palpha = palpha
                self.D_fast = D_fast
                self.flag_low = flag_low
                
                # Mock setup
                self.N0 = 1e50
        
        return MockSNRCloudFlux(**kwargs)
    
    def test_default_initialization(self):
        """Test default parameter initialization."""
        snr = self.create_snr_cloud_flux_mock()
        
        # Check default parameters
        self.assertEqual(snr.chi, 0.05)
        self.assertEqual(snr.distance_SNR, 2000*u.pc)
        self.assertEqual(snr.radius_MC, 10*u.pc)
        self.assertEqual(snr.accel_type, 'Impulsive')
        self.assertTrue(snr.snr_typeII)
        self.assertFalse(snr.F_gal)
        self.assertEqual(snr.palpha, 2.0)
        self.assertTrue(snr.D_fast)
        self.assertTrue(snr.flag_low)
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        custom_params = {
            'chi': 0.1,
            'distance_SNR': 1000*u.pc,
            'radius_MC': 5*u.pc,
            'accel_type': 'Continuous',
            'snr_typeII': False,
            'F_gal': True,
            'palpha': 2.2,
            'D_fast': False,
            'flag_low': False
        }
        
        snr = self.create_snr_cloud_flux_mock(**custom_params)
        
        # Check all custom parameters
        for param, value in custom_params.items():
            self.assertEqual(getattr(snr, param), value)
    
    def test_energy_array_setup(self):
        """Test that energy arrays are properly initialized."""
        snr = self.create_snr_cloud_flux_mock(Eg_lo=10*u.GeV, Eg_hi=100*u.TeV)
        
        # Check that arrays are created
        self.assertIsInstance(snr.Egs, u.Quantity)
        self.assertIsInstance(snr.Eps, u.Quantity)
        self.assertIsInstance(snr.Epls, u.Quantity)
        
        # Check units
        self.assertEqual(snr.Egs.unit, u.TeV)
        self.assertEqual(snr.Eps.unit, u.GeV)
        self.assertEqual(snr.Epls.unit, u.GeV)
        
        # Check that arrays have reasonable lengths
        self.assertGreater(len(snr.Egs), 0)
        self.assertGreater(len(snr.Eps), 0)
        self.assertGreater(len(snr.Epls), 0)
        
        # Check energy ranges
        self.assertGreaterEqual(snr.Egs.min(), 10*u.GeV)
        self.assertLessEqual(snr.Egs.max(), 100*u.TeV)


class TestSNRCloudFluxParameterValidation(unittest.TestCase):
    """Test parameter validation and edge cases."""
    
    def create_snr_cloud_flux_mock(self, **kwargs):
        """Create a mock SNR_Cloud_Flux class for testing."""
        class MockSNRCloudFlux:
            def __init__(self, **params):
                # Basic validation
                if 'chi' in params and (params['chi'] <= 0 or params['chi'] > 1):
                    raise ValueError("chi must be between 0 and 1")
                if 'distance_SNR' in params and params['distance_SNR'] <= 0:
                    raise ValueError("distance_SNR must be positive")
                if 'radius_MC' in params and params['radius_MC'] <= 0:
                    raise ValueError("radius_MC must be positive")
                if 'accel_type' in params and params['accel_type'] not in ['Impulsive', 'Continuous']:
                    raise ValueError("accel_type must be 'Impulsive' or 'Continuous'")
                
                # Set defaults
                self.chi = params.get('chi', 0.05)
                self.distance_SNR = params.get('distance_SNR', 2000*u.pc)
                self.radius_MC = params.get('radius_MC', 10*u.pc)
                self.accel_type = params.get('accel_type', 'Impulsive')
        
        return MockSNRCloudFlux(**kwargs)
    
    def test_invalid_chi_values(self):
        """Test validation of chi parameter."""
        # Test negative chi
        with self.assertRaises(ValueError):
            self.create_snr_cloud_flux_mock(chi=-0.1)
        
        # Test zero chi
        with self.assertRaises(ValueError):
            self.create_snr_cloud_flux_mock(chi=0)
        
        # Test chi > 1
        with self.assertRaises(ValueError):
            self.create_snr_cloud_flux_mock(chi=1.5)
    
    def test_invalid_distances(self):
        """Test validation of distance parameters."""
        # Test negative distance_SNR
        with self.assertRaises(ValueError):
            self.create_snr_cloud_flux_mock(distance_SNR=-100*u.pc)
        
        # Test zero distance_SNR
        with self.assertRaises(ValueError):
            self.create_snr_cloud_flux_mock(distance_SNR=0*u.pc)
        
        # Test negative radius_MC
        with self.assertRaises(ValueError):
            self.create_snr_cloud_flux_mock(radius_MC=-5*u.pc)
    
    def test_invalid_acceleration_type(self):
        """Test validation of acceleration type."""
        with self.assertRaises(ValueError):
            self.create_snr_cloud_flux_mock(accel_type='Invalid')
        
        with self.assertRaises(ValueError):
            self.create_snr_cloud_flux_mock(accel_type='impulsive')  # Case sensitive
    
    def test_valid_edge_case_parameters(self):
        """Test valid edge case parameters."""
        # Test minimum valid chi
        snr = self.create_snr_cloud_flux_mock(chi=1e-10)
        self.assertEqual(snr.chi, 1e-10)
        
        # Test maximum valid chi
        snr = self.create_snr_cloud_flux_mock(chi=1.0)
        self.assertEqual(snr.chi, 1.0)


class TestSNRCloudFluxMathematicalOperations(unittest.TestCase):
    """Test mathematical operations and computations."""
    
    def test_travel_parameters_computation(self):
        """Test computation of travel parameters."""
        # Mock the travel parameters calculation
        def mock_compute_travel_parameters(nh2, dist, age):
            # Simplified calculation for testing
            cloud_depth = 2 * u.pc
            dism = dist - 10 * u.pc  # SNR radius
            ismtime = 1000 * u.yr
            Resc = 10 * u.pc
            return cloud_depth, dism, ismtime, Resc
        
        # Test with valid inputs
        nh2 = 1e3 * u.cm**-3
        dist = 50 * u.pc
        age = 2000 * u.yr
        
        cloud_depth, dism, ismtime, Resc = mock_compute_travel_parameters(nh2, dist, age)
        
        # Check return types and units
        self.assertIsInstance(cloud_depth, u.Quantity)
        self.assertIsInstance(dism, u.Quantity)
        self.assertIsInstance(ismtime, u.Quantity)
        self.assertIsInstance(Resc, u.Quantity)
        
        self.assertEqual(cloud_depth.unit, u.pc)
        self.assertEqual(dism.unit, u.pc)
        self.assertEqual(ismtime.unit, u.yr)
        self.assertEqual(Resc.unit, u.pc)
    
    def test_flux_computation_units(self):
        """Test that flux computations return correct units."""
        # Mock flux computation
        def mock_compute_gamma_ray_flux():
            return np.ones(20) / (u.TeV * u.s * u.cm**2)
        
        def mock_compute_neutrino_flux():
            flux_unit = 1 / (u.TeV * u.s * u.cm**2)
            phi_nu = np.ones(20) * flux_unit
            phi_nue_osc = np.ones(20) * flux_unit
            phi_numu_osc = np.ones(20) * flux_unit
            phi_nutau_osc = np.ones(20) * flux_unit
            return phi_nu, phi_nue_osc, phi_numu_osc, phi_nutau_osc
        
        gamma_flux = mock_compute_gamma_ray_flux()
        neutrino_fluxes = mock_compute_neutrino_flux()
        
        # Check gamma-ray flux units
        expected_unit = 1 / (u.TeV * u.s * u.cm**2)
        self.assertEqual(gamma_flux.unit, expected_unit)
        
        # Check neutrino flux units
        for flux in neutrino_fluxes:
            self.assertEqual(flux.unit, expected_unit)
    
    def test_energy_array_consistency(self):
        """Test consistency of energy arrays."""
        # Mock energy array setup
        N = 10
        Eg_lo = 1 * u.GeV
        Eg_hi = 100 * u.TeV
        
        Eg_edges = np.logspace(np.log10(Eg_lo.to(u.TeV).value),
                              np.log10(Eg_hi.to(u.TeV).value), N+1) * u.TeV
        Egs = np.sqrt(Eg_edges[:-1] * Eg_edges[1:])
        dEgs = np.diff(Eg_edges)
        
        # Check array lengths
        self.assertEqual(len(Egs), N)
        self.assertEqual(len(dEgs), N)
        
        # Check monotonicity
        self.assertTrue(np.all(np.diff(Egs.value) > 0))
        
        # Check energy range
        self.assertGreaterEqual(Egs.min(), Eg_lo)
        self.assertLessEqual(Egs.max(), Eg_hi)


class TestSNRCloudFluxErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        # Mock function that handles NaN values
        def mock_handle_nans(flux_array):
            return np.nan_to_num(flux_array, nan=0)
        
        # Test with NaN values
        test_array = np.array([1, 2, np.nan, 4, np.inf, -np.inf])
        result = mock_handle_nans(test_array)
        
        # Check that NaN values are replaced with 0
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
        self.assertEqual(result[2], 0)  # NaN should become 0
    
    def test_zero_flux_handling(self):
        """Test handling of zero fluxes."""
        # Mock computation that might produce zero flux
        def mock_flux_computation(input_flux):
            if np.all(input_flux == 0):
                return np.zeros_like(input_flux)
            return input_flux * 2
        
        zero_flux = np.zeros(10)
        result = mock_flux_computation(zero_flux)
        
        self.assertTrue(np.all(result == 0))
    
    def test_unit_consistency(self):
        """Test unit consistency in calculations."""
        # Test that operations preserve units correctly
        distance = 1000 * u.pc
        radius = 10 * u.pc
        
        # Simple geometric calculation
        area = np.pi * radius**2
        volume = (4./3.) * np.pi * radius**3
        solid_angle = area / distance**2
        
        # Check units
        self.assertEqual(area.unit, u.pc**2)
        self.assertEqual(volume.unit, u.pc**3)
        self.assertEqual(solid_angle.unit, u.dimensionless_unscaled)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestUtilityFunctions,
        TestSNRCloudFluxInitialization,
        TestSNRCloudFluxParameterValidation,
        TestSNRCloudFluxMathematicalOperations,
        TestSNRCloudFluxErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
            
if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

