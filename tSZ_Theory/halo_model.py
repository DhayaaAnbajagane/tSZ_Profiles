#Import useful packages.
import pyccl as ccl
import numpy as np
from scipy import interpolate

#Define relevant physical constants
Msun_to_Kg = ccl.physical_constants.SOLAR_MASS
Mpc_to_m   = ccl.physical_constants.MPC_TO_METER
G          = ccl.physical_constants.GNEWT / Mpc_to_m**3 * Msun_to_Kg
m_to_cm    = 1e2

#Just define some useful conversions/constants
sigma_T = 6.652458e-29 / Mpc_to_m**2
m_e     = 9.10938e-31 / Msun_to_Kg
c       = 2.99792458e8 / Mpc_to_m

#The main class doing the heavy lifting.
#Inherits properties of the CCL profile class
class HaloProfileBattaglia(ccl.halos.profiles.HaloProfile):

    '''
    Class that implements a Battaglia profile using the
    CCL profile class. This class inheritance allows us to
    easily compute relevant observables using the CCL
    machinery.

    ------------------
    Params:
    ------------------

    Model_def : str
        The available mode calibrations from Battaglia+ 2012.
        Can be one of '200_AGN', '500_AGN', and '500_SH'. The former
        two were calibrated using simulations w/ AGN feedback, whereas
        the latter did not. The initial three-digit number denoted the
        spherical overdensity definition used in the calibration (Either
        M200c or M500c).

    truncate : float
        The radius (in units of R/Rdef, where Rdef is the halo radius defined
        via some chosen spherical overdensity definition) at which to cutoff
        the profiles and set them to zero. Default is False.
     '''

    def __init__(self, Model_def, truncate = False):

        #Set mass definition using the input Model_def
        if Model_def == '200_AGN':
            self.mdef = ccl.halos.massdef.MassDef(200, 'critical')

        elif Model_def == '500_AGN':
            self.mdef = ccl.halos.massdef.MassDef(500, 'critical')

        elif Model_def == '500_SH':
            self.mdef = ccl.halos.massdef.MassDef(500, 'critical')

        else:

            raise ValueError("Input Model_def not valid. Select one of: 200_AGN, 500_AGN, 500_SH")

        self.Model_def = Model_def
        self.truncate  = truncate

        #Import all other parameters from the base CCL Profile class
        super(HaloProfileBattaglia, self).__init__()

        #Constant that helps with the fourier transform convolution integral.
        #This value minimized the ringing due to the transforms
        self.precision_fftlog['plaw_fourier'] = -2

        #Need this to prevent projected profile from artificially cutting off
        self.precision_fftlog['padding_lo_fftlog'] = 1e-4
        self.precision_fftlog['padding_hi_fftlog'] = 1e4

    def _real(self, cosmo, r, M, a, mass_def=None):

        '''
        Function that computes the Battaglia pressure profile for halos.
        Can use three different definitions: 200_AGN, 500_AGN, and 500_SH.

        Based on arxiv:1109.3711

        ------------------
        Params:
        ------------------

        cosmo : pyccl.Cosmology object
            A CCL cosmology object that contains the relevant
            cosmological parameters

        r : float, numpy array, list
            Radii (in comoving Mpc) to evaluate the profiles at.

        M : float, numpy array, list
            The list of halo masses (in Msun) to compute the profiles
            around.

        a : float
            The cosmic scale factor

        mass_def : ccl.halos.massdef.MassDef object
            The mass definition associated with the input, M


        ------------------
        Output:
        ------------------

        numpy array :
            An array of size (M.size, R.size) that contains the electron
            pressure values at radius R of each cluster of a mass given
            by array M. If M.size and/or R.size are simply 1, then the output
            is flattened along that dimension.
        '''

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1
        mass_def = self.mdef

        #Setup parameters as they were calibrated in Battaglia+ 2012
        if self.Model_def == '200_AGN':

            P_0  = 18.1  * (M_use/1e14)**0.154    * (1 + z)**-0.758
            x_c  = 0.497 * (M_use/1e14)**-0.00865 * (1 + z)**0.731
            beta = 4.35  * (M_use/1e14)**0.0393   * (1 + z)**0.415

        elif self.Model_def == '500_AGN':

            P_0  = 7.49  * (M_use/1e14)**0.226   * (1 + z)**-0.957
            x_c  = 0.710 * (M_use/1e14)**-0.0833 * (1 + z)**0.853
            beta = 4.19  * (M_use/1e14)**0.0480  * (1 + z)**0.615

        elif self.Model_def == '500_SH':

            P_0  = 20.7  * (M_use/1e14)**-0.074 * (1 + z)**-0.743
            x_c  = 0.428 * (M_use/1e14)**0.011  * (1 + z)**1.01
            beta = 3.82  * (M_use/1e14)**0.0375 * (1 + z)**0.535


        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        x = r_use[None, :]/R[:, None]

        #The overdensity constrast related to the mass definition
        Delta    = mass_def.get_Delta(cosmo, a)

        #Cosmological parameters
        Omega_m  = cosmo.cosmo.params.Omega_m
        Omega_b  = cosmo.cosmo.params.Omega_b
        Omega_g  = cosmo.cosmo.params.Omega_g
        h        = cosmo.cosmo.params.h

        RHO_CRIT = ccl.physical_constants.RHO_CRITICAL*h**2 * ccl.background.h_over_h0(cosmo, a)**2 #This is in physical coordinates

        #Thermodynamic/abundance quantities
        Y         = 0.24 #Helium mass ratio
        Pth_to_Pe = (4 - 2*Y)/(8 - 5*Y) #Factor to conver gas temp. to electron temp


        # The self-similar expectation for Pressure
        # Need R*a to convert comoving Mpc to physical
        P_delta = Delta*RHO_CRIT * Omega_b/Omega_m * G * (M_use)/(2*R*a)
        alpha, gamma = 1, -0.3

        P_delta, P_0, beta, x_c = P_delta[:, None], P_0[:, None], beta[:, None], x_c[:, None]
        prof = Pth_to_Pe * P_delta * P_0 * (x/x_c)**gamma * (1 + (x/x_c)**alpha)**-beta

        # Battaglia profile has validity limits for redshift, mass, and distance from halo center.
        # Here, we enforce the distance limit at R/R_Delta > X, where X is input by user
        if self.truncate:
            prof[x > self.truncate] = 0

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof

def PowSpec_HaloPressure(cosmo, k, M, a, mass_def = None, Model_def = '500_SH', truncate = False):
    '''
    Routine to compute the halo-pressure cross power spectrum.
    It is easier to think of this as the halo profile in fourier space

    ------------------
    Params:
    ------------------

    cosmo : pyccl.Cosmology object
        A CCL cosmology object that contains the relevant
        cosmological parameters

    k : float, numpy array, list
        wavenumber (in comoving 1/Mpc) to evaluate the spectrum at.

    M : float, numpy array, list
        The list of halo masses (in Msun) to compute the cross power spectrum
        around.

    a : float
        The cosmic scale factor

    mass_def : ccl.halos.massdef.MassDef object
        The mass definition associated with the input, M

    Model_def : str
        The available mode calibrations from Battaglia+ 2012.
        Can be one of '200_AGN', '500_AGN', and '500_SH'. The former
        two were calibrated using simulations w/ AGN feedback, whereas
        the latter did not. The initial three-digit number denoted the
        spherical overdensity definition used in the calibration (Either
        M200c or M500c).

    truncate : float
        The radius (in units of R/Rdef, where Rdef is the halo radius defined
        via some chosen spherical overdensity definition) at which to cutoff
        the profiles and set them to zero. Default is False.

    ------------------
    Output:
    ------------------

    numpy array :
        An array of size (M.size, R.size) that contains the halo pressure
        cross spectrum values at wavenumber k, for each cluster in array M.
    '''

    #Setup Halo mass Function (HMF) and Halo bias objects
    HMF   = ccl.halos.hmfunc.MassFuncTinker08(cosmo, mass_def = mass_def, mass_def_strict = True)
    Hbias = ccl.halos.hbias.HaloBiasTinker10(cosmo, mass_def = mass_def, mass_def_strict = True)

    #Setup integration object
    Integral_method = ccl.halos.halo_model.HMCalculator(cosmo, HMF, Hbias, mass_def)

    #Get linear matter power spectrum, and Halo bias
    Pk_lin = ccl.boltzmann.get_camb_pk_lin(cosmo).eval(k, a, cosmo)
    b_M    = Hbias.get_halo_bias(cosmo, M, a, mass_def)

    b_M    = np.atleast_1d(b_M)

    #Compute cross power spectrum using Battaglia profiles
    Pk_hp = b_M[:, None] * Pk_lin * Integral_method.I_1_1(cosmo, k, a, HaloProfileBattaglia(Model_def, truncate))

    return Pk_hp

def CorrFunc_HaloPressure(cosmo, r, M, a, mass_def = None, Model_def = '500_SH', truncate = False):

    '''
    Routine that computes the halo-pressure cross correlation.
    Computes everything in fourier space, and the fourier transforms
    the power spectra to get the correlation functions in
    dimensions of Msun/Mpc/s^2 (pressure)

    ------------------
    Params:
    ------------------

    cosmo : pyccl.Cosmology object
        A CCL cosmology object that contains the relevant
        cosmological parameters

    r : float, numpy array, list
        radius (in comoving Mpc) to evaluate the spectrum at.

    M : float, numpy array, list
        The list of halo masses (in Msun) to compute the cross power spectrum
        around.

    a : float
        The cosmic scale factor

    mass_def : ccl.halos.massdef.MassDef object
        The mass definition associated with the input, M

    Model_def : str
        The available mode calibrations from Battaglia+ 2012.
        Can be one of '200_AGN', '500_AGN', and '500_SH'. The former
        two were calibrated using simulations w/ AGN feedback, whereas
        the latter did not. The initial three-digit number denoted the
        spherical overdensity definition used in the calibration (Either
        M200c or M500c).

    truncate : float
        The radius (in units of R/Rdef, where Rdef is the halo radius defined
        via some chosen spherical overdensity definition) at which to cutoff
        the profiles and set them to zero. Default is False.

    ------------------
    Output:
    ------------------

    numpy array :
        An array of size (M.size, R.size) that contains the halo pressure
        cross spectrum values at distance R, for each cluster in array M.
    '''

    M_use = np.atleast_1d(M)
    r_use = np.atleast_1d(r)

    #Get array of wavenumbers to sample fourier profile at
    #The choice of the range/sampling is a bit arbitrary.
    #Was chosen so transformed profile didn't have any artifacts,
    #and we confirmed the result was converged to changes in range.
    k = np.geomspace(1e-4, 1e4, 1000)
    output_r, output_Xi_M = ccl.pyutils._fftlog_transform(k, PowSpec_HaloPressure(cosmo, k, M, a, mass_def, Model_def, truncate), 2, 0, -1)

    Xi = np.zeros([M_use.size, r_use.size])

    ln_r_use    = np.log(r_use)
    ln_output_r = np.log(output_r)

    #Loop over each cluster with mass M
    #for each, sample the transformed output to
    #get the Corr. Func at the exact distance we need
    for im, output_Xi in enumerate(output_Xi_M):

        # Resample into input r_t values
        Xi[im, :] = ccl.pyutils.resample_array(ln_output_r, output_Xi, ln_r_use,
                                               'linx_liny', 'linx_liny', 0, 0)

    if np.ndim(r) == 0:
        Xi = np.squeeze(Xi, axis=-1)
    if np.ndim(M) == 0:
        Xi = np.squeeze(Xi, axis=0)

    #Multiply by
    # Xi_hp = sigma_T/(m_e*c**2) * a * Xi

    return Xi, r_use

def Total_halo_model(cosmo, r, M, a, mass_def = None, Model_def = '500_SH', truncate = False):
    '''
    Wrapper that computes both one halo and two halo terms
    for a given set of halo masses and radii.

    ------------------
    Params:
    ------------------

    cosmo : pyccl.Cosmology object
        A CCL cosmology object that contains the relevant
        cosmological parameters

    r : float, numpy array, list
        radius (in comoving Mpc) to evaluate the spectrum at.

    M : float, numpy array, list
        The list of halo masses (in Msun) to compute the cross power spectrum
        around.

    a : float
        The cosmic scale factor

    mass_def : ccl.halos.massdef.MassDef object
        The mass definition associated with the input, M

    Model_def : str
        The available mode calibrations from Battaglia+ 2012.
        Can be one of '200_AGN', '500_AGN', and '500_SH'. The former
        two were calibrated using simulations w/ AGN feedback, whereas
        the latter did not. The initial three-digit number denoted the
        spherical overdensity definition used in the calibration (Either
        M200c or M500c).

    truncate : float
        The radius (in units of R/Rdef, where Rdef is the halo radius defined
        via some chosen spherical overdensity definition) at which to cutoff
        the profiles and set them to zero. Default is False.

    ------------------
    Output:
    ------------------

    numpy array :
        An array of size (M.size, R.size) that contains the one halo term
        of the halo pressure cross spectrum at distance R, for each
        cluster in array M.
    '''

    one_halo = sigma_T/(m_e*c**2) * a * HaloProfileBattaglia(Model_def, truncate).projected(cosmo, r, M, a, mass_def)
    two_halo = sigma_T/(m_e*c**2) * a * CorrFunc_HaloPressure(cosmo, r, M, a, mass_def, Model_def, truncate)[0]

    return one_halo, two_halo


def Smoothed_Total_halo_model(cosmo, r, M, a, FWHM_arcmin,
                              mass_def = None, Model_def = '500_SH', truncate = False,
                              f_miscen = 0, tau_miscen = 0):

    '''
    Compute a beam-smoothed version of the one halo and two halo
    terms given set of halo masses and radii.

    ------------------
    Params:
    ------------------

    cosmo : pyccl.Cosmology object
        A CCL cosmology object that contains the relevant
        cosmological parameters

    r : float, numpy array, list
        radius (in comoving Mpc) to evaluate the spectrum at.

    M : float, numpy array, list
        The list of halo masses (in Msun) to compute the cross power spectrum
        around.

    a : float
        The cosmic scale factor.

        Note: The input of a = 1 will cause the code to crash as
        the angular diameter distance is then D_a(a = 1) = 0.
        This is because the smoothing step requires the quantity 1/D_a,
        which, at a = 1, becomes 1/D_a = infinity. One can simply replace
        a = 1 with a = 0.999999 without any issue


    FWHM_arcmin : float
        The full-width half-max of a gaussian beam smoothing, in units of arcmin

    mass_def : ccl.halos.massdef.MassDef object
        The mass definition associated with the input, M

    Model_def : str
        The available mode calibrations from Battaglia+ 2012.
        Can be one of '200_AGN', '500_AGN', and '500_SH'. The former
        two were calibrated using simulations w/ AGN feedback, whereas
        the latter did not. The initial three-digit number denoted the
        spherical overdensity definition used in the calibration (Either
        M200c or M500c).

    truncate : float
        The radius (in units of R/Rdef, where Rdef is the halo radius defined
        via some chosen spherical overdensity definition) at which to cutoff
        the profiles and set them to zero. Default is False.

    f_miscen : float
        Fraction of halos that are miscentered in this sample. If 0, then
        no miscentering is applied. Else, apply it similar to 1702.01722.

    tau_miscen : float, optional
        Parameter of the miscentering model used in 1702.01722

    lambda : float, optional
        Cluster richness. Needed only if f_miscen != 0 and
        miscentering must be applied. Part of model in 1702.01722

    ------------------
    Output:
    ------------------

    numpy array :
        An array of size (M.size, R.size) that contains the one halo term
        of the halo pressure cross spectrum at distance R, for each
        cluster in array M.
    '''

    M_use = np.atleast_1d(M)
    r_use = np.atleast_1d(r)

    #Need this because we are transforming to fourier space
    r_for_smoothing    = np.geomspace(1e-4, 1e4, 1000)
    one_halo, two_halo = Total_halo_model(cosmo, r_for_smoothing, M, a, mass_def, Model_def, truncate)

    #Convert to angular space
    D_A   = ccl.background.angular_diameter_distance(cosmo, a)
    theta = r_for_smoothing*a/D_A #in radians

    # Convert to ell-space
    l_one, Cl_one = ccl.pyutils._fftlog_transform(theta, one_halo, 2, 0, -1.5)
    l_two, Cl_two = ccl.pyutils._fftlog_transform(theta, two_halo, 2, 0, -1.5)

    #Smooth via beam
    FWHM_rad    = FWHM_arcmin * 1/60 * np.pi/180
    sigma_beam  = FWHM_rad / np.sqrt(8 * np.log(2))

    #Compute the beam profile, B(ell)
    Beam_l_one  = np.exp(-l_one*(l_one + 1)*sigma_beam**2/2)
    Beam_l_two  = np.exp(-l_two*(l_two + 1)*sigma_beam**2/2)

    #Convert back from ell space to angular space
    theta_one, Xi_smoothed_one = ccl.pyutils._fftlog_transform(l_one, Cl_one * Beam_l_one, 2, 0, -1)
    theta_two, Xi_smoothed_two = ccl.pyutils._fftlog_transform(l_two, Cl_two * Beam_l_two, 2, 0, -1)

    #get the radial scale of each smoothed Corr Func value
    output_r_one = theta_one * D_A / a
    output_r_two = theta_two * D_A / a

    Xi_one = np.zeros([M_use.size, r_use.size])
    Xi_two = np.zeros([M_use.size, r_use.size])

    ln_r_use = np.log(r_use)

    if np.ndim(Xi_smoothed_one) == 1: Xi_smoothed_one = Xi_smoothed_one[None, :]
    if np.ndim(Xi_smoothed_two) == 1: Xi_smoothed_two = Xi_smoothed_two[None, :]

    ln_output_r = np.log(output_r_one)

    #Interpolate to get Xi are correct radii.
    for im, output_Xi in enumerate(Xi_smoothed_one):
        # Resample into input r_t values
        Xi_one[im, :] = ccl.pyutils.resample_array(ln_output_r, output_Xi, ln_r_use, 'linx_liny', 'linx_liny', 0, 0)


    ln_output_r = np.log(output_r_two)
    for im, output_Xi in enumerate(Xi_smoothed_two):
        # Resample into input r_t values
        Xi_two[im, :] = ccl.pyutils.resample_array(ln_output_r, output_Xi, ln_r_use, 'linx_liny', 'linx_liny', 0, 0)


    if np.ndim(r) == 0:
        Xi_one = np.squeeze(Xi_one, axis=-1)
        Xi_two = np.squeeze(Xi_two, axis=-1)

    if np.ndim(M) == 0:
        Xi_one = np.squeeze(Xi_one, axis=0)
        Xi_two = np.squeeze(Xi_two, axis=0)

    #Factor of (2*np.pi)**2 comes from the multiple fourier transformations
    return (2*np.pi)**2 * Xi_one, (2*np.pi)**2 * Xi_two
