"""CMB noise curves in temperature or intensity units

This simple module computes angular power spectra of instrumental noise
for various CMB experiments, either in temperature or intensity at
a given frequency.

Simon Foreman, Perimeter Institute, 2019
"""

import numpy as np
from scipy.constants import arcmin, degree

# Fundamental constants
hPlanck = 6.626e-34  # J s
cLight = 3e8  # m s^-1
kB = 1.38e-23  # J K^-1

# CMB temperature at z=0
TCMB = 2.726  # K


def uKCMB_to_Jypersr(nu, T):
    """Convert temperature in uK_CMB to Jy sr^-1

    Based on a routine in the ACT tILe-C code, found in
        https://github.com/ACTCollaboration/tilec/blob/master/tilec/fg.py ,
    which is itself based on Sec. 3.2 of
        https://arxiv.org/abs/1303.5070 .
    Another useful reference is here:
        https://www.cfa.harvard.edu/~kkarkare/map_units.pdf .

    The conversion is given by
        uK_CMB * 10^{26} \frac{dB_\nu}{dT}|_{T_0} = Jy sr^{-1} ,
    where
        \frac{dB_\nu}{dT} = \frac{2 h \nu^3}{c^2} \frac{e^x}{(e^x-1)^2} \frac{x}{T [in \mu K]} ,
    with x \equiv h\nu/(k_B T) and evaluating everything at the CMB temperature T_0.

    Parameters
    ----------
    nu : float
        Frequency, in GHz.
    T : float
        Temperature, in microKelvin_CMB.

    Returns
    -------
    I : float
        Equivalent value in Jy sr^{-1}.
    """
    x = hPlanck * nu * 1e9 / kB / TCMB
    prefactor = 1e26 * 2 * hPlanck * (nu * 1e9) ** 3 / cLight ** 2
    return T * prefactor * np.exp(x) / (np.exp(x) - 1) ** 2 * x / (TCMB * 1e6)


def Jypersr_to_uKCMB(nu, n):
    """Convert from Jy sr^-1 to uK_CMB.

    Simply inverts the conversion done by uKCMB_to_Jypersr.

    Parameters
    ----------
    nu : float
        Frequency, in GHz.
    n : float
        Noise in Jy sr^-1.

    Returns
    -------
    T : float
        Value in uK_CMB.
    """
    x = hPlanck * nu * 1e9 / kB / TCMB
    prefactor = 1e26 * 2 * hPlanck * (nu * 1e9) ** 3 / cLight ** 2
    return n / (prefactor * np.exp(x) / (np.exp(x) - 1) ** 2 * x / (TCMB * 1e6))


def uKarcmin_to_Jyperrad(nu, n):
    """Convert map noise value in uK_CMB arcmin to Jy rad^-1

    This simply converts the input value using uKCMB_to_Jypersr,
    and also converts the arcmin value to rad.

    Parameters
    ----------
    nu : float
        Frequency, in GHz.
    n : float
        Noise, in microKelvin_CMB arcmin.

    Returns
    -------
    I : float
        Equivalent value in Jy rad^{-1}.
    """
    return uKCMB_to_Jypersr(nu, n) * arcmin


def Jyperrad_to_uKarcmin(nu, n):
    """Convert from Jy rad^-1 to uK_CMB arcmin.

    Simply inverts the conversion done by uKarcmin_to_Jyperrad.

    Parameters
    ----------
    nu : float
        Frequency, in GHz.
    n : float
        Noise in Jy rad^-1.

    Returns
    -------
    N : float
        Value in uK_CMB arcmin.
    """
    return Jypersr_to_uKCMB(nu, n) / arcmin


def uKarcmin_to_Jypersr(nu, n, fwhm):
    """Convert map noise value in uK_CMB arcmin to Jy sr^-1

    One must specify a beam FWHM for this.

    Parameters
    ----------
    nu : float
        Frequency, in GHz.
    n : float
        Noise, in uK_CMB arcmin.
    fwhm : float
        Beam FWHM (or other measure of linear size), in arcmin.

    Returns
    -------
    I : float
        Equivalent value in Jy sr^{-1}.
    """
    return uKCMB_to_Jypersr(nu, n) * arcmin / (fwhm * arcmin)


def Jypersr_to_uKarcmin(nu, n, fwhm):
    """Convert from Jy sr^-1 to uK_CMB arcmin.

    Simply inverts the conversion done by uKarcmin_to_Jypersr.
    
    One must specify a beam FWHM for this.

    Parameters
    ----------
    nu : float
        Frequency, in GHz.
    n : float
        Noise in Jy sr^-1.
    fwhm : float
        Beam FWHM (or other measure of linear size), in arcmin.

    Returns
    -------
    N : float
        Value in uK_CMB arcmin.
    """
    return Jypersr_to_uKCMB(nu, n) / arcmin * (fwhm * arcmin)


def test_TtoI():
    """Verify uK-arcmin to Jy rad^{-1} conversion against some results from the
    literature.
    """
    print("822 uK-arcmin at 545GHz =\t%g Jy rad^-1" % uKarcmin_to_Jyperrad(545, 822))
    print("Schaan+ 2018, Table I:\t\t13.5 Jy rad^-1")
    print("\n20 uK-arcmin at 405GHz =\t%g Jy rad^-1" % uKarcmin_to_Jyperrad(405, 20))
    print("Schaan+ 2018, Table I:\t\t1.2 Jy rad^-1")
    print(
        "\n23.9 uK-arcmin at 21GHz with a 38.4arcmin FWHM beam = \t%g Jy sr^-1"
        % (uKarcmin_to_Jypersr(21, 23.9, 38.4))
    )
    print("Hanany+ 2019, Table 1.2:\t\t\t\t8.3 Jy sr^-1")
    print(
        "\n4 uK-arcmin at 186GHz with a 4.3arcmin FWHM beam = \t%g Jy sr^-1"
        % (uKarcmin_to_Jypersr(186, 4, 4.3))
    )
    print("Hanany+ 2019, Table 1.2:\t\t\t\t433 Jy sr^-1")


class Experiment:
    """Base class defining a CMB experiment.

    As inputs, takes lists of midpoints (or other representative values) of
    each frequency band, the beam FWHMs at each frequency, and the map noise
    levels at each frequency. Subroutines return noise curves for temperature or
    intensity at a given frequency, either as C_l or D_l = l(l+1) C_l / 2pi.

    The noise curves are computed using the standard Knox 1995 formula: in
    \mu K_CMB units,
        C_\ell^N = w^{-1} \exp( \ell (\ell+1) \sigma^2 / [8\ln 2] ) ,
    where w^{-1/2} is the map noise level converted to \mu K rad, and
    \sigma is the beam FWHM in rad. Appropriate unit conversions are made for
    other cases.

    So far, there is no functionality for including atmospheric noise in this
    base class.

    Attributes
    -------
    freqs
    beam_FWHMs
    map_noise_levels

    Methods
    -------
    Cl_N_uK2
    Cl_N_dToverT
    Dl_N_uK2
    Dl_N_dToverT
    Cl_N_Jy2persr
    Dl_N_Jy2persr
    """

    def __init__(self, freqs=[], beam_FWHMs=[], map_noise_levels=[]):
        if len(freqs) != len(beam_FWHMs):
            raise Exception(
                "Need to input the same number of frequencies and beam widths! %d vs. %d"
                % (len(freqs), len(beam_FWHMs))
            )
        if len(freqs) != len(map_noise_levels):
            raise Exception(
                "Need to input the same number of frequencies and map noise levels! %d vs. %d"
                % (len(freqs), len(map_noise_levels))
            )

        self.freqs = np.array(freqs)
        self.beam_FWHMs = np.array(beam_FWHMs)
        self.map_noise_levels = np.array(map_noise_levels)

        self.beam_FWHM = dict(zip(self.freqs, self.beam_FWHMs))
        self.map_noise = dict(zip(self.freqs, self.map_noise_levels))

    def Cl_N_uK2(self, nu, ell):
        if nu not in self.freqs:
            raise Exception("No stored information for desired frequency!")

        return (self.map_noise[nu] * arcmin) ** 2 * np.exp(
            ell * (ell + 1) * (self.beam_FWHM[nu] * arcmin) ** 2 / (8 * np.log(2))
        )

    def Cl_N_dToverT(self, nu, ell):
        return self.Cl_N_uK2(nu, ell) / (TCMB * 1e6) ** 2

    def Dl_N_uK2(self, nu, ell):
        return ell * (ell + 1) * self.Cl_N_uK2(nu, ell) / (2 * np.pi)

    def Dl_N_dToverT(self, nu, ell):
        return ell * (ell + 1) * self.Cl_N_dToverT(nu, ell) / (2 * np.pi)

    def Cl_N_Jy2persr(self, nu, ell):
        return uKarcmin_to_Jyperrad(nu, self.map_noise[nu]) ** 2 * np.exp(
            ell * (ell + 1) * (self.beam_FWHM[nu] * arcmin) ** 2 / (8 * np.log(2))
        )

    def Dl_N_Jy2persr(self, nu, ell):
        return ell * (ell + 1) * self.Cl_N_Jy2persr(nu, ell) / (2 * np.pi)


class Planck(Experiment):
    """Noise curves for Planck.

    Numbers taken from Table 4 of
        https://arxiv.org/abs/1807.06205 .
    """

    def __init__(self):
        # The table gives the noise levels for all frequencies except the upper
        # two in uK_CMB deg, which we convert to uK arcmin
        map_noise_lower = np.array([2.5, 2.7, 3.5, 1.29, 0.55, 0.78, 2.56]) * 60.0

        # The table gives the noise levels for 545 and 857 GHz in kJy sr^-1 deg,
        # which we convert to Jy rad^-1 and then to uK arcmin.
        map_noise_upper = np.array(
            [
                Jyperrad_to_uKarcmin(545, 0.78 * 1e3 * 60),
                Jyperrad_to_uKarcmin(857, 0.72 * 1e3 * 60),
            ]
        )

        Experiment.__init__(
            self,
            freqs=[28.4, 44.1, 70.4, 100, 143, 217, 353, 545, 857],
            beam_FWHMs=[32.29, 27.94, 13.08, 9.66, 7.22, 4.90, 4.92, 4.67, 4.22],
            map_noise_levels=np.concatenate((map_noise_lower, map_noise_upper)),
        )


class _PICO_base(Experiment):
    """Base class for PICO noise curves, defining frequency bands and beam widths.

    Numbers taken from Table 1 of
        https://arxiv.org/abs/1902.10541 .

    A derived class must specify the map noise levels.
    """

    def __init__(self, map_noise_levels):
        Experiment.__init__(
            self,
            freqs=[
                21,
                25,
                30,
                36,
                43,
                52,
                62,
                75,
                90,
                108,
                129,
                155,
                186,
                223,
                268,
                321,
                385,
                462,
                555,
                666,
                799,
            ],
            beam_FWHMs=[
                38.4,
                32.0,
                28.3,
                23.6,
                22.2,
                18.4,
                12.8,
                10.7,
                9.5,
                7.9,
                7.4,
                6.2,
                4.3,
                3.6,
                3.2,
                2.6,
                2.5,
                2.1,
                1.5,
                1.3,
                1.1,
            ],
            map_noise_levels=map_noise_levels,
        )


class PICO_baseline(_PICO_base):
    """Baseline PICO configuration.

    Numbers taken from Table 1 of
        https://arxiv.org/abs/1902.10541 .
    """

    def __init__(self):
        _PICO_base.__init__(
            self,
            map_noise_levels=[
                23.9,
                18.4,
                12.4,
                7.9,
                7.9,
                5.7,
                5.4,
                4.2,
                2.8,
                2.3,
                2.1,
                1.8,
                4.0,
                4.5,
                3.1,
                4.2,
                4.5,
                9.1,
                45.8,
                177,
                1050,
            ],
        )


class PICO_CBE(_PICO_base):
    """CBE ("current best estimate") PICO configuration.

    Numbers taken from Table 1 of
        https://arxiv.org/abs/1902.10541 .
    """

    def __init__(self):
        _PICO_base.__init__(
            self,
            map_noise_levels=[
                16.9,
                13.0,
                8.7,
                5.6,
                5.6,
                4.0,
                3.8,
                3.0,
                2.0,
                1.6,
                1.5,
                1.3,
                2.8,
                3.2,
                2.2,
                3.0,
                3.2,
                6.4,
                32.4,
                125,
                740,
            ],
        )


class LiteBIRD(Experiment):
    """Noise curves for LiteBIRD.

    Numbers taken from Table 1 of
        Hazumi et al., "LiteBIRD: A Satellite for the Studies of B-Mode
        Polarization and Inflation from Cosmic Background Radiation Detection",
        Journal of Low Temperature Physics, 194:443 (2019).
    """

    def __init__(self):
        # The map noise in Hazumi et al. is given for polarization, so we divide by
        # sqrt(2) to convert to temperature noise
        Experiment.__init__(
            self,
            freqs=[40, 50, 60, 68, 78, 89, 100, 119, 140, 166, 195, 235, 280, 337, 402],
            beam_FWHMs=[69, 56, 48, 43, 39, 35, 29, 25, 23, 21, 20, 19, 24, 20, 17],
            map_noise_levels=np.array(
                [
                    37.5,
                    24,
                    19.9,
                    16.2,
                    13.5,
                    11.7,
                    9.2,
                    7.6,
                    5.9,
                    6.5,
                    5.8,
                    7.7,
                    13.2,
                    19.5,
                    37.5,
                ]
            )
            / np.sqrt(2),
        )

        
class CMB_HD(Experiment):
    """Noise curves for CMB-HD.

    Numbers taken from Table 1 of
        https://arxiv.org/abs/2002.12714
        
    These do not include atmospheric noise.
    """

    def __init__(self):
        Experiment.__init__(
            self,
            freqs=[30, 40, 90, 150, 220, 280, 350],
            beam_FWHMs=[1.25, 0.94, 0.42, 0.25, 0.17, 0.13, 0.11],
            map_noise_levels=[6.5, 3.4, 0.7, 0.8, 2.0, 2.7, 100.0]
        )