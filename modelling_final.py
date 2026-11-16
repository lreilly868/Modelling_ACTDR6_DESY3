import os
import numpy as np
import pyccl as ccl
import pylab as plt
import matplotlib.cm as cm
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy
import emcee
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
import h5py
from scipy.integrate import quad
import scipy.integrate as integrate
from pyccl.halos.profiles import HaloProfileNFW
from pyccl.halos.profiles import HaloProfileMatter
from scipy.interpolate import CubicSpline
from functools import lru_cache
from scipy.integrate import cumulative_trapezoid
from scipy.special import logsumexp
from scipy.linalg import block_diag
from functools import partial
import sys
import multiprocessing as mp

import time
T_START = time.time()
MAX_RUNTIME_H = 46

def out_of_time():
    return (time.time() - T_START) > MAX_RUNTIME_H *3600

_trapz = getattr(np, "trapezoid", None) or np.trapz

#incase it goes to shit and doenst hog up server
max_attempts = 5

task_tag = ("stage1" if os.environ.get("stage1_only")
            else os.environ.get("SLURM_ARRAY_TASK_ID", "all"))
flag_file = f"pipeline_not_done_{task_tag}.flag"
attempt_file = f"attempts_{task_tag}.txt"

n_attempt = 1
if os.path.exists(attempt_file):
    with open(attempt_file) as fh:
        n_attempt = int(fh.read().strip() or 0) + 1
with open(attempt_file, "w") as fh:
    fh.write(str(n_attempt))
print(f"[{task_tag}] attempt {n_attempt}/{max_attempts}", flush=True)

if n_attempt > max_attempts:
    if os.path.exists(flag_file):
        os.remove(flag_file)
    open(f"GAVE_UP_{task_tag}.flag", "w").close()
    sys.exit(f"[{task_tag}] giving up after {max_attempts} attempts: check the logs")

open(flag_file, "w").close()

def mark_finished():
    for f in (flag_file, attempt_file):
        if os.path.exists(f):
            os.remove(f)
        

n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
print("cwd:", os.getcwd())
print("stage1_chain.h5 absolute path:", os.path.abspath("stage1_chain.h5"))
print("file exists there:", os.path.exists("stage1_chain.h5"))

    



#--------------------------------------
#------------------------------------------
#test for paper data to make sure it is running correctly
#recreating gt_model as I belive it needs to be done for each individual cluster,
#not as a batch

#need to make a galaxy redshift distribution for each cluster


#Modelling for 1 and 2 halo terms

#need to pass in certain variables for the individual clusters, will figure out once done


#new halo profile
class HaloProfileNFWCDM(HaloProfileNFW):

    __repr_attrs__ = __eq_attrs__ = HaloProfileNFW.__repr_attrs__ + ("fbary",)
    
    def __init__(self, *, mass_def = "500c", concentration, fbary = None,
                 fourier_analytic=True,
                 projected_analytic=False,
                 cumul2d_analytic = False, truncated = False):
        self.fbary = fbary
        super().__init__(mass_def=mass_def, concentration = concentration,
                         fourier_analytic = fourier_analytic,
                         projected_analytic=projected_analytic,
                         cumul2d_analytic=cumul2d_analytic,
                         truncated=truncated)

    def _get_fbary(self, cosmo):
        if self.fbary is not None:
            return self.fbary
        return cosmo["Omega_b"] / (cosmo["Omega_m"])

    def _norm(self, M, Rs, c, cosmo=None):
        norm_total = super()._norm(M, Rs, c)
        if cosmo is None:

            fb = self.fbary
        else:
            fb = self._get_fbary(cosmo)
        return (1 -fb) * norm_total

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        R_M = self.mass_def.get_radius(cosmo, M_use, a )/a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M

        x = r_use[None, :] / R_s[:, None]
        prof = 1. / (x * (1 + x)**2)

        if self.truncated:
            prof[r_use[None, :] > R_M[:, None]] = 0

        norm = self._norm(M_use, R_s, c_M, cosmo=cosmo)
        prof = prof[:,:] * norm[:,None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis = -1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis = 0)
        return prof

    def _projected_analytic(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        R_M = self.mass_def.get_radius(cosmo, M_use, a)/a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M

        x = r_use[None, :] / R_s[:, None]
        prof = self._fx_projected(x)
        norm = 2 * R_s * self._norm(M_use, R_s, c_M, cosmo=cosmo)
        prof *= norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis = -1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis =0)
        return prof

    def _cumul2d_analytic(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        R_M = self.mass_def.get_radius(cosmo, M_use, a)/a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M

        x = r_use[None, :] / R_s[:, None]
        prof = self._fx_cumul2d(x)
        norm = 2 * R_s * self._norm(M_use, R_s, c_M, cosmo=cosmo)
        prof = prof[:,:] * norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis = -1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis =0)
        return prof



    def _fourier_analytic(self, cosmo, k, M, a):
        from scipy.special import sici
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M
        x = k_use[None, :] * R_s[:, None]
        Si2, Ci2 = sici(x)
        fb = self._get_fbary(cosmo)
        P1 = (1 -fb) * M_use / (np.log(1 + c_M) - c_M / (1 + c_M))
        if self.truncated:
            Si1, Ci1 = sici((1 + c_M[:, None]) * x)
            P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
            P3 = np.sin(c_M[:, None] * x) / (( 1 + c_M[:, None]) * x)
            prof = P1[:, None] * (P2 - P3)
        else:
            P2 = np.sin(x) * (0.5 * np.pi - Si2) - np.cos(x) * Ci2
            prof = P1[:, None] * P2
        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis = -1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis = 0)
        return prof
            
@lru_cache(maxsize=4096)
def _enclose_shape_integral(alpha, beta, gamma, Xmax):

    def integrand(u):
        lnu = np.log(u)
        return u**2 / np.exp(gamma * lnu + (beta - gamma) * alpha * np.logaddexp(0.0,  lnu / alpha))
    if Xmax <= 1.0:
        val, _ = quad(integrand, 0.0, Xmax, limit = 200, epsabs = 1e-8)
    else:
        v1, _ = quad(integrand, 0.0, 1.0, limit = 200, epsabs=0.0, epsrel=1e-8)
        v2, _ = quad(integrand, 1.0, Xmax, limit = 200, epsabs=0.0, epsrel=1e-8)
        val = v1 + v2
    return val

class HaloProfileGNFWBaryon(HaloProfileMatter):

    __repr_attrs__ = __eq_attrs__ = ("alpha", "beta", "gamma", "x_c", "fbary", "truncated", "mass_def", "concentration", "precision_fftlog")

    def __init__(self, *, mass_def ="500c", concentration, alpha, beta, gamma=0.2, x_c=0.5, fbary=None, R_b=3.3, truncated = False):
        self._validate_shape_params(alpha=alpha, beta=beta, gamma=gamma, x_c=x_c)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.x_c = x_c
        self.fbary = fbary
        self.truncated = truncated
        self._dimless_shape_cache = {}
        super().__init__(mass_def=mass_def, concentration=concentration)

    def update_parameters(self, *, alpha = None, beta=None, gamma=None, x_c=None, fbary=None):

        self._validate_shape_params(
            alpha = alpha if alpha is not None else self.alpha,
            beta = beta if beta is not None else self.beta,
            gamma = gamma if gamma is not None else self.gamma,
            x_c = x_c if x_c is not None else self.x_c)
        
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if x_c is not None:
            self.x_c = x_c
        if fbary is not None:
            self.fbary = fbary


    @staticmethod
    def _validate_shape_params(*, alpha, beta, gamma, x_c):

        if not (alpha > 0):
            raise ValueError("alpha must be >0")
        if not (beta > gamma):
            raise ValueError("beta must be > gamma")
        if not (x_c > 0):
            raise ValueError("x_c must be > 0")


        
    def _get_fbary(self, cosmo):
        if self.fbary is not None:
            return self.fbary
        return cosmo["Omega_b"] / cosmo["Omega_m"]

    def _shape(self, u):
        return 1.0 / (u**self.gamma * (1.0 + u**(1.0/self.alpha)) ** ((self.beta - self.gamma) * self.alpha))
                      
    def _norm(self, M, R_s, c):

        Xmax = c / self.x_c
        I = np.array([_enclose_shape_integral(self.alpha, self.beta, self.gamma, float(X))
                      for X in Xmax])
        return M / (4 * np.pi * (R_s * self.x_c)**3 * I)

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M

        u = r_use[None, :] / (R_s[:, None] * self.x_c)
        prof = self._shape(u)
        if self.truncated:
            prof[r_use[None, :] > R_M[:, None]] = 0

        fb = self._get_fbary(cosmo)
        rho_b = self._norm(M_use, R_s, c_M)
        norm = fb * rho_b
        prof = prof * norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis = 0)
        return prof


    def _get_dimless_shape_splines(self):
        key = (self.alpha, self.beta, self.gamma)
        cached = self._dimless_shape_cache.get(key)
        if cached is not None:
            return cached

        def sigma_shape(u):
            integrand = lambda w: self._shape(np.sqrt(u**2 + w**2))
            scale = max(abs(self._shape(u)), 1e-12)
            val, _ =  quad(integrand, 0.0, 1e2, limit=400, epsabs=scale*1e-8, epsrel=1e-6)
            return 2*val
            

        u_grid = np.geomspace(1e-4, 1e3, 100)
        sigma_vals = np.array([sigma_shape(u) for u in u_grid])
        self._check_finite(sigma_vals, u_grid, "Sigma", key)
        sigma_spline = CubicSpline(np.log10(u_grid), sigma_vals, extrapolate=True)

        def sigma_bar_shape(u):
            integrand = lambda up: up* sigma_spline(np.log10(up))
            scale = max(abs(u * sigma_spline(np.log10(u))), 1e-12)
            val, _ = quad(integrand, u_grid[0] * 1e-3, u, limit=200, epsabs=scale * 1e-8, epsrel=1e-6)
            return 2 * val / u**2
        
        sigma_bar_vals = np.array([sigma_bar_shape(u) for u in u_grid])
        self._check_finite(sigma_bar_vals, u_grid, "Sigma_bar", key)
        sigma_bar_spline = CubicSpline(np.log10(u_grid), sigma_bar_vals, extrapolate = True)

        self._dimless_shape_cache[key] = (sigma_spline, sigma_bar_spline)
        return sigma_spline, sigma_bar_spline

    @staticmethod
    def _check_finite(vals, u_grid, name, key):

        bad = ~np.isfinite(vals)
        if np.any(bad):
            alpha, beta, gamma = key
            raise ValueError("non finite value bad")
            
    def projected(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M
        Lc = R_s * self.x_c
        u = r_use[None, :] / Lc[:, None]

        sigma_spline, _ = self._get_dimless_shape_splines()
        shape = sigma_spline(np.log10(u))

        fb = self._get_fbary(cosmo)
        rho_b = self._norm(M_use, R_s, c_M)
        prof = shape * (fb * rho_b * Lc)[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) ==0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def cumul2d(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M
        Lc = R_s * self.x_c
        u = r_use[None, :] / Lc[:, None]

        _, sigma_bar_spline = self._get_dimless_shape_splines()
        shape = sigma_bar_spline(np.log10(u))

        fb = self._get_fbary(cosmo)
        rho_b = self._norm(M_use, R_s, c_M)
        prof = shape * (fb * rho_b * Lc)[:, None]

        if np.ndim(r) ==0:
            prof = np.squeeze(prof, axis = -1)
        if np.ndim(M) ==0:
            prof = np.squeeze(prof, axis = 0)
        return prof

        
        



class HaloProfile2Halo(HaloProfileMatter):

    __repr_attrs__ = __eq_attrs__ = ("a2h", "p_of_k_a", "mass_def", "bias_model")

    def __init__(self, *, mass_def="500c", a2h=1.0, bias_model=None,
                 p_of_k_a ="delta_matter:delta_matter", r_grid = None,
                 R_shape_grid = None, los_max = 3000.0):

        self.a2h = a2h
        self.p_of_k_a = p_of_k_a
        self._r_grid = (np.geomspace(1e-3, 2e3, 400)
                        if r_grid is None else np.asarray(r_grid))
        self._R_shape_grid = (np.geomspace(1e-3, 100, 60)
                              if R_shape_grid is None
                              else np.asarray(R_shape_grid))
        self._los_max = los_max
        self._xi_cache = {}
        self._shape_cache = {}
        super().__init__(mass_def=mass_def, concentration =None)
        self.bias_model = (bias_model if bias_model is not None else
                           ccl.halos.HaloBiasTinker10(mass_def=self.mass_def))

    def update_parameters(self, *, a2h=None):

        if a2h is not None:
            self.a2h = a2h

    def _xi_mm(self, cosmo, a, r):

        key = (id(cosmo), round(float(a), 10), str(self.p_of_k_a))
        spline = self._xi_cache.get(key)

    
        if spline is None:
            xi_grid = ccl.correlations.correlation_3d(cosmo, r=self._r_grid, a=a,
                                                      p_of_k_a =self.p_of_k_a)
            bad = ~np.isfinite(xi_grid)
            if np.any(bad):
                raise ValueError("issue with xi grid")
            spline = CubicSpline(np.log10(self._r_grid), xi_grid, extrapolate = True)
            self._xi_cache[key] = spline
        return spline(np.log10(r))

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        rho_m = ccl.background.rho_x(cosmo, a, "matter", is_comoving=True)
        b_M = np.atleast_1d(self.bias_model(cosmo, M_use, a))
        xi = self._xi_mm(cosmo, a, r_use)

        prof = self.a2h * rho_m * b_M[:, None] * xi[None, :]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _get_shape_splines(self, cosmo, a):
        key = (id(cosmo), round(float(a), 2), str(self.p_of_k_a))
        cached = self._shape_cache.get(key)
        if cached is not None:
            return cached

        def xi_scalar(r):
            return float(self._xi_mm(cosmo, a, np.array([r]))[0])

        def sigma_shape(R):
            integrand = lambda l: xi_scalar(np.sqrt(R**2 + l**2))

            scale = max(abs(xi_scalar(R)), 1e-12)
            val, _ = quad(integrand, 0.0, self._los_max, limit=400,
                      epsabs = scale * 1e-6, epsrel = 1e-6)
            return 2 * val

        R_grid = self._R_shape_grid
        sigma_vals = np.array([sigma_shape(R) for R in R_grid])
        sigma_spline = CubicSpline(np.log10(R_grid), sigma_vals, extrapolate = True)
        bad = ~np.isfinite(sigma_vals)

        if np.any(bad):
            raise ValueError("non finite sigma value")
        def sigma_bar_shape(R):
            integrand = lambda Rp: Rp * sigma_spline(np.log10(Rp))
            val, _ = quad(integrand, R_grid[0] * 1e-3, R, limit = 200)
            return 2 * val / R**2

        sigma_bar_vals = np.array([sigma_bar_shape(R) for R in R_grid])
        sigma_bar_spline = CubicSpline(np.log10(R_grid), sigma_bar_vals, extrapolate = True)

        self._shape_cache[key] = (sigma_spline, sigma_bar_spline)
        return sigma_spline, sigma_bar_spline

    def _amplitude(self, cosmo, M, a):
        rho_m = ccl.background.rho_x(cosmo, a, "matter", is_comoving = True)
        b_M = np.atleast_1d(self.bias_model(cosmo, np.atleast_1d(M), a))
        return self.a2h * rho_m * b_M

    def projected(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        sigma_spline, _ = self._get_shape_splines(cosmo, a)
        shape = sigma_spline(np.log10(r_use))
        amp = self._amplitude(cosmo, M_use, a)
        prof = amp[:, None] * shape[None, :]
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis = -1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def cumul2d(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        _, sigma_bar_spline = self._get_shape_splines(cosmo, a)
        shape = sigma_bar_spline(np.log10(r_use))
        amp = self._amplitude(cosmo, M_use, a)
        prof = amp[:, None] * shape[None, :]
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis = 0)
        return prof

    
class HaloProfileClusterTotal(HaloProfileMatter):

    __repr_attrs__ = __eq_attrs__ = ("cdm", "baryon", "twohalo", "mass_def", "concentration")

    def __init__(self, *, mass_def="500c", concentration, alpha, beta, gamma=0.2,
                 x_c =0.5, a2h=1.0, fbary =None, bias_model=None,
                 p_of_k_a ="delta_matter:delta_matter", r_grid = None, truncated = False):
        self.cdm = HaloProfileNFWCDM(mass_def = mass_def, concentration=concentration, fbary=fbary, truncated=truncated, projected_analytic=True, cumul2d_analytic=True)
        self.baryon = HaloProfileGNFWBaryon(mass_def = mass_def, concentration=concentration, alpha = alpha, beta = beta, gamma=gamma, x_c = x_c, fbary =fbary, truncated = truncated)
        self.twohalo = HaloProfile2Halo(mass_def=mass_def, a2h=a2h, bias_model=bias_model,
                                        p_of_k_a = p_of_k_a, r_grid = r_grid)

        
        super().__init__(mass_def=self.cdm.mass_def, concentration=self.cdm.concentration)

    def update_parameters(self, *, alpha=None, beta=None, gamma=None, x_c=None, a2h=None, fbary=None):
        self.baryon.update_parameters(alpha=alpha, beta=beta, gamma=gamma, x_c=x_c, fbary=fbary)

        self.twohalo.update_parameters(a2h=a2h)
        if fbary is not None:
            self.cdm.fbary = fbary

    def _real(self, cosmo, r, M, a):
        return (self.cdm._real(cosmo, r, M, a)
                + self.baryon._real(cosmo, r, M, a)
                + self.twohalo._real(cosmo, r, M, a))

    def projected(self, cosmo, r, M, a):
        return (self.cdm.projected(cosmo, r, M, a)
                + self.baryon.projected(cosmo, r, M, a)
                + self.twohalo.projected(cosmo, r, M, a))

    def cumul2d(self, cosmo, r, M, a):
                return (self.cdm.cumul2d(cosmo, r, M, a)
                        + self. baryon.cumul2d(cosmo, r, M, a)
                        + self.twohalo.cumul2d(cosmo, r, M, a))


class HaloProfileNFWOnly(HaloProfileMatter):
    #just for the NFW only run, essentially f_b is just set to 0

    __repr_attrs__ = __eq_attrs__ = ("cdm", "twohalo", "mass_def", "concentration")

    def __init__(self, *, mass_def="500c", concentration, a2h =1.0, bias_model=None,
                 p_of_k_a="delta_matter:delta_matter", r_grid=None, truncated=False):
        self.cdm = HaloProfileNFWCDM(mass_def=mass_def, concentration=concentration,fbary=0.0, truncated=truncated,
                                     projected_analytic=True, cumul2d_analytic=True)
        self.twohalo = HaloProfile2Halo(mass_def=mass_def, a2h=a2h, bias_model=bias_model,
                                         p_of_k_a=p_of_k_a, r_grid=r_grid)
        super().__init__(mass_def=self.cdm.mass_def, concentration=self.cdm.concentration)

    def update_parameters(self, *, alpha=None, beta=None, gamma=None, x_c=None, a2h=None,fbary=None):
        self.twohalo.update_parameters(a2h=a2h)

    def _real(self, cosmo, r, M, a):
        return self.cdm._real(cosmo, r, M, a) + self.twohalo._real(cosmo, r, M, a)

    def projected(self, cosmo, r, M, a):
        return self.cdm.projected(cosmo, r, M, a) + self.twohalo.projected(cosmo, r, M, a)

    def cumul2d(self, cosmo, r, M, a):
        return self.cdm.cumul2d(cosmo, r, M, a) + self.twohalo.cumul2d(cosmo, r, M, a)
    

DZ_S_PRIOR = {
    3: (0.0, 0.011),
    4: (0.0, 0.017),
    }

M_GAMMA_PRIOR = {
    3: ( -0.024, 0.008),
    4: (-0.037, 0.008),
    }

def _check_bin(source_bin):
    if source_bin not in (3, 4):
        raise ValueError("source must be in bin 3 or 4")

def draw_dz_s(source_bin, rng =None):

    _check_bin(source_bin)
    if rng is None:
        rng = np.random.default_rng()
    mean, std = DZ_S_PRIOR[source_bin]
    return rng.normal(mean, std)

def draw_m_gamma(source_bin, rng = None):
    _check_bin(source_bin)
    if rng is None:
        rng = np.random.default_rng()
    mean, std = M_GAMMA_PRIOR[source_bin]
    return rng.normal(mean, std)

def draw_nuisance_params(source_bins, rng=None):

    if rng is None:
        rng = np.random.default_rng()
    dz_s = {b: draw_dz_s(b, rng = rng) for b in source_bins}
    m_gamma = {b: draw_m_gamma(b, rng=rng) for b in source_bins}
    return dz_s, m_gamma

def shift_source_nz(z_grid, nz, dz_s):
    return np.interp(z_grid - dz_s, z_grid, nz, left=0.0, right= 0.0)

#section 4.3 
def true_mass_from_sz(M_SZ, z, A_mass, eta=0.0, zeta=0.0, M_pivot=3e14, z_pivot=0.45):

    M_SZ = np.asarray(M_SZ, dtype=float)
    z = np.asarray(z, dtype=float)

    mass_term = (M_SZ / M_pivot)** eta
    z_term = ((1.0 + z) / (1.0 + z_pivot)) ** zeta
    bias_factor = A_mass * mass_term * z_term

    M_true = M_SZ * bias_factor
    return M_true, bias_factor

#calculates 1-b 
def hydrostatic_bias(A_mass, eta=0.0, zeta=0.0, M_SZ=3e14, z=0.45, M_pivot=3e14, z_pivot=0.45):


    _, bias_factor = true_mass_from_sz(M_SZ, z, A_mass, eta, zeta, M_pivot, z_pivot)

    return 1.0 / bias_factor

def draw_true_masses(M_true_mean, sigma_lnM, rng=None):

    if rng is None:
        rng = np.random.default_rng()
    M_true_mean = np.asarray(M_true_mean, dtype = float)
    ln_M = rng.normal(loc=np.log(M_true_mean), scale = sigma_lnM)
    return np.exp(ln_M)

def margninalized_log_likelihood(M_true_mean, sigma_lnM, log_likelihood_fn, n_draws=7, rng=None):

    if rng is None:
        rng = np.random.default_rng()
    log_likes = np.empty(n_draws)
    for j in range(n_draws):
        M_i_draw = draw_true_masses(M_true_mean, sigma_lnM, rng=rng)
        log_likes[j] = log_likelihood_fn(M_i_draw)
    return logsumexp(log_likes) - np.log(n_draws)


#miscentering
def sample_is_centered(f_mis, n_clusters, rng=None):

    if rng is None:
        rng = np.random.default_rng()
    return rng.random(n_clusters) >= f_mis

def sample_r_mis(sigma_mis, n_draws, rng=None):

    if rng is None:
        rng = np.random.default_rng()
    return rng.gamma(shape=2.0, scale=sigma_mis, size=n_draws)

def sigma_miscentered(sigma_func, r_query, R_mis, n_phi=64):

    r_query = np.atleast_1d(r_query)
    phi = (np.arange(n_phi) +0.5) * (2* np.pi / n_phi)
    s = np.sqrt(r_query[:, None]**2 + R_mis**2
                + 2 * r_query[:, None] * R_mis * np.cos(phi)[None, :])
    vals = sigma_func(s.ravel()).reshape(s.shape)
    return vals.mean(axis=1)

def sigma_bar_miscentered(sigma_func, r_query, R_mis, n_phi=64, n_R=120):

    r_query = np.atleast_1d(r_query)
    R_max = max(r_query.max(), R_mis) * 1.01
    R_min = min(r_query.min(), 1e-3) * 1e-2

    Rp_grid = np.geomspace(R_min, R_max, n_R)
    sigma_vals = sigma_miscentered(sigma_func, Rp_grid, R_mis, n_phi = n_phi)
    bad = ~np.isfinite(sigma_vals)
    if np.any(bad):
        raise ValueError("non-finite sigma value miscentered")

    integrand = Rp_grid * sigma_vals
    cumulative = cumulative_trapezoid(integrand, Rp_grid, initial=0.0)
    bad = ~np.isfinite(cumulative)
    if np.any(bad):
        raise ValueError("non-finite cumulative integral")
    spline = CubicSpline(Rp_grid, cumulative,extrapolate = True)
    return 2 * spline(r_query) / r_query**2

 

def _F(x):

    x = np.asarray(x, dtype =float)
    F = np.empty_like(x)

    hi = x > 1
    lo = x < 1

    sqrt_hi = np.sqrt(x[hi]**2 - 1.0)
    F[hi] = np.arctan(sqrt_hi) / sqrt_hi

    sqrt_lo = np.sqrt(1.0 - x[lo]**2)
    F[lo] = np.arctanh(sqrt_lo) / sqrt_lo

    return F

def boost_factor(R, B0, Rs, x_tol=1e-4):


    R = np.asarray(R, dtype=float)
    scalar_input = (R.ndim ==0)
    R  = np.atleast_1d(R)
    x = R/ Rs

    result = np.empty_like(x)
    near_one = np.abs(x - 1.0) < x_tol
    far = ~near_one

    if np.any(far):
        F_far = _F(x[far])
        result[far] = 1.0 + B0 * (1.0 - F_far) / (x[far] **2 - 1.0)
    if np.any(near_one):
        result[near_one] = 1.0 + B0 / 3.0

    return result.item() if scalar_input else result


G = 4.30081e-9
c = 299792.458


def apply_miscentering_to_clusters(sigma_funcs, r_query, f_mis, sigma_mis, n_phi=64, n_R=60, rng=None):

    if rng is None:
        rng = np.randomg.default_rng()
    n_clusters = len(sigma_funcs)
    r_query = np.atleast_1d(r_query)
    is_centered = sample_is_centered(f_mis, n_cluster, rng=rng)

    Sigma_used = np.empty((n_cluster, len(r_query)))
    Sigma_bar_used = np.empty((n_cluster, len(r_query)))

    n_miscentered = int(np.sum(~is_centered))
    R_mis_draws = (sample_r_mis(sigma_mis, n_miscentered, rng=rng)
                   if n_miscentered else np.array([]))

    j = 0
    for i, (sigma_func, sigma_bar_func) in enumerate(sigma_funcs):
        if is_centered[i]:
            Sigma_used[i] = sigma_func(r_query)
            Sigma_bar_used[i] = sigma_bar_func(r_query)
        else:
            R_mis_i = R_mis_draws[j]
            j += 1
            Sigma_used[i] = sigma_miscentered(sigma_func. r_query, R_mis_i, n_phi=n_phi)
            Sigma_bar_used[i] = sigma_bar_miscentered(sigma_func, r_query,
                                                      R_mis_i, n_phi = n_phi,
                                                      n_R = n_R)
    return Sigma_used, Sigma_bar_used


G = 4.30091e-9
c = 299792.458

def sigma_critical(cosmo, z_l, z_s_grid):

    z_s_grid = np.asarray(z_s_grid, dtype = float)
    a_l = 1.0 / (1.0 + z_l)
    a_s = 1.0 / (1.0 + z_s_grid)

    D_A_l = ccl.angular_diameter_distance(cosmo, a_l)
    D_A_s = ccl.angular_diameter_distance(cosmo, a_s)
    D_C_l = ccl.comoving_radial_distance(cosmo, a_l)
    D_C_s = ccl.comoving_radial_distance(cosmo, a_s)
    D_A_ls = (D_C_s - D_C_l) / (1.0 + z_s_grid)

    sigma_crit = np.full_like(z_s_grid, np.inf)
    valid = z_s_grid > z_l
    sigma_crit[valid] = (c **2 / (4 * np.pi * G)
                         * D_A_s[valid] / (D_A_l * D_A_ls[valid]))
    return sigma_crit

def theta_to_comoving_R(cosmo, z_l, theta_rad):

    a_l = 1.0 / (1.0 + z_l)
    D_C_l = ccl.comoving_angular_distance(cosmo, a_l)
    return theta_rad * D_C_l

def gt_integrated_over_source_nz(theta_rad, z_l, Sigma, SigmaBar, z_s_grid, nz_s, cosmo):
    DeltaSigma = SigmaBar - Sigma
    sigma_crit = sigma_critical(cosmo, z_l, z_s_grid)
    with np.errstate(divide = "ignore"):
        inv_sc = np.where(np.isfinite(sigma_crit), 1.0/sigma_crit, 0.0)

    num = DeltaSigma[None, :] * inv_sc[:, None]
    denom = 1.0 - Sigma[None, :] * inv_sc[:, None]
    gt_zs_theta = num / denom

    numerator = _trapz(nz_s[:, None] * gt_zs_theta, z_s_grid, axis = 0)
    normalization = _trapz(nz_s, z_s_grid)
    return numerator / normalization

def lens_redshift_nodes(z_mean, sigma_z, n_nodes = 5):

    if sigma_z <= 0:
        return np.array([z_mean]), np.array([1.0])
    x, w = np.polynomial.hermite.hermgauss(n_nodes)
    z_nodes = z_mean + np.sqrt(2.0) * sigma_z * x
    weights = w / np.sqrt(np.pi)

    z_nodes = np.clip(z_nodes, 1e-4, None)
    weights = weights / weights.sum()
    return z_nodes, weights

def assemble_gt_model(profile, cosmo, M_SZ_arr, z_arr, w_arr, theta_arcmin,
                      source_bins, nz_grids, A_mass, eta, zeta, alpha, beta, a2h,
                      f_mis, sigma_mis, boost_params, sigma_lnM = 0.2,
                      sigma_z_arr = None, n_zl_nodes = 5, rng = None):
    if rng is None:
        rng = np.random.default_rng()

    profile.update_parameters(alpha=alpha, beta=beta, a2h=a2h)

    theta_rad = np.deg2rad(np.asarray(theta_arcmin) / 60.0)
    n_cluster = len(M_SZ_arr)
    n_theta = len(theta_rad)

    if sigma_z_arr is None:
        sigma_z_arr = np.zeros(n_cluster)

    w_arr = np.asarray(w_arr, dtype = float)
    n_bins = len(source_bins)
    if w_arr.ndim == 1:
        w_norm_by_bin = {j: np.tile(w_arr[:, None], (1, n_theta)) for j in source_bins}
    elif w_arr.shape[1] == n_theta:
        w_norm_by_bin = {j: w_arr for j in source_bins}
    elif w_arr.shape[1] == n_theta * n_bins:
        w_norm_by_bin = {j: w_arr[:, k*n_theta:(k+1)*n_theta]
                         for k, j in enumerate(source_bins)}
    else:
        raise ValueError("weight has bad shape")

    for j, w_j in w_norm_by_bin.items():
        col_sum = w_j.sum(axis=0)
        if np.any(col_sum == 0):
            raise ValueError("sourcebin error ")
        w_norm_by_bin[j] = w_j / col_sum[None, :]
        
    M_true_mean, _ = true_mass_from_sz(M_SZ_arr, z_arr, A_mass, eta, zeta)
    M_i_draw = draw_true_masses(M_true_mean, sigma_lnM, rng=rng)

    is_centered = sample_is_centered(f_mis, n_cluster, rng=rng)
    R_mis_draws = {}
    for i in range(n_cluster):
        if not is_centered[i]:
            R_mis_draws[i] = sample_r_mis(sigma_mis, 1, rng=rng)[0]

    dz_s, m_gamma = draw_nuisance_params(source_bins, rng=rng)

    R_boost_by_bin = {}
    for j, w_j in w_norm_by_bin.items():
        w_eff_j = w_j.mean(axis=1)
        z_eff_j = np.sum(w_eff_j * z_arr) / np.sum(w_eff_j)
        R_boost_by_bin[j] = theta_to_comoving_R(cosmo, z_eff_j, theta_rad)


    if isinstance(boost_params, dict):
        boost_params_by_bin = boost_params
    else:
        boost_params_by_bin = {j: boost_params for j in source_bins}
        

    cluster_node_data = []

    for i in range(n_cluster):
        M_i = M_i_draw[i]
        z_nodes, z_weights = lens_redshift_nodes(z_arr[i], sigma_z_arr[i],
                                                 n_nodes=n_zl_nodes)
        node_list = []
        for z_node, w_node in zip(z_nodes, z_weights):
            a_node = 1.0 / (1.0 + z_node)
            R_node = theta_to_comoving_R(cosmo, z_node, theta_rad)
            sigma_func = lambda r, M=M_i, a=a_node: profile.projected(cosmo, r, M, a)
            sigma_bar_func = lambda r, M=M_i, a=a_node: profile.cumul2d(cosmo, r, M, a)
        
        
            if is_centered[i]:
                Sigma_node = sigma_func(R_node)
                SigmaBar_node = sigma_bar_func(R_node)
            else:
                Sigma_node = sigma_miscentered(sigma_func, R_node, R_mis_draws[i])
                SigmaBar_node = sigma_bar_miscentered(sigma_func, R_node, R_mis_draws[i])


            Sigma_node = Sigma_node / a_node**2
            SigmaBar_node = SigmaBar_node / a_node**2
            node_list.append((z_node, w_node, Sigma_node, SigmaBar_node))
        cluster_node_data.append(node_list)

    gt_model = {}

    for j in source_bins:
        z_s_grid, nz_raw = nz_grids[j]
        nz_shifted = shift_source_nz(z_s_grid, nz_raw, dz_s[j])

        gt_stack = np.zeros(len(theta_rad))
        for i in range(n_cluster):
            gt_ij = np.zeros(len(theta_rad))
            for z_node, w_node, Sigma_node, SigmaBar_node in cluster_node_data[i]:
                gt_ij_node = gt_integrated_over_source_nz(theta_rad, z_node,
                                                          Sigma_node, SigmaBar_node,
                                                          z_s_grid, nz_shifted, cosmo)
                gt_ij += w_node * gt_ij_node
            gt_stack += w_norm_by_bin[j][i, :] * gt_ij

        B0_j, Rs_boost_j = boost_params_by_bin[j]
        B = boost_factor(theta_arcmin, B0_j, Rs_boost_j)
        gt_model[j] = (1.0 + m_gamma[j]) / B * gt_stack
    return gt_model

def flatten_gt_model(gt_model, source_bins):
    return np.concatenate([gt_model[j] for j in source_bins])

def gaussian_log_likelihood(data_vector, model_vector, cov):
    resid = data_vector - model_vector
    L = np.linalg.cholesky(cov)

    y = np.linalg.solve(L, resid)
    chi2 = np.dot(y, y)
    return -0.5 * chi2

def log_likelihood_lensing_one_draw(profile, cosmo, theta_params, M500c,
                                    z, sigma_z, weight, thmid,
                                    source_bins, nz_grids,
                                    gammat, cov, boost_params,
                                    sigma_lnM = 0.2, n_zl_nodes= 5, rng = None):
    
    gt_model = assemble_gt_model(
        profile = profile, cosmo = cosmo, M_SZ_arr = M500c, z_arr = z, w_arr = weight,
        theta_arcmin = thmid, source_bins = source_bins, nz_grids=nz_grids,
        A_mass = theta_params["A_mass"], eta=theta_params["eta"],
        zeta = theta_params["zeta"], alpha = theta_params["alpha"],
        beta = theta_params["beta"], a2h=theta_params["a2h"],
        f_mis = theta_params["f_mis"], sigma_mis = theta_params["sigma_mis"],
        boost_params = boost_params,
        sigma_lnM = sigma_lnM, sigma_z_arr=sigma_z, n_zl_nodes = n_zl_nodes,
        rng = rng,
        )
    model_vector = flatten_gt_model(gt_model, source_bins)
    return gaussian_log_likelihood(gammat, model_vector, cov)

def log_likelihood_boost_only(boost_params_by_bin, thmid, source_bins, boost_pz,  cov_boost_pz):
    thmid = np.asarray(thmid, dtype=float)
    model = np.concatenate([
        boost_factor(thmid, *boost_params_by_bin[j]) for j in source_bins])
    return gaussian_log_likelihood(boost_pz, model, cov_boost_pz)


def log_prior_main(theta, fit_evolution=True):

    if fit_evolution:
        A_mass, eta, zeta, alpha, beta, a2h, f_mis, ln_sigma_mis = theta
    else:
        A_mass, alpha, beta, a2h, f_mis, ln_sigma_mis =theta
        eta = zeta = 0.0

    if not (0.1 < A_mass < 10.0):
        return -np.inf
    if fit_evolution:
        if not (-10.0 < eta < 10.0):
            return -np.inf
        if not (-10.0 < zeta < 10.0):
            return -np.inf
    if not (0.1 < alpha < 2.0):
        return -np.inf
    if not (2.0 < beta < 5.0):
        return -np.inf
    if not (0.01 < a2h < 2.0):
        return -np.inf
    if not (0.01 <= f_mis <= 1.0):
        return -np.inf

    lp = 0.0
    lp += -0.5 * ((f_mis - 0.18) / 0.08) **2
    lp += -0.5 * ((ln_sigma_mis - (-0.82)) / 0.22) ** 2
    return lp

def log_prior_boost(theta_boost, source_bins):

    if len(theta_boost) != 2 *len(source_bins):
        raise ValueError(
            "wrong number of bins")
    
    for k in range(len(source_bins)):
        B0, Rs = theta_boost[2*k], theta_boost[2*k +1]
        
        if not (0.1 < B0 < 7.0):
            return -np.inf
        if not (0.1 < Rs < 7.0):
            return -np.inf
    return 0.0

def log_probability_boost(theta_boost, *, thmid,source_bins, boost_pz, cov_boost_pz):
    lp = log_prior_boost(theta_boost, source_bins)
    if not np.isfinite(lp):
        return -np.inf
    boost_params_by_bin = {j: (theta_boost[2*k], theta_boost[2*k +1])
                           for k, j in enumerate(source_bins)}
    
    return lp + log_likelihood_boost_only(boost_params_by_bin,  thmid, source_bins, boost_pz, cov_boost_pz)

def draw_boost_params_from_chain(boost_chain_samples, source_bins, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, len(boost_chain_samples))
    row = boost_chain_samples[idx]
    return {j: (row[2*k], row[2*k +1]) for k, j in enumerate(source_bins)}

def log_probability_main(theta, *, profile, cosmo, M500c, z, sigma_z, weight,
                         thmid, source_bins, nz_grids, gammat, cov, boost_chain_samples,
                         sigma_lnM=0.2, n_draws=7, fit_evolution=True,
                         rng = None):

    if rng is None:
        rng = np.random.default_rng()

    lp = log_prior_main(theta, fit_evolution=fit_evolution)
    if not np.isfinite(lp):
        return -np.inf

    if fit_evolution:
        A_mass, eta, zeta, alpha, beta, a2h, f_mis, ln_sigma_mis = theta
    else:
        A_mass, alpha, beta, a2h, f_mis, ln_sigma_mis = theta
        eta = zeta = 0.0
    sigma_mis = np.exp(ln_sigma_mis)
    theta_params = dict(A_mass=A_mass, eta=eta, zeta=zeta, alpha =alpha, beta=beta,
                        a2h = a2h, f_mis= f_mis, sigma_mis = sigma_mis)

    log_likes = np.empty(n_draws)
    for k in range(n_draws):
        boost_params = draw_boost_params_from_chain(boost_chain_samples, source_bins, rng=rng)
        log_likes[k] = log_likelihood_lensing_one_draw(
            profile = profile, cosmo=cosmo, theta_params = theta_params,
            M500c=M500c, z=z, sigma_z=sigma_z, weight=weight, thmid=thmid,
            source_bins=source_bins, nz_grids=nz_grids, gammat=gammat, cov=cov,
            boost_params=boost_params,
            sigma_lnM=sigma_lnM, rng=rng)

    ll_marginal = logsumexp(log_likes) - np.log(n_draws)
    return lp +ll_marginal


def log_probability_main_join(theta, *, profile, cosmo, datasets, boost_chains,
                              sigma_lnM=0.2, n_draws = 7, fit_evolution=True, rng=None):

    if rng is None:
        rng = np.random.default_rng()
    if len(datasets) != len(boost_chains):
        raise ValueError("data sets and boost chanins different lengths")

    lp = log_prior_main(theta, fit_evolution=fit_evolution)
    if not np.isfinite(lp):
        return ~np.inf

    if fit_evolution:
        A_mass, eta, zeta, alpha, beta, a2h, f_mis,ln_sigma_mis = theta
    else:
        A_mass, alpha, beta, a2h, f_mis, ln_sigma_mis = theta
        eta = zeta = 0.0
    sigma_mis = np.exp(ln_sigma_mis)

    theta_params = dict(A_mass = A_mass, eta=eta, zeta=zeta, alpha=alpha,
                        beta=beta, a2h=a2h, f_mis=f_mis,
                        sigma_mis=sigma_mis)
    total_ll = 0.0

    for dataset, boost_chain in zip(datasets, boost_chains):
        log_likes = np.empty(n_draws)
        for k in range(n_draws):
            boost_params = draw_boost_params_from_chain(
                boost_chain, dataset["source_bins"], rng=rng)
            log_likes[k] = log_likelihood_lensing_one_draw(
                profile=profile, cosmo=cosmo, theta_params=theta_params,
                M500c=dataset["M500c"], z=dataset["z"], sigma_z=dataset["sigma_z"],
                weight=datase["weight"], thmid=dataset["thmid"], source_bins=dataset["source_bins"],
                nz_grids=dataset["nz_grids"], gammat = dataset["gammat"],
                cov=dataset["cov"], boost_params=boost_params, sigma_lnM=sigmalnMy,
                rng = rng)
        total_ll += logsumexp(log_likes) - np.log(n_draws)
    return lp + total_ll



rng = np.random.default_rng(42)

cosmo = ccl.Cosmology(Omega_c = 0.3075 -0.0486, Omega_b=0.0486, h=0.6774, n_s= 0.96, sigma8=0.81)

mdef = ccl.halos.MassDef(500, "critical")
conc = ccl.halos.ConcentrationConstant(c=2.45, mass_def=mdef)
profile = HaloProfileClusterTotal(mass_def=mdef, concentration=conc, alpha = 1.0, beta = 3.5,
                                  a2h = 1.0, p_of_k_a = cosmo.get_linear_power())

check_2halo = False
if check_2halo:
    import sys
    M_mean = 1.23 * np.median(M500c)
    z_eff = np.median(z)
    a_eff = 1.0 / (1.0 + z_eff)
    theta_rad = np.deg2rad(np.asarray(thmid) / 60.0)
    R = theta_to_comoving_R(cosmo, z_eff, theta_rad)
    print(f"M = {M_mean:.2e} Msun, z = {z_eff:.2f}")
    print("theta [arcmin]:", np.round(thmid, 2))
    print("R [comoving Mpc]:", np.round(R, 2))

    def dsig(p):
        return p.cumul2d(cosmo, R, M_mean, a_eff) - p.projected(cosmo, R, M_mean, a_eff)

    fig, ax = plt.subplots(figsize=(6,4))
    for label, pk, ls in [("nonlinear", "delta_matter:delta_matter", "--"),
                          ("linear", cosmo.get_linear_power(), "-")]:
        p = HaloProfileClusterTotal(mass_def=mdef, concentration=conc, alpha=1.0,
                                    beta=3.5, a2h=1.0, p_of_k_a=pk)
        ds_1h = dsig(p.cdm) + dsig(p.baryon)
        ds_2h = dsig(p.twohalo)
        print(f"\n{label}: DeltaSigma_2h / DeltaSigma_1h =", np.round(ds_2h / ds_1h, 3))
        ax.loglog(thmid, ds_2h, ls, label=f"2-halo ({label})")
    ax.loglog(thmid, ds_1h, "k-", lw=2, label="1-halo")
    ax.set_xlabel(r"$\theta$ [arcmin]")
    ax.set_ylabel(r"$\Delta\Sigma$ (comoving units)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("twohalo_check.png", dpi=150)
    print("\nsaved twohalo_check.png")
    sys.exit(0)

#replace this with actual data from paper first

#M500c = dataset["M500c"]
#z = dataset["z"]
#sigma_z = dataset["sigma_z"]
#thmid = dataset["thmid"]
source_bins = [3,4]


with h5py.File("/cephfs/soukdata/shared_projects/sz_cfc/mass_cal/data/DES/y3kp_cats/DESY3_sompz_v0.50.h5", "r") as f:
    pz = f["catalog/sompz/pzdata"]


    print(list(pz.keys()))
    for k in pz.keys():
        a = pz[k][:]
        print(f"{k:10s} shape ={a.shape} min ={a.min():.4f} max={a.max():.4f}")


    z_edges = np.linspace(0.0, 3.0, 301)
    zg = 0.5 * (z_edges[:-1] + z_edges[1:])

    h5_key = {3: "bin2", 4: "bin3"}
    nz_grids = {}
    for j in source_bins:
        nz = pz[h5_key[j]][:].astype(float)
        nz /= np.trapezoid(nz, zg)
        nz_grids[j] = (zg, nz)

        
required_keys = ["gammat", "cov", "weight", "boost_pz", "cov_boost_pz", "thmid", "M500c", "z", "sigma_z"]

#loads data in, makes like easier
def load_bin(path):
    with np.load(path) as d:
        missing = [k for k in required_keys if k not in d.files]
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        out = {k: np.asarray(d[k]) for k in required_keys}
    out["M500c"] = out["M500c"].astype(float) * 1e14
    out["z"] = out["z"].astype(float)
    out["sigma_z"] = np.nan_to_num(out["sigma_z"].astype(float), nan=0.0)
    return out
    

#nneed to change this for relevent bins

scratch = "/shared_home/lreilly/scratch"
data_files = {
    "lowz_lowy": f"{scratch}/tangentialshear_0z0y_DR6.npz",
    "lowz_highy": f"{scratch}/tangentialshear_0z1y_DR6.npz",
    "highz_lowy": f"{scratch}/tangentialshear_1z0y_DR6.npz",
    "highz_highy": f"{scratch}/tangentialshear_1z1y_DR6.npz",
    }

bins = {tag: load_bin(path) for tag, path in data_files.items()}
for b in bins.values():
    b["nz_grids"] = nz_grids

for tag, b in bins.items():
    n_data, n_cl = len(b["gammat"]), len(b["z"])
    assert n_data == len(source_bins) * len(b["thmid"]), tag
    assert b["cov"].shape == (n_data, n_data), tag
    assert b["cov_boost_pz"].shape == (len(b["boost_pz"]),) * 2, tag
    assert b["weight"].shape == (n_cl, n_data), (tag, b["weight"].shape)
    assert len(b["M500c"]) == len(b["sigma_z"]) == n_cl, tag
    for k in ("gammat", "cov", "boost_pz", "cov_boost_pz", "M500c", "z"):
        assert np.all(np.isfinite(b[k])), f"{tag}: non-finite {k}"
    print(f"{tag}: {n_cl} clusters, thmid = {np.round(b['thmid'], 2)}")

for zt in ("lowz", "highz"):
    assert np.allclose(bins[f"{zt}_lowy"]["thmid"], bins[f"{zt}_highy"]["thmid"]), zt

def summarise(chain, names, label):
    print(f"\n{label}")
    print(f"{'param':<14}{'median':>10}{'16-84%':>24}")
    for i, name in enumerate(names):
        p16, p50, p84 = np.percentile(chain[:, i], [16, 50, 84])
        print(f"{name:<14}{p50:>10.3f} [{p16:.3f}, {p84:.3f}] "
              f"(-{p50 - p16:.3f} / +{p84 - p50:.3f})")

fork = mp.get_context("fork")


#stage 1, runs for boost factors

nwalkers1, ndim1 = 112, 4
names1 = ["B0_3", "Rs_3", "B0_4", "Rs_4"]

def stage1_cache(tag):
    return f"stage1_flastchain_{tag}_DR6own.npy"

def run_stage1(tag, b):
    if os.path.exists(stage1_cache(tag)):
        print(f"stage 1 {tag}: loading converged chain from {stage1_cache(tag)}")
        return np.load(stage1_cache(tag))

    lp_boost = partial(log_probability_boost, thmid=b["thmid"], source_bins=source_bins,
                       boost_pz=b["boost_pz"], cov_boost_pz=b["cov_boost_pz"])
    print(f"\n=== stage 1 ({tag}): boost MCMC ===")
    max_steps1, check_every1, old_tau1 = 300_000, 2000, np.inf
    fname = f"stage1_chain_{tag}_DR6own.h5"
    backend1 = emcee.backends.HDFBackend(fname)
    fresh = not os.path.exists(fname) or backend1.iteration == 0
    if fresh:
        backend1.reset(nwalkers1, ndim1)
        initial_state1 = np.column_stack([
            rng.uniform(1.0, 2.0, nwalkers1),
            rng.uniform(0.6, 1.2, nwalkers1),
            rng.uniform(0.8, 1.6, nwalkers1),
            rng.uniform(1.0, 1.6, nwalkers1),
            ])

        bad = ~np.isfinite([lp_boost(p) for p in initial_state1])
        if bad.any():
            raise ValueError(f"{bad.sum()} stage 1 walkers start outside prior ({tag})")
    else:
        initial_state1 = backend1.get_last_sample()

    converged_flag = False
    with fork.Pool(processes=n_cores) as pool1:
        sampler1 = emcee.EnsembleSampler(nwalkers1, ndim1, lp_boost, pool=pool1, backend =backend1)
        for _ in sampler1.sample(initial_state1, iterations=max_steps1, progress=False):
            if out_of_time():
                break
            if sampler1.iteration % check_every1:
                continue
            tau1 = sampler1.get_autocorr_time(tol=0)
            converged = sampler1.iteration > 20000
            converged &= np.all(tau1 * 50 < sampler1.iteration)
            converged &= np.all(np.abs(old_tau1 - tau1) / tau1 < 0.01)
            print(f"\n step {sampler1.iteration}: tau={tau1}, converged={converged}")
            if converged:
                converged_flag = True
                break
            old_tau1 = tau1

    if not converged_flag:
        if out_of_time():
            print(f"stage 1 {tag}: out of time, stopping so the job can resubmit", flush=True)
            sys.exit(0)
        raise RuntimeError(f"stage1 {tag} did not converge: need resubmission")

    tau1 = sampler1.get_autocorr_time(tol=0)
    burn_in1 = int(5*np.max(tau1))
    chain = sampler1.get_chain(discard=burn_in1, flat=True)
    summarise(chain, names1, f"stage 1 {tag}: boost parameters (burn-in {burn_in1})")
    np.save(stage1_cache(tag), chain)
    return chain


#stage 2 stuff

priors = {
    "A_mass": (0.1, 10.0, 1.0, 1.8),
    "eta": (-10.0, 10.0, -0.3, 0.3),
    "zeta": (-10.0, 10.0, -0.5, 0.5),
    "alpha": (0.1, 2.0, 0.7, 1.3),
    "beta": (2.0, 5.0, 3.2, 3.8),
    "a2h": (0.01, 2.0, 0.7, 1.3),
    "f_mis": (0.01, 1.0, 0.14, 0.22),
    "ln_sigma_mis": (-np.inf, np.inf, -0.95, -0.7),
    }

gauss_priors = {"f_mis": (0.18, 0.08), "ln_sigma_mis": (-0.82, 0.22)}
fixed = dict(eta=0.0, zeta=0.0, alpha=1.0, beta=3.5)
n_draws = 7

def param_names(model, free_evolution):
    names = ["A_mass"]
    if free_evolution:
        names += ["eta", "zeta"]
    if model == "full":
        names += ["alpha", "beta"]
    return names + ["a2h", "f_mis", "ln_sigma_mis"]

def log_prior_generic(theta, names):
    lp = 0.0
    for name, val in zip(names, theta):
        lo, hi = priors[name][:2]
        if not (lo < val < hi):
            return -np.inf
        if name in gauss_priors:
            mu_p, sd_p = gauss_priors[name]
            lp += -0.5 * ((val - mu_p) / sd_p) **2
    return lp

profiles = {
    "full": profile,
    "nfw": HaloProfileNFWOnly(mass_def=mdef, concentration=conc, a2h=1.0,
                              p_of_k_a=cosmo.get_linear_power()),
    }

_run = {}

def lp_run(theta):
    names = _run["names"]
    if out_of_time():
        return None, param_names(run["model"], run["free_evolution"])
    lp = log_prior_generic(theta, names)
    if not np.isfinite(lp):
        return -np.inf
    p = {**fixed, **dict(zip(names, theta))}
    theta_params = dict(A_mass = p["A_mass"], eta=p["eta"], zeta=p["zeta"],
                        alpha=p["alpha"], beta=p["beta"], a2h=p["a2h"],
                        f_mis=p["f_mis"], sigma_mis=np.exp(p["ln_sigma_mis"]))
    total = 0.0
    try:
        for tag in _run["datasets"]:
            b = bins[tag]
            ll = np.empty(n_draws)
            for k in range(n_draws):
                bp = draw_boost_params_from_chain(boost_chains[tag], source_bins)
                ll[k] = log_likelihood_lensing_one_draw(
                    profile=_run["profile"], cosmo=cosmo, theta_params=theta_params,
                    M500c=b["M500c"], z=b["z"], sigma_z=b["sigma_z"], weight=b["weight"],
                    thmid=b["thmid"], source_bins=source_bins, nz_grids=b["nz_grids"],
                    gammat=b["gammat"], cov=b["cov"], boost_params=bp, sigma_lnM=0.2)
            total += logsumexp(ll) - np.log(n_draws)
    except (ValueError, np.linalg.LinAlgError, ccl.CCLError):
        return -np.inf
    if not np.isfinite(total):
        return -np.inf
    return lp + total

def run_stage2(run, nwalkers=112, max_steps=300_000, check_every=200):
    name = run["name"]
    done_file = f"sstage2_flatchain_{name}_DR6own.npy"
    names = param_names(run["model"], run["free_evolution"])
    if os.path.exists(done_file):
        print(f"stage 2 {name}: already converged, loading {done_file}")
        return np.load(done_file), names

    ndim = len(names)
    _run.clear()
    _run.update(names=names, datasets=run["datasets"], profile=profiles[run["model"]])


    print(f"\n=== Stage 2: {name} | model = {run['model']} | data={run['datasets']} | params {names} ===")
    fname = f"stage2_chain_{name}_DR6own.h5"
    backend = emcee.backends.HDFBackend(fname)
    fresh = not os.path.exists(fname) or backend.iteration == 0
    if fresh:
        backend.reset(nwalkers, ndim)
        p0 = np.column_stack([rng.uniform(priors[n][2], priors[n][3], nwalkers) for n in names])
    else:
        p0 = backend.get_last_sample()

    converged, old_tau = False, np.inf
    with fork.Pool(processes=n_cores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp_run, pool=pool, backend=backend)
        
        for _ in sampler.sample(p0, iterations=max_steps, progress=False):
            if out_of_time():
                break
            if sampler.iteration % check_every:
                continue
            tau = sampler.get_autocorr_time(tol=0)
            converged = np.all(tau * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            print(f"\n {name} step {sampler.iteration}: tau={tau}, converged={converged}")
            if converged:
                break
            old_tau = tau

    if not converged:
        print(f"stage 2 {name} not converged: requires resubmission from {fname}")
        return None, names

    tau = sampler.get_autocorr_time(tol=0)
    burn_in = int(5 * np.max(tau))
    chain = sampler.get_chain(discard=burn_in, flat=True)
    summarise(chain, names, f"stage 2 {name} (burn-in {burn_in}, "
              f"acceptance { np.mean(sampler.acceptance_fraction):.3f})")
    np.save(done_file, chain)
    return chain, names

lowz = ["lowz_lowy", "lowz_highy"]
highz = ["highz_lowy", "highz_highy"]
total = lowz + highz

runs = []
for free_evo in (False, True):
    evo = "evo" if free_evo else "noevo"
    runs += [
        dict(name=f"lowz_full_{evo}", datasets=lowz, model="full", free_evolution=free_evo),
        dict(name=f"highz_full_{evo}", datasets=highz, model="full", free_evolution=free_evo),
        dict(name=f"combined_full_{evo}", datasets=total, model="full", free_evolution=free_evo),
        dict(name=f"combined_nfw__{evo}", datasets=total, model="nfw", free_evolution=free_evo),
        
]

for i, r in enumerate(runs):
    print(f"run {i}: {r['name']}")

task = os.environ.get("SLURM_ARRAY_TASK_ID")
selected = runs if task is None else [runs[int(task)]]
needed = sorted({t for r in selected for t in r["datasets"]})

if task is not None:
    missing = [t for t in needed if not os.path.exists(stage1_cache(t))]
    if missing:
        raise RuntimeError(f"stage 1 chains missing for {missing}; "
                           f"run once with stage1_only=1 before the array job")

boost_chains = {t: run_stage1(t, bins[t]) for t in needed}

if os.environ.get("stage1_only"):
    print("stage 1 finished for:", needed)
    mark_finished()
    sys.exit(0)


for tag in needed:
    b, bc = bins[tag], boost_chains[tag]
    bp_check = {j: tuple(np.median(bc[:, 2*k:2*k+2], axis=0)) for k, j in enumerate(source_bins)}
    gt_check = assemble_gt_model(
        profile=profiles["full"], cosmo=cosmo, M_SZ_arr = b["M500c"], z_arr=b["z"],
        w_arr=b["weight"], theta_arcmin=b["thmid"], source_bins= source_bins,
        nz_grids = b["nz_grids"], A_mass=1.23, eta =0.0, zeta =0.0, alpha=1.0, beta=3.5,
        a2h=1.0, f_mis=0.18, sigma_mis=np.exp(-0.82), boost_params=bp_check,
        sigma_lnM=0.0, sigma_z_arr=b["sigma_z"], rng = np.random.default_rng(0))
    print(f"{tag} gammat / model at A_mass=1.23:",
          np.round(b["gammat"] / flatten_gt_model(gt_check, source_bins), 2))

results = {r["name"]: run_stage2(r) for r in selected}

print("\n============== summary ===============")

for name, (chain, names) in results.items():
    if chain is None:
        print(f"{name:<24} not converged")
        continue
    line = f"{name:<24}"
    for par in ("A_mass", "eta", "zeta"):
        if par in names:
            p16, p50, p84 = np.percentile(chain[:, names.index(par)], [16, 50, 84])
            line += f" {par} ={p50:.3f} (-{p50-p16:.3f}/+{p84-p50:.3f})"
    print(line)

if all(ch is not None for ch, _ in results.values()):
    mark_finished()
    print("\nAll selected runs converged.")
            


                                





#truth stuff was here but got rid of it

#gammat = np.asarray(gammat)
#cov = bock_diag(*[np.asarray(cov[j]) for j in source_bins])

#boost_pz = np.concatenate([np.asarray(boost_pz[j]) for j in source_bins])
#cov_bost_pz = block_diag(*[np.asarray(cov_boost_pz[j]) for j in source_bins])
