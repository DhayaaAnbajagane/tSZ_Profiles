# tSZ profile models

This repository contains:

1. The code used in Anbajagane+ 2021 to construct the theoretical model for the halo x y correlation (alternatively called the halo-y profile, or also the halo-tSZ profile)
2. The profile and log-derivative results shown in Anbajagane+ 2021


## Quickstart: tSZ Theory Model

We provide code to compute the theoretical one-halo and two-halo terms shown in Anbajagane+ 2021. This code is fully
consistent with the CCL framework, and so can be used to easily compute cross-correlations between tSZ and other
probes (galaxy, matter etc.). It is also easy to swap the Battaglia+ 2012 pressure profile model used in this work
with other existing models.

```
import sys
sys.path.append("<path to 'tSZ_Profiles'>/tSZ_Profiles")

import pyccl as ccl
import numpy as np
import tSZ_Theory

cosmo = ccl.Cosmology(Omega_c = 0.26, Omega_b = 0.04, h=0.7, sigma8 = 0.8, n_s = 0.96)

r = np.geomspace(0.1, 30, 100)
z = 0.5
a = 1/(1 + z)

#Unsmoothed case
one_halo, two_halo = tSZ_Theory.Total_halo_model(cosmo, r, M = 1e15, a = a, mass_def = None, Model_def = '500_SH', truncate = None)

total_halo_model   = one_halo + two_halo

#Smoothed case
one_halo, two_halo = tSZ_Theory.Smoothed_Total_halo_model(cosmo, r, M = 1e15, a = a, FWHM_arcmin = 1, 
                                                          mass_def = None, Model_def = '500_SH', truncate = None)

```

## Quickstart: tSZ Observational Results
