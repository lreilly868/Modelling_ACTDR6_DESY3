# Modelling_ACTDR6_DESY3
Modelling code + data files containing all relevant ACT DR6 data

The main places that I am guessing that something went wrong are in the custom built functions based on CCLs NFW and GNFW models.

The reasoning behind custom building them is that I was running into issues primarily when trying to sum the profiles together when the one and two halo term are combined. Doing it this way bypasses that and to my knowledge works.

The functions that I think you should look at are:
- HaloProfileNFWCDM() : cold dark matter NFW profile
- HaloProfileGNFWBaryon() : baryon GNFW profile
- HaloProfile2Halo() : two-halo term profile (most likely culprit)
- HaloProfileCLusterTotal() : combination of the one and two halo profiles

There's also the possibility that there are issues with the actual implementation of the MCMC, but I doubt those would be giving us the discrepancies in the corner plots. Even so, wouldn't be a terrible idea to look into the likelihood functions (there are a few), as prior to including extra bins the result was closer to the papers.

Additionally, I have also uploaded my data and the ACT DR5 papers data in case you are interested in running the code. They are stored with the same names for each set of data, but DR5 just saves a few extra things that I don't use in this modelling. It should still work with DR5, but there is a small possibility it doesn't.

Alongside the code is the slurm file I am running it on. The modelling can be split into 4 seperate parts:
-low redshift (includes both compton y bins)
-high redshift (same as above)
-low+high redshift
-NFW only (no baryon correction, just sets f_b = 0)

and these are split between two modes, one that sets redshift and mass evolution (eta = 0, zeta = 0) and one that leaves them as free parameters.

Let me know if you need anything else clarifying for this hellish code, and good luck (lord knows I needed it)

Luke


