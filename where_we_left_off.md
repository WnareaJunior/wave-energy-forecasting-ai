Everything is in motion. Here's the state of your feature list and what's running:

Data now being collected — all over the same PNW box (130–124°W, 46–50.5°N)

┌──────────────────────────────┬─────────────────────────────┬─────────────┐
│            Input             │           Source            │   Status    │
├──────────────────────────────┼─────────────────────────────┼─────────────┤
│ Significant wave height (Hs) │ Copernicus VHM0 / NOAA swh  │ downloading │
├──────────────────────────────┼─────────────────────────────┼─────────────┤
│ Peak period (Tp)             │ Copernicus VTPK / NOAA      │ downloading │
│                              │ perpw                       │             │
├──────────────────────────────┼─────────────────────────────┼─────────────┤
│ Mean period                  │ Copernicus VTM02 + VTM10 /  │ downloading │
│                              │ NOAA mp1                    │             │
├──────────────────────────────┼─────────────────────────────┼─────────────┤
│ Wave direction               │ Copernicus VMDR / NOAA      │ downloading │
│                              │ dirpw                       │             │
├──────────────────────────────┼─────────────────────────────┼─────────────┤
│ Swell height                 │ Copernicus VHM0_SW1/SW2 /   │ added today │
│                              │ NOAA SWELL                  │             │
├──────────────────────────────┼─────────────────────────────┼─────────────┤
│ Wind-sea height              │ Copernicus VHM0_WW / NOAA   │ added today │
│                              │ WVHGT                       │             │
├──────────────────────────────┼─────────────────────────────┼─────────────┤
│ Wind speed + direction       │ NOAA u/v (10 m)             │ downloading │
├──────────────────────────────┼─────────────────────────────┼─────────────┤
│ Ocean current                │ GLORYS uo/vo (surface)      │ downloading │
├──────────────────────────────┼─────────────────────────────┼─────────────┤
│ Sea surface temperature      │ GLORYS thetao               │ downloading │
├──────────────────────────────┼─────────────────────────────┼─────────────┤
│ Bathymetry / water depth /   │ static layer (GEBCO/ETOPO)  │ next step   │
│ distance from coast          │ — not built yet             │             │
├──────────────────────────────┼─────────────────────────────┼─────────────┤
│ Season, month, lat, lon      │ derived at table-build time │ trivial     │
└──────────────────────────────┴─────────────────────────────┴─────────────┘

Three self-terminating EC2 instances are running: the Japan wave backfill (finishing what you started), the NOAA reforecast (restarted with the widened 8-variable set before it had produced any files, so the archive will be uniform), and a new PNW instance that runs the 1980–2023 PNW wave hindcast then the 2000–2019 GLORYS currents/SST. GLORYS was validated end to end on year 2000 before deployment (surface layer, plausible SST and current ranges). All committed on copernicus-backfill-v2 (commit 0a3a110).

On the objective — you're right, and the data supports it

For P ∝ H²T, the correct T is the energy period Te, and Copernicus's VTM10 is Te — so power flux can be computed exactly on the hindcast side (NOAA's mp1 approximates it for the forecast side). The framing "power minus fatigue minus risk minus maintenance" maps cleanly onto features we'll have: fatigue drivers are exactly the wind-sea/swell partition, direction spread, and Hs extremes; deployment risk correlates with depth and distance from coast. My suggestion when we get to labels: have XGBoost predict the physical quantities (Hs, Te → power flux) rather than the reward directly, and keep the reward function as a separate, tunable layer on top — you'll want to iterate on the fatigue and cost weights without retraining.

What remains before "XGBoost-ready"

1. Static layer: one small NetCDF with bathymetry, water depth, and distance-from-coast on the PNW grid (a few hours of work, no EC2 needed).
2. The join job: align the three dynamic sources onto a common grid/timebase, dedupe the old chunk-boundary timestamps, compute power flux + derived features, and write a training table (Parquet) to panthalassa-ocean-processed.

I'd start on the static layer next while the backfills run — want me to go ahead?

✻ Churned for 7m 54s · 1 shell still running

────────────