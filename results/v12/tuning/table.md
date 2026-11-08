200 songs (RWC-P and GuitarSet, train + val). Cache matches: both 90, 24 bins 110, 12 bins 0, neither 0.
26 songs whose 12-bin and 24-bin estimates differ by more than 3 cents:

| song | split | 24 bins | 12 bins | change | cache vs 24-bin features | cache vs 12-bin features | cache built with |
|---|---|---:|---:|---:|---:|---:|---|
| guitarset/02_Rock1-130-A_comp | train | -24.5 | +28.0 | +52.5 | 0.0000 | 0.0600 | 24 bins |
| rwc/RWC_P038 | train | -3.5 | -50.0 | -46.5 | 0.0001 | 0.1397 | 24 bins |
| rwc/RWC_P045 | train | +3.0 | -16.0 | -19.0 | 0.0000 | 0.0517 | 24 bins |
| guitarset/02_Jazz1-130-D_comp | train | -6.5 | +11.0 | +17.5 | 0.0000 | 0.0219 | 24 bins |
| rwc/RWC_P009 | train | +18.5 | +4.0 | -14.5 | 0.0001 | 0.0522 | 24 bins |
| guitarset/03_BN1-147-Gb_comp | train | +19.5 | +10.0 | -9.5 | 0.0000 | 0.0112 | 24 bins |
| guitarset/03_Jazz1-200-B_comp | train | +10.5 | +1.0 | -9.5 | 0.0000 | 0.0064 | 24 bins |
| guitarset/01_Funk3-112-C#_comp | train | +8.5 | +0.0 | -8.5 | 0.0000 | 0.0182 | 24 bins |
| rwc/RWC_P095 | train | +10.0 | +18.0 | +8.0 | 0.0001 | 0.0234 | 24 bins |
| guitarset/01_Rock1-130-A_comp | train | +5.5 | +13.0 | +7.5 | 0.0000 | 0.0112 | 24 bins |
| guitarset/04_Rock3-148-C_comp | val | +2.5 | +10.0 | +7.5 | 0.0000 | 0.0135 | 24 bins |
| rwc/RWC_P073 | train | -5.0 | +2.0 | +7.0 | 0.0000 | 0.0180 | 24 bins |
| guitarset/02_SS3-98-C_comp | val | +12.0 | +5.0 | -7.0 | 0.0000 | 0.0122 | 24 bins |
| guitarset/03_SS3-84-Bb_comp | val | +12.0 | +5.0 | -7.0 | 0.0000 | 0.0186 | 24 bins |
| guitarset/01_Rock3-148-C_comp | val | +14.0 | +8.0 | -6.0 | 0.0000 | 0.0103 | 24 bins |
| rwc/RWC_P037 | val | +5.0 | +0.0 | -5.0 | 0.0001 | 0.0156 | 24 bins |
| guitarset/02_Funk3-112-C#_comp | train | +10.5 | +15.0 | +4.5 | 0.0000 | 0.0059 | 24 bins |
| guitarset/05_SS3-84-Bb_comp | val | +10.5 | +15.0 | +4.5 | 0.0000 | 0.0110 | 24 bins |
| guitarset/01_Jazz2-110-Bb_comp | val | +9.5 | +5.0 | -4.5 | 0.0000 | 0.0079 | 24 bins |
| rwc/RWC_P054 | train | +10.5 | +6.0 | -4.5 | 0.0000 | 0.0139 | 24 bins |
| rwc/RWC_P072 | train | +0.0 | +4.0 | +4.0 | 0.0000 | 0.0132 | 24 bins |
| guitarset/04_BN1-147-Gb_comp | train | +17.0 | +13.0 | -4.0 | 0.0000 | 0.0044 | 24 bins |
| guitarset/02_Funk2-119-G_comp | val | +4.0 | +8.0 | +4.0 | 0.0000 | 0.0048 | 24 bins |
| guitarset/01_BN1-147-Gb_comp | train | +6.5 | +10.0 | +3.5 | 0.0000 | 0.0059 | 24 bins |
| guitarset/04_Funk2-108-Eb_comp | val | +9.5 | +6.0 | -3.5 | 0.0000 | 0.0077 | 24 bins |
| rwc/RWC_P066 | train | +11.0 | +8.0 | -3.0 | 0.0000 | 0.0102 | 24 bins |

Largest difference between the cache and its 24-bin features: 0.00007.
