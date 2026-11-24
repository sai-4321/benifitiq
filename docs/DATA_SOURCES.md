# Data sources

BenefitIQ uses **only public documents and synthetic data**. No real patient or member data, ever.

## Knowledge base: public Medicare.gov publications

Publisher: Centers for Medicare & Medicaid Services (CMS), U.S. Department of Health & Human Services.
License: **public domain** - works of the U.S. federal government are not subject to copyright (17 U.S.C. § 105).

| Product # | Title | Download URL |
|---|---|---|
| 10050 | Medicare & You (official handbook) | https://www.medicare.gov/publications/10050-Medicare-and-You.pdf |
| 10110 | Your Guide to Medicare Preventive Services | https://www.medicare.gov/publications/10110-Your-Guide-to-Medicare-Preventive-Services.pdf |
| 11109 | Your Guide to Medicare Prescription Drug Coverage | https://www.medicare.gov/publications/11109-your-guide-to-medicare-prescription-drug-coverage.pdf |
| 11045 | Medicare Coverage of Durable Medical Equipment & Other Devices | https://www.medicare.gov/publications/11045-medicare-coverage-of-dme-and-other-devices.pdf (saved as `11045-medicare-coverage-of-dme.pdf`) |

Get them with `python scripts/download_docs.py`. The PDFs are git-ignored (`data/raw/`); the script writes
`data/raw/manifest.json` with each file's page count, size and SHA-256 fingerprint, because these links always
serve the newest edition. If a download fails, open https://www.medicare.gov/publications, search the product
number, click **Download PDF** and save it into `data/raw/` with the file name above.

## Member and plan data: synthetic

`python scripts/generate_synthetic_data.py` builds `data/synthetic/members.db` (SQLite, git-ignored) with Faker and a
fixed seed (42), so every run is identical. Every row has `synthetic = 1`.

| Table | Rows | Contents |
|---|---|---|
| `plans` | 4 | PLN-001 Silver PPO, PLN-002 Gold HMO, PLN-003 Bronze EPO, PLN-004 Medicare Advantage HMO: deductible, copays, OOP max, coinsurance, referral rule |
| `members` | 50 | M1001-M1050, fake name/DOB/phone, plan, coverage start, status |
| `prior_auth_rules` | 40 | 10 services x 4 plans (MRI, CT, specialist, PT, DME, home health, SNF, outpatient surgery, sleep study, genetic testing) |

Stable sample IDs: M1001 -> PLN-001, M1002 -> PLN-002, M1003 -> PLN-003, M1004 -> PLN-004; every 10th member
(M1010, M1020 ...) is inactive. Plan values are invented for the demo and do not describe any real insurer.
