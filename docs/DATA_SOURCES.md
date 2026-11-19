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
| 11045 | Medicare Coverage of Durable Medical Equipment & Other Devices | https://www.medicare.gov/publications/11045-medicare-coverage-of-dme.pdf |

Get them with `python scripts/download_docs.py`. The PDFs are git-ignored (`data/raw/`); the script writes
`data/raw/manifest.json` with each file's page count, size and SHA-256 fingerprint, because these links always
serve the newest edition. If a download fails, open https://www.medicare.gov/publications, search the product
number, click **Download PDF** and save it into `data/raw/` with the file name above.

## Member and plan data: synthetic

Generated locally with Faker by `scripts/generate_synthetic_data.py` (BIQ-7). Every name, ID and date is fake.
