# DataBrew — ICT324 Data Mining Project

**Market Basket Analysis and Temporal Basket Segmentation in Bakery Retail: A Combined Data Mining Study**

Rangsit University International College · Academic Year 2026

## Research question

How does temporal and behavioral transaction segmentation affect the confidence and lift of association rules relative to global mining?

## Dataset

[French Bakery Daily Sales — Matthieu Gimbert (Kaggle)](https://www.kaggle.com/datasets/matthieugimbert/french-bakery-daily-sales). The original CSV contains 234,005 sales lines from January 2021 to September 2022. Download `Bakery sales.csv` from the original source and place it beside the Python script. Raw data is not redistributed here; consult the source for licensing and attribution. Original business provenance has not been independently authenticated.

SHA-256 of the input used in the project: `af5eede55b6eb2efb6bebb0e6dd1a51568c7eb73554d2f8ad2563081025681b2`.

## Reproduce the experiments

Install Python 3.11 or newer (the packaging checks were performed on Python 3.13.5). From the repository directory:

```bash
python -m pip install -r requirements.txt
python DataBrew_bakery_analysis.py "Bakery sales.csv" reproduced_results
```

The dependency versions are those available in the packaging environment; the original run's exact environment was not recorded in the archived log. Minor numerical differences can arise across versions or platforms. Runtime is hardware-dependent. The script uses a random seed of 42, ten K-Means initializations, and a 5,000-receipt silhouette sample.

## Method

1. Audit the data; exclude tickets containing negative quantities and remaining nonpositive price/quantity rows. Retain repeated-looking lines with distinct source IDs.
2. Aggregate products by ticket number; construct distinct-product baskets and spending features.
3. Log-transform basket size and spending, standardize them alongside hour of day, and fit K-Means for K=2 through K=8.
4. Mine global Apriori and cluster-specific Apriori/FP-Growth using minimum support 1%, confidence 20%, and lift > 1. Both algorithms use custom Python implementations.
5. Compare frequent itemsets, median mining time across three runs, and matched global versus cluster-specific rules.

## Archived results

| Metric | Value |
|---|---:|
| Cleaned lines | 232,655 |
| Valid baskets | 135,672 |
| Multi-product baskets | 58,235 |
| Selected K | 3 |
| Sample silhouette score | 0.4525 |
| Global rules | 13 |
| Cluster-specific rule instances | 64 |
| Instances with higher confidence | 57/64 |
| Instances with higher lift | 14/64 |
| Baguette → Croissant global confidence | 9.67% |
| Baguette → Croissant cluster 0 confidence | 27.48% |

## Repository files

- `DataBrew_bakery_analysis.py`: reproducible analysis, with custom Apriori and FP-Growth.
- `requirements.txt`: dependency versions for the packaging environment.
- `DataBrew_bakery_results/audit.json`: data quality and cleaning counts.
- `DataBrew_bakery_results/kmeans_validation.csv`: WCSS and silhouette scores.
- `DataBrew_bakery_results/cluster_profiles.csv`: cluster summaries.
- `DataBrew_bakery_results/association_summary.csv`: rule counts and mining benchmarks.
- `DataBrew_bakery_results/association_rules.csv`: mined rules and support/confidence/lift.
- `DataBrew_bakery_results/matched_rules.csv`: matched global and segmented rule metrics.
- `DataBrew_bakery_results/run_log.txt`: original experiment log.

## Limitations

These are descriptive, in-sample results; selecting rules and clusters using completed baskets can inflate apparent recommendation relevance. The clusters are not strictly separated dayparts. Rule confidence is not held-out predictive precision, and higher confidence does not imply revenue growth. Live POS deployment would require a segment assignment using information available at recommendation time. Timings exclude preprocessing and rule generation. No real-world sales intervention or margin optimization was tested.

## Project website

[DataBrew — ICT324 project](https://sites.google.com/rsu.ac.th/data-brew-ict324-data-mining-p/home)
