---
title: 'FSEVAL: Feature Selection Evaluation Toolbox and Dashboard'
tags:
  - Python
  - feature selection
  - machine learning
  - evaluation
  - visualization
  - dashboard
authors:
  - name: Muhammad Rajabinasab
    corresponding: true
    affiliation: 1
  - name: Arthur Zimek
    affiliation: 1
affiliations:
  - name: Department of Mathematics and Computer Science, University of Southern Denmark, Odense, Denmark
    index: 1
date: 3 September 2026
bibliography: paper.bib
---

# Summary

Feature selection is the task of identifying the most informative variables in a dataset while discarding redundant or irrelevant ones. It is a core step in machine learning and data mining because it reduces the "curse of dimensionality" without sacrificing the interpretability that alternative dimensionality reduction techniques, such as Principal Component Analysis, autoencoders, t-SNE, or UMAP, give up in exchange for compactness [@pearson1901; @bourlard1988; @vandermaaten2008; @mcinnes2018]. Judging which feature selection algorithm is "best," however, is not a single-number question: the answer depends on the downstream task (classification versus clustering), on how many features are retained, and on how consistently the algorithm selects the same features when the data or its ordering changes. `FSEVAL` is a Python toolbox, paired with an interactive visualization dashboard, that packages these considerations into one reproducible pipeline. It runs supervised, unsupervised, and model-agnostic benchmarks across a grid of feature-retention ratios, computes several established and recently proposed evaluation metrics, profiles runtime scalability, and turns the resulting output into publication-ready figures and statistical rank analyses [@guyon2003introduction; @li2017feature].

# Statement of need

Researchers who develop or apply feature selection algorithms typically assemble their own evaluation scripts by hand, combining a classifier or a clustering routine [@lloyd1982] with ad hoc plotting code. Such scripts rarely account for stochastic variance across repeated runs, and they usually report performance at a single, arbitrarily chosen number of retained features, obscuring how a method behaves across the full range from mild to aggressive selection [@dy2003feature]. `FSEVAL` is built for researchers and practitioners who need a standardized, reproducible way to benchmark feature selection methods and communicate the results. It combines supervised metrics (accuracy, AUC), unsupervised metrics (clustering accuracy, normalized mutual information), a model-agnostic metric based on principal-component alignment (Average Angle Difference, AAD, [@rajabinasab2025metrics]), a dynamic performance-and-stability metric (FSDEM, [@rajabinasab2024fsdem]), and user-supplied custom metrics, all evaluated over feature-ratio grids rather than at a single point. Results feed directly into the `FSEVAL` dashboard, which produces multi-metric performance profiles, rank analyses and critical-difference diagrams based on both classical rank statistics [@demsar2006statistical] and the more recent magnitude-aware rank statistic (MARS) [@rajabinasab2026mars], and runtime-scalability plots, all exportable as publication-ready PDF figures or LaTeX table code. The intended audience is the feature selection research community, together with practitioners in domains such as malware detection [@jain2026enfestdroid] or high-dimensional and streaming data analysis [@dai2026online; @zou2026feature] who need to select and justify a feature selection method for their own pipeline.

# State of the field

Several tools address parts of the feature selection evaluation problem, but, to our knowledge, none combine them into a single comprehensive benchmarking and visualization pipeline. The scikit-feature repository, which underlies many of the benchmark datasets used with `FSEVAL`, provides algorithm implementations rather than an evaluation harness [@li2017feature]. `featsel` is closer in spirit: it is a mathematically rigorous C++ framework for benchmarking the efficiency of search algorithms and cost functions within a Boolean lattice [@vieirareis2017featsel]. Its focus, however, is on the computational optimization of the search trajectory itself, rather than on the downstream predictive quality or stability of the selected features, and it exposes no interactive visualization layer. `FSEVAL` was built rather than extended from these tools because its purpose is different: to give researchers a single, Python-native pipeline that evaluates feature selection methods from several complementary angles at once (supervised, unsupervised, and model-agnostic), sweeps feature-ratio grids by default instead of requiring the user to script that sweep, and turns the resulting output directly into publication-ready figures, rank statistics, and critical-difference diagrams through a companion dashboard. This combination of an extensible, metric-agnostic evaluation harness with an interactive results dashboard is, to our knowledge, not offered elsewhere.

# Software design

`FSEVAL` follows three design principles: modularity, extensibility, and treating feature selection as a dynamic process rather than a static, single-point result. The `FSEVAL` class exposes a small, declarative configuration surface (output directory, number of cross-validation folds, stochastic repetitions, evaluation types, metrics, and stability toggles), so that a full benchmark can be launched with a single call to `run()`, while a separate `timer()` method profiles runtime scalability against the number of instances and/or features. Rather than scoring a method at one arbitrarily chosen number of retained features, `FSEVAL` by default sweeps the first 10% of the feature-ratio grid at fine (0.5%) resolution and the full range at coarser (5%) resolution, so that an algorithm's "elbow point," where further feature removal starts to hurt performance, becomes visible directly in the output. Evaluation metrics are pluggable: the built-in supervised, unsupervised, and model-agnostic metrics are complemented by user-defined custom metrics passed as plain Python callables, and any scikit-learn classifier can replace the default Random Forest. A built-in random baseline [@rajabinasab2026random] is included to give unsupervised feature selection methods a reference point they are expected to beat. This design keeps `FSEVAL` usable both as a drop-in benchmark for common cases and as an extensible base for methodological research on evaluation itself, which motivated the group's development of the FSDEM and AAD metrics that the toolbox now implements natively [@rajabinasab2024fsdem; @rajabinasab2025metrics]. The dashboard is a separate, decoupled component that only consumes the CSV output of the toolbox, can deployed for interactive use locally or accessed directly via <https://fseval.imada.sdu.dk>.

The dashboard turns this evaluation output into publication-ready figures. Multi-metric performance profiles show how each metric evolves across the feature-ratio grid (\autoref{fig:lineplot}), and rank analyses are rendered as critical-difference diagrams under both the standard rank statistics [@demsar2006statistical] and the magnitude-aware rank statistic (MARS) [@rajabinasab2026mars] (\autoref{fig:cdd-standard}, \autoref{fig:cdd-mars}). Runtime scalability is visualized separately against the number of features (\autoref{fig:runtime-features}) and the number of instances (\autoref{fig:runtime-instances}), letting users trade off predictive utility against computational cost directly from the exported figures. 

The full API reference for the FSEVAL class and its parameters, together with additional worked examples, is available in the extended version of this paper [@rajabinasab2026fseval].

![Publication-ready line plot generated by the dashboard, demonstrating the performance trajectory and variance of multiple algorithms.\label{fig:lineplot}](lineplot.pdf)

![Standard rank statistics critical-difference diagram (10% feature ratio) provided by FSEVAL.\label{fig:cdd-standard}](CD_Standard_AUC_10Percent.pdf){ width=49% }
![MARS magnitude-aware rank statistics critical-difference diagram (10% feature ratio) provided by FSEVAL.\label{fig:cdd-mars}](CD_MARS_AUC_10Percent.pdf){ width=49% }

![Scalability analysis showing runtime (seconds) against the number of features.\label{fig:runtime-features}](runtime_features.pdf)

![Scalability analysis showing runtime (seconds) against the number of instances.\label{fig:runtime-instances}](runtime_instances.pdf)

# Research impact statement

`FSEVAL` is openly available on GitHub (<https://github.com/mrajabinasab/FSEVAL>) and installable from the Python Package Index (`pip install sdufseval`). Its dashboard is deployed publicly at <https://fseval.imada.sdu.dk>, pre-loaded with benchmark results across a range of scikit-feature datasets that visitors can filter, inspect, and export as publication-ready figures or LaTeX code without installing anything locally. Within the group's own research, `FSEVAL` provides a shared, standardized implementation of several evaluation metrics developed as part of the same research program, including FSDEM [@rajabinasab2024fsdem] and AAD [@rajabinasab2025metrics], and its dashboard's rank-analysis component incorporates the magnitude-aware rank statistic (MARS)[@rajabinasab2026mars] alongside the standard rank statistics, illustrating its role as an active research tool rather than only a post-hoc reporting utility. The project also invites external contributions: researchers who evaluate their own feature selection algorithms with `FSEVAL` are encouraged to submit their results by email so they can be included in the public dashboard, allowing the benchmark to grow as a shared, comparable resource for the feature selection community rather than a fixed snapshot.

# AI usage disclosure

Generative AI tools were used for code cleanup and for parts of the user-interface design of the static pages of the `FSEVAL` dashboard website. All AI-assisted code and design output was reviewed and tested by the authors before inclusion. 

# Acknowledgements

This study was funded by Innovation Fund Denmark in the project "PREPARE: Personalized Risk Estimation and Prevention of Cardiovascular Disease."

# References
