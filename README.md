# Network Orchestration Framework — Simulation & Validation Code

Simulation code, result logs, and figures supporting the manuscript:

> **Network Orchestration Framework Design Using AI-Driven Automation And Cybersecurity**
> Tasneem Annahdi, Albandari Alsumayt, Majid Alshammari
> Submitted to *Future Internet* (MDPI). DOI to be added upon publication.

## What this repository contains

| Path | Description |
|---|---|
| `simulation_v3.py` | Self-contained simulation script (current, round-2 revision). Reproduces every number and figure in Section 4 of the manuscript. |
| `results/results_v3.json` | Full machine-readable results: per-seed raw runs, CI summaries, specialist-accuracy sweep, and all cost-model parameters. |
| `results/policy_summary_ci_indist.csv` | Table 4 of the manuscript (in-distribution, mean ± 95% CI over 20 runs). |
| `results/policy_summary_ci_shifted.csv` | Table 5 of the manuscript (distribution-shifted, mean ± 95% CI over 20 runs). |
| `figures/` | All manuscript figures at 300 dpi (mapping below). |

## What the simulation does

The script validates an AI-driven, gradually deployed network-orchestration framework over 30 simulated operational days (80 tasks/day):

1. **Non-circular ground truth.** Task labels derive from *latent* outcome variables (true recurrence, automation feasibility, operational risk) that the classifier never observes; a Monte-Carlo estimate places the achievable accuracy ceiling at ≈79.1%. The Random Forest decision-maker reaches 78.3%.
2. **Six operating policies** are compared under a cost model with human errors, automation failures, fragile automations, incidents, and AI infrastructure/maintenance costs: fully manual, rule-based router, automate-everything (with and without a protective sensitivity guard), full AI without review, and the proposed gradual framework.
3. **Explicit protective guard.** Tasks with the observable sensitivity flag route to the IT specialist, taking precedence over the classifier. Zero incidents in guarded policies are attributed to this rule, not to the learned model.
4. **Imperfect, costed specialist review.** Development requests are reviewed by a specialist whose judgment is correct with probability *p* (default 0.85) at a cost of 10 minutes per review; *p* is swept over [0.5, 1.0].
5. **Statistical rigor.** All headline results are means ± 95% confidence intervals over 20 independent seeded runs, with all policies evaluated on identical task streams within each run (common random numbers).

## Reproducing the results

**Google Colab (recommended):** upload `simulation_v3.py` (or paste its contents into one cell) and run. All figures render inline and a ZIP of results is offered for download.

**Locally:**

```bash
pip install -r requirements.txt
python simulation_v3.py
```

Runtime is a few minutes. All randomness is seeded (`SEED = 42`; per-run seeds 1000–1019), so outputs are bit-reproducible on the pinned library versions.

## Figure mapping to the manuscript

| File | Manuscript figure |
|---|---|
| `fig_classifier_comparison.png` | Figure 3 — classifier accuracy vs. achievable ceiling |
| `fig_confusion_matrix.png` | Figure 4 — Random Forest confusion matrix |
| `fig_rf_hyperparams.png` | Figure 5 — hyperparameter validation curves |
| `fig_policy_ci.png` | Figure 6 — savings by policy, both environments, mean ± 95% CI |
| `fig_sensitivity.png` | Figure 7 — cost-parameter and threshold sensitivity |
| `fig_task_mix.png` | Figure 8 — daily task mix (representative run) |
| `fig_library_growth.png` | Figure 9 — automation-library growth (representative run) |
| `fig_specialist_sweep.png` | Figure 10 — effect of specialist accuracy *p* |

(Figure numbers may shift slightly in production typesetting; captions in the manuscript are authoritative. Figures 3–5, 7–9 are produced by the round-1 classifier/sensitivity code paths retained inside the script lineage; `results_v3.json` contains the exact configuration used.)

## Key parameters (defaults)

Manual task 12 min · automated task 0.5 min · development 60 min · specialist rate 1.20 USD/min · human error 3%/30 min rework · robot failure 0.5%/15 min · fragile automation 30%/20 min · incident 10%/180 min · AI infrastructure 12 USD/day · maintenance ≈0.5 h/week · review 10 min · specialist accuracy p = 0.85 · threshold 5 recommendations · 2,400 tasks over 30 days. Every influential parameter is varied in the sensitivity analyses.

## License

Code is released under the MIT License (see `LICENSE`). Figures and result data are released under CC BY 4.0.

## Citation

If you use this code or data, please cite the archived version via its Zenodo DOI (see the badge/metadata on the Zenodo record) and the manuscript once published. Machine-readable citation metadata is in `CITATION.cff`.
