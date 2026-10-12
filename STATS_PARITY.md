# Patient-level statistics + DE: Python (`arsc.stats`, `arsc.de`) vs R

R references: DESeq2 1.42.1, limma 3.58.1, edgeR 4.0.16, fgsea 1.28.0, lme4 1.1.35.1 / lmerTest, msigdbr 26.1.1 (R 4.3.1).
Python: pydeseq2 0.5.2, statsmodels 0.15, scipy 1.17. Reproduce: `scratch/arsc/stats_parity/{prep.py,prep2.py,ref.R,ref_small.R,cmp_de.py,cmp_stats.py}`;
tests: `tests/test_stats_de.py` (17 tests, ~110 s; `-m "not slow"` skips DESeq2/GLMM/limma-tp parity).

## Data
gdT_1 store (`scratch/arsc/exp_gd`, 38,440 cells), cell_type `gd`, `arsc.pseudobulk` (exact integer sums) with a 20-cell floor.
* **tp**: patient x timepoint, Baseline vs Follow Up 1. 25 samples; 14 form 7 complete pairs. DESeq2 `~patient+timepoint` on the 7 pairs;
  limma-voom + duplicateCorrelation on all 25 (unbalanced, `~timepoint`, block = patient). Weak biology (0 genes at padj < 0.1 in R) so used for estimate/stat parity.
* **ct**: gd vs cd8_tem in the same patient x timepoint sample (7 matched pairs, 14 samples): DESeq2 `~pt+cell_type`, voom `~cell_type` block = pt.
* Composition table: `stats.composition` on all cell types (2,040 rows), GLMM/LMM on gd, cd8_tem, cd4_t_memory (67 samples, 7 timepoints).
* Gene sets: MSigDB Hallmark (50) and C2:CP:REACTOME (1,839) exported once by msigdbr and cached in `E:/autoresearch/.cache/genesets/`.

## Results (Python vs R)

| Component | Metric | tp | ct |
|---|---|---|---|
| DESeq2 (pydeseq2) | baseMean | identical | identical |
| | log2FC Pearson r / Spearman | 0.9998 / 0.99996 | 0.99986 / 0.99993 |
| | Wald stat r | 0.9998 | 0.9993 |
| | -log10 p r | 0.9996 | 0.9979 |
| | top-100 genes by p, overlap | 100 | 98 |
| | padj < 0.1 calls | 0 vs 0 | 2 vs 2 (same genes); padj r 0.998 |
| voom + dupCor | logFC / t Pearson r | 1.00000 / 1.00000 | 1.00000 / 1.00000 |
| | max abs diff logFC, t, adj.P | 3.6e-4, 9.3e-4, 5.7e-4 | 6e-5, 6e-5, 8e-5 |
| | consensus correlation (Py / R) | 0.2183 / 0.2187 | 0.4547 / 0.4547 |
| | df.prior, s2.prior | 5.1641 / 5.1642, 0.7015 / 0.7015 | 4.7911 / 4.7911, 0.8082 / 0.8082 |
| | genes kept by filterByExpr | 3,638 = 3,638 | 5,698 = 5,698 |
| fgsea (Hallmark) | ES max abs diff | 8e-16 | 0 |
| | NES r (max abs) | 0.99995 (0.037) | 0.99997 (0.011) |
| | size identical; padj<0.05 set | yes; 1 = 1 | yes; 3 = 3 |
| fgsea (Reactome) | NES r; padj r | 0.99997; 0.982 | 0.99996; 0.997 |
| | padj<0.05: Py / R / both | 0 / 0 / 0 | 36 / 36 / 34 |
| | leading-edge Jaccard (mean) | 0.96 | 0.96 |

Other stats (R refs in `R_small.csv`, `R_glmer.csv`, `R_lmer_*.csv`):
* `wilcoxon_signed_rank` vs `wilcox.test`: 60 random vectors (n = 3..60, with ties and zeros): max |dp| 6e-16.
* `perm_test` (exact enumeration): 60 cases, p and statistic identical (5e-16 / 6e-15).
* `glmm_prop` vs `lme4::glmer(cbind(k,n-k) ~ timepoint + (1|patient) + (1|sample))`: fixed effects max |dEst| 2.2e-4, SE max diff 2e-3, SD(patient) 1.2492 vs 1.2490, SD(sample) 1.1590 vs 1.1589, logLik within 5e-5; 6-16 s per fit.
* `lmm_score` vs `lmer(logit(prop) ~ timepoint + (1|patient))`: estimates identical (< 1e-6), SE diff <= 0.012 (statsmodels uses the REML Wald SE; lmerTest reports Satterthwaite t/p which are not reproduced), variance components identical to 6 digits.

## Implementation notes and differences

**Composition / paired / perm** (`arsc.stats`): moved from `tools/R/ar.R` and `tools/py/arstats.py` (those files are untouched).
Differences from the old Python: Wilcoxon now follows R exactly (exact for n < 50 *only without ties/zeros*, else normal approximation with tie + continuity correction;
the old code switched at n > 25 and scipy's treatment of zeros differs). Bootstrap CIs use numpy's RNG and so agree with R only within Monte-Carlo error; the exact tests are deterministic.
`perm_test` is vectorised and takes `stat=None` (median difference, fast path) or any `stat(a, b)`.

**Binomial GLMM** (`glmm_prop`): own Laplace (nAGQ = 1) maximum-likelihood implementation of the glmer objective for any number of independent random intercepts
(penalised Newton for the conditional modes, BFGS + Nelder-Mead over (beta, sigma)), so it matches glmer rather than approximating it. Standard errors follow lme4's default
(numerical Hessian of the deviance in (sigma, beta)); `vcov_method="conditional"` gives the RX^-1 variant (1-5% smaller). Not used: `BinomialBayesMixedGLM` (statsmodels) needs
0/1 responses and returns variational posterior means, not ML estimates, so it cannot reproduce glmer; `gee_prop` (binomial GEE, exchangeable) is provided as a *marginal* alternative and its coefficients
are NOT comparable to the conditional glmer ones (and sandwich SEs are anti-conservative with < ~10 patients). Random-slope or correlated random effects are not supported
(`score ~ time + (1|patient)` via `lmm_score`, statsmodels MixedLM).

**DESeq2** (`de.deseq2`): pydeseq2 with `~patient+time` designs. Same pre-filter and `make.names` handling as `ar_de_deseq2`, `alpha=0.1` (R `results()` default). Known difference fixed here: pydeseq2 0.5.2's
parametric dispersion-trend fit (L-BFGS) fails to converge on small designs and silently falls back to a flat mean trend; the ct contrast then gave log2FC r = 0.979 and only 56/100 top genes.
`de._desq2_dataset()` swaps in DESeq2's own parametricDispersionFit (Gamma-identity GLM on genes with dispersion > 100*minDisp, iterative drop of disp/fit outside (1e-4, 15)), restoring r = 0.9999.
Residual differences (largest lfcSE gaps in a handful of low-count genes) come from pydeseq2's separate MAP-dispersion/Cook's-refit code paths. Wald test, no LFC shrinkage, as in the R helper.
Speed: ~40-55 s for 7-11k genes on 14 samples with `n_cpus=1`.

**limma-voom + duplicateCorrelation** (`de.voom`): own port of filterByExpr (group = NULL branch), calcNormFactors(TMM, upper-quartile reference), voom (lowess span 0.5 via statsmodels, 1/f^4 weights), duplicateCorrelation
(per-gene REML profiled over rho, statmod `mixedModel2Fit` parametrisation: y and X weighted, Z not; negative rho allowed; limma clipping and 15% trimmed mean of atanh), gls.series lmFit, eBayes (`fitFDist` with trigamma inverse,
`squeezeVar`; no trend/robust), moderated t, BH. As in R, a block factor fully encoded in the design should not be used (R returns correlation 0 with a warning; here the REML simply becomes degenerate): use `~timepoint` with `block="patient"`.
Not implemented: `trend=TRUE`, `robust=TRUE`, `B` statistics, contrasts beyond a single coefficient. ~10-30 s for 4-6k genes.

**fgsea** (`de.fgsea`): ES, leading edge, size filtering and NES (ES / mean of same-sign null) follow fgsea exactly (ES agrees to 1e-15). p-values come from `nperm` sampled gene sets per set size (numba, deterministic per `seed`),
i.e. fgseaSimple: the smallest attainable p is 1/(nperm+1) (default 1e-4), whereas fgsea >= 1.14 (multilevel) resolves p to ~1e-50. Consequences seen above: p-values/padj agree in rank (Pearson r of -log10 p 0.90-0.99; 0.71 for the
strong Reactome ct case, driven by the floor) and significance calls agree except for borderline sets (2 of 36 in ct/Reactome). Raise `nperm` (1e5 is fast) for finer p; multilevel is not ported. gseapy.prerank was not used: its classic-GSEA permutation
(phenotype/gene permutation, different NES normalisation) is a different estimator from fgsea.

**MSigDB** (`de.msigdb`): `Rscript src/arsc/R/msigdb_export.R` once per (species, collection, subcollection) -> `E:/autoresearch/.cache/genesets/<species>__<coll>__<sub>.csv.gz` (`gs_name, gene_symbol, ...`);
afterwards no R is needed. Cached now: H, C2:CP:REACTOME, C5:GO:BP. `refresh=True` re-exports (set `RSCRIPT` env var to override the executable).

## Exports to add to `arsc/__init__.py` (not edited here)
```python
from . import stats, de
from .stats import composition, paired_test, perm_test, glmm_prop
from .de import deseq2, voom, fgsea, msigdb
```
Name clash note: `arsc.stats` / `arsc.de` as submodules are safe; `deseq2`/`voom`/`fgsea` are new top-level names.
