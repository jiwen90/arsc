---
name: arsc
description: Single-cell RNA-seq / CITE-seq / TCR analysis with the arsc Python package - loading h5ad/10x/Seurat data into memory-mapped stores, QC, normalization, clustering, markers, signature scores, pseudobulk DE (DESeq2, voom) with the patient as unit, GSEA, cell-type composition, clone expansion (Cyclone, STARTRAC) and patient-level statistics. Use when a request asks a biology question of single-cell data or mentions arsc.
---

# arsc skill

1. Read `AGENTS.md` at the root of the arsc repository (mental model, task map, recipes, statistics rules,
   gotchas). Do not read the source unless AGENTS.md and `docs/api.md` do not answer the question.
2. Find the recipe for the question: `python -c "import arsc; arsc.recipes('<the question in English>')"`, or the
   "Translate English to arsc" table in AGENTS.md. Run it on `arsc.datasets.toy()` first, then swap in the
   user's data (`arsc.open(store)` / `arsc.read(file)`) and their column names.
3. Statistics: the patient is the unit. Aggregate (pseudobulk, composition, per-sample means) before testing;
   report n patients, effect sizes with CIs, per-patient values and the leave-one-out range.
4. arsc does not plot: write tidy tables (`arsc.write_table`) for the plotting tool.

Minimal check that arsc works:

```python
import arsc
ds = arsc.datasets.toy()
print(ds)
arsc.help("pseudobulk")
```
