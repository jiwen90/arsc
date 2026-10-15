"""Agreement between tools for each benchmark step, from the outputs saved by suite_py.py / suite_seurat.R.

``compare_all(bench_dir, out_dir, sizes)`` writes ``agreement.csv`` (one row per size x step x pair x metric)
and ``summary.json`` / ``summary.csv`` (timings + memory joined with the headline agreement metric).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def _read_mat(prefix):
    meta = json.loads(Path(str(prefix) + ".json").read_text())
    nr, nc = meta["dim"]
    a = np.fromfile(str(prefix) + ".f64", dtype="<f8").reshape(nc, nr).T
    return pd.DataFrame(a, index=meta.get("rownames"), columns=meta.get("colnames"))


def _ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def _r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan


def _mlog(p):
    return -np.log10(np.clip(np.asarray(p, float), 1e-300, 1))


def _step_metrics(step, d: Path, n: int):
    """Yield (ref, other, metric, value) tuples. ``ref`` is always arsc unless noted."""
    o = d / "out"
    A, S, R = o / "arsc", o / "scanpy", o / "seurat"
    if step == "normalize":
        a = sp.load_npz(A / "normalize.npz").toarray()
        if (S / "normalize.npz").exists():
            s = sp.load_npz(S / "normalize.npz").toarray()
            yield "arsc", "scanpy", "max_abs_diff", float(np.abs(a - s).max())
        if (R / "normalize.mtx").exists():
            r = scipy.io.mmread(R / "normalize.mtx").T.toarray()
            yield "arsc", "seurat", "max_abs_diff", float(np.abs(a - r).max())
    elif step == "hvg":
        a = pd.read_csv(A / "hvg.csv", index_col=0)
        sa = set(a.index[a.highly_variable])
        for tool, f in (("scanpy", S / "hvg.csv"), ("seurat", R / "hvg.csv")):
            if f.exists():
                b = pd.read_csv(f, index_col=0)
                sb = set(b.index[b.highly_variable.astype(bool)])
                yield "arsc", tool, "hvg_set_jaccard", len(sa & sb) / len(sa | sb)
                yield "arsc", tool, "hvg_shared_of_2000", len(sa & sb)
                b = b.reindex(a.index)
                yield "arsc", tool, "var_std_r", _r(a.variance_standardized, b.variance_standardized)
                yield "arsc", tool, "var_std_max_rel_diff", float(np.nanmax(np.abs(a.variance_standardized - b.variance_standardized)
                                                                             / np.maximum(np.abs(b.variance_standardized), 1e-12)))
    elif step == "pca":
        a = np.load(A / "pca.npy").astype(float)
        for tool, f in (("scanpy", S / "pca.npy"), ("seurat", R / "pca")):
            if (f.exists() if tool == "scanpy" else Path(str(f) + ".f64").exists()):
                b = np.load(f).astype(float) if tool == "scanpy" else _read_mat(f).to_numpy()
                rs = [abs(_r(a[:, k], b[:, k])) for k in range(a.shape[1])]
                yield "arsc", tool, "pc_abs_r_min_top10", float(min(rs[:10]))
                yield "arsc", tool, "pc_abs_r_min_all30", float(min(rs))
    elif step == "neighbors":
        a = np.load(A / "neighbors.npy")
        k = a.shape[1]
        sa = np.sort(a, axis=1)
        for tool, f in (("scanpy", S / "neighbors.npy"), ("seurat", R / "neighbors.i32")):
            if f.exists():
                b = np.load(f) if tool == "scanpy" else np.fromfile(f, dtype="<i4").reshape(-1, k)
                sb = np.sort(b, axis=1)
                ov = np.array([len(np.intersect1d(sa[i], sb[i], assume_unique=True)) for i in range(len(sa))])
                yield "arsc", tool, "knn_overlap_mean", float((ov - 1).mean() / (k - 1))
        # exactness of arsc: brute force on 1,000 random cells
        E = np.load(d / "common" / "pca.npy").astype(np.float64)
        rng = np.random.default_rng(0)
        q = rng.choice(len(E), min(1000, len(E)), replace=False)
        rec = []
        for i in q:
            dd = ((E - E[i]) ** 2).sum(1)
            nb = np.argpartition(dd, k)[:k]
            rec.append(len(np.intersect1d(nb, a[i])) / k)
        yield "arsc", "brute_force", "knn_recall", float(np.mean(rec))
    elif step == "cluster":
        truth = pd.read_csv(d / "r" / "obs.csv", index_col=0, usecols=["Unnamed: 0", "cell_type"]).cell_type.to_numpy() \
            if (d / "r" / "obs.csv").exists() else None
        labs = {}
        if (A / "cluster.npy").exists():
            labs["arsc"] = np.load(A / "cluster.npy")
        if (S / "cluster.npy").exists():
            labs["scanpy"] = np.load(S / "cluster.npy")
        if (R / "cluster.txt").exists():
            labs["seurat"] = np.array((R / "cluster.txt").read_text().split())
        for t, l in labs.items():
            yield t, "truth", "ari", _ari(l, truth)
            yield t, "truth", "n_clusters", len(np.unique(l))
        for t in ("scanpy", "seurat"):
            if t in labs and "arsc" in labs:
                yield "arsc", t, "ari", _ari(labs["arsc"], labs[t])
    elif step == "markers":
        a = pd.read_parquet(A / "markers.parquet")
        a["key"] = a.group.astype(str) + "|" + a.feature.astype(str)
        a = a.set_index("key")
        if (R / "markers.csv").exists():
            r = pd.read_csv(R / "markers.csv")
            r.index = r.group.astype(str) + "|" + r.feature.astype(str)
            r = r.reindex(a.index)
            yield "arsc", "seurat_presto", "auc_max_abs_diff", float(np.nanmax(np.abs(a.auc - r.auc)))
            rel = np.abs(a.pval - r.pval) / np.maximum(r.pval, 1e-300)
            yield "arsc", "seurat_presto", "pval_max_rel_diff", float(np.nanmax(rel[r.pval > 1e-290]))
    elif step == "rank_genes":
        if (A / "rank_genes.parquet").exists() and (S / "rank_genes.parquet").exists():
            from scipy.stats import spearmanr
            a = pd.read_parquet(A / "rank_genes.parquet")
            a.index = a.group.astype(str) + "|" + a.feature.astype(str)
            s = pd.read_parquet(S / "rank_genes.parquet")
            s.index = s.group.astype(str) + "|" + s.feature.astype(str)
            s = s.reindex(a.index)
            yield "arsc", "scanpy", "score_max_abs_diff", float(np.nanmax(np.abs(a.statistic - s.statistic)))
            rel = np.abs(a.pval - s.pval) / np.maximum(s.pval, 1e-300)
            yield "arsc", "scanpy", "pval_max_rel_diff", float(np.nanmax(rel[s.pval > 1e-290]))
            ov = []
            for g, ga in a.groupby("group"):
                ta = list(ga.sort_values("statistic", ascending=False, kind="stable").feature[:50])
                ts = list(s.loc[ga.index].sort_values("statistic", ascending=False, kind="stable").feature[:50])
                ov.append(len(set(ta) & set(ts)) / 50)
            yield "arsc", "scanpy", "top50_overlap_mean", float(np.mean(ov))
    elif step == "pseudobulk":
        z = np.load(A / "pseudobulk.npz", allow_pickle=True)
        Xa, la = z["X"], [str(x) for x in z["labels"]]
        if (S / "pseudobulk.npz").exists():
            zs = np.load(S / "pseudobulk.npz", allow_pickle=True)
            ls = [str(x) for x in zs["labels"]]
            if ls == la:
                yield "arsc", "scanpy", "max_abs_diff", float(np.abs(Xa - zs["X"]).max())
            else:
                yield "arsc", "scanpy", "label_mismatch", 1.0
        if Path(str(R / "pseudobulk") + ".f64").exists():
            r = _read_mat(R / "pseudobulk")  # genes x groups
            cols = [c.replace("-", "_") for c in r.columns]
            r.columns = cols
            common = [l for l in la if l in set(cols)]
            if len(common) == len(la):
                Ra = r[la].to_numpy().T
                yield "arsc", "seurat", "max_abs_diff", float(np.abs(Xa - Ra).max())
            else:
                yield "arsc", "seurat", "labels_matched_frac", len(common) / len(la)
    elif step == "de_deseq2":
        truth = set(json.loads((d / "truth.json").read_text())["de_up_T0_post"])
        res = {t: pd.read_csv(o / t / "de_deseq2.csv").set_index("gene") for t in ("arsc", "scanpy", "seurat")
               if (o / t / "de_deseq2.csv").exists()}
        for t, r in res.items():
            sig = set(r.index[r.padj < 0.1])
            yield t, "truth", "sensitivity_padj0.1", len(sig & truth) / len(truth)
            yield t, "truth", "false_discovery_prop", (len(sig - truth) / len(sig)) if sig else 0.0
        if "seurat" in res:
            R_ = res["seurat"]
            for t in ("arsc", "scanpy"):
                if t in res:
                    x = res[t].reindex(R_.index)
                    yield t, "seurat_DESeq2", "log2fc_r", _r(x.log2FoldChange, R_.log2FoldChange)
                    yield t, "seurat_DESeq2", "mlog10p_r", _r(_mlog(x.pvalue), _mlog(R_.pvalue))
                    sa, sr = set(x.index[x.padj < 0.1]), set(R_.index[R_.padj < 0.1])
                    yield t, "seurat_DESeq2", "padj0.1_jaccard", len(sa & sr) / max(1, len(sa | sr))
    elif step == "de_voom":
        if (A / "de_voom.csv").exists() and (R / "de_voom.csv").exists():
            a = pd.read_csv(A / "de_voom.csv").set_index("gene")
            r = pd.read_csv(R / "de_voom.csv").set_index("gene").reindex(a.index)
            yield "arsc", "seurat_limma", "logfc_max_abs_diff", float(np.nanmax(np.abs(a.logFC - r.logFC)))
            yield "arsc", "seurat_limma", "t_max_abs_diff", float(np.nanmax(np.abs(a.t - r.t)))
            yield "arsc", "seurat_limma", "t_r", _r(a.t, r.t)
    elif step in ("score_module", "score_ucell"):
        if (A / f"{step}.parquet").exists() and Path(str(R / step) + ".f64").exists():
            a = pd.read_parquet(A / f"{step}.parquet")
            r = _read_mat(R / step)
            r.columns = [c.replace("_UCell", "") for c in r.columns]
            a.columns = [c.replace("_UCell", "") for c in a.columns]
            yield "arsc", "seurat", "max_abs_diff", float(np.abs(a.to_numpy() - r[a.columns].to_numpy()).max())
    elif step == "score_genes":
        if (A / "score_genes.parquet").exists() and (S / "score_genes.parquet").exists():
            a = pd.read_parquet(A / "score_genes.parquet")
            s = pd.read_parquet(S / "score_genes.parquet")[a.columns]
            yield "arsc", "scanpy", "max_abs_diff", float(np.abs(a.to_numpy() - s.to_numpy()).max())


HEADLINE = {  # (pair other, metric) used in the summary row for each step and comparator
    "normalize": "max_abs_diff", "hvg": "hvg_shared_of_2000", "pca": "pc_abs_r_min_top10",
    "neighbors": "knn_overlap_mean", "cluster": "ari", "markers": None, "rank_genes": None, "pseudobulk": "max_abs_diff",
    "de_deseq2": "mlog10p_r", "de_voom": "t_max_abs_diff", "score_module": "max_abs_diff",
    "score_genes": "max_abs_diff", "score_ucell": "max_abs_diff",
}


def compare_all(bench: Path, out: Path, sizes) -> None:
    rows = []
    for n in sizes:
        d = Path(bench) / f"n{n}"
        if not (d / "out").exists():
            continue
        for step in HEADLINE.keys() | {"clones"}:
            if not (d / "out" / "arsc").exists():
                continue
            try:
                for ref, other, metric, val in _step_metrics(step, d, n):
                    rows.append(dict(n=n, step=step, tool=ref, vs=other, metric=metric, value=val))
            except FileNotFoundError:
                continue
            except Exception as e:  # noqa: BLE001
                rows.append(dict(n=n, step=step, tool="arsc", vs="?", metric="compare_error", value=np.nan, note=str(e)[:200]))
    ag = pd.DataFrame(rows)
    ag.to_csv(out / "agreement.csv", index=False)
    runs = []
    rf = out / "runs.jsonl"
    if rf.exists():
        for l in rf.read_text().splitlines():
            runs.append(json.loads(l))
    rr = pd.DataFrame(runs)
    if not rr.empty:
        rr = rr.drop_duplicates(["n", "tool", "step"], keep="last")
        keep = [c for c in ["n", "tool", "step", "ok", "seconds", "rss_peak_gb", "rss_base_gb", "process_peak_gb", "warmup_seconds",
                            "error", "skipped", "t_start"] if c in rr]
        rr[keep].sort_values(["n", "step", "tool"]).to_csv(out / "summary.csv", index=False)
        summ = {"runs": json.loads(rr[keep].to_json(orient="records")),
                "agreement": json.loads(ag.to_json(orient="records")) if not ag.empty else []}
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"[compare] {len(ag)} agreement rows -> {out / 'agreement.csv'}")
