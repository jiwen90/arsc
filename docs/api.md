# API reference

Generated from the docstrings by `docs/build_docs.py`; do not edit by hand. Start with `arsc.help()`.

* [arsc](#arsc): Top level (I/O, preprocessing shortcuts, markers, scores, graphs)
* [arsc.datasets](#arscdatasets): Toy datasets
* [arsc.pp](#arscpp): Preprocessing (Seurat conventions)
* [arsc.formats](#arscformats): Readers and writers
* [arsc.io](#arscio): Store, Dataset and the Seurat bridge
* [arsc.views](#arscviews): Subset stores as views of their parent stores
* [arsc.markers](#arscmarkers): Markers (presto)
* [arsc.scores](#arscscores): Signature scores
* [arsc.neighbors](#arscneighbors): Neighbors and WNN
* [arsc.harmony](#arscharmony): Harmony
* [arsc.symphony](#arscsymphony): Symphony reference mapping
* [arsc.de](#arscde): Differential expression and GSEA
* [arsc.stats](#arscstats): Patient-level statistics
* [arsc.clones](#arscclones): TCR clones
* [arsc.clonotrace](#arscclonotrace): CloneTrace (clonal fate)
* [arsc.genesets](#arscgenesets): Gene sets
* [arsc.bam](#arscbam): BAM reading, per-cell allele counts, genotype checks
* [arsc.seurat_extra](#arscseurat_extra): Seurat parity
* [arsc.scanpy_extra](#arscscanpy_extra): scanpy parity

## arsc

Top level (I/O, preprocessing shortcuts, markers, scores, graphs).

### `arsc.ArscFormatError(...)`

```text
Input file is not in a supported format, or has an inconsistent structure (a ``ValueError``).

Attributes
----------
args : the message (says which file and what was expected).

Examples
--------
>>> import arsc
>>> try:
...     arsc.detect_format("data.xyz")
... except arsc.ArscFormatError as e:
...     print("unknown format" in str(e))
True
```

### `arsc.Assay(name: 'str', features: 'pd.Index', layers: 'dict' = <factory>, variable_features: 'list' = <factory>, var: 'pd.DataFrame | None' = None, varm: 'dict' = <factory>) -> None`

```text
One modality (RNA, ADT, ...). Every layer is a cells x features ``scipy.sparse.csr_matrix``.

Attributes
----------
name : assay name.
features : feature names.
layers : ``{layer: csr_matrix}`` (e.g. ``counts``, ``data``).
variable_features : variable features recorded at import.
var : optional per-feature annotations.
varm : per-feature matrices.

Examples
--------
>>> import arsc
>>> a = arsc.datasets.toy()["ADT"]
>>> a.name, len(a.features), list(a.layers)
('ADT', 6, ['counts'])
```

### `arsc.Dataset(obs: 'pd.DataFrame', assays: 'dict' = <factory>, reductions: 'dict' = <factory>, graphs: 'dict' = <factory>, manifest: 'dict' = <factory>, uns: 'dict' = <factory>) -> None`

```text
In-memory (or mmap-backed) container: ``obs``, ``assays``, ``reductions``, ``graphs``.

Returned by :func:`arsc.read` (no ``store=``), :meth:`Store.subset` and :func:`arsc.datasets.toy`.
Every matrix is cells x features (CSR).

Attributes
----------
obs : per-cell metadata (index = barcodes).
assays : ``{name: Assay}``; ``ds.X(assay, layer)`` returns a matrix.
reductions : ``{name: Reduction}``; ``ds.embedding(name)``.
graphs : ``{name: csr_matrix}``.
manifest : store manifest (provenance).
uns : unstructured metadata.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> ds.n_cells, list(ds.assays), ds.layers("RNA")
(3000, ['RNA', 'ADT'], ['counts'])
>>> cd8 = ds.subset((ds.obs["cell_type"] == "CD8 T").to_numpy())
>>> cd8.X("RNA", "counts").shape[1]
300
```

### `arsc.detect_format(path: 'PathLike') -> 'str'`

```text
Detect the format of a path from its name (and HDF5 layout for ``.h5``).

Parameters
----------
path : file or directory.

Returns
-------
str, one of ``store, h5ad, 10x_h5, 10x_mtx, seurat, h5seurat, loom``; raises
:class:`ArscFormatError` otherwise.

Examples
--------
>>> import arsc
>>> arsc.detect_format("obj.qs"), arsc.detect_format("pbmc.h5ad")
('seurat', 'h5ad')
```

### `arsc.export_seurat(obj_path, out_dir, *, assays=None, layers=('counts',), graphs: 'bool' = True, reductions: 'bool' = True, rscript: 'str' = 'Rscript', threads: 'int' = 16, full: 'bool' = False, scale_f64: 'bool' = False) -> 'Path'`

```text
Run the R exporter on a ``.rds``/``.qs`` Seurat object (v4 or v5); prefer :func:`arsc.read_seurat`.

Parameters
----------
obj_path : ``.rds`` or ``.qs`` file.
out_dir : store directory to write.
assays : assays to export (default all).
layers : layers to export (``"counts"``, ``"data"``, ``"scale.data"``).
graphs, reductions : export graphs / reductions.
rscript : path of ``Rscript``.
threads : R threads.
full : export everything the object holds (every assay and layer, scale.data as a dense array,
    per-feature metadata, Idents, reductions with projected loadings, graphs, Neighbor objects,
    misc/commands/tools as ``uns``); what cannot be converted is listed in
    ``manifest["unconvertible"]``. ``assays``/``layers``/``graphs``/``reductions`` are ignored.
scale_f64 : with ``full``, keep scale.data as float64 (default float32: half the disk).

Returns
-------
pathlib.Path ``out_dir`` (open it with :func:`open`). Needs R with SeuratObject (and qs).

Examples
--------
>>> import arsc
>>> arsc.export_seurat("obj.qs", "stores/obj", layers=("counts", "data"))   # doctest: +SKIP
```

### `arsc.exporter_script() -> 'Path'`

```text
Path of the bundled R exporter (``Rscript <path> <obj.rds|.qs> <out_dir>``).

Returns
-------
pathlib.Path of ``export_seurat.R`` inside the installed package.

Examples
--------
>>> import arsc
>>> arsc.exporter_script().name
'export_seurat.R'
```

### `arsc.from_anndata(h5ad, out_dir, *, layers=('counts',), assay: 'str' = 'RNA', use_raw_for=None, obsm: 'bool' = True, chunk: 'int | None' = None, obsp: 'bool' = False) -> 'Path'`

```text
Convert an .h5ad into an arsc store so that many processes share one memory-mapped copy.

Compatibility wrapper around :func:`arsc.read_h5ad`; prefer ``arsc.read(h5ad, store=out_dir)``,
which imports every layer by default.

``layers``: AnnData layer names to export (``"X"`` exports ``.X`` as layer ``X``).
``use_raw_for``: name under which to export ``.raw.X``. This is a layer of ``assay`` when raw has
the same genes, and otherwise a separate assay of that name. The matrices are streamed from
disk in chunks (memory-safe; ``chunk`` is accepted for backward compatibility and ignored).
``obsm`` embeddings become reductions. Idempotent: if ``out_dir`` already holds a store, it is
returned untouched.

Parameters
----------
h5ad : input ``.h5ad``.
out_dir : store directory.
layers : AnnData layers to export.
assay : assay name.
use_raw_for : name for ``.raw.X``.
obsm : export embeddings.
chunk : ignored (backward compatibility).
obsp : export graphs.

Returns
-------
pathlib.Path ``out_dir``.

Examples
--------
>>> import arsc
>>> f = arsc.datasets.toy_h5ad("toy_fa.h5ad", n_cells=300)
>>> arsc.open(arsc.from_anndata(f, "toy_fa_store", layers=("X",))).n_cells
300
```

### `arsc.help(topic: 'str | None' = None, *, show: 'bool' = True) -> 'str'`

```text
Print the task -> function map ("I want to ..." -> the arsc call).

Parameters
----------
topic : optional filter; only rows whose section, task or call contain this text (case-insensitive)
    are shown, e.g. ``"clone"``, ``"DE"``, ``"markers"``.
show : print the table (default). The text is returned either way.

Returns
-------
str, the formatted table.

Examples
--------
>>> import arsc
>>> txt = arsc.help("pseudobulk", show=False)
>>> "arsc.pseudobulk" in txt
True
```

### `arsc.make_unique(names: 'Sequence[str]', join: 'str' = '-') -> 'list[str]'`

```text
Deduplicate names exactly like ``anndata.utils.make_index_unique`` (``a, a -> a, a-1``).

Parameters
----------
names : sequence of names.
join : separator before the counter.

Returns
-------
list of str, unique, first occurrences unchanged.

Examples
--------
>>> import arsc
>>> arsc.make_unique(["TBCE", "TBCE", "CD8A"])
['TBCE', 'TBCE-1', 'CD8A']
```

### `arsc.open(path, finalize: 'bool' = True, build_csc: 'bool' = True) -> 'Store'`

```text
Open an arsc store lazily (memory-mapped, zero-copy). See :class:`Store`.

The first open of a fresh export writes ``obs.parquet`` and the per-feature
CSC copies (one-time; concurrent first opens wait on a lock instead of
duplicating work). Pass ``finalize=False`` for strictly read-only use.

Parameters
----------
path : store directory.
finalize : build missing one-time files.
build_csc : build per-feature CSC copies.

Returns
-------
arsc.Store.

Examples
--------
>>> import arsc
>>> _ = arsc.datasets.toy(store="toy_store_open")
>>> st = arsc.open("toy_store_open")
>>> st.n_cells
3000
```

### `arsc.read(path: 'PathLike', store: 'PathLike | None' = None, **kw) -> 'Store | Dataset'`

```text
Read any supported single-cell file.

Returns an in-memory :class:`~arsc.io.Dataset`, or, with ``store=<dir>``, converts once into a
memory-mapped :class:`~arsc.io.Store` and opens it. Arsc store directories always open as a Store.

=========================  ===============================  =========================
input                      reader                           needs
=========================  ===============================  =========================
arsc store directory       :func:`arsc.open`                --
``.h5ad``                  :func:`read_h5ad`                anndata, h5py
10x ``.h5``                :func:`read_10x_h5`              h5py
10x mtx directory          :func:`read_10x_mtx`             --
``.qs`` / ``.rds``         :func:`read_seurat`              R + SeuratObject (store=)
=========================  ===============================  =========================

``.h5Seurat`` is not parsed natively. In R, ``SeuratDisk::Convert("x.h5Seurat", dest = "h5ad")``
gives an h5ad, or ``qs::qsave(LoadH5Seurat("x.h5Seurat"), "x.qs")`` gives a qs for the exporter.

Parameters
----------
path : file or directory (any format above).
store : output store directory for a conversion (required for Seurat objects).

Returns
-------
arsc.Dataset (in memory) or arsc.Store (memory-mapped).

Examples
--------
>>> import arsc
>>> f = arsc.datasets.toy_h5ad("toy_b.h5ad", n_cells=500)
>>> ds = arsc.read(f)
>>> ds.n_cells
500
>>> st = arsc.read(f, store="toy_store2")
>>> arsc.read("toy_store2").n_cells      # a store directory reopens as a Store
500
```

### `arsc.read_10x_h5(path: 'PathLike', store: 'PathLike | None' = None, *, var_names: "Literal['gene_symbols', 'gene_ids']" = 'gene_symbols', feature_types: 'Sequence[str] | None' = None, join: 'str' = '-', dtype: 'DtypeArg' = 'auto', chunk_nnz: 'int' = 33554432, overwrite: 'bool' = False) -> 'Store | Dataset'`

```text
Read a CellRanger ``filtered_feature_bc_matrix.h5`` (v3+; v2 single-genome also supported).

Features are split by ``feature_type`` into separate assays: Gene Expression -> ``RNA``,
Antibody Capture -> ``ADT``, CRISPR Guide Capture -> ``CRISPR``, Multiplexing Capture -> ``HTO``,
Peaks -> ``ATAC``. Each assay gets one layer, ``counts``. Names are deduplicated within each assay
like ``var_names_make_unique`` (``join``). ``var`` keeps ``gene_ids``, ``feature_types`` and the
other per-feature fields, as ``scanpy.read_10x_h5`` does.

Parameters
----------
path : the ``.h5`` file.
store : output store directory; ``None`` returns an in-memory Dataset.
var_names : ``"gene_symbols"`` or ``"gene_ids"`` as feature names.
feature_types : restrict the import, e.g. ``["Gene Expression"]``.
join : separator for deduplicated names.
dtype : value dtype on disk.
chunk_nnz : non-zeros per streamed chunk.
overwrite : replace an existing store.

Returns
-------
arsc.Store (with ``store=``) or arsc.Dataset with assays ``RNA`` (and ``ADT``, ...), layer ``counts``.

Examples
--------
>>> import arsc
>>> st = arsc.read_10x_h5("filtered_feature_bc_matrix.h5", store="run1")   # doctest: +SKIP
>>> st.assays                                                             # doctest: +SKIP
['RNA', 'ADT']
```

### `arsc.read_10x_mtx(path: 'PathLike', store: 'PathLike | None' = None, *, var_names: "Literal['gene_symbols', 'gene_ids']" = 'gene_symbols', feature_types: 'Sequence[str] | None' = None, join: 'str' = '-', dtype: 'DtypeArg' = 'auto', overwrite: 'bool' = False) -> 'Store | Dataset'`

```text
Read a 10x Matrix Market directory (``matrix.mtx[.gz]``, ``features.tsv[.gz]`` or ``genes.tsv``,
``barcodes.tsv[.gz]``). Feature types are split into assays as in :func:`read_10x_h5`.

The triplets are parsed in one pass (scipy's fast Matrix Market reader), so peak memory is about
16 bytes per non-zero. For very large matrices, the 10x ``.h5`` is the memory-bounded route.

Parameters
----------
path : the directory (or a file inside it).
store : output store directory; ``None`` returns an in-memory Dataset.
var_names : ``"gene_symbols"`` or ``"gene_ids"``.
feature_types : restrict the import.
join : separator for deduplicated names.
dtype : value dtype on disk.
overwrite : replace an existing store.

Returns
-------
arsc.Store (with ``store=``) or arsc.Dataset.

Examples
--------
>>> import arsc
>>> ds = arsc.read_10x_mtx("filtered_feature_bc_matrix/")    # doctest: +SKIP
```

### `arsc.read_h5ad(path: 'PathLike', store: 'PathLike | None' = None, *, assay: 'str' = 'RNA', x_layer: 'str' = 'X', layers: 'Sequence[str] | None' = None, raw: 'str | None' = 'raw', raw_as_layer: 'bool' = False, obsm: 'bool' = True, varm: 'bool' = True, obsp: 'bool' = True, uns: 'bool' = True, dtype: 'DtypeArg' = 'auto', chunk_nnz: 'int' = 33554432, overwrite: 'bool' = False) -> 'Store | Dataset'`

```text
Read an ``.h5ad`` file into a memory-mapped store (``store=dir``) or an in-memory Dataset.

The expression matrices (``X``, ``layers``, ``raw/X``) are streamed from disk in chunks of
about ``chunk_nnz`` non-zeros, so a file of any size converts in bounded memory. Dense, CSR and
CSC encodings are all supported; for CSC the row-major copy is built out of core. Mapping:

===================  =====================================================================
``X``                layer ``x_layer`` (default ``"X"``) of assay ``assay``
``layers[k]``        layer ``k`` of ``assay`` (``layers=None``: all; or a list; ``"X"`` = X)
``raw``              assay ``raw`` (own feature set, layer ``"X"``); ``raw=None`` skips it.
                     ``raw_as_layer=True`` stores it as layer ``raw`` of ``assay`` instead
                     (only if raw has the same genes)
``obs`` / ``var``    ``obs.parquet`` / ``assays/<assay>/var.parquet`` (categoricals, nullable
                     ints/bools, strings preserved); ``var.highly_variable`` -> variable features
``obsm[k]``          reduction ``k`` without the ``X_`` prefix (DataFrames kept with columns)
``varm[k]``          ``Store.varm(assay, k)``; ``varm["PCs"]`` also becomes the PCA loadings
``obsp[k]``          graph ``k``
``uns``              JSON-able part (scalars, strings, lists, arrays, nested dicts) -> ``Store.uns``
===================  =====================================================================

With ``store`` set, the store is written atomically and then opened. If the same source was
already converted there, the existing store is opened and nothing is rewritten.

Parameters
----------
path : the ``.h5ad`` file.
store : output store directory; ``None`` returns an in-memory Dataset.
assay : assay name for ``X`` and ``layers``.
x_layer : layer name for ``X``.
layers : AnnData layers to import (``None`` = all).
raw : assay name for ``.raw`` (``None`` skips it).
raw_as_layer : store ``.raw`` as a layer of ``assay`` (same genes only).
obsm, varm, obsp, uns : import these AnnData slots.
dtype : value dtype on disk (``"auto"``: float32 when exact, else the source dtype).
chunk_nnz : non-zeros per streamed chunk.
overwrite : replace an existing store. Without it, an existing store built from a file of the same name
    is reopened as is (with a warning if the file is newer than the store).

Returns
-------
arsc.Store (with ``store=``) or arsc.Dataset. Needs ``anndata`` and ``h5py`` (``pip install "arsc[h5ad]"``).

Examples
--------
>>> import arsc
>>> f = arsc.datasets.toy_h5ad("toy_a.h5ad", n_cells=500)
>>> ds = arsc.read_h5ad(f)                      # in memory
>>> st = arsc.read_h5ad(f, store="toy_h5ad_store")   # memory-mapped store, converted once
>>> st.n_cells, st.layers("RNA")
(500, ['X'])
```

### `arsc.read_seurat(path: 'PathLike', store: 'PathLike', **export_kw) -> 'Store'`

```text
Convert a Seurat v4/v5 object (``.qs`` preferred, ``.rds`` accepted) with the bundled R exporter.

Needs R with SeuratObject (and ``qs`` for ``.qs``). Writes ``store`` once; later calls open it.
``export_kw`` are passed to :func:`arsc.export_seurat` (``assays``, ``layers``, ``graphs``, ...).
For ``.h5Seurat`` files, see :func:`read` (convert to h5ad with SeuratDisk, or save as qs in R).

Parameters
----------
path : ``.qs`` or ``.rds`` file with a Seurat object.
store : output store directory (required).

Returns
-------
arsc.Store.

Examples
--------
>>> import arsc
>>> st = arsc.read_seurat("obj.qs", store="stores/obj")      # doctest: +SKIP
```

### `arsc.read_seurat_export(path, *, assays=None, layers=None, graphs: 'bool' = True, reductions: 'bool' = True) -> 'Dataset'`

```text
Load an arsc store as a :class:`Dataset` (memory-mapped arrays; nothing is copied).

Parameters
----------
path : store directory.
assays, layers : restrict to these assays / layers.
graphs, reductions : include graphs / reductions.

Returns
-------
arsc.Dataset.

Examples
--------
>>> import arsc
>>> _ = arsc.datasets.toy(store="toy_store_rse")
>>> arsc.read_seurat_export("toy_store_rse", assays=["RNA"]).n_cells
3000
```

### `arsc.recipes(query: 'str | None' = None, *, show: 'bool' = True) -> 'str'`

```text
List the recipes, or print the one matching an id or a plain-English question.

Parameters
----------
query : ``None`` lists every recipe (id, title, example questions). A recipe id (e.g. ``"de_paired"``)
    prints that recipe's code. Any other text is matched against the titles and example questions
    (word overlap), e.g. ``"compare responders and non-responders in CD8 T cells"``.
show : print the result (default). The text is returned either way.

Returns
-------
str, the listing or the recipe code (runnable on ``arsc.datasets.toy()``; swap in your data and column names).

Examples
--------
>>> import arsc
>>> code = arsc.recipes("do these clones expand after treatment", show=False)
>>> "clone_dynamics" in code
True
>>> "de_paired" in arsc.recipes(show=False)
True
```

### `arsc.Reduction(embedding: 'np.ndarray', columns: 'list', loadings: 'pd.DataFrame | None' = None, stdev: 'np.ndarray | None' = None, assay: 'str | None' = None, projected: 'pd.DataFrame | None' = None, key: 'str | None' = None) -> None`

```text
Cell embedding (cells x k), optional feature loadings and stdev.

Attributes
----------
embedding : cells x k array.
columns : column names (e.g. ``PC_1``...).
loadings : optional features x k DataFrame.
stdev : optional per-component s.d.
assay : assay the reduction was computed from.
projected : optional projected loadings (features x k DataFrame; Seurat ``ProjectDim``).
key : column key (e.g. ``PC_``).

Examples
--------
>>> import arsc, numpy as np
>>> r = arsc.Reduction(np.zeros((5, 2)), ["PC_1", "PC_2"])
>>> r.to_frame().shape
(5, 2)
```

### `arsc.reset_seeds_used() -> 'None'`

```text
Clear the seed log of this process (``seeds_used``).

Returns
-------
None

Examples
--------
>>> import arsc
>>> arsc.reset_seeds_used()
>>> arsc.seeds_used()
[]
```

### `arsc.seeds_used(reset: 'bool' = False) -> 'list'`

```text
Seeds used by stochastic arsc calls in this process (outermost calls only).

Parameters
----------
reset : clear the log after reading it.

Returns
-------
list of dicts ``{"function", "param", "seed", "calls"}`` in first-call order.

Examples
--------
>>> import arsc
>>> arsc.reset_seeds_used()
>>> _ = arsc.stats.boot_ci([1.0, 2.0, 3.0, 4.0], seed=7)
>>> arsc.seeds_used()
[{'function': 'stats.boot_ci', 'param': 'seed', 'seed': 7, 'calls': 1}]
```

### `arsc.Store(path, finalize: 'bool' = True, build_csc: 'bool' = True)`

```text
Lazy, memory-mapped handle on an arsc store directory (see :func:`open`).

Nothing is read until asked for: ``st.obs`` reads the metadata, ``st.X(assay, layer)`` returns a
zero-copy CSR backed by the OS page cache (shared by all processes), ``st.get_features`` reads
only the requested genes, ``st.subset(mask)`` materializes only those cells.

Parameters
----------
path : store directory (contains ``manifest.json``).
finalize : build the one-time ``obs.parquet`` / CSC copies if missing.
build_csc : build per-feature CSC copies (needed by ``get_features``).

Attributes
----------
path : the store directory.
manifest : parsed ``manifest.json``.

Examples
--------
>>> import arsc
>>> st = arsc.datasets.toy(store="toy_store_cls")   # writes a memory-mapped store
>>> st.n_cells, st.assays
(3000, ['RNA', 'ADT'])
>>> X = st.X("RNA", "counts")                     # memory-mapped, no copy
>>> st.get_features(["CD8A", "GZMK"]).shape       # reads only these genes
(3000, 2)
```

### `arsc.StoreBuilder(out: 'PathLike | None', *, source: 'str' = '', overwrite: 'bool' = False)`

```text
Assemble an arsc store (``out`` = directory) or an in-memory Dataset (``out=None``) piece by piece.

The order is cells, then assays and layers, then the optional reductions, graphs, varm and uns,
and finally :meth:`finish`. The directory version writes into ``<out>.tmp<pid>`` and renames it
when finished, so an interrupted conversion never leaves a half-written store behind.

Parameters
----------
out : store directory, or ``None`` for an in-memory :class:`arsc.Dataset`.
source : free-text provenance recorded in the manifest.
overwrite : replace an existing store at ``out``.

Examples
--------
>>> import arsc, numpy as np, pandas as pd, scipy.sparse as sp
>>> counts = sp.random(100, 20, density=0.2, format="csr", random_state=0) * 10
>>> b = arsc.StoreBuilder("my_store")
>>> b.set_cells([f"c{i}" for i in range(100)], pd.DataFrame({"patient": ["P1", "P2"] * 50}))
>>> b.add_assay("RNA", [f"g{j}" for j in range(20)])
>>> b.add_layer("RNA", "counts", counts.floor())
>>> st = b.finish()
>>> st.n_cells, st.layers("RNA")
(100, ['counts'])
```

### `arsc.to_anndata(obj: 'Store | Dataset', cells=None, genes=None, *, assay: 'str | None' = None, x_layer: 'str | None' = None, layers: 'Sequence[str] | None' = None, raw: 'bool' = True, obsm: 'bool' = True, varm: 'bool' = True, obsp: 'bool' = False, uns: 'bool' = True, restore_dtype: 'bool' = True)`

```text
Build an in-memory :class:`anndata.AnnData` from a store or dataset (scanpy hand-off).

``cells`` / ``genes``: boolean mask, positions or names (``None`` = all). Only the selected rows
are read from the memory map. ``X`` is ``x_layer``: the original ``.X`` for h5ad-derived stores,
otherwise ``counts`` when present. The other layers of ``assay`` go to ``.layers``. ``raw``
restores ``.raw`` when the store came from an h5ad that had one. ``obsp=True`` also exports the
graphs. Values are cast back to the source dtype recorded at import (``restore_dtype``).

Parameters
----------
obj : arsc.Store or arsc.Dataset.
cells : cells to export (mask, positions or barcodes).
genes : genes to export (mask, positions or names).
assay : assay for ``X`` (default: the store's default assay).
x_layer : layer used as ``X``.
layers : other layers to export (default all).
raw : restore ``.raw``.
obsm, varm, obsp, uns : export these slots.
restore_dtype : cast to the source dtype.

Returns
-------
anndata.AnnData.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> ad = arsc.to_anndata(ds, cells=(ds.obs.cell_type == "CD8 T").to_numpy(), genes=["CD8A", "GZMK"], assay="RNA")
>>> ad.n_vars, list(ad.var_names)
(2, ['CD8A', 'GZMK'])
```

### `arsc.write_h5ad(obj: 'Store | Dataset', path: 'PathLike', cells=None, genes=None, *, assay: 'str | None' = None, x_layer: 'str | None' = None, layers: 'Sequence[str] | None' = None, raw: 'bool' = True, obsm: 'bool' = True, varm: 'bool' = True, obsp: 'bool' = False, uns: 'bool' = True, restore_dtype: 'bool' = True, compression: 'str | None' = None, chunk_nnz: 'int' = 33554432, overwrite: 'bool' = False) -> 'Path'`

```text
Write a store, dataset or subset to ``.h5ad`` without materializing the full matrix.

The expression matrices are streamed in row chunks (CSR on disk), so a store larger than RAM
exports in bounded memory. Arguments are as in :func:`to_anndata`. ``compression`` is passed to
h5py (``"gzip"``, ``"lzf"``, or ``None``, which is uncompressed and fastest). The file is
written to ``<path>.tmp`` and renamed. The output reads back with ``anndata.read_h5ad`` and
``scanpy.read_h5ad``.

Parameters
----------
obj : arsc.Store or arsc.Dataset.
path : output ``.h5ad``.
cells, genes, assay, x_layer, layers, raw, obsm, varm, obsp, uns, restore_dtype : as in :func:`to_anndata`.
compression : h5py compression.
chunk_nnz : non-zeros per streamed chunk.
overwrite : replace an existing file.

Returns
-------
pathlib.Path of the written file.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> p = arsc.write_h5ad(ds, "cd8.h5ad", cells=(ds.obs.cell_type == "CD8 T").to_numpy(), assay="RNA")
>>> p.name
'cd8.h5ad'
```

### `arsc.write_table(df: 'pd.DataFrame', path, index: 'bool | None' = None) -> 'Path'`

```text
Write a tidy table (e.g. for plotting in R); format from the extension (.parquet/.csv/.tsv).

Read in R with ``arrow::read_parquet`` or ``data.table::fread``. Parent directories are created.

Parameters
----------
df : the table.
path : output path ending in ``.parquet``, ``.csv`` or ``.tsv``.
index : write the index (default: only if it is not a RangeIndex).

Returns
-------
pathlib.Path of the written file.

Examples
--------
>>> import arsc, pandas as pd
>>> arsc.write_table(pd.DataFrame({"a": [1, 2]}), "out/table.csv").name
'table.csv'
```

## arsc.datasets

Toy datasets.

### `arsc.datasets.toy(n_cells: 'int' = 3000, n_genes: 'int' = 300, *, seed: 'int' = 0, store=None, overwrite: 'bool' = False)`

```text
Simulate a small PBMC-like study (6 patients x pre/post, known planted effects).

Parameters
----------
n_cells : number of cells (spread over the 12 samples).
n_genes : number of genes (the named marker/effect/cell-cycle genes first, then fillers).
seed : integer seed; the same seed gives an identical dataset.
store : optional directory; if given, the data is written as a memory-mapped arsc store and an
    :class:`arsc.Store` is returned instead of an in-memory :class:`arsc.Dataset`.
overwrite : replace an existing store at ``store``.

Returns
-------
arsc.Dataset (or arsc.Store with ``store=``) with assays ``RNA`` (layer ``counts``) and ``ADT``
(layer ``counts``), and ``obs`` columns ``patient``, ``timepoint``, ``response``, ``sample``,
``cell_type``, ``clone_id``.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy(n_cells=1200)
>>> sorted(ds.obs["cell_type"].unique())
['B', 'CD4 T', 'CD8 T', 'Mono', 'NK']
>>> ds.X("RNA", "counts").shape
(1200, 300)
```

### `arsc.datasets.toy_bam(path, *, n_cells_per_donor: 'int' = 40, n_sites: 'int' = 30, seed: 'int' = 0) -> 'dict'`

```text
Write a small synthetic 10x-style BAM (with ``.bai``) of 3 donors' cells, for :mod:`arsc.bam`.

Two references (``chr1`` 60 kb, ``chr2`` 30 kb, random sequence); ``n_sites`` SNPs with a known genotype per
donor (``D1``-``D3``, 0 / 1 / 2 ALT copies). Each cell has ``CB`` (16 bases + ``-1``) and its reads ``UB``;
a UMI has 1-3 reads (all but the first flagged duplicate) with a rare sequencing error. Reads are 60-90 bp
with spliced (``N``), deleted (``D``), inserted (``I``) and soft-clipped (``S``) CIGARs, so some reads skip a
site. Filter traps: MAPQ 3 reads, secondary and QC-fail reads, low base quality at a site, a barcode not in
the list, reads without ``UB``, and unplaced unmapped reads at the end.

Parameters
----------
path : output ``.bam`` path (the index is written to ``path + ".bai"``).
n_cells_per_donor : cells per donor.
n_sites : SNPs (spread over both references).
seed : random seed (integer).

Returns
-------
dict with ``path``, ``sites`` (DataFrame chrom, pos (1-based), ref, alt), ``genotypes`` (sites x donors ALT
copies), ``barcodes`` (list), ``donor`` and ``half`` (Series indexed by barcode: donor, and an ``A`` / ``B``
split of each donor's cells), ``n_records``.

Examples
--------
>>> import arsc, tempfile, os
>>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
>>> len(toy["barcodes"]), len(toy["sites"])
(120, 30)
```

### `arsc.datasets.toy_h5ad(path, n_cells: 'int' = 3000, n_genes: 'int' = 300, *, seed: 'int' = 0) -> 'Path'`

```text
Write :func:`toy` to an ``.h5ad`` file (RNA counts as ``X``), e.g. to try :func:`arsc.read`.

Parameters
----------
path : output ``.h5ad`` path.
n_cells, n_genes, seed : passed to :func:`toy`.

Returns
-------
pathlib.Path of the written file. Needs ``anndata`` (``pip install "arsc[h5ad]"``).

Examples
--------
>>> import arsc, tempfile, os
>>> f = arsc.datasets.toy_h5ad(os.path.join(tempfile.mkdtemp(), "toy.h5ad"), n_cells=500)
>>> arsc.read(f).n_cells
500
```

### `arsc.datasets.toy_view(path, n_cells: 'int' = 600, *, seed: 'int' = 0) -> 'dict'`

```text
A toy parent store, a full subset store of every other cell, and a view of that subset on the parent.

Parameters
----------
path : a new or empty directory; ``parent/``, ``full/`` (the subset as a full copy) and ``view/`` are written
    inside it (an existing set is reused).
n_cells : cells of the parent (the subset keeps every other one).
seed : integer seed of the toy data.

Returns
-------
dict with the three store paths (``parent``, ``full``, ``view``) and the subset's parent rows (``rows``).

Examples
--------
>>> import arsc, tempfile
>>> t = arsc.datasets.toy_view(tempfile.mkdtemp())
>>> v = arsc.open(t["view"])
>>> v.is_view, v.n_cells
(True, 300)
```

## arsc.pp

Preprocessing (Seurat conventions).

### `arsc.pp.clr(X, margin: 'int' = 2) -> 'sp.csr_matrix'`

```text
Centered log-ratio transform as in Seurat ``NormalizeData(method = "CLR")`` (for ADT).

Seurat's function is ``log1p(x / exp(sum(log1p(x[x > 0])) / length(x)))``.

Parameters
----------
X : cells x features counts (e.g. the ``ADT`` assay).
margin : ``2`` (Seurat's ADT default; Seurat's margin refers to its features x cells
    layout) normalizes each *cell* across features; ``1`` normalizes each feature across cells.

Returns
-------
scipy.sparse.csr_matrix (float64) with the same sparsity pattern as ``X``.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> A = arsc.pp.clr(ds.X("ADT", "counts"), margin=2)
>>> A.shape
(3000, 6)
```

### `arsc.pp.hvg_vst(counts, nfeatures: 'int' = 2000, features=None, loess_span: 'float' = 0.3, clip_max='auto', surface: 'str' = 'interpolate') -> 'pd.DataFrame'`

```text
Seurat v3 "vst" highly-variable features on raw counts (``FindVariableFeatures``).

Fits ``log10(variance) ~ log10(mean)`` with R-compatible loess (span 0.3),
standardizes each feature by the fitted s.d. (clipping at ``sqrt(n_cells)``)
and ranks by the variance of the standardized values.

Parameters
----------
counts : cells x features raw counts (not log-normalized).
nfeatures : number of features flagged ``highly_variable``.
features : feature names (index of the result; default ``0..p-1``).
loess_span : loess span (Seurat ``loess.span``).
clip_max : clipping value for standardized values; ``"auto"`` = ``sqrt(n_cells)``.
surface : loess surface, ``"interpolate"`` (R default) or ``"direct"``.

Returns
-------
DataFrame indexed by feature with ``mean, variance, variance_expected,
variance_standardized, rank`` (1 = most variable; NaN for zero-mean
features) and boolean ``highly_variable`` (top ``nfeatures``).
``df.index[df.highly_variable]`` is *not* ranked; use
``df.sort_values("rank").index[:n]`` for Seurat's ``VariableFeatures`` order.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> hv = arsc.pp.hvg_vst(ds.X("RNA", "counts"), nfeatures=100, features=ds.features("RNA"))
>>> int(hv.highly_variable.sum())
100
>>> top = hv.sort_values("rank").index[:100]       # Seurat VariableFeatures order
```

### `arsc.pp.log_normalize(X, scale_factor: 'float' = 10000.0, dtype=<class 'numpy.float64'>) -> 'sp.csr_matrix'`

```text
Log-normalize counts per cell: ``log1p(x / total_counts(cell) * scale_factor)``.

Matches Seurat's ``LogNormalize`` (C++ ``LogNorm``) to floating-point precision
(< 1e-10 vs ``NormalizeData``). For bitwise scanpy parity use
:func:`arsc.normalize_total_log1p`.

Parameters
----------
X : cells x features raw counts (sparse or dense).
scale_factor : target total per cell (Seurat ``scale.factor``, default 1e4).
dtype : output dtype (float64 default).

Returns
-------
scipy.sparse.csr_matrix with the same sparsity pattern as ``X``.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> L.shape
(3000, 300)
```

### `arsc.pp.pca(Z: 'np.ndarray', n_comps: 'int' = 30, method: 'str' = 'auto', seed: 'int' = 42, n_oversamples: 'int | None' = None, n_iter: 'int' = 15) -> 'PCAResult'`

```text
Truncated PCA of an already-centered (scaled) matrix, Seurat ``RunPCA`` conventions.

Seurat computes ``irlba(t(scale.data), nv = npcs)`` without re-centering, so
neither do we. ``method``:

* ``"gram"`` -- exact: eigendecomposition of the p x p Gram matrix (fast when
  p, the number of HVGs, is a few thousand). Default for p <= 6000.
* ``"arpack"`` -- exact truncated SVD via ARPACK (``scipy.sparse.linalg.svds``),
  default for p > 6000.
* ``"randomized"`` -- Halko randomized SVD (``n_iter`` power iterations,
  ``n_oversamples`` defaults to max(20, k)). Fast but approximate on the flat
  tail of the spectrum; leading PCs are exact.

Component signs are fixed so the largest-|loading| gene of each PC is positive
(Seurat's signs are arbitrary; compare with ``abs(corr)``).

``seed`` (default 42 = Seurat ``RunPCA(seed.use = 42)``): ARPACK start vector (``"arpack"``)
or the Gaussian test matrix (``"randomized"``); ``"gram"`` is deterministic and ignores it.

Parameters
----------
Z : cells x features dense, centered (scaled) matrix, e.g. the first output of :func:`scale`.
n_comps : number of components (capped at ``min(n, p) - 1``; default 30 since 0.3.0, the RunPCA
    npcs of the CD8 T-cell workflow arsc was validated on; 0.2.x defaulted to Seurat's 50).
method : ``"auto"``, ``"gram"``, ``"arpack"`` or ``"randomized"`` (see above).
seed : integer seed (``None`` is rejected).
n_oversamples : randomized-SVD oversampling.
n_iter : randomized-SVD power iterations.

Returns
-------
PCAResult with ``embedding``, ``loadings``, ``stdev`` and ``variance_ratio``.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> counts = ds.X("RNA", "counts")
>>> hv = arsc.pp.hvg_vst(counts, nfeatures=100)
>>> L = arsc.pp.log_normalize(counts)
>>> Z, _, _ = arsc.pp.scale(L[:, hv.sort_values("rank").index[:100]])
>>> pc = arsc.pp.pca(Z, n_comps=20)
>>> pc.embedding.shape
(3000, 20)
```

### `arsc.pp.PCAResult(embedding: 'np.ndarray', loadings: 'np.ndarray', stdev: 'np.ndarray', total_variance: 'float') -> None`

```text
Result of :func:`pca`.

Attributes
----------
embedding : cells x k cell embeddings (= U * d, Seurat ``cell.embeddings``).
loadings : features x k loadings (Seurat ``feature.loadings``).
stdev : per-PC standard deviation (= d / sqrt(n - 1), Seurat ``stdev``).
total_variance : total variance of the input.
variance_ratio : fraction of total variance per PC (property).

Examples
--------
>>> import arsc, numpy as np
>>> r = arsc.pp.pca(np.random.default_rng(0).standard_normal((100, 20)), n_comps=5)
>>> r.embedding.shape, r.loadings.shape, r.variance_ratio.shape
((100, 5), (20, 5), (5,))
```

### `arsc.pp.scale(X, center: 'bool' = True, do_scale: 'bool' = True, scale_max: 'float' = 10.0, dtype=<class 'numpy.float64'>) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]'`

```text
Seurat ``ScaleData`` (no regression): dense z-scores, s.d. with n-1, values > ``scale_max`` clipped.

Seurat clips only the upper tail (``FastSparseRowScale``). Zero-variance features become 0.
Select features first (``X[:, hvg_idx]``): Seurat scales only ``VariableFeatures`` by default,
and the output is dense. For regression of covariates use :func:`arsc.scale_data`.

Parameters
----------
X : cells x features (log-normalized) matrix, sparse or dense.
center : subtract the per-feature mean.
do_scale : divide by the per-feature s.d. (n-1).
scale_max : upper clip (Seurat ``scale.max``); ``None`` or ``inf`` disables clipping.
dtype : output dtype.

Returns
-------
tuple ``(Z, mean, sd)``: dense cells x features array, per-feature mean and s.d.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> Z, mu, sd = arsc.pp.scale(L[:, :50])
>>> Z.shape
(3000, 50)
```

## arsc.formats

Readers and writers.

### `arsc.formats.ArscFormatError(...)`

```text
Input file is not in a supported format, or has an inconsistent structure (a ``ValueError``).

Attributes
----------
args : the message (says which file and what was expected).

Examples
--------
>>> import arsc
>>> try:
...     arsc.detect_format("data.xyz")
... except arsc.ArscFormatError as e:
...     print("unknown format" in str(e))
True
```

### `arsc.formats.detect_format(path: 'PathLike') -> 'str'`

```text
Detect the format of a path from its name (and HDF5 layout for ``.h5``).

Parameters
----------
path : file or directory.

Returns
-------
str, one of ``store, h5ad, 10x_h5, 10x_mtx, seurat, h5seurat, loom``; raises
:class:`ArscFormatError` otherwise.

Examples
--------
>>> import arsc
>>> arsc.detect_format("obj.qs"), arsc.detect_format("pbmc.h5ad")
('seurat', 'h5ad')
```

### `arsc.formats.make_unique(names: 'Sequence[str]', join: 'str' = '-') -> 'list[str]'`

```text
Deduplicate names exactly like ``anndata.utils.make_index_unique`` (``a, a -> a, a-1``).

Parameters
----------
names : sequence of names.
join : separator before the counter.

Returns
-------
list of str, unique, first occurrences unchanged.

Examples
--------
>>> import arsc
>>> arsc.make_unique(["TBCE", "TBCE", "CD8A"])
['TBCE', 'TBCE-1', 'CD8A']
```

### `arsc.formats.read(path: 'PathLike', store: 'PathLike | None' = None, **kw) -> 'Store | Dataset'`

```text
Read any supported single-cell file.

Returns an in-memory :class:`~arsc.io.Dataset`, or, with ``store=<dir>``, converts once into a
memory-mapped :class:`~arsc.io.Store` and opens it. Arsc store directories always open as a Store.

=========================  ===============================  =========================
input                      reader                           needs
=========================  ===============================  =========================
arsc store directory       :func:`arsc.open`                --
``.h5ad``                  :func:`read_h5ad`                anndata, h5py
10x ``.h5``                :func:`read_10x_h5`              h5py
10x mtx directory          :func:`read_10x_mtx`             --
``.qs`` / ``.rds``         :func:`read_seurat`              R + SeuratObject (store=)
=========================  ===============================  =========================

``.h5Seurat`` is not parsed natively. In R, ``SeuratDisk::Convert("x.h5Seurat", dest = "h5ad")``
gives an h5ad, or ``qs::qsave(LoadH5Seurat("x.h5Seurat"), "x.qs")`` gives a qs for the exporter.

Parameters
----------
path : file or directory (any format above).
store : output store directory for a conversion (required for Seurat objects).

Returns
-------
arsc.Dataset (in memory) or arsc.Store (memory-mapped).

Examples
--------
>>> import arsc
>>> f = arsc.datasets.toy_h5ad("toy_b.h5ad", n_cells=500)
>>> ds = arsc.read(f)
>>> ds.n_cells
500
>>> st = arsc.read(f, store="toy_store2")
>>> arsc.read("toy_store2").n_cells      # a store directory reopens as a Store
500
```

### `arsc.formats.read_10x_h5(path: 'PathLike', store: 'PathLike | None' = None, *, var_names: "Literal['gene_symbols', 'gene_ids']" = 'gene_symbols', feature_types: 'Sequence[str] | None' = None, join: 'str' = '-', dtype: 'DtypeArg' = 'auto', chunk_nnz: 'int' = 33554432, overwrite: 'bool' = False) -> 'Store | Dataset'`

```text
Read a CellRanger ``filtered_feature_bc_matrix.h5`` (v3+; v2 single-genome also supported).

Features are split by ``feature_type`` into separate assays: Gene Expression -> ``RNA``,
Antibody Capture -> ``ADT``, CRISPR Guide Capture -> ``CRISPR``, Multiplexing Capture -> ``HTO``,
Peaks -> ``ATAC``. Each assay gets one layer, ``counts``. Names are deduplicated within each assay
like ``var_names_make_unique`` (``join``). ``var`` keeps ``gene_ids``, ``feature_types`` and the
other per-feature fields, as ``scanpy.read_10x_h5`` does.

Parameters
----------
path : the ``.h5`` file.
store : output store directory; ``None`` returns an in-memory Dataset.
var_names : ``"gene_symbols"`` or ``"gene_ids"`` as feature names.
feature_types : restrict the import, e.g. ``["Gene Expression"]``.
join : separator for deduplicated names.
dtype : value dtype on disk.
chunk_nnz : non-zeros per streamed chunk.
overwrite : replace an existing store.

Returns
-------
arsc.Store (with ``store=``) or arsc.Dataset with assays ``RNA`` (and ``ADT``, ...), layer ``counts``.

Examples
--------
>>> import arsc
>>> st = arsc.read_10x_h5("filtered_feature_bc_matrix.h5", store="run1")   # doctest: +SKIP
>>> st.assays                                                             # doctest: +SKIP
['RNA', 'ADT']
```

### `arsc.formats.read_10x_mtx(path: 'PathLike', store: 'PathLike | None' = None, *, var_names: "Literal['gene_symbols', 'gene_ids']" = 'gene_symbols', feature_types: 'Sequence[str] | None' = None, join: 'str' = '-', dtype: 'DtypeArg' = 'auto', overwrite: 'bool' = False) -> 'Store | Dataset'`

```text
Read a 10x Matrix Market directory (``matrix.mtx[.gz]``, ``features.tsv[.gz]`` or ``genes.tsv``,
``barcodes.tsv[.gz]``). Feature types are split into assays as in :func:`read_10x_h5`.

The triplets are parsed in one pass (scipy's fast Matrix Market reader), so peak memory is about
16 bytes per non-zero. For very large matrices, the 10x ``.h5`` is the memory-bounded route.

Parameters
----------
path : the directory (or a file inside it).
store : output store directory; ``None`` returns an in-memory Dataset.
var_names : ``"gene_symbols"`` or ``"gene_ids"``.
feature_types : restrict the import.
join : separator for deduplicated names.
dtype : value dtype on disk.
overwrite : replace an existing store.

Returns
-------
arsc.Store (with ``store=``) or arsc.Dataset.

Examples
--------
>>> import arsc
>>> ds = arsc.read_10x_mtx("filtered_feature_bc_matrix/")    # doctest: +SKIP
```

### `arsc.formats.read_h5ad(path: 'PathLike', store: 'PathLike | None' = None, *, assay: 'str' = 'RNA', x_layer: 'str' = 'X', layers: 'Sequence[str] | None' = None, raw: 'str | None' = 'raw', raw_as_layer: 'bool' = False, obsm: 'bool' = True, varm: 'bool' = True, obsp: 'bool' = True, uns: 'bool' = True, dtype: 'DtypeArg' = 'auto', chunk_nnz: 'int' = 33554432, overwrite: 'bool' = False) -> 'Store | Dataset'`

```text
Read an ``.h5ad`` file into a memory-mapped store (``store=dir``) or an in-memory Dataset.

The expression matrices (``X``, ``layers``, ``raw/X``) are streamed from disk in chunks of
about ``chunk_nnz`` non-zeros, so a file of any size converts in bounded memory. Dense, CSR and
CSC encodings are all supported; for CSC the row-major copy is built out of core. Mapping:

===================  =====================================================================
``X``                layer ``x_layer`` (default ``"X"``) of assay ``assay``
``layers[k]``        layer ``k`` of ``assay`` (``layers=None``: all; or a list; ``"X"`` = X)
``raw``              assay ``raw`` (own feature set, layer ``"X"``); ``raw=None`` skips it.
                     ``raw_as_layer=True`` stores it as layer ``raw`` of ``assay`` instead
                     (only if raw has the same genes)
``obs`` / ``var``    ``obs.parquet`` / ``assays/<assay>/var.parquet`` (categoricals, nullable
                     ints/bools, strings preserved); ``var.highly_variable`` -> variable features
``obsm[k]``          reduction ``k`` without the ``X_`` prefix (DataFrames kept with columns)
``varm[k]``          ``Store.varm(assay, k)``; ``varm["PCs"]`` also becomes the PCA loadings
``obsp[k]``          graph ``k``
``uns``              JSON-able part (scalars, strings, lists, arrays, nested dicts) -> ``Store.uns``
===================  =====================================================================

With ``store`` set, the store is written atomically and then opened. If the same source was
already converted there, the existing store is opened and nothing is rewritten.

Parameters
----------
path : the ``.h5ad`` file.
store : output store directory; ``None`` returns an in-memory Dataset.
assay : assay name for ``X`` and ``layers``.
x_layer : layer name for ``X``.
layers : AnnData layers to import (``None`` = all).
raw : assay name for ``.raw`` (``None`` skips it).
raw_as_layer : store ``.raw`` as a layer of ``assay`` (same genes only).
obsm, varm, obsp, uns : import these AnnData slots.
dtype : value dtype on disk (``"auto"``: float32 when exact, else the source dtype).
chunk_nnz : non-zeros per streamed chunk.
overwrite : replace an existing store. Without it, an existing store built from a file of the same name
    is reopened as is (with a warning if the file is newer than the store).

Returns
-------
arsc.Store (with ``store=``) or arsc.Dataset. Needs ``anndata`` and ``h5py`` (``pip install "arsc[h5ad]"``).

Examples
--------
>>> import arsc
>>> f = arsc.datasets.toy_h5ad("toy_a.h5ad", n_cells=500)
>>> ds = arsc.read_h5ad(f)                      # in memory
>>> st = arsc.read_h5ad(f, store="toy_h5ad_store")   # memory-mapped store, converted once
>>> st.n_cells, st.layers("RNA")
(500, ['X'])
```

### `arsc.formats.read_seurat(path: 'PathLike', store: 'PathLike', **export_kw) -> 'Store'`

```text
Convert a Seurat v4/v5 object (``.qs`` preferred, ``.rds`` accepted) with the bundled R exporter.

Needs R with SeuratObject (and ``qs`` for ``.qs``). Writes ``store`` once; later calls open it.
``export_kw`` are passed to :func:`arsc.export_seurat` (``assays``, ``layers``, ``graphs``, ...).
For ``.h5Seurat`` files, see :func:`read` (convert to h5ad with SeuratDisk, or save as qs in R).

Parameters
----------
path : ``.qs`` or ``.rds`` file with a Seurat object.
store : output store directory (required).

Returns
-------
arsc.Store.

Examples
--------
>>> import arsc
>>> st = arsc.read_seurat("obj.qs", store="stores/obj")      # doctest: +SKIP
```

### `arsc.formats.StoreBuilder(out: 'PathLike | None', *, source: 'str' = '', overwrite: 'bool' = False)`

```text
Assemble an arsc store (``out`` = directory) or an in-memory Dataset (``out=None``) piece by piece.

The order is cells, then assays and layers, then the optional reductions, graphs, varm and uns,
and finally :meth:`finish`. The directory version writes into ``<out>.tmp<pid>`` and renames it
when finished, so an interrupted conversion never leaves a half-written store behind.

Parameters
----------
out : store directory, or ``None`` for an in-memory :class:`arsc.Dataset`.
source : free-text provenance recorded in the manifest.
overwrite : replace an existing store at ``out``.

Examples
--------
>>> import arsc, numpy as np, pandas as pd, scipy.sparse as sp
>>> counts = sp.random(100, 20, density=0.2, format="csr", random_state=0) * 10
>>> b = arsc.StoreBuilder("my_store")
>>> b.set_cells([f"c{i}" for i in range(100)], pd.DataFrame({"patient": ["P1", "P2"] * 50}))
>>> b.add_assay("RNA", [f"g{j}" for j in range(20)])
>>> b.add_layer("RNA", "counts", counts.floor())
>>> st = b.finish()
>>> st.n_cells, st.layers("RNA")
(100, ['counts'])
```

### `arsc.formats.to_anndata(obj: 'Store | Dataset', cells=None, genes=None, *, assay: 'str | None' = None, x_layer: 'str | None' = None, layers: 'Sequence[str] | None' = None, raw: 'bool' = True, obsm: 'bool' = True, varm: 'bool' = True, obsp: 'bool' = False, uns: 'bool' = True, restore_dtype: 'bool' = True)`

```text
Build an in-memory :class:`anndata.AnnData` from a store or dataset (scanpy hand-off).

``cells`` / ``genes``: boolean mask, positions or names (``None`` = all). Only the selected rows
are read from the memory map. ``X`` is ``x_layer``: the original ``.X`` for h5ad-derived stores,
otherwise ``counts`` when present. The other layers of ``assay`` go to ``.layers``. ``raw``
restores ``.raw`` when the store came from an h5ad that had one. ``obsp=True`` also exports the
graphs. Values are cast back to the source dtype recorded at import (``restore_dtype``).

Parameters
----------
obj : arsc.Store or arsc.Dataset.
cells : cells to export (mask, positions or barcodes).
genes : genes to export (mask, positions or names).
assay : assay for ``X`` (default: the store's default assay).
x_layer : layer used as ``X``.
layers : other layers to export (default all).
raw : restore ``.raw``.
obsm, varm, obsp, uns : export these slots.
restore_dtype : cast to the source dtype.

Returns
-------
anndata.AnnData.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> ad = arsc.to_anndata(ds, cells=(ds.obs.cell_type == "CD8 T").to_numpy(), genes=["CD8A", "GZMK"], assay="RNA")
>>> ad.n_vars, list(ad.var_names)
(2, ['CD8A', 'GZMK'])
```

### `arsc.formats.write_h5ad(obj: 'Store | Dataset', path: 'PathLike', cells=None, genes=None, *, assay: 'str | None' = None, x_layer: 'str | None' = None, layers: 'Sequence[str] | None' = None, raw: 'bool' = True, obsm: 'bool' = True, varm: 'bool' = True, obsp: 'bool' = False, uns: 'bool' = True, restore_dtype: 'bool' = True, compression: 'str | None' = None, chunk_nnz: 'int' = 33554432, overwrite: 'bool' = False) -> 'Path'`

```text
Write a store, dataset or subset to ``.h5ad`` without materializing the full matrix.

The expression matrices are streamed in row chunks (CSR on disk), so a store larger than RAM
exports in bounded memory. Arguments are as in :func:`to_anndata`. ``compression`` is passed to
h5py (``"gzip"``, ``"lzf"``, or ``None``, which is uncompressed and fastest). The file is
written to ``<path>.tmp`` and renamed. The output reads back with ``anndata.read_h5ad`` and
``scanpy.read_h5ad``.

Parameters
----------
obj : arsc.Store or arsc.Dataset.
path : output ``.h5ad``.
cells, genes, assay, x_layer, layers, raw, obsm, varm, obsp, uns, restore_dtype : as in :func:`to_anndata`.
compression : h5py compression.
chunk_nnz : non-zeros per streamed chunk.
overwrite : replace an existing file.

Returns
-------
pathlib.Path of the written file.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> p = arsc.write_h5ad(ds, "cd8.h5ad", cells=(ds.obs.cell_type == "CD8 T").to_numpy(), assay="RNA")
>>> p.name
'cd8.h5ad'
```

## arsc.io

Store, Dataset and the Seurat bridge.

### `arsc.io.Assay(name: 'str', features: 'pd.Index', layers: 'dict' = <factory>, variable_features: 'list' = <factory>, var: 'pd.DataFrame | None' = None, varm: 'dict' = <factory>) -> None`

```text
One modality (RNA, ADT, ...). Every layer is a cells x features ``scipy.sparse.csr_matrix``.

Attributes
----------
name : assay name.
features : feature names.
layers : ``{layer: csr_matrix}`` (e.g. ``counts``, ``data``).
variable_features : variable features recorded at import.
var : optional per-feature annotations.
varm : per-feature matrices.

Examples
--------
>>> import arsc
>>> a = arsc.datasets.toy()["ADT"]
>>> a.name, len(a.features), list(a.layers)
('ADT', 6, ['counts'])
```

### `arsc.io.Dataset(obs: 'pd.DataFrame', assays: 'dict' = <factory>, reductions: 'dict' = <factory>, graphs: 'dict' = <factory>, manifest: 'dict' = <factory>, uns: 'dict' = <factory>) -> None`

```text
In-memory (or mmap-backed) container: ``obs``, ``assays``, ``reductions``, ``graphs``.

Returned by :func:`arsc.read` (no ``store=``), :meth:`Store.subset` and :func:`arsc.datasets.toy`.
Every matrix is cells x features (CSR).

Attributes
----------
obs : per-cell metadata (index = barcodes).
assays : ``{name: Assay}``; ``ds.X(assay, layer)`` returns a matrix.
reductions : ``{name: Reduction}``; ``ds.embedding(name)``.
graphs : ``{name: csr_matrix}``.
manifest : store manifest (provenance).
uns : unstructured metadata.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> ds.n_cells, list(ds.assays), ds.layers("RNA")
(3000, ['RNA', 'ADT'], ['counts'])
>>> cd8 = ds.subset((ds.obs["cell_type"] == "CD8 T").to_numpy())
>>> cd8.X("RNA", "counts").shape[1]
300
```

### `arsc.io.export_seurat(obj_path, out_dir, *, assays=None, layers=('counts',), graphs: 'bool' = True, reductions: 'bool' = True, rscript: 'str' = 'Rscript', threads: 'int' = 16, full: 'bool' = False, scale_f64: 'bool' = False) -> 'Path'`

```text
Run the R exporter on a ``.rds``/``.qs`` Seurat object (v4 or v5); prefer :func:`arsc.read_seurat`.

Parameters
----------
obj_path : ``.rds`` or ``.qs`` file.
out_dir : store directory to write.
assays : assays to export (default all).
layers : layers to export (``"counts"``, ``"data"``, ``"scale.data"``).
graphs, reductions : export graphs / reductions.
rscript : path of ``Rscript``.
threads : R threads.
full : export everything the object holds (every assay and layer, scale.data as a dense array,
    per-feature metadata, Idents, reductions with projected loadings, graphs, Neighbor objects,
    misc/commands/tools as ``uns``); what cannot be converted is listed in
    ``manifest["unconvertible"]``. ``assays``/``layers``/``graphs``/``reductions`` are ignored.
scale_f64 : with ``full``, keep scale.data as float64 (default float32: half the disk).

Returns
-------
pathlib.Path ``out_dir`` (open it with :func:`open`). Needs R with SeuratObject (and qs).

Examples
--------
>>> import arsc
>>> arsc.export_seurat("obj.qs", "stores/obj", layers=("counts", "data"))   # doctest: +SKIP
```

### `arsc.io.exporter_script() -> 'Path'`

```text
Path of the bundled R exporter (``Rscript <path> <obj.rds|.qs> <out_dir>``).

Returns
-------
pathlib.Path of ``export_seurat.R`` inside the installed package.

Examples
--------
>>> import arsc
>>> arsc.exporter_script().name
'export_seurat.R'
```

### `arsc.io.from_anndata(h5ad, out_dir, *, layers=('counts',), assay: 'str' = 'RNA', use_raw_for=None, obsm: 'bool' = True, chunk: 'int | None' = None, obsp: 'bool' = False) -> 'Path'`

```text
Convert an .h5ad into an arsc store so that many processes share one memory-mapped copy.

Compatibility wrapper around :func:`arsc.read_h5ad`; prefer ``arsc.read(h5ad, store=out_dir)``,
which imports every layer by default.

``layers``: AnnData layer names to export (``"X"`` exports ``.X`` as layer ``X``).
``use_raw_for``: name under which to export ``.raw.X``. This is a layer of ``assay`` when raw has
the same genes, and otherwise a separate assay of that name. The matrices are streamed from
disk in chunks (memory-safe; ``chunk`` is accepted for backward compatibility and ignored).
``obsm`` embeddings become reductions. Idempotent: if ``out_dir`` already holds a store, it is
returned untouched.

Parameters
----------
h5ad : input ``.h5ad``.
out_dir : store directory.
layers : AnnData layers to export.
assay : assay name.
use_raw_for : name for ``.raw.X``.
obsm : export embeddings.
chunk : ignored (backward compatibility).
obsp : export graphs.

Returns
-------
pathlib.Path ``out_dir``.

Examples
--------
>>> import arsc
>>> f = arsc.datasets.toy_h5ad("toy_fa.h5ad", n_cells=300)
>>> arsc.open(arsc.from_anndata(f, "toy_fa_store", layers=("X",))).n_cells
300
```

### `arsc.io.open(path, finalize: 'bool' = True, build_csc: 'bool' = True) -> 'Store'`

```text
Open an arsc store lazily (memory-mapped, zero-copy). See :class:`Store`.

The first open of a fresh export writes ``obs.parquet`` and the per-feature
CSC copies (one-time; concurrent first opens wait on a lock instead of
duplicating work). Pass ``finalize=False`` for strictly read-only use.

Parameters
----------
path : store directory.
finalize : build missing one-time files.
build_csc : build per-feature CSC copies.

Returns
-------
arsc.Store.

Examples
--------
>>> import arsc
>>> _ = arsc.datasets.toy(store="toy_store_open")
>>> st = arsc.open("toy_store_open")
>>> st.n_cells
3000
```

### `arsc.io.read_seurat_export(path, *, assays=None, layers=None, graphs: 'bool' = True, reductions: 'bool' = True) -> 'Dataset'`

```text
Load an arsc store as a :class:`Dataset` (memory-mapped arrays; nothing is copied).

Parameters
----------
path : store directory.
assays, layers : restrict to these assays / layers.
graphs, reductions : include graphs / reductions.

Returns
-------
arsc.Dataset.

Examples
--------
>>> import arsc
>>> _ = arsc.datasets.toy(store="toy_store_rse")
>>> arsc.read_seurat_export("toy_store_rse", assays=["RNA"]).n_cells
3000
```

### `arsc.io.Reduction(embedding: 'np.ndarray', columns: 'list', loadings: 'pd.DataFrame | None' = None, stdev: 'np.ndarray | None' = None, assay: 'str | None' = None, projected: 'pd.DataFrame | None' = None, key: 'str | None' = None) -> None`

```text
Cell embedding (cells x k), optional feature loadings and stdev.

Attributes
----------
embedding : cells x k array.
columns : column names (e.g. ``PC_1``...).
loadings : optional features x k DataFrame.
stdev : optional per-component s.d.
assay : assay the reduction was computed from.
projected : optional projected loadings (features x k DataFrame; Seurat ``ProjectDim``).
key : column key (e.g. ``PC_``).

Examples
--------
>>> import arsc, numpy as np
>>> r = arsc.Reduction(np.zeros((5, 2)), ["PC_1", "PC_2"])
>>> r.to_frame().shape
(5, 2)
```

### `arsc.io.Store(path, finalize: 'bool' = True, build_csc: 'bool' = True)`

```text
Lazy, memory-mapped handle on an arsc store directory (see :func:`open`).

Nothing is read until asked for: ``st.obs`` reads the metadata, ``st.X(assay, layer)`` returns a
zero-copy CSR backed by the OS page cache (shared by all processes), ``st.get_features`` reads
only the requested genes, ``st.subset(mask)`` materializes only those cells.

Parameters
----------
path : store directory (contains ``manifest.json``).
finalize : build the one-time ``obs.parquet`` / CSC copies if missing.
build_csc : build per-feature CSC copies (needed by ``get_features``).

Attributes
----------
path : the store directory.
manifest : parsed ``manifest.json``.

Examples
--------
>>> import arsc
>>> st = arsc.datasets.toy(store="toy_store_cls")   # writes a memory-mapped store
>>> st.n_cells, st.assays
(3000, ['RNA', 'ADT'])
>>> X = st.X("RNA", "counts")                     # memory-mapped, no copy
>>> st.get_features(["CD8A", "GZMK"]).shape       # reads only these genes
(3000, 2)
```

### `arsc.io.write_table(df: 'pd.DataFrame', path, index: 'bool | None' = None) -> 'Path'`

```text
Write a tidy table (e.g. for plotting in R); format from the extension (.parquet/.csv/.tsv).

Read in R with ``arrow::read_parquet`` or ``data.table::fread``. Parent directories are created.

Parameters
----------
df : the table.
path : output path ending in ``.parquet``, ``.csv`` or ``.tsv``.
index : write the index (default: only if it is not a RangeIndex).

Returns
-------
pathlib.Path of the written file.

Examples
--------
>>> import arsc, pandas as pd
>>> arsc.write_table(pd.DataFrame({"a": [1, 2]}), "out/table.csv").name
'table.csv'
```

## arsc.views

Subset stores as views of their parent stores.

### `arsc.views.array_sha256(a) -> 'str'`

```text
sha256 of an array's data bytes (C order); equal for a file's array and the view's reconstruction.

Parameters
----------
a : array-like.

Returns
-------
str (hex digest).

Examples
--------
>>> import numpy as np, arsc
>>> arsc.views.array_sha256(np.arange(3)) == arsc.views.array_sha256(np.array([0, 1, 2]))
True
```

### `arsc.views.build(full, parents, out, rows, *, min_fraction: 'float' = 0.1, min_bytes: 'int' = 0, dry_run: 'bool' = False, log=<built-in function print>) -> 'dict'`

```text
Build a view of the full store ``full`` on ``parents``.

Parameters
----------
full : path of the full store (or an open Store) whose content the view must reproduce exactly.
parents : list of parent store paths (full stores, not views).
out : the new view directory (must not exist; ignored with ``dry_run``).
rows : ``{parent index: (view_rows, parent_rows)}`` candidate pairs (int arrays). Each pair is only a candidate:
    a layer inherits a row only when the parent row equals it bit for bit (indices and values, after the
    feature map and the full store's dtype).
min_fraction : a layer inherits from parents only when at least this fraction of its rows matched (else it stays
    whole in the view).
min_bytes : no view when it would save fewer bytes than this (the report says so).
dry_run : only match and report.

log : a function taking one message (default print).

Returns
-------
dict report: per layer the rows matched per source, whether it is inherited and the bytes saved; ``view``
(built or not, with ``why_not``), ``out`` and the full and view sizes.

Examples
--------
>>> import arsc, tempfile, numpy as np
>>> t = arsc.datasets.toy_view(tempfile.mkdtemp())
>>> rep = arsc.views.build(t["full"], [t["parent"]], None, {0: (np.arange(300), t["rows"])}, dry_run=True,
...                        log=lambda *a: None)
>>> rep["layers"]["RNA/counts"]["matched"], rep["layers"]["RNA/counts"]["inherit"]
(300, True)
```

### `arsc.views.compare(view, full, *, feature_batch: 'int' = 4096, subset_n: 'int' = 2000, seed: 'int' = 0, log=None) -> 'list'`

```text
Compare a view with the full store it replaces on every accessor, bit for bit.

Returns a list of ``{"check", "ok", "detail"}``; all ``ok`` means the view is a drop-in replacement. Checked:
manifest, cells, n_cells, obs, assays, layers, features, var, varm, variable_features, dense layers, splits,
neighbors, idents, unconvertible, uns, reductions (embedding, loadings, projected, stdev, key, assay, columns),
graphs, ``X`` and ``X_csc`` of every layer (aliases included), ``get_features`` of every feature (sparse) and of
one batch (dense), ``subset`` of random cells (every layer, obs, reductions) and the parent store files' sizes.

Parameters
----------
view : the view (path or Store).
full : the full store it replaces (path or Store).
feature_batch : features per ``get_features`` call.
subset_n : random cells for the ``subset`` check.
seed : integer seed of that draw.
log : optional function called with each failure.

Returns
-------
list of dict (``check``, ``ok``, ``detail``).

Examples
--------
>>> import arsc, tempfile
>>> t = arsc.datasets.toy_view(tempfile.mkdtemp())
>>> res = arsc.views.compare(t["view"], t["full"])
>>> all(r["ok"] for r in res), len(res) > 30
(True, True)
```

### `arsc.views.is_view(path) -> 'bool'`

```text
True when ``path`` is a store directory holding a view (``view.json``).

Parameters
----------
path : a store directory.

Returns
-------
bool.

Examples
--------
>>> import arsc, tempfile
>>> t = arsc.datasets.toy_view(tempfile.mkdtemp())
>>> arsc.views.is_view(t["view"]), arsc.views.is_view(t["full"])
(True, False)
```

### `arsc.views.logical_bytes(path) -> 'int'`

```text
Bytes the full store would take on disk (recorded at build; the store's own size for a full store).

Parameters
----------
path : a store directory.

Returns
-------
int.

Examples
--------
>>> import arsc, tempfile
>>> t = arsc.datasets.toy_view(tempfile.mkdtemp())
>>> arsc.views.logical_bytes(t["view"]) > 0
True
```

### `arsc.views.materialize(view, out) -> 'Path'`

```text
Write the full store a view stands for (every file of the view except view files, plus each inherited layer's
CSR and CSC arrays in the dtypes of the original files). Its arrays and other files equal the store the view was
built from.

Parameters
----------
view : the view (path or Store); its parents must be reachable.
out : the new full store directory (must not exist).

Returns
-------
Path of the full store.

Examples
--------
>>> import arsc, tempfile, os
>>> t = arsc.datasets.toy_view(tempfile.mkdtemp())
>>> m = arsc.views.materialize(t["view"], os.path.join(tempfile.mkdtemp(), "full_again"))
>>> arsc.open(m).is_view
False
```

### `arsc.views.parent_paths(path) -> 'list'`

```text
Resolved parent store directories of a view (``[]`` for a full store). Raises ViewError when one is missing.

Parameters
----------
path : a store directory.

Returns
-------
list of Path.

Examples
--------
>>> import arsc, tempfile
>>> t = arsc.datasets.toy_view(tempfile.mkdtemp())
>>> [p.name for p in arsc.views.parent_paths(t["view"])]
['parent']
```

### `arsc.views.ViewError(...)`

```text
A view's parent is missing or changed, or a view cannot be built.

Attributes
----------
args : the message (as for any exception).

Examples
--------
>>> from arsc.views import ViewError
>>> issubclass(ViewError, RuntimeError)
True
```

## arsc.markers

Markers (presto).

### `arsc.markers.top_markers(res: 'pd.DataFrame', n: 'int' = 10, auc_min: 'float' = 0.0, padj_max: 'float' = 1.0, pct_in_min: 'float' = 0.0, by: 'str' = 'auc') -> 'pd.DataFrame'`

```text
Top ``n`` features per group from :func:`wilcoxauc` output (filters, then sort by ``by``).

Parameters
----------
res : output of :func:`wilcoxauc`.
n : features kept per group.
auc_min : minimum AUC.
padj_max : maximum BH-adjusted p.
pct_in_min : minimum percent of in-group cells expressing.
by : column to sort by (descending), default ``"auc"``.

Returns
-------
DataFrame with the same columns as ``res``, up to ``n`` rows per group.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> mk = arsc.wilcoxauc(L, ds.obs["cell_type"], features=ds.features("RNA"))
>>> top = arsc.top_markers(mk, n=5, auc_min=0.6, padj_max=0.05)
>>> int(top.groupby("group").size().max())
5
```

### `arsc.markers.wilcoxauc(X, groups, *, features=None, groups_use=None, obs: 'pd.DataFrame | None' = None) -> 'pd.DataFrame'`

```text
Wilcoxon rank-sum test of every group vs. all other cells, for every feature.

Parameters
----------
X : cells x features matrix (sparse CSR/CSC or dense), e.g. log-normalized data.
    Values are assumed non-negative (zeros are ranked as one tied block, as presto).
groups : labels per cell, or a column name of ``obs``. NA cells are dropped.
features : feature names (default ``0..p-1``).
groups_use : restrict to cells in these groups (presto ``groups_use``).
obs : cell metadata, needed only when ``groups`` is a column name.

Constant (all-zero) features get ``auc = 0.5, pval = 1`` as in presto. ``pval`` is a cell-level
test: use it to rank markers, not as evidence for a difference between conditions (for that,
pseudobulk per patient and use :func:`arsc.de.deseq2` / :func:`arsc.de.voom`).

Returns
-------
Tidy DataFrame (n_groups * n_features rows, group-major like presto):
``feature, group, avgExpr, logFC, statistic, auc, pval, padj, pct_in, pct_out``.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> mk = arsc.wilcoxauc(L, ds.obs["cell_type"], features=ds.features("RNA"))
>>> list(mk.columns)
['feature', 'group', 'avgExpr', 'logFC', 'statistic', 'auc', 'pval', 'padj', 'pct_in', 'pct_out']
>>> top = arsc.top_markers(mk, n=3)
>>> top[top.group == "B"].feature.tolist()[0] in ("MS4A1", "CD79A", "CD19")
True
```

## arsc.scores

Signature scores.

### `arsc.scores.add_module_score(X, features, signatures, *, pool=None, nbin: 'int' = 24, ctrl: 'int' = 100, seed: 'int' = 1, name: 'str' = 'Cluster', cells=None, return_controls: 'bool' = False)`

```text
Seurat ``AddModuleScore``: mean expression of each gene set minus a binned control set.

Parameters
----------
X : cells x genes matrix of normalized expression (Seurat's ``data`` slot).
features : gene names.
signatures : list of gene lists, or dict ``{name: genes}`` (dict keys become
    column names; otherwise ``f"{name}{i}"`` like Seurat).
pool : genes to draw controls from (default all features).
nbin : number of expression bins (Seurat ``nbin``).
ctrl : random control genes drawn from each signature gene's bin (Seurat ``ctrl``); must not
    exceed the genes per bin (about ``len(pool) / nbin``).
seed : R ``set.seed`` value; ``seed=1`` (Seurat's default) reproduces R exactly. The
    tie-breaking noise and control-gene draws use a private R-compatible RNG
    (``_rrng.RRandom``), so any integer reproduces R's ``AddModuleScore(seed = s)``.
    ``None`` (Seurat: do not reseed) is rejected.
name : column prefix when ``signatures`` is a list (Seurat ``name``).
cells : index of the output (e.g. ``ds.cells``).
return_controls : also return the control-gene lists.

Returns
-------
DataFrame (cells x signatures); with ``return_controls`` also the list of
control-gene lists.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> ms = arsc.add_module_score(L, ds.features("RNA"), {"Bcell": ["MS4A1", "CD79A", "CD19"]},
...                            ctrl=5, cells=ds.cells)
>>> ms.groupby(ds.obs["cell_type"], observed=True)["Bcell"].mean().idxmax()
'B'
```

### `arsc.scores.parse_signature(genes) -> 'tuple[list, list]'`

```text
Split UCell-style genes into (positive, negative): trailing ``-`` = negative, ``+`` stripped.

Parameters
----------
genes : iterable of gene names, optionally suffixed ``+`` / ``-``.

Returns
-------
tuple ``(positive, negative)`` of gene-name lists.

Examples
--------
>>> import arsc
>>> arsc.scores.parse_signature(["CD8A", "GZMK+", "CD4-"])
(['CD8A', 'GZMK'], ['CD4'])
```

### `arsc.scores.score_ucell(X, features, signatures: 'dict', *, max_rank: 'int' = 1500, w_neg: 'float' = 1.0, suffix: 'str' = '_UCell', cells=None) -> 'pd.DataFrame'`

```text
UCell scores (``ScoreSignatures_UCell`` defaults; use raw counts like UCell's default).

Parameters
----------
X : cells x genes sparse matrix (non-negative).
features : gene names (columns of ``X``).
signatures : ``{name: [genes]}``; ``"GENE-"`` marks a negative gene, ``"GENE+"``
    is the same as ``"GENE"``. Score = max(0, U_pos - w_neg * U_neg).
max_rank : rank cap (``maxRank``); lowered to the number of genes if larger.
w_neg : weight of the negative genes.
suffix : appended to each signature name in the output columns.
cells : index of the output (e.g. ``ds.cells``); default a RangeIndex.

Returns
-------
DataFrame, cells x signatures, columns ``f"{name}{suffix}"``; scores in [0, 1].

Notes
-----
Faithful to UCell 2.6 quirks: genes missing from the data (union across all
signatures) are appended as rows of ``min(X)`` before ranking, which enlarges
the zero-tie block; duplicated genes within a signature count twice.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> sc = arsc.score_ucell(ds.X("RNA", "counts"), ds.features("RNA"),
...                       {"cyto": ["GZMK", "GZMB", "PRF1", "NKG7"]}, cells=ds.cells)
>>> list(sc.columns)
['cyto_UCell']
>>> sc.groupby(ds.obs["cell_type"], observed=True)["cyto_UCell"].mean().idxmax()
'NK'
```

## arsc.neighbors

Neighbors and WNN.

### `arsc.neighbors.find_neighbors(emb: 'np.ndarray', k: 'int' = 20, prune: 'float' = 0.06666666666666667, l2_norm: 'bool' = False, method: 'str' = 'auto', seed: 'int' = 0) -> 'dict'`

```text
Seurat ``FindNeighbors``: k-NN (self included) and SNN graphs from an embedding.

``method`` and ``seed`` are passed to :func:`knn`: by default (``"auto"``) the k-NN is exact up to
4,096 cells and approximate (arsc's clustered search) above; ``method="exact"`` forces exact search.
Seurat's Annoy is approximate, so SNN graphs differ slightly from Seurat's (see the README validation
table).

Parameters
----------
emb : (n x d) embedding (PCs, harmony).
k : neighbors including self (Seurat ``k.param``).
prune : SNN Jaccard cutoff (Seurat ``prune.SNN``).
l2_norm : L2-normalize rows first (Seurat ``l2.norm``).
method : ``"auto"`` (default), ``"exact"`` or ``"approximate"`` (see :func:`knn`).
seed : integer seed.

Returns
-------
dict with ``"idx"``, ``"dist"`` (n x k arrays), ``"nn"`` (binary CSR) and ``"snn"`` (CSR), the
input for :func:`arsc.cluster`.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> Z, _, _ = arsc.pp.scale(L)
>>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
>>> g = arsc.find_neighbors(pcs, k=20)
>>> sorted(g)
['dist', 'idx', 'nn', 'snn']
>>> labels = arsc.cluster(g["snn"], resolution=0.5)
>>> labels.shape
(3000,)
```

### `arsc.neighbors.knn(X, k: 'int', query=None, *, method: 'str' = 'auto', seed: 'int' = 0, n_clusters: 'int | None' = None)`

```text
k nearest neighbors (Euclidean): exact up to 4,096 points, approximate above (``"auto"``).

Both methods start from a seeded k-means partition of the points (numba and BLAS, no other dependency) and
report exact float64 distances (the plain sum of squared differences, as brute force), ordered by (distance,
index).

``method="exact"``: each query scans clusters in order of their center's distance, skipping a cluster or a
ring of it when the triangle inequality proves it cannot hold a closer point; candidates pass a float32 filter
with a proven rounding margin before the float64 check. The answer is the exact k-NN at any size; ``seed`` and
``n_clusters`` change only the speed. Deterministic; the query point itself comes first when ``query`` is None.

``method="approximate"`` (alias ``"approx"``): about 2 sqrt(n) clusters; each query scans every point of its
24 nearest clusters. Distances to a cluster's points are float32 matrix products of the queries that probe it
(grouped by nearest center) and the cluster's points, both centered on the cluster's center; the k + 8 best
candidates are re-scored exactly in float64. Recall@k on real PCs: 0.96-0.996 (100k and 1M cells, k = 15 and
30). Deterministic for a given ``seed`` and independent of the thread count. Speed against FAISS, hnswlib,
Voyager, Annoy, pynndescent and scanpy: ``SCANPY_PARITY.md`` ("k-NN backend").

``method="auto"`` (default): exact when n <= 4,096 or ``query`` is given, approximate above, as
``scanpy.pp.neighbors``.

Parameters
----------
X : (n x d) dense embedding (e.g. PCs).
k : neighbors per query (self included when ``query`` is None); capped at n.
query : optional (m x d) query points; default ``X`` itself.
method : ``"auto"``, ``"exact"`` or ``"approximate"`` (``"approx"``).
seed : integer seed of the k-means partition.
n_clusters : number of partitions (default about sqrt(n)).

Returns
-------
tuple ``(idx, dist)``: int64 and float64 arrays, each (n_query x k), nearest first.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> Z, _, _ = arsc.pp.scale(L)
>>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
>>> idx, dist = arsc.knn(pcs, 15)
>>> idx.shape, bool((idx[:, 0] == range(3000)).all())
((3000, 15), True)
```

### `arsc.neighbors.l2_normalize(X: 'np.ndarray') -> 'np.ndarray'`

```text
Row-wise L2 normalization; zero rows stay 0 (Seurat ``L2Norm``).

Parameters
----------
X : (n x d) array.

Returns
-------
numpy.ndarray (float64) with unit-norm rows.

Examples
--------
>>> import arsc, numpy as np
>>> arsc.neighbors.l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
array([[0.6, 0.8],
       [0. , 0. ]])
```

### `arsc.neighbors.snn_graph(idx: 'np.ndarray', prune: 'float' = 0.06666666666666667) -> 'sp.csr_matrix'`

```text
Seurat ``ComputeSNN``: Jaccard similarity of k-NN sets, entries < ``prune`` removed.

Parameters
----------
idx : (n x k) neighbor index matrix, e.g. from :func:`knn` (include self for
    ``FindNeighbors`` semantics).
prune : Jaccard cutoff (Seurat ``prune.SNN``, default 1/15).

Returns
-------
scipy.sparse.csr_matrix, symmetric (n x n) SNN weights.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> Z, _, _ = arsc.pp.scale(L)
>>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
>>> idx, _ = arsc.knn(pcs, 20)
>>> arsc.snn_graph(idx).shape
(3000, 3000)
```

### `arsc.neighbors.wnn(embeddings: 'list', k_nn: 'int' = 20, knn_range: 'int' = 200, prune_snn: 'float' = 0.06666666666666667, sd_scale: 'float' = 1.0, cross_constant=0.0001, l2_norm: 'bool' = True, smooth: 'bool' = False, knn_method: 'str' = 'auto', seed: 'int' = 0) -> 'WNNResult'`

```text
Weighted nearest neighbors (Seurat ``FindMultiModalNeighbors`` / ``FindModalityWeights``).

Parameters
----------
embeddings : list of (n x d_m) arrays, one per modality (e.g. ``[harmony_rna[:, :30],
    harmony_adt[:, :15]]``).
k_nn, knn_range, prune_snn, sd_scale, cross_constant, l2_norm, smooth : as in Seurat
    (``cross.contant.list`` -> ``cross_constant``, scalar or per modality).
knn_method, seed : passed to :func:`knn` (``"auto"``: exact up to 4,096 cells, approximate above).

Returns
-------
WNNResult with ``weights``, ``idx``, ``dist``, ``wknn``, ``wsnn``, ``sigma``, ``nearest_dist``.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> rna = arsc.pp.pca(arsc.pp.scale(arsc.pp.log_normalize(ds.X("RNA", "counts")))[0], n_comps=10).embedding
>>> adt = arsc.pp.pca(arsc.pp.scale(arsc.pp.clr(ds.X("ADT", "counts")))[0], n_comps=4).embedding
>>> w = arsc.wnn([rna, adt], k_nn=20, knn_range=50)
>>> labels = arsc.cluster(w.wsnn, resolution=0.5)
```

### `arsc.neighbors.WNNResult(weights: 'np.ndarray', idx: 'np.ndarray', dist: 'np.ndarray', wknn: 'sp.csr_matrix', wsnn: 'sp.csr_matrix', sigma: 'np.ndarray', nearest_dist: 'np.ndarray') -> None`

```text
Output of :func:`wnn`.

Attributes
----------
weights : n x M modality weights (rows sum to 1; Seurat ``RNA.weight`` etc.).
idx, dist : n x k weighted neighbors, self excluded (Seurat ``weighted.nn``).
wknn : binary, symmetrized k-NN graph, diagonal 1.
wsnn : pruned Jaccard SNN graph (cluster this).
sigma : per-cell kernel bandwidths, per modality.
nearest_dist : distance to the nearest neighbor, per modality.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> rna = arsc.pp.pca(arsc.pp.scale(arsc.pp.log_normalize(ds.X("RNA", "counts")))[0], n_comps=10).embedding
>>> adt = arsc.pp.pca(arsc.pp.scale(arsc.pp.clr(ds.X("ADT", "counts")))[0], n_comps=4).embedding
>>> w = arsc.wnn([rna, adt], k_nn=20, knn_range=50)
>>> w.weights.shape, w.wsnn.shape
((3000, 2), (3000, 3000))
```

## arsc.harmony

Harmony.

### `arsc.harmony.Harmony(Z, codes, B_vec, *, theta, sigma, lambda_vec, alpha=0.2, max_iter_cluster=20, epsilon_cluster=0.001, epsilon_harmony=0.01, K=100, block_size=0.05, seed=0, version='1.2', batch_prop_cutoff=1e-05)`

```text
Stateful Harmony object (port of the C++ class of R harmony 1.2 and 2.x). Most users want :func:`run_harmony`.

Parameters
----------
Z : (n_cells, d) embedding.
codes : (n_cells, n_variables) integer batch codes, global across variables (variable v's levels follow
    variable v-1's).
B_vec : number of levels per variable.
theta : per-level diversity penalty (length sum(B_vec)).
sigma : per-cluster soft k-means width (length ``K``).
lambda_vec : ridge penalties ``[0, per level...]``, or ``None`` for automatic lambda.
alpha : automatic-lambda factor (default 0.2).
max_iter_cluster : clustering iterations per round (default 20).
epsilon_cluster, epsilon_harmony : convergence tolerances.
K : number of clusters.
block_size : fraction of cells per online-update block (default 0.05; must be < 1 for version 1.2).
seed : integer seed for k-means init and block order.
version : R harmony release to reproduce: ``"1.2"`` (default here), ``"2.0"`` or ``"2.1"`` (see :func:`run_harmony`).
batch_prop_cutoff : 2.x: minimum share of a batch level in a cluster for it to enter that cluster's regression.

Attributes
----------
Z_corr : corrected embedding after :meth:`harmonize`; ``R``, ``Y``, objective traces as in
:class:`HarmonyResult`.

Examples
--------
>>> import numpy as np
>>> from arsc.harmony import Harmony
>>> rng = np.random.default_rng(0)
>>> Z = rng.normal(size=(200, 5)); codes = (np.arange(200) % 2)[:, None]
>>> h = Harmony(Z, codes, [2], theta=np.full(2, 2.0), sigma=np.full(5, 0.1), lambda_vec=np.array([0, 1.0, 1.0]), K=5)
>>> h.init_cluster()
>>> _ = h.harmonize(3)
>>> h.Z_corr.shape
(200, 5)
```

### `arsc.harmony.HarmonyResult(Z_corr: 'np.ndarray', R: 'np.ndarray', Y: 'np.ndarray', objective_harmony: 'list' = <factory>, objective_kmeans: 'list' = <factory>, kmeans_rounds: 'list' = <factory>, converged: 'bool' = False, batch_levels: 'dict' = <factory>) -> None`

```text
Result of :func:`run_harmony` with ``return_object=True``: corrected embedding plus final state and traces.

Attributes
----------
Z_corr : (n_cells, d) corrected embedding (what :func:`run_harmony` returns by default).
R : (n_cells, nclust) soft cluster assignments (cells first; R harmony stores the transpose).
Y : (nclust, d) cluster centroids (L2-normalized rows).
objective_harmony : objective after each harmony round.
objective_kmeans : objective after each clustering iteration.
kmeans_rounds : clustering iterations per harmony round.
converged : whether the harmony objective converged before ``max_iter``.
batch_levels : ``{variable: levels}`` in R factor order.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> Z, _, _ = arsc.pp.scale(L.toarray())
>>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
>>> res = arsc.run_harmony(pcs, ds.obs, "patient", return_object=True)
>>> res.Z_corr.shape, list(res.batch_levels)
((3000, 10), ['patient'])
```

### `arsc.harmony.run_harmony(Z, meta, vars_use, *, theta=None, sigma=0.1, lambda_='default', nclust=None, max_iter=10, early_stop=True, alpha=0.2, tau=0.0, block_size=0.05, max_iter_cluster=None, epsilon_cluster=0.001, epsilon_harmony=0.01, batch_prop_cutoff=1e-05, version='2.0', seed=0, init_centroids=None, kmeans_n_init=10, n_threads=None, return_object=False, verbose=False)`

```text
Run Harmony batch correction on an embedding (cells x d), matching ``harmony::RunHarmony``.

``version`` picks the R release to reproduce: ``"2.0"`` (default; R harmony 2.0.5, the current CRAN release),
``"2.1"`` (2.0 with the multi-variable fix of GitHub tag v2.1.0, not yet on CRAN) or ``"1.2"`` (harmony
1.2.x, used by Seurat-era pipelines and by :mod:`arsc.symphony`). The versions give different embeddings:

* 2.x starts from k-means++-style seeds plus 10 Lloyd steps; 1.2 from k-means with 10 restarts.
* 2.x keeps the centroids fixed while clustering and sets them from the regression intercept in the
  correction step; 1.2 recomputes them from R at every clustering iteration.
* 2.x diversity penalty ``((2E + 1) / (O + E + 1)) ** theta`` (pseudocounts) vs ``(E / (O + E)) ** theta``.
* 2.x leaves batch levels that are (almost) absent from a cluster out of that cluster's regression
  (``batch_prop_cutoff``); 1.2 regresses every level in every cluster.
* Defaults: 2.x ``lambda = NULL`` (automatic, ``alpha * E``) and 4 clustering iterations per round; 1.2
  ``lambda = 1`` and 20.
* With several batch variables, 1.2 and 2.0 add the per-variable penalties; 2.1 multiplies them, so a
  variable with ``theta = 0`` no longer changes the clustering.

Each version is validated against the matching R release with one, two and three batch variables and
distinct theta / lambda per variable (README, harmony rows): started from R's k-means centroids, per-dim r
>= 0.98 on every dimension (median >= 0.999) and the same objective path; from arsc's own start the residual
is the spread between two R seeds. Not the same algorithm port as ``harmonypy``.

Parameters
----------
Z : (n_cells, d) array, typically PCA cell embeddings (not rescaled).
meta : DataFrame with the batch columns, or a 1-D array of batch labels.
vars_use : column name or list of names (ignored if ``meta`` is an array).
theta : diversity penalty per variable (default 2 each), in the order of ``vars_use``.
sigma : soft k-means width (scalar or length ``nclust``).
lambda_ : ridge penalty, scalar or one per variable (order of ``vars_use``); ``None`` = automatic
    (``alpha * E``), as ``lambda = NULL`` in R; ``"default"`` = the version's R default (1.2: 1; 2.x: None).
nclust : number of clusters (default ``min(round(N / 30), 100)``).
max_iter : harmony rounds (R ``max_iter``).
early_stop : stop when the objective converges (default True; False runs all ``max_iter`` rounds).
alpha : automatic-lambda factor (default 0.2).
tau : protection against over-correcting small batches (default 0 = off).
block_size : fraction of cells per online-update block (default 0.05; 2.x also accepts 1).
max_iter_cluster : clustering iterations per round (default: 20 for 1.2, 4 for 2.x, as R's
    ``harmony_options``).
epsilon_cluster, epsilon_harmony : convergence tolerances (defaults 1e-3, 1e-2).
batch_prop_cutoff : 2.x only: a batch level enters cluster k's regression when its share of the cluster
    (``O[k, b] / N_b``) exceeds this (default 1e-5, R ``harmony_options(batch.prop.cutoff)``).
version : ``"2.0"`` (default), ``"2.1"`` or ``"1.2"`` (also accepted: ``"2.0.5"``, ``"2.1.0"``, ``"1.2.0"``).
seed : integer seed (default 0) for the k-means start (1.2: sklearn ``random_state``; 2.x: the seeding draws)
    and the per-iteration random block order (local ``np.random.default_rng``). Same input + seed ->
    bit-identical output, independent of ``n_threads`` (fixed reduction order).
init_centroids : optional (nclust x d) starting centroids (bypasses k-means; use R's to match R exactly).
kmeans_n_init : 1.2 only: k-means restarts (default 10).
n_threads : numba threads (default: all).
return_object : return :class:`HarmonyResult` (with traces) instead of the array.
verbose : print the objective per round.

Returns
-------
(n_cells, d) corrected embedding, or :class:`HarmonyResult`.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> Z, _, _ = arsc.pp.scale(L.toarray())
>>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
>>> H = arsc.run_harmony(pcs, ds.obs, "patient")
>>> H.shape
(3000, 10)
```

## arsc.symphony

Symphony reference mapping.

### `arsc.symphony.build_reference(X, genes, obs, vars=None, *, normalize: 'bool' = True, topn: 'int' = 2000, vargenes_groups=None, hvg=None, additional_genes=None, d: 'int' = 20, K: 'int' = 100, theta=2.0, sigma: 'float' = 0.1, lambda_: 'float' = 1.0, tau: 'float' = 0.0, max_iter: 'int' = 10, umap: 'bool' = True, umap_min_dist: 'float' = 0.1, umap_n_neighbors: 'int' = 30, seed: 'int' = 111) -> 'Reference'`

```text
Build a Symphony reference (R ``symphony::buildReference``).

Parameters
----------
X : reference cells x genes matrix: raw counts (``normalize=True``, the default, as R ``do_normalize``) or
    log-normalized values (``normalize=False``). A memory-mapped store matrix works (``st.X("RNA", "counts")``).
genes : gene names aligned to the columns of ``X``.
obs : reference metadata, row-aligned with ``X`` (index = barcodes); kept for label transfer.
vars : batch column(s) for Harmony (R ``vars``); None = no Harmony (soft k-means clusters instead).
normalize : log-normalize counts first (LogNormalize, scale factor 1e4).
topn : variable genes (per group when ``vargenes_groups`` is given; groups' top genes are united).
vargenes_groups : obs column whose groups each contribute ``topn`` genes (R ``vargenes_groups``).
hvg : optional explicit gene list instead of the vst selection (e.g. the HVGs of an existing Seurat object).
additional_genes : genes added to the selection (R ``additional_genes``).
d : number of PCs (R ``d``).
K : Harmony clusters (R ``K``).
theta, sigma, lambda_, tau, max_iter : Harmony settings (see module notes: R with harmony 1.2 runs
    ``max_iter = 10`` and ignores ``tau``).
umap : fit a reference UMAP (kept as a model so query cells can be placed on it).
umap_min_dist, umap_n_neighbors : UMAP settings (R: 0.1, 30).
seed : integer seed (R ``seed = 111``) for every stochastic step: Harmony's k-means init and block order,
    the k-means of the no-Harmony case, and the UMAP layout (SEEDS.md rule 5).

Returns
-------
Reference.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> ref = arsc.symphony.build_reference(ds.X("RNA", "counts"), ds.features("RNA"), ds.obs, vars="patient",
...                                     topn=200, d=10, K=20, umap=False)
>>> ref.loadings.shape, round(float(ref.R.sum(0).mean()), 6)
((200, 10), 1.0)
```

### `arsc.symphony.load_reference(path: 'str') -> 'Reference'`

```text
Read a reference written by :meth:`Reference.save`.

The UMAP model is a pickle: load only references you (or a trusted pipeline) wrote.

Parameters
----------
path : directory written by ``Reference.save``.

Returns
-------
Reference.

Examples
--------
>>> import arsc, tempfile
>>> ds = arsc.datasets.toy()
>>> ref = arsc.symphony.build_reference(ds.X("RNA", "counts"), ds.features("RNA"), ds.obs, vars="patient",
...                                     topn=200, d=10, K=20, umap=False)
>>> with tempfile.TemporaryDirectory() as d:
...     len(arsc.symphony.load_reference(ref.save(d)).genes)
200
```

### `arsc.symphony.map_query(X, genes, obs, ref: 'Reference', vars=None, *, normalize: 'bool' = True, sigma: 'float' = 0.1, lambda_: 'float' = 1.0, umap: 'bool' = True) -> 'QueryMap'`

```text
Map query cells onto a Symphony reference (R ``symphony::mapQuery``).

Parameters
----------
X : query cells x genes, raw counts (``normalize=True``) or log-normalized. Only the reference's variable
    genes are densified; library sizes use all genes. A memory-mapped store matrix works.
genes : gene names of ``X``'s columns.
obs : query metadata, row-aligned with ``X``.
ref : the :class:`Reference`.
vars : query batch column(s) to correct (R ``vars``). The correction removes what these columns explain, so
    never include the biology you want to compare (e.g. timepoint or response).
normalize : log-normalize counts first.
sigma : soft-clustering width (R ``sigma = 0.1``).
lambda_ : ridge penalty on the batch terms (symphony fixes 1).
umap : place query cells on the reference UMAP (needs ``ref.umap_model``).

Returns
-------
QueryMap.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> X, g = ds.X("RNA", "counts"), ds.features("RNA")
>>> isref = (ds.obs["timepoint"] == "pre").to_numpy()
>>> ref = arsc.symphony.build_reference(X[isref], g, ds.obs[isref], vars="patient", topn=200, d=10, K=20,
...                                     umap=False)
>>> q = arsc.symphony.map_query(X[~isref], g, ds.obs[~isref], ref, vars="patient", umap=False)
>>> q.R.shape
(20, 1517)
```

### `arsc.symphony.mapping_score(ref: 'Reference', query: 'QueryMap', use_orig: 'bool' = True) -> 'np.ndarray'`

```text
Per-cell mapping distance (R ``symphony::calcPerCellMappingMetric``, Mahalanobis).

For each reference cluster k, the R_k-weighted mean and covariance of the reference PCs (``Z_orig``; or the
corrected ``Z_corr`` with ``use_orig=False``); each query cell's Mahalanobis distance of its uncorrected
projection to every cluster, averaged with the cell's soft assignment. High values = the cell sits far from
the reference distribution (possibly a state the reference lacks); compare against the reference's own cells
or a held-out reference batch to set a cut-off.

Parameters
----------
ref : the :class:`Reference`.
query : the :class:`QueryMap`.
use_orig : use the reference PCs before Harmony (R default ``Z_orig = TRUE``).

Returns
-------
numpy.ndarray, one distance per query cell.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> X, g = ds.X("RNA", "counts"), ds.features("RNA")
>>> isref = (ds.obs["timepoint"] == "pre").to_numpy()
>>> ref = arsc.symphony.build_reference(X[isref], g, ds.obs[isref], vars="patient", topn=200, d=10, K=20,
...                                     umap=False)
>>> q = arsc.symphony.map_query(X[~isref], g, ds.obs[~isref], ref, vars="patient", umap=False)
>>> arsc.symphony.mapping_score(ref, q).shape
(1517,)
```

### `arsc.symphony.QueryMap(Z: 'np.ndarray', Z_pca: 'np.ndarray', R: 'np.ndarray', obs: 'pd.DataFrame', umap: 'np.ndarray | None' = None, n_shared_genes: 'int' = 0) -> None`

```text
A query mapped onto a :class:`Reference` (``mapQuery`` output), cells first.

Attributes
----------
Z : (cells x d) corrected query embedding, in the reference's Harmony space (R ``Z``).
Z_pca : (cells x d) query projected on the reference loadings, before correction (R ``Zq_pca``).
R : (K x cells) soft assignment of query cells to the reference clusters.
obs : query metadata.
umap : (cells x 2) query cells placed on the reference UMAP, or None.
n_shared_genes : reference variable genes found in the query (missing ones are set to 0 after scaling).

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> X, g = ds.X("RNA", "counts"), ds.features("RNA")
>>> isref = (ds.obs["timepoint"] == "pre").to_numpy()
>>> ref = arsc.symphony.build_reference(X[isref], g, ds.obs[isref], vars="patient", topn=200, d=10, K=20,
...                                     umap=False)
>>> q = arsc.symphony.map_query(X[~isref], g, ds.obs[~isref], ref, vars="patient", umap=False)
>>> q.Z.shape, q.n_shared_genes
((1517, 10), 200)
```

### `arsc.symphony.Reference(genes: 'pd.Index', mean: 'np.ndarray', sd: 'np.ndarray', loadings: 'np.ndarray', Z_orig: 'np.ndarray', Z_corr: 'np.ndarray', R: 'np.ndarray', centroids: 'np.ndarray', Nr: 'np.ndarray', C: 'np.ndarray', obs: 'pd.DataFrame', umap: 'np.ndarray | None' = None, umap_model: 'object | None' = None, params: 'dict' = <factory>) -> None`

```text
A compressed Symphony reference (``buildReference`` output), cells-first orientation.

Attributes
----------
genes : Index of the reference variable genes (order of ``mean``, ``sd`` and the ``loadings`` rows).
mean, sd : per-gene mean and s.d. (n - 1) of the log-normalized reference expression.
loadings : (genes x d) PCA gene loadings (R ``loadings``).
Z_orig : (cells x d) reference PCs before Harmony.
Z_corr : (cells x d) Harmony-corrected reference embedding.
R : (K x cells) soft cluster assignments.
centroids : (K x d) L2-normalized Harmony centroids (R ``centroids``, transposed).
Nr : (K,) cluster sizes ``R.sum(1)`` (R ``cache[[1]]``).
C : (K x d) ``R @ Z_corr`` (R ``cache[[2]]``).
obs : reference cell metadata (index = cell barcodes, row-aligned with ``Z_corr``).
umap : (cells x 2) reference UMAP, or None.
umap_model : fitted umap-learn model (for placing query cells), or None.
params : the settings used to build it.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> ref = arsc.symphony.build_reference(ds.X("RNA", "counts"), ds.features("RNA"), ds.obs, vars="patient",
...                                     topn=200, d=10, K=20, umap=False)
>>> ref.Z_corr.shape, ref.centroids.shape
((3000, 10), (20, 10))
```

### `arsc.symphony.reference_from_harmony(Z_orig, Z_corr, R, genes, mean, sd, loadings, obs, *, umap: 'bool' = True, umap_min_dist: 'float' = 0.1, umap_n_neighbors: 'int' = 30, seed: 'int' = 111) -> 'Reference'`

```text
Build a reference from an existing PCA + Harmony run (R ``buildReferenceFromHarmonyObj``).

Use it to wrap a reference made elsewhere, e.g. the matrices of a saved R symphony reference or a Seurat
object (``Z_orig`` = PCs, ``Z_corr`` = Harmony embedding, ``R`` = Harmony's soft clusters, ``loadings`` =
feature loadings, ``mean``/``sd`` = the variable genes' log-normalized mean and s.d.).

Parameters
----------
Z_orig, Z_corr : (cells x d) PCs and Harmony embedding.
R : (K x cells) or (cells x K) soft cluster assignments.
genes : the variable genes (rows of ``loadings``).
mean, sd : per-gene mean and s.d. used to scale the reference.
loadings : (genes x d) loadings.
obs : reference metadata, row-aligned with ``Z_corr``.
umap : fit a reference UMAP model on ``Z_corr``.
umap_min_dist, umap_n_neighbors, seed : UMAP settings.

Returns
-------
Reference.

Examples
--------
>>> import arsc, numpy as np
>>> ds = arsc.datasets.toy()
>>> r = arsc.symphony.build_reference(ds.X("RNA", "counts"), ds.features("RNA"), ds.obs, vars="patient",
...                                   topn=200, d=10, K=20, umap=False)
>>> r2 = arsc.symphony.reference_from_harmony(r.Z_orig, r.Z_corr, r.R, r.genes, r.mean, r.sd, r.loadings,
...                                          r.obs, umap=False)
>>> bool(np.allclose(r2.C, r.C))
True
```

### `arsc.symphony.transfer_labels(ref: 'Reference', query: 'QueryMap', labels, k: 'int' = 5, seed: 'int' = 0) -> 'pd.DataFrame'`

```text
k-NN label transfer from the reference to the mapped query (R ``symphony::knnPredict``).

Parameters
----------
ref : the :class:`Reference`.
query : the :class:`QueryMap`.
labels : a column of ``ref.obs`` or an array of reference labels (row-aligned with ``ref.Z_corr``).
k : neighbors (R ``k = 5``).
seed : integer seed for breaking vote ties.

Returns
-------
DataFrame indexed by query cell: ``label`` (majority among the k nearest reference cells, Euclidean on the
corrected embeddings) and ``prob`` (share of the k votes for it; R ``_prob``). Low ``prob`` marks cells whose
neighborhood is mixed, e.g. a state absent from the reference.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> X, g = ds.X("RNA", "counts"), ds.features("RNA")
>>> isref = (ds.obs["timepoint"] == "pre").to_numpy()
>>> ref = arsc.symphony.build_reference(X[isref], g, ds.obs[isref], vars="patient", topn=200, d=10, K=20,
...                                     umap=False)
>>> q = arsc.symphony.map_query(X[~isref], g, ds.obs[~isref], ref, vars="patient", umap=False)
>>> lab = arsc.symphony.transfer_labels(ref, q, "cell_type", k=5)
>>> float((lab["label"] == ds.obs.loc[~isref, "cell_type"]).mean()) > 0.8   # the toy types overlap
True
```

## arsc.de

Differential expression and GSEA.

### `arsc.de.deseq2(counts: 'pd.DataFrame', coldata: 'pd.DataFrame', design: 'str', contrast, min_count: 'int' = 10, min_samples: 'int' = 3, alpha: 'float' = 0.1, quiet: 'bool' = True, n_cpus: 'int' = 1) -> 'pd.DataFrame'`

```text
DESeq2 Wald-test DE on pseudobulk counts (pydeseq2 with DESeq2's parametric dispersion trend).

Parity with R DESeq2: log2FC r 0.9998, Wald stat r 0.9993-0.9998, same padj < 0.1 calls (STATS_PARITY.md).
No LFC shrinkage.

Parameters
----------
counts : genes x samples integer table (``Pseudobulk.to_frame()``).
coldata : sample table indexed by sample name (the columns of ``counts``), e.g. ``Pseudobulk.groups``.
design : formula, e.g. ``"~ patient + timepoint"`` (paired: patient as a fixed block).
contrast : ``(variable, level, reference)``, e.g. ``("timepoint", "post", "pre")``.
min_count, min_samples : genes with fewer than ``min_samples`` samples at count >= ``min_count`` are
    removed first.
alpha : independent-filtering / padj target (DESeq2 ``results()`` default 0.1).
quiet : silence pydeseq2.
n_cpus : pydeseq2 worker processes.

Returns
-------
DataFrame with R's columns ``baseMean, log2FoldChange, lfcSE, stat, pvalue, padj, gene``, sorted by
pvalue; ``.attrs`` holds ``n_samples`` and ``n_genes_tested``. Needs ``pip install pydeseq2``.

Examples
--------
>>> import arsc, pandas as pd
>>> ds = arsc.datasets.toy()
>>> m = ((ds.obs.cell_type == "CD8 T") & (ds.obs.response == "R")).to_numpy()
>>> pb = arsc.pseudobulk(ds.X("RNA", "counts")[m], ds.obs[m], ["patient", "timepoint"],
...                      features=ds.features("RNA"), min_cells=10)
>>> cd = pb.groups.copy()
>>> cd["timepoint"] = pd.Categorical(cd["timepoint"], ["pre", "post"])   # reference level first
>>> res = arsc.de.deseq2(pb.to_frame(), cd, "~ patient + timepoint",
...                      ("timepoint", "post", "pre"))                   # doctest: +SKIP
>>> res.head(5).gene.tolist()                                            # doctest: +SKIP
```

### `arsc.de.duplicate_correlation(y, weights, design, block, trim: 'float' = 0.15) -> 'float'`

```text
limma::duplicateCorrelation(..., block) consensus correlation via per-gene REML (statmod mixedModel2Fit model:
whitened var = s_e^2 I + s_a^2 ZZ').  The REML objective is profiled over rho = s_a^2/(s_e^2+s_a^2) and maximized
over (1/(1-maxblock), 0.999); per-gene values are then clipped to [1/(1-maxblock)+0.01, 0.99] as in limma.

Used inside :func:`voom` (``block=``); call it directly only to inspect the correlation.

Parameters
----------
y : genes x samples log-expression (voom ``E``).
weights : genes x samples precision weights (voom weights).
design : samples x p design matrix.
block : per-sample block labels (e.g. patient).
trim : trimmed-mean fraction for the consensus.

Returns
-------
float consensus within-block correlation (0.0 when no block has more than one sample).

Examples
--------
>>> import arsc, numpy as np
>>> rng = np.random.default_rng(0)
>>> pat = rng.normal(0, 1, size=(100, 4))
>>> y = np.repeat(pat, 2, axis=1) + rng.normal(0, 1, size=(100, 8))
>>> design = np.column_stack([np.ones(8), np.tile([0, 1], 4)])
>>> rho = arsc.de.duplicate_correlation(y, np.ones_like(y), design, np.repeat(["a", "b", "c", "d"], 2))
>>> 0.2 < rho < 0.8
True
```

### `arsc.de.fgsea(stats: 'pd.Series', genesets: 'dict', min_size: 'int' = 10, max_size: 'int' = 500, nperm: 'int' = 10000, gsea_param: 'float' = 1.0, seed: 'int' = 1) -> 'pd.DataFrame'`

```text
Preranked GSEA following fgsea.  ES, leading edge and the NES definition (ES / mean of the
same-sign null) follow fgsea exactly; p-values come from ``nperm`` sampled gene sets per set size
(fgseaSimple), so the smallest attainable p is 1/(nperm+1) -- fgsea >= 1.14 default (multilevel) resolves
far smaller p.  Columns: pathway, pval, padj, ES, NES, nMoreExtreme, size, leadingEdge (comma-joined).
``seed`` (default 1): the null for set size k uses permutation streams seeded from ``(seed, k, i)``
(splitmix64, one independent stream per permutation), so results are bit-identical regardless of the
number of threads.

Parameters
----------
stats : gene-level ranking statistic, a Series indexed by gene (e.g. voom ``t`` or DESeq2 ``stat``).
genesets : ``{name: [genes]}`` (e.g. :func:`msigdb`, :func:`arsc.genesets.read_gmt`).
min_size, max_size : gene-set size limits after intersecting with ``stats``.
nperm : permutations per set size.
gsea_param : weight exponent (fgsea ``gseaParam``).
seed : integer seed.

Returns
-------
DataFrame ``pathway, pval, padj, ES, NES, nMoreExtreme, size, leadingEdge`` sorted by pval
(empty if no set passes the size limits).

Examples
--------
>>> import arsc, pandas as pd
>>> ds = arsc.datasets.toy()
>>> m = ((ds.obs.cell_type == "CD8 T") & (ds.obs.response == "R")).to_numpy()
>>> pb = arsc.pseudobulk(ds.X("RNA", "counts")[m], ds.obs[m], ["patient", "timepoint"],
...                      features=ds.features("RNA"), min_cells=10)
>>> cd = pb.groups.copy()
>>> cd["timepoint"] = pd.Categorical(cd["timepoint"], ["pre", "post"])   # reference level first
>>> res = arsc.de.voom(pb.to_frame(), cd, "~ timepoint", coef=("timepoint", "post"), block="patient")
>>> sets = {"effect": arsc.datasets.TOY_EFFECT_GENES, "B": ["MS4A1", "CD79A", "CD19"]}
>>> fg = arsc.de.fgsea(res.set_index("gene")["t"], sets, min_size=3)
>>> fg.pathway.tolist()[0], bool(fg.NES.iloc[0] > 0)
('effect', True)
```

### `arsc.de.filter_by_expr(counts: 'np.ndarray', design: 'np.ndarray', min_count: 'float' = 10, min_total_count: 'float' = 15, large_n: 'int' = 10, min_prop: 'float' = 0.7) -> 'np.ndarray'`

```text
edgeR ``filterByExpr(DGEList(counts), design = ...)`` (group = NULL branch).

Parameters
----------
counts : genes x samples count array.
design : samples x p design matrix.
min_count : minimum count (scaled to CPM at the median library size).
min_total_count : minimum total count per gene.
large_n : sample-size threshold for the ``min_prop`` rule.
min_prop : proportion rule for large groups.

Returns
-------
numpy.ndarray of bool, one per gene (True = keep).

Examples
--------
>>> import arsc, numpy as np
>>> counts = np.array([[0, 0, 1, 0], [50, 60, 40, 55], [100, 90, 120, 80]])
>>> design = np.column_stack([np.ones(4), [0, 0, 1, 1]])
>>> arsc.de.filter_by_expr(counts, design).tolist()
[False, True, True]
```

### `arsc.de.msigdb(collection: 'str' = 'H', subcollection: 'str | None' = None, species: 'str' = 'Homo sapiens', as_frame: 'bool' = False, refresh: 'bool' = False)`

```text
MSigDB gene sets as ``{gs_name: [symbols]}``, exported once from R msigdbr and cached.

The first call runs msigdbr (Rscript, ``arsc/R/msigdb_export.R``) and writes
``<cache>/genesets/<species>__<coll>__<sub>.csv.gz``, where ``<cache>`` is ``$ARSC_CACHE``, else
``$AR_ROOT/.cache``, else ``~/.cache/arsc``; later calls read the cache (no R needed).

Parameters
----------
collection : MSigDB collection, e.g. ``"H"`` (hallmark), ``"C2"``, ``"C7"``.
subcollection : e.g. ``"CP:REACTOME"``.
species : msigdbr species name.
as_frame : return the long table instead of a dict.
refresh : re-export even if cached.

Returns
-------
dict ``{gs_name: [gene symbols]}`` (or a DataFrame with ``as_frame=True``).

Examples
--------
>>> import arsc
>>> hallmark = arsc.de.msigdb("H")                       # doctest: +SKIP
>>> len(hallmark)                                        # doctest: +SKIP
50
```

### `arsc.de.tmm_factors(counts: 'np.ndarray', logratio_trim: 'float' = 0.3, sum_trim: 'float' = 0.05, acutoff: 'float' = -10000000000.0)`

```text
edgeR ``calcNormFactors(method = "TMM")``: weighted trimmed mean of M-values, upper-quartile reference.

Parameters
----------
counts : genes x samples count array.
logratio_trim : trim fraction for M values.
sum_trim : trim fraction for A values.
acutoff : cutoff on A values.

Returns
-------
numpy.ndarray of normalization factors, one per sample (geometric mean 1).

Examples
--------
>>> import arsc, numpy as np
>>> rng = np.random.default_rng(0)
>>> f = arsc.de.tmm_factors(rng.poisson(20, size=(200, 4)))
>>> f.shape
(4,)
```

### `arsc.de.voom(counts: 'pd.DataFrame', coldata: 'pd.DataFrame', formula: 'str', coef, block: 'str | None' = None, min_count: 'float' = 10, min_total_count: 'float' = 15, filter: 'bool' = True) -> 'pd.DataFrame'`

```text
limma-voom DE on pseudobulk counts, with optional duplicateCorrelation for repeated patients.

Pipeline (as in R): filterByExpr -> TMM -> voom -> duplicateCorrelation -> voom(block, cor)
-> lmFit(block, cor) -> eBayes -> topTable. Parity with R limma: logFC / t r = 1.00000, max |diff| < 1e-3
(STATS_PARITY.md).

Parameters
----------
counts : genes x samples counts (``Pseudobulk.to_frame()``).
coldata : sample table indexed by sample name (the columns of ``counts``), e.g. ``Pseudobulk.groups``.
    The reference level of a factor is its first level: string columns sort alphabetically
    (``"post"`` < ``"pre"``), so set ``coldata[var] = pd.Categorical(coldata[var], [reference, level])``.
formula : patsy / R model.matrix formula, e.g. ``"~ timepoint"`` or ``"~ patient + timepoint"`` (paired).
coef : design column name, or ``(variable, level)`` -> ``"variable[T.level]"``.
block : column with the repeated-measure unit (patient), for duplicateCorrelation. As in R, the block
    factor must NOT also be in the design (else the correlation is 0).
min_count, min_total_count : filterByExpr thresholds.
filter : apply filterByExpr.

Returns
-------
DataFrame ``logFC, AveExpr, t, P.Value, adj.P.Val, gene`` sorted by P.Value; ``.attrs`` holds
``consensus_cor``, ``df_prior``, ``s2_prior``, ``df_residual``, ``design_columns``, ``n_genes``.

Examples
--------
>>> import arsc, pandas as pd
>>> ds = arsc.datasets.toy()
>>> m = ((ds.obs.cell_type == "CD8 T") & (ds.obs.response == "R")).to_numpy()
>>> pb = arsc.pseudobulk(ds.X("RNA", "counts")[m], ds.obs[m], ["patient", "timepoint"],
...                      features=ds.features("RNA"), min_cells=10)
>>> cd = pb.groups.copy()
>>> cd["timepoint"] = pd.Categorical(cd["timepoint"], ["pre", "post"])   # reference level first
>>> res = arsc.de.voom(pb.to_frame(), cd, "~ timepoint", coef=("timepoint", "post"), block="patient")
>>> sorted(res.gene[:5])                    # the planted responder genes
['CXCL13', 'GZMB', 'GZMK', 'IFNG', 'PRF1']
```

## arsc.stats

Patient-level statistics.

### `arsc.stats.arm_contrast(a, b, B: 'int' = 20000, seed: 'int' = 1) -> 'dict'`

```text
Between-arm comparison of per-patient deltas (e.g. post - pre in responders vs non-responders).

Exact permutation on the median difference, Hodges-Lehmann shift and leave-one-patient-out range /
sign reversals.

Parameters
----------
a, b : per-patient values (deltas) of the two arms.
B : passed to :func:`perm_test`.
seed : passed to :func:`perm_test` (used only when the permutation is sampled).

Returns
-------
dict with ``n_a, n_b, median_a, median_b, diff_medians, hodges_lehmann, p_perm, loo_min, loo_max,
loo_sign_reversals``.

Examples
--------
>>> import arsc
>>> r = arsc.stats.arm_contrast([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
>>> r["hodges_lehmann"], r["loo_sign_reversals"]
(-3.0, 0)
```

### `arsc.stats.bh(p) -> 'np.ndarray'`

```text
Benjamini-Hochberg adjusted p-values; NaN preserved (== R ``p.adjust(method = "BH")``).

Parameters
----------
p : array-like of p-values.

Returns
-------
numpy.ndarray of adjusted p-values (same shape; NaN where ``p`` is NaN).

Examples
--------
>>> import arsc
>>> arsc.stats.bh([0.01, 0.04, 0.03, 0.5]).round(4).tolist()
[0.04, 0.0533, 0.0533, 0.5]
```

### `arsc.stats.boot_ci(x, f=median, B: 'int' = 4000, seed: 'int' = 1)`

```text
Percentile bootstrap 95% CI of a statistic of per-patient values (NaN when < 3 finite values).

Parameters
----------
x : per-patient values.
f : statistic (default ``np.median``).
B : bootstrap resamples.
seed : seeds a local ``np.random.default_rng`` for the ``B`` resamples (default 1).

Returns
-------
tuple ``(lo, hi)``.

Examples
--------
>>> import arsc
>>> lo, hi = arsc.stats.boot_ci([1.0, 2.0, 3.0, 4.0, 5.0])
>>> float(lo), float(hi)
(1.0, 5.0)
```

### `arsc.stats.boot_ci_diff(x, y, f=median, paired: 'bool' = False, conf_level: 'float' = 0.95, B: 'int' = 4000, seed: 'int' = 1) -> 'tuple'`

```text
Percentile bootstrap CI of ``f(x) - f(y)`` (default: the difference of medians) with patients resampled.

Unpaired: ``x`` and ``y`` are resampled separately, each within its own group (group sizes kept). Paired
(``x[i]`` and ``y[i]`` from the same patient): patients are resampled as pairs and the statistic is
``f(x*) - f(y*)``; for the median OF the paired differences use :func:`boot_ci` on ``x - y`` (with
``f=np.mean`` the two agree exactly). At fewer than ~10 patients per group a percentile bootstrap CI of a
median is too narrow (it can only take observed values); prefer the exact CI of :func:`hodges_lehmann` for a
location shift. Non-finite values are dropped (pairwise when paired); fewer than 3 values in a group gives NaN.

Parameters
----------
x, y : per-patient values of the two groups (or the two conditions when paired).
f : statistic of one group; ``np.median`` / ``np.mean`` are vectorized, any other callable is applied per
    resample.
paired : resample pairs (equal lengths required).
conf_level : central coverage of the percentile interval.
B : bootstrap resamples.
seed : seeds a local ``np.random.default_rng``.

Returns
-------
tuple ``(lo, hi)``.

Examples
--------
>>> import numpy as np, arsc
>>> r = np.array([0.8, 1.1, 0.4, 1.6, 0.9, 1.3])
>>> nr = np.array([0.2, 0.5, -0.1, 0.3, 0.6])
>>> lo, hi = arsc.stats.boot_ci_diff(r, nr)
>>> bool(lo < float(np.median(r) - np.median(nr)) < hi)
True
```

### `arsc.stats.composition(obs: 'pd.DataFrame', celltype: 'str', sample_cols=('patient_alias', 'timepoint'), min_parent: 'int' = 50, extra_cols=None) -> 'pd.DataFrame'`

```text
Per-sample counts and proportions of every level of a cell-type column.

Zeros are filled (a cell type absent from a sample gets ``k = 0``); samples whose parent total ``n``
is below ``min_parent`` are dropped (a message says how many).

Parameters
----------
obs : cell metadata (one row per cell).
celltype : column with the cell-type labels.
sample_cols : columns defining a sample (default ``("patient_alias", "timepoint")``; pass yours).
min_parent : minimum cells per sample.
extra_cols : sample-level columns carried along (e.g. ``["response"]``).

Returns
-------
DataFrame with the key columns, ``celltype``, ``k`` (cells of that type), ``n`` (cells in the
sample) and ``prop`` = k / n; one row per sample x cell type.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> comp = arsc.stats.composition(ds.obs, "cell_type", ["patient", "timepoint"], extra_cols=["response"])
>>> cd8 = comp[comp.celltype == "CD8 T"]
>>> list(comp.columns)
['patient', 'timepoint', 'response', 'celltype', 'k', 'n', 'prop']
>>> len(cd8)                                     # 6 patients x 2 timepoints
12
```

### `arsc.stats.count_logit_delta(k0, n0, k1, n1)`

```text
Within-patient logit change from counts with a (k+0.5)/(n+1) correction; flags sparse (< 10) numerators.

Parameters
----------
k0, n0 : positive and total counts at the first timepoint (per patient).
k1, n1 : the same at the second timepoint.

Returns
-------
tuple ``(delta, sparse)``: logit(k1/n1) - logit(k0/n0) and a boolean flag ``min(k0, k1) < 10``.

Examples
--------
>>> import arsc
>>> d, sparse = arsc.stats.count_logit_delta([5], [100], [20], [100])
>>> round(float(d[0]), 3), bool(sparse[0])
(1.487, True)
```

### `arsc.stats.downsample_counts(X, target, method: 'str' = 'hypergeometric', seed: 'int' = 1)`

```text
Thin a cells x genes count matrix to a common depth per cell (removes depth differences before a comparison).

``method="hypergeometric"`` draws exactly ``target`` UMIs without replacement from each cell (scanpy
``pp.downsample_counts``, Seurat ``SampleUMI``, vegan ``rrarefy``): every thinned cell has total ``target``.
``method="binomial"`` keeps each UMI with probability ``target / total`` (expected total ``target``; the
variance is larger and totals vary). Cells whose total is at or below ``target`` are returned unchanged:
drop them (``totals < target``) when an exact common depth matters. Normalize thinned cells by their own
new total (or by ``target`` for the hypergeometric method, where they are equal).

Parameters
----------
X : cells x genes integer counts (CSR or dense).
target : common depth (int) or one target per cell.
method : ``"hypergeometric"`` (exact depth) or ``"binomial"``.
seed : seeds a local ``np.random.default_rng``.

Returns
-------
scipy.sparse.csr_matrix of the same shape and dtype float64.

Examples
--------
>>> import numpy as np, arsc
>>> ds = arsc.datasets.toy()
>>> X = ds.X()
>>> Y = arsc.stats.downsample_counts(X, 200)
>>> tot = np.asarray(X.sum(1)).ravel(); new = np.asarray(Y.sum(1)).ravel()
>>> bool(np.all(new[tot >= 200] == 200)), bool(np.all(new[tot < 200] == tot[tot < 200]))
(True, True)
```

### `arsc.stats.feasibility(sample_ids, parent, positive=None, parent_floor: 'int' = 50, pos_floor: 'int' = 10)`

```text
Per-sample cell counts for a planned readout: run BEFORE committing to an analysis plan.

Counts parent cells (e.g. CD8 T) and positive cells (e.g. GZMK+) per sample, and says whether
enough samples reach the floors to support a patient-level test. Uses no outcome data.

Parameters
----------
sample_ids : per-cell sample labels.
parent : per-cell boolean, cell belongs to the parent population.
positive : optional per-cell boolean, cell is positive for the readout.
parent_floor : minimum parent cells per usable sample.
pos_floor : minimum median positive cells per sample.

Returns
-------
tuple ``(table, summary)``: per-sample counts DataFrame and a dict with ``n_samples``,
``median_parent``, ``usable`` (and ``verdict`` when ``positive`` is given).

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> tab, summ = arsc.stats.feasibility(ds.obs["sample"], ds.obs["cell_type"] == "CD8 T")
>>> summ["n_samples"], summ["usable"]
(12, True)
```

### `arsc.stats.freedman_lane(y, x, covariates=None, B: 'int' = 10000, max_exact: 'int' = 8, alternative: 'str' = 'two-sided', seed: 'int' = 1) -> 'dict'`

```text
Permutation test of one regression coefficient adjusted for covariates (Freedman & Lane 1983).

Tests ``x`` in ``y ~ 1 + covariates + x``: fit the reduced model ``y ~ 1 + covariates``, permute its residuals,
add them back to its fitted values, refit the full model and compare the t statistic of ``x``. Permuting
``y`` itself (or ``x``) ignores the covariates and is not a valid adjusted test when ``x`` and the
covariates are correlated. All ``n!`` permutations for ``n <= max_exact`` (8! = 40,320), else ``B`` random ones
with p = (1 + #) / (B + 1). Rows with any missing value are dropped.

Parameters
----------
y : outcome per unit (e.g. per patient).
x : the predictor tested (one column).
covariates : optional array / DataFrame of nuisance covariates (an intercept is always added).
B : random permutations beyond ``max_exact``.
max_exact : enumerate every permutation up to this many units.
alternative : ``"two-sided"`` (``|t|``), ``"greater"`` or ``"less"``.
seed : seeds a local ``np.random.default_rng`` (unused when exact).

Returns
-------
dict with ``beta`` and ``t`` of ``x`` (OLS), ``p_param`` (the OLS t-test), ``p_perm``, ``n``, ``exact``,
``n_perm``.

Examples
--------
>>> import numpy as np, arsc
>>> rng = np.random.default_rng(0)
>>> z = rng.normal(size=30); x = z + rng.normal(scale=0.5, size=30); y = 2 * z + rng.normal(size=30)
>>> r = arsc.stats.freedman_lane(y, x, covariates=z, B=2000)
>>> r["p_perm"] > 0.05, bool(r["exact"])
(True, False)
```

### `arsc.stats.gee_prop(comp: 'pd.DataFrame', fixed: 'str' = 'timepoint', group: 'str' = 'patient_alias', k: 'str' = 'k', n: 'str' = 'n', cov_struct: 'str' = 'exchangeable') -> 'pd.DataFrame'`

```text
Marginal (population-averaged) alternative: binomial GEE with exchangeable working correlation and robust
SEs.  Coefficients are marginal log-ORs, NOT the conditional glmer ones (they attenuate toward 0 with
between-patient variance); use for few-cluster sanity checks only (sandwich SEs are anti-conservative at
< ~10 clusters).

Parameters
----------
comp : per-sample count table with ``k`` and ``n`` columns.
fixed : fixed-effect formula right-hand side (patsy).
group : cluster (patient) column.
k, n : count columns.
cov_struct : statsmodels working correlation (``"exchangeable"``, ``"independence"``).

Returns
-------
DataFrame of coefficients (``Estimate, Std. Error, z value, Pr(>|z|), term, OR``).

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> comp = arsc.stats.composition(ds.obs, "cell_type", ["patient", "timepoint"], extra_cols=["response"])
>>> cd8 = comp[comp.celltype == "CD8 T"]
>>> arsc.stats.gee_prop(cd8, fixed="timepoint", group="patient").term.tolist()
['Intercept', 'timepoint[T.post]']
```

### `arsc.stats.glmm_prop(comp: 'pd.DataFrame', fixed: 'str' = 'timepoint', groups=('patient_alias', 'sample_id'), k: 'str' = 'k', n: 'str' = 'n', vcov_method: 'str' = 'hessian') -> 'pd.DataFrame'`

```text
Binomial GLMM  ``cbind(k, n-k) ~ <fixed> + (1|g1) + (1|g2) ...``  (== lme4::glmer, nAGQ = 1).

``groups`` are independent random-intercept factors; a missing ``sample_id`` is created as observation-level
(one level per row), which is the overdispersion term of the R default.  Estimated by Laplace-approximated ML
over (beta, sigma), so estimates / variance components match glmer to optimizer tolerance; Std. Error uses
lme4's default Vcov (numerical Hessian of the deviance in (sigma, beta); ``vcov_method="conditional"`` gives
the RX^-1 version).

Parameters
----------
comp : per-sample count table with ``k`` and ``n`` columns, e.g. one cell type of :func:`composition`.
fixed : fixed-effect formula right-hand side (patsy), e.g. ``"timepoint"`` or ``"timepoint * response"``.
groups : random-intercept columns (default ``("patient_alias", "sample_id")``; pass your patient column).
k, n : count columns (successes, totals).
vcov_method : ``"hessian"`` (lme4 default) or ``"conditional"``.

Returns
-------
DataFrame of coefficients (``Estimate, Std. Error, z value, Pr(>|z|), term, OR``); ``.attrs`` carries
``sigma`` (dict of random-effect SDs), ``logLik`` and ``converged``.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> comp = arsc.stats.composition(ds.obs, "cell_type", ["patient", "timepoint"], extra_cols=["response"])
>>> cd8 = comp[comp.celltype == "CD8 T"]
>>> fit = arsc.stats.glmm_prop(cd8, fixed="timepoint", groups=("patient", "sample_id"))
>>> fit.term.tolist()
['Intercept', 'timepoint[T.post]']
```

### `arsc.stats.hodges_lehmann(x, y=None, paired: 'bool' = False, conf_level: 'float' = 0.95, alternative: 'str' = 'two-sided', mu: 'float' = 0.0, exact=None, correct: 'bool' = True, tol_root: 'float' = 0.0001) -> 'dict'`

```text
Hodges-Lehmann location estimate with its Wilcoxon confidence interval and p, replicating R ``wilcox.test(conf.int = TRUE)``.

One sample (or ``paired=True``): the pseudomedian (median of the Walsh averages) of ``x`` (or ``x - y``).
Two samples: the median of all ``x_i - y_j`` differences. Exact (n < 50 and no ties or zeros, as R): the CI
is read off the sorted Walsh averages / differences at the ``qsignrank`` / ``qwilcox`` quantiles; when the
requested level cannot be reached (tiny n) the achieved level is reported, as R does. Otherwise R's normal
approximation: the CI inverts the continuity-corrected z by Brent root finding (R's uniroot, ``tol_root``),
and the estimate is the root without correction. Use this instead of bootstrap CIs of the HL estimate,
which are degenerate with 4-8 patients (they collapse onto the few distinct resampled medians).

Parameters
----------
x : values (one per patient) of group 1, or the paired differences.
y : values of group 2 (two-sample), or the second member of each pair with ``paired=True``.
paired : treat ``x`` and ``y`` as pairs (``x - y`` on complete pairs).
conf_level : requested confidence level.
alternative : ``"two-sided"``, ``"greater"`` or ``"less"`` (one-sided intervals are half-infinite).
mu : null location (shift).
exact : exact distribution (default: n < 50, both groups for two samples; ignored with ties / zeros).
correct : continuity correction in the normal approximation.
tol_root : root-finding tolerance (R's ``tol.root``).

Returns
-------
dict with ``estimate``, ``ci_lo``, ``ci_hi``, ``conf_level`` (achieved), ``statistic`` (R's V or W),
``p``, ``method`` (``"exact"`` or ``"normal"``), ``n_x``, ``n_y``.

Examples
--------
>>> import arsc
>>> r = arsc.stats.hodges_lehmann([1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30],
...                               [0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29], paired=True)
>>> round(r["estimate"], 3), round(r["ci_lo"], 3), round(r["ci_hi"], 3), round(r["p"], 5)
(0.46, 0.01, 0.786, 0.03906)
```

### `arsc.stats.icc_oneway(y, groups, strata=None, conf_level: 'float' = 0.95, clamp: 'bool' = False) -> 'dict'`

```text
One-way random-effects intraclass correlation ICC(1) by ANOVA (unbalanced groups allowed), with its F test and CI.

``ICC = (MSB - MSW) / (MSB + (n0 - 1) MSW)`` with Searle's ``n0 = (N - sum n_i^2 / N) / (k - 1)``
(e.g. y = a per-cell score, groups = clone: how much of the variance is between clones). With ``strata``
(e.g. patient) the groups are nested in strata and the strata are removed as fixed effects: sums of squares
and degrees of freedom are pooled over strata, and ``n0 = sum_s (N_s - sum n_si^2 / N_s) / sum_s (k_s - 1)``,
the coefficient of the between-group variance in E[MSB]. (A plain mean of per-stratum n0 values, used by
hand-written versions, is a different and biased weighting when strata differ in size.) The CI is the F-based
Shrout-Fleiss interval, exact for balanced groups and approximate otherwise. Groups of size 1 add nothing
to MSW but count in MSB, as in the ANOVA.

Parameters
----------
y : per-observation values.
groups : group label per observation (e.g. clone id).
strata : optional stratum label per observation (e.g. patient) within which groups are nested.
conf_level : CI level.
clamp : truncate a negative estimate (and CI bounds) at 0.

Returns
-------
dict with ``icc``, ``ci_lo``, ``ci_hi``, ``F``, ``df_b``, ``df_w``, ``p`` (F test of no between-group
variance), ``msb``, ``msw``, ``n0``, ``k`` (groups), ``n`` (observations).

Examples
--------
>>> import arsc
>>> r = arsc.stats.icc_oneway([1.0, 1.2, 0.9, 3.0, 3.3, 2.8, 2.0, 2.1, 1.7], list("aaabbbccc"))
>>> round(r["icc"], 3), round(r["p"], 5)
(0.958, 7e-05)
```

### `arsc.stats.lmm_score(df: 'pd.DataFrame', formula: 'str', group: 'str' = 'patient_alias', reml: 'bool' = True) -> 'pd.DataFrame'`

```text
Linear mixed model ``score ~ time + (1|patient)`` (== lme4::lmer; statsmodels MixedLM REML).  Coefficients,
SEs and variance components agree with lmer; statsmodels reports Wald z (lmerTest's Satterthwaite df/p are not
reproduced).

Parameters
----------
df : per-sample (or per-cell) table.
formula : statsmodels formula, e.g. ``"score ~ timepoint"``.
group : random-intercept column (default ``"patient_alias"``; pass yours).
reml : REML (True, lmer default) or ML.

Returns
-------
DataFrame of fixed effects (``Estimate, Std. Error, z value, Pr(>|z|), term``); ``.attrs`` carries
``sigma_group`` and ``sigma_resid``.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> obs = ds.obs.assign(nUMI=ds.X("RNA", "counts").sum(axis=1).A1)
>>> sm = arsc.stats.sample_means(obs, ["nUMI"], ["patient", "timepoint"])
>>> arsc.stats.lmm_score(sm, "nUMI ~ timepoint", group="patient").term.tolist()
['Intercept', 'timepoint[T.post]']
```

### `arsc.stats.loo_effect(df: 'pd.DataFrame', patient: 'str', fn) -> 'pd.Series'`

```text
Leave-one-patient-out robustness: ``fn(df without patient p)`` for each patient p.

Parameters
----------
df : per-sample table.
patient : patient column.
fn : ``fn(df_subset) -> float``, the effect estimate.

Returns
-------
pandas.Series indexed by the left-out patient, named ``effect_without``.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> comp = arsc.stats.composition(ds.obs, "cell_type", ["patient", "timepoint"], extra_cols=["response"])
>>> cd8 = comp[comp.celltype == "CD8 T"]
>>> loo = arsc.stats.loo_effect(cd8, "patient", lambda d: d.prop.mean())
>>> loo.index.tolist()
['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
```

### `arsc.stats.mantel_haenszel(a, b=None, c=None, d=None, conf_level: 'float' = 0.95, correct: 'bool' = True) -> 'dict'`

```text
Mantel-Haenszel common odds ratio over strata (e.g. patients) with its CI and the CMH test, as R ``mantelhaen.test``.

Stratum k is the 2 x 2 table ``[[a_k, b_k], [c_k, d_k]]`` (rows = exposure yes/no, columns = outcome yes/no).
Odds ratio = sum(a d / n) / sum(b c / n); CI from the Robins-Breslow-Greenland variance; the chi-square
statistic with continuity correction (R's default ``correct = TRUE``, applied when |sum(a - E a)| >= 0.5).
No pseudo-counts are added (adding 0.5 to every cell, as hand-written versions did, pulls the odds ratio
towards 1). Strata with fewer than 2 observations carry no information and are dropped (R stops on them).

Parameters
----------
a : cell counts a_k, or a K x 2 x 2 array of the tables (then ``b``, ``c``, ``d`` are omitted).
b, c, d : the other cell counts per stratum.
conf_level : CI level.
correct : continuity correction of the CMH statistic.

Returns
-------
dict with ``or_mh``, ``ci_lo``, ``ci_hi``, ``log_or``, ``se_log_or``, ``statistic`` (CMH chi-square, 1 df),
``p``, ``n_strata`` (used), ``n_dropped``.

Examples
--------
>>> import arsc
>>> r = arsc.stats.mantel_haenszel([10, 6, 8], [5, 7, 4], [4, 3, 5], [12, 9, 10])
>>> round(r["or_mh"], 3), round(r["p"], 4)
(4.041, 0.006)
```

### `arsc.stats.mde(n, n2=None, sd: 'float' = 1.0, alpha: 'float' = 0.05, power: 'float' = 0.8) -> 'dict'`

```text
Minimum detectable effect (in units of ``sd``) for a paired (n) or two-group (n, n2) design.

Checks the exact-test floor first (min p = 2/2^n paired, 2/C(n+n2, n) two-group): if that floor exceeds
``alpha`` no effect is detectable. Otherwise a t-based MDE with 1.15 rank-test inflation.

Parameters
----------
n : patients (paired design) or patients in group 1.
n2 : patients in group 2 (two-group design); ``None`` = paired.
sd : standard deviation of the per-patient value (the MDE is also returned in these units).
alpha : two-sided significance level.
power : target power.

Returns
-------
dict with ``design``, ``n`` (``n2``), ``min_p``, ``mde_sd``, ``mde`` and ``feasible``.

Examples
--------
>>> import arsc
>>> arsc.stats.mde(3, 3)["feasible"]          # 3 vs 3 patients: min p = 0.1 > 0.05
False
>>> round(float(arsc.stats.mde(6)["mde_sd"]), 2)  # 6 paired patients
1.64
```

### `arsc.stats.paired_test(df: 'pd.DataFrame', value: 'str' = 'value', t0=None, t1=None, patient: 'str' = 'patient_alias', time: 'str' = 'timepoint', by=None, transform: 'str' = 'none', eps: 'float' = 0.001, B: 'int' = 4000, seed: 'int' = 1) -> 'pd.DataFrame'`

```text
Within-patient change from ``t0`` to ``t1`` on a per-sample table (the patient is the unit).

Per group: number of patients with both timepoints, sign counts, median delta, bootstrap CI, exact
Wilcoxon signed-rank p, leave-one-patient-out range of the median, minimum attainable p, BH fdr.

Parameters
----------
df : per-sample table (one row per patient x timepoint [x group]), e.g. from :func:`composition`.
value : column with the per-sample value (e.g. ``"prop"``).
t0, t1 : the two timepoint levels (values of ``time``), e.g. ``"pre"``, ``"post"``.
patient : patient column (default ``"patient_alias"``; pass yours).
time : timepoint column.
by : optional grouping column(s); one result row per group (e.g. ``"celltype"``).
transform : ``"none"``, ``"logit"`` (proportions) or ``"log2ratio"``, applied before differencing.
eps : clipping / pseudo-count for the transforms.
B : bootstrap resamples.
seed : seeds the bootstrap CI; every group uses the same seed (default 1).

Returns
-------
DataFrame, one row per group: ``group, n_patients, k_up, k_down, median_delta, ci_lo, ci_hi,
loo_min, loo_max, p_wilcox_exact, min_possible_p, transform, fdr``.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> comp = arsc.stats.composition(ds.obs, "cell_type", ["patient", "timepoint"], extra_cols=["response"])
>>> cd8 = comp[comp.celltype == "CD8 T"]
>>> res = arsc.stats.paired_test(cd8, "prop", t0="pre", t1="post", patient="patient",
...                              by="response", transform="logit")
>>> res[["group", "n_patients", "min_possible_p"]].values.tolist()
[['R', 3, 0.25], ['NR', 3, 0.25]]
```

### `arsc.stats.perm_pvalue(null, obs, alternative: 'str' = 'two-sided', exact: 'bool' = False) -> 'float'`

```text
p-value of an observed statistic against a null distribution you built yourself (random gene sets, a custom
permutation scheme, shuffled clone labels).

Use it for the p whenever the null is not one of the standard schemes (:func:`perm_test`,
:func:`strat_perm_test`, :func:`signflip_test`, :func:`freedman_lane`): it does the three things hand-written
versions got wrong. The Monte Carlo p is ``(1 + #extreme) / (B + 1)`` and never 0 (the observed labeling counts
as one draw); non-finite null values are dropped before counting (a NaN makes every comparison False, so a
hand count returns p = 0); "as or more extreme" uses a tie tolerance of 1e-12 + 1e-10 * |obs|, so equal sums
that differ by floating-point rounding count as ties.

Parameters
----------
null : statistics of the null draws (not including the observed value).
obs : the observed statistic.
alternative : ``"two-sided"`` (``|null| >= |obs|``: the null must be centered at 0, e.g. a difference or a
    centered sum), ``"greater"`` (``null >= obs``) or ``"less"`` (``null <= obs``).
exact : True when ``null`` is the complete enumeration of the randomization distribution (every
    permutation, observed labeling included): then p = ``#extreme / len(null)``.

Returns
-------
float p-value; NaN when ``obs`` is not finite or no null value is finite.

Examples
--------
>>> import numpy as np, arsc
>>> null = np.array([0.1, -0.3, 0.2, np.nan, -0.05, 0.4])
>>> arsc.stats.perm_pvalue(null, 0.35)          # (1 + 1) / (5 + 1); the NaN is dropped
0.3333333333333333
>>> arsc.stats.perm_pvalue(null, 0.35, alternative="less")   # (1 + 4) / (5 + 1)
0.8333333333333334
```

### `arsc.stats.perm_test(x, y, stat=None, B: 'int' = 20000, seed: 'int' = 1) -> 'dict'`

```text
Two-group permutation test on per-patient values (e.g. responders vs non-responders).

Exact enumeration when ``choose(n_x + n_y, n_x) <= B``, otherwise ``B`` random permutations.
Two-sided: exact p = fraction of permutations with ``|stat| >= |observed|``; sampled p =
``(1 + #extreme) / (B + 1)``, which is never 0 (the observed labeling counts as one permutation).
Non-finite values are dropped first; an empty group or a non-finite observed statistic gives p = NaN.

Parameters
----------
x, y : per-patient values of the two groups (one value per patient, not per cell).
stat : ``stat(a, b) -> float``; default difference of medians. Medians tie across permutations, so with
    complete separation the median p can exceed the exact floor of :func:`mde` (4 vs 4: 0.057, not 0.029);
    pass ``stat=lambda a, b: a.mean() - b.mean()`` to reach that floor.
B : enumeration limit / number of sampled permutations.
seed : seeds a local ``np.random.default_rng`` for sampled permutations (unused when exact).

Returns
-------
dict with ``n_x``, ``n_y``, ``stat`` (observed), ``p_perm`` and ``exact`` (True when enumerated). With 3 vs 3
patients the smallest possible p is 2/20 = 0.1 (ties of the median statistic can make it larger, as below).

Examples
--------
>>> import arsc
>>> r = arsc.stats.perm_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
>>> r["stat"], r["p_perm"]
(-3.0, 0.2)
```

### `arsc.stats.poisson_binomial(p) -> 'np.ndarray'`

```text
Distribution of the number of successes among independent trials with different probabilities.

E.g. the chance that a "at least k of n patients pass" kill rule passes when patient i passes with
probability ``p_i`` (from a per-patient planted-effect calibration): ``pmf[k:].sum()``. Exact dynamic
programming over the trials (no normal approximation).

Parameters
----------
p : success probability per trial (patient), each in [0, 1].

Returns
-------
numpy array ``pmf`` of length ``len(p) + 1``: ``pmf[k] = P(X = k)``.

Examples
--------
>>> import arsc
>>> pmf = arsc.stats.poisson_binomial([0.9, 0.8, 0.7, 0.6])
>>> round(float(pmf[3:].sum()), 4)          # P(at least 3 of 4 pass)
0.7428
```

### `arsc.stats.rarefy_richness(counts, depth, se: 'bool' = False)`

```text
Expected number of distinct clones (or genes) in a random subsample of ``depth`` cells, as vegan ``rarefy``.

Closed form (Hurlbert 1971; variance Heck et al. 1975): ``E[S_d] = sum_i 1 - C(N - n_i, d) / C(N, d)``. No
Monte Carlo is needed (repeated random subsampling estimates the same number with noise). A depth above
the sample total returns the observed richness, as vegan does: check ``depth <= N``.

Parameters
----------
counts : clone sizes of one sample (1-D), or samples x clones (2-D; one result per row).
depth : subsample size (int).
se : also return the standard error.

Returns
-------
float (1-D input) or array (2-D); with ``se=True`` a tuple ``(expected, se)``.

Examples
--------
>>> import arsc
>>> round(float(arsc.stats.rarefy_richness([10, 5, 3, 1, 1], 5)), 4)
2.8909
```

### `arsc.stats.sample_means(obs: 'pd.DataFrame', cols, sample_cols=('patient_alias', 'timepoint'), by=None, min_cells: 'int' = 10)`

```text
Per-sample means of numeric per-cell columns (e.g. a module score), with a cell floor.

Parameters
----------
obs : cell metadata with the numeric columns.
cols : numeric columns to average.
sample_cols : columns defining a sample (default ``("patient_alias", "timepoint")``; pass yours).
by : optional extra grouping column(s) (e.g. ``"cell_type"``).
min_cells : drop samples with fewer cells.

Returns
-------
DataFrame, one row per sample (x ``by``): keys, the means of ``cols`` and ``n_cells``.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> obs = ds.obs.assign(nUMI=ds.X("RNA", "counts").sum(axis=1).A1)
>>> sm = arsc.stats.sample_means(obs, ["nUMI"], ["patient", "timepoint"], by="cell_type")
>>> list(sm.columns)
['patient', 'timepoint', 'cell_type', 'nUMI', 'n_cells']
```

### `arsc.stats.signflip_maxt(D, stat: 'str' = 'mean', method: 'str' = 'step-down', max_exact: 'int' = 16, B: 'int' = 20000, seed: 'int' = 1) -> 'pd.DataFrame'`

```text
Family-wise sign-flip test over K readouts measured on the same patients (Westfall-Young maxT).

Each permutation flips the sign of a whole patient row (all K readouts together), so the correlation
between readouts is kept. The family-wise p of readout k is the share of patterns whose maximum
``|T*|`` over the family (single-step) or over the readouts not yet rejected (free step-down, then
monotone) reaches ``|t_k|``. Use when a kill rule asks whether "the largest of K channels" moves.
Missing values (NaN) are allowed: each readout uses its finite patients (``np.nanmean``).

Parameters
----------
D : patients x readouts array or DataFrame of per-patient differences (columns = readouts).
stat : ``"mean"`` (default) or ``"t"`` (studentised mean; use when readouts have different scales).
method : ``"step-down"`` (default; more powerful, same FWER control) or ``"single-step"``.
max_exact : enumerate all ``2**n`` patterns up to this many patients.
B : random patterns beyond ``max_exact`` (Monte Carlo p = (1 + #) / (B + 1)).
seed : seeds a local ``np.random.default_rng`` for the random patterns (unused when exact).

Returns
-------
DataFrame, one row per readout: ``readout, n, stat, p_raw, p_fwer, exact, n_patterns``.

Examples
--------
>>> import numpy as np, arsc
>>> D = np.array([[1.0, 0.2], [0.8, -0.1], [1.3, 0.3], [0.9, 0.1], [1.1, -0.2], [0.7, 0.0]])
>>> r = arsc.stats.signflip_maxt(D)
>>> r[["p_raw", "p_fwer"]].round(4).values.tolist()
[[0.0312, 0.0312], [0.6875, 0.6875]]
```

### `arsc.stats.signflip_test(d, stat='mean', alternative: 'str' = 'two-sided', max_exact: 'int' = 20, B: 'int' = 100000, seed: 'int' = 1) -> 'dict'`

```text
One-sample sign-flip (randomization) test of per-patient differences: H0 = each difference is symmetric about 0.

Exact over all ``2**n`` sign patterns for ``n <= max_exact``; otherwise ``B`` random patterns with the
never-zero Monte Carlo p-value ``(1 + #extreme) / (B + 1)``. Non-finite values are dropped first (a NaN
left in a hand-written version makes every comparison False and returns p = 0). "As or more extreme"
uses a tolerance of 1e-12 + 1e-10 * |observed| so floating-point rounding of equal sums counts as a tie.

Parameters
----------
d : per-patient values (e.g. post - pre deltas), one per patient.
stat : ``"mean"`` (default), ``"sum"`` (same p as the mean), ``"median"`` or a callable on a 1-D array.
alternative : ``"two-sided"`` (``|null| >= |obs|``), ``"greater"`` or ``"less"``.
max_exact : enumerate all patterns up to this many patients (2**20 = 1,048,576 patterns, chunked).
B : number of random patterns beyond ``max_exact``.
seed : seeds a local ``np.random.default_rng`` for the random patterns (unused when exact).

Returns
-------
dict with ``n``, ``stat`` (observed), ``p``, ``exact``, ``n_patterns`` and ``min_p`` (the smallest p the
design can give: ``2 / 2**n`` two-sided, ``1 / 2**n`` one-sided, exact case; ``1 / (B + 1)`` otherwise).

Examples
--------
>>> import arsc
>>> r = arsc.stats.signflip_test([0.5, 1.2, 0.8, 2.0, 1.1, 0.3])
>>> r["p"], r["exact"], r["min_p"]
(0.03125, True, 0.03125)
```

### `arsc.stats.strat_perm_test(y, labels, strata, stat='mean_diff', alternative: 'str' = 'two-sided', B: 'int' = 20000, max_exact: 'int' = 20000, seed: 'int' = 1) -> 'dict'`

```text
Permutation test of a two-group difference with labels permuted WITHIN strata (blocks).

For designs where only within-stratum exchangeability holds: clones within patient (each patient's clones keep
their patient), patients within batch, sort or cohort, cells within sample. Group sizes are kept in every
stratum. Exact enumeration of every within-stratum arrangement when there are at most ``max_exact`` of them,
otherwise ``B`` random ones with the never-zero p ``(1 + #extreme) / (B + 1)``. Strata with only one group
carry no information and are left out of the statistic (they do not move under the permutation). Units with a
non-finite ``y`` or a missing label / stratum are dropped first.

Parameters
----------
y : value per unit (patient, clone, cell).
labels : group per unit; exactly two distinct values. The statistic is "first minus second" with the label
    values in sorted order, or True minus False for boolean labels.
strata : stratum per unit (patient for clone-level data, batch for patient-level data).
stat : ``"mean_diff"`` (default: mean over informative strata of the within-stratum difference of means),
    ``"sum"`` (sum of ``y`` in the first group over informative strata, centered at its permutation mean: the
    stratified linear statistic; it weights strata by size), or a callable
    ``stat(y, is_first_group, strata) -> float``
    (applied to the units that remain after dropping missing values).
alternative : ``"two-sided"``, ``"greater"`` or ``"less"``.
B : random within-stratum permutations when not exact.
max_exact : enumerate when the number of arrangements (product of per-stratum binomial coefficients) is at
    most this.
seed : seeds a local ``np.random.default_rng`` for random permutations (unused when exact).

Returns
-------
dict with ``stat`` (observed), ``p``, ``exact``, ``n_perm``, ``n_units``, ``n_strata`` (informative strata),
``min_p`` (the smallest p the design can give) and ``groups`` (first, second label).

Examples
--------
>>> import numpy as np, arsc
>>> y = np.array([5.0, 6.0, 1.0, 2.0, 7.0, 8.0, 3.0, 2.5])
>>> g = np.array(["R", "R", "NR", "NR", "R", "R", "NR", "NR"])
>>> s = np.array([1, 1, 1, 1, 2, 2, 2, 2])       # two batches; responders compared within batch
>>> r = arsc.stats.strat_perm_test(y, g, s)
>>> r["groups"], r["exact"], r["n_perm"], round(r["p"], 4)
(('NR', 'R'), True, 36, 0.0556)
```

### `arsc.stats.wilcoxon_signed_rank(x, exact: 'bool | None' = None, correct: 'bool' = True) -> 'float'`

```text
Two-sided one-sample / paired Wilcoxon signed-rank p-value replicating R ``wilcox.test(x, exact=...)``.

Zeros are dropped; ``exact`` defaults to n < 50 and is only honored without zeros/ties (R falls back to the
normal approximation with continuity correction and tie-corrected variance otherwise).

Parameters
----------
x : per-patient paired differences (post - pre).
exact : exact null distribution (default: n < 50).
correct : continuity correction for the normal approximation.

Returns
-------
float p-value (NaN if no non-zero observations remain). With n patients the smallest possible
exact p is 2 / 2**n (0.03125 for n = 6).

Examples
--------
>>> import arsc
>>> arsc.stats.wilcoxon_signed_rank([0.5, 1.2, 0.8, 2.0, 1.1, 0.3])
0.03125
```

## arsc.clones

TCR clones.

### `arsc.clones.ari(a, b) -> 'float'`

```text
Adjusted Rand index between two labelings.

Parameters
----------
a, b : array-like of labels, one per item (same length; label values need not match).

Returns
-------
float; 1.0 = identical partitions, about 0 = chance agreement.

Examples
--------
>>> from arsc import clones
>>> clones.ari([0, 0, 1, 1], ["x", "x", "y", "y"])
1.0
```

### `arsc.clones.clone_diversity(ct, sample_cols=('patient_alias', 'timepoint'), depth=None, floor=200, reps=20, seed=1, verbose=True) -> 'pd.DataFrame'`

```text
Clonal diversity per sample after rarefying to a common depth.

Metrics (means over ``reps`` draws): shannon (nats), simpson, richness, top10_frac, clonality. Rarefaction
uses an exact multivariate hypergeometric draw (== sampling cells without replacement). ``seed`` seeds a
local ``np.random.default_rng`` consumed in sorted sample order.

Parameters
----------
ct : clone table from :func:`clone_table`.
sample_cols : the sample columns used to build ``ct``.
depth : rarefaction depth (default: the smallest sample depth >= ``floor``).
floor : minimum depth; samples below ``depth`` are dropped (default 200).
reps : number of rarefaction draws (default 20).
seed : integer seed (default 1).
verbose : print the depth and the number of retained samples.

Returns
-------
DataFrame, one row per retained sample: ``shannon``, ``simpson``, ``richness``, ``top10_frac``,
``clonality``, ``depth``.

Examples
--------
>>> import arsc
>>> from arsc import clones
>>> ct = clones.clone_table(arsc.datasets.toy().obs, "clone_id", sample_cols=("patient", "timepoint"))
>>> dv = clones.clone_diversity(ct, sample_cols=("patient", "timepoint"), floor=50, verbose=False)
>>> dv.shape
(12, 8)
```

### `arsc.clones.clone_dynamics(ct, t0, t1, patient='patient_alias', time='timepoint', min_cells=3) -> 'pd.DataFrame'`

```text
Clone dynamics between two timepoints within each patient (Fisher test per clone, BH within patient).

Parameters
----------
ct : clone table from :func:`clone_table` (must contain ``patient``, ``time``, ``clone``, ``n``).
t0, t1 : the two timepoint labels (t0 = reference).
patient : patient column (default ``"patient_alias"``; pass yours).
time : timepoint column.
min_cells : minimum cells of a clone over both timepoints to be tested (default 3).

Returns
-------
DataFrame, one row per tested clone: ``n_0``, ``n_1``, ``p``, ``fdr``, ``lfc`` (log2 frequency ratio),
``class`` (``novel_expanded`` / ``expanded`` / ``contracted`` / ``stable`` at FDR < 0.05), patient,
``N0``, ``N1``. Cell-level Fisher p-values describe single clones within a patient; compare patients
or arms with patient-level summaries (e.g. the number of expanded clones per patient).

Examples
--------
>>> import arsc
>>> from arsc import clones
>>> ct = clones.clone_table(arsc.datasets.toy().obs, "clone_id", sample_cols=("patient", "timepoint"))
>>> dyn = clones.clone_dynamics(ct, "pre", "post", patient="patient", time="timepoint")
>>> sorted(dyn["patient"].unique())
['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
```

### `arsc.clones.clone_expansion_null(meta, hd, pairs=((0, 3), (3, 6), (6, 9)), hd_times=('T1', 'T2'), clone_col='clone_id', patient_col='patient_alias', time_col='timepoint', ref_depth=None, null_q=0.95, reps=1, seed=1, downsample=True, hd_time_col=None, hd_clone_col=None)`

```text
Per-clone expansion/contraction between timepoints, calibrated by a healthy-donor (HD) null.

Port of ``ar_clone_expansion_null``: per patient and timepoint pair, each timepoint is downsampled to at most
the HD depth (``ref_depth``, default the HD cell counts); per clone a Fisher test (BH within patient x pair ->
``fdr``) and a signed sqrt(Yates chi-square) ``chi``. Null: HD T1 vs T2 downsampled to the patient's depths,
cutoff = quantile(|chi_HD|, ``null_q``); ``sig_null`` if |chi| > cutoff. Random draws use numpy, so they agree
with R statistically but not bitwise. ``seed`` seeds one local ``np.random.default_rng`` consumed in input
order (patients by first appearance, then ``pairs``, then reps).

Parameters
----------
meta : cell-level metadata of the patients (clone, patient, timepoint columns).
hd : cell-level healthy-donor table (e.g. from :func:`read_10x_clonotypes`) with two timepoints.
pairs : timepoint pairs ``(t1, t2)`` to test, as labels found in ``time_col``.
hd_times : the two HD timepoint labels.
clone_col, patient_col, time_col : column names in ``meta``.
ref_depth : ``(depth_1, depth_2)`` downsampling depths (default: the HD cell counts).
null_q : quantile of |chi_HD| used as the cutoff (default 0.95).
reps : repeated downsamplings, averaged in the patient summary.
seed : integer seed (default 1).
downsample : downsample patient timepoints to ``ref_depth`` (default True).
hd_time_col, hd_clone_col : column names in ``hd`` (default: same as ``time_col`` / ``clone_col``).

Returns
-------
tuple ``(clones, patients)``: per-clone table (``clone``, ``n_1``, ``n_2``, ``chi``, ``p``, ``fdr``,
``cutoff``, ``sig_null``...) and per patient x pair summary (``n_sig_up``, ``n_sig_down``,
``frac_cells_sig_up_t2``...).

Examples
--------
>>> import arsc, pandas as pd
>>> from arsc import clones
>>> obs = arsc.datasets.toy().obs
>>> hd = pd.DataFrame({"timepoint": ["T1"] * 200 + ["T2"] * 200,
...                    "clone_id": [f"h{i % 150}" for i in range(400)]})
>>> per_clone, per_patient = clones.clone_expansion_null(obs, hd, pairs=(("pre", "post"),),
...     clone_col="clone_id", patient_col="patient", time_col="timepoint")
>>> per_patient.shape[0]
6
```

### `arsc.clones.clone_table(meta, clone_col, sample_cols=('patient_alias', 'timepoint'), by=None) -> 'pd.DataFrame'`

```text
Clone counts and within-sample frequency per sample (cells without a clone dropped).

Parameters
----------
meta : cell-level metadata.
clone_col : clone id column.
sample_cols : columns that define a sample (default ``("patient_alias", "timepoint")``; pass yours).
by : optional extra column(s) to split counts by (e.g. cell state); ``freq`` stays within sample.

Returns
-------
DataFrame with ``sample_cols`` (+ ``by``), ``clone``, ``n`` (cells) and ``freq`` (fraction of the sample's
clonotyped cells).

Examples
--------
>>> import arsc
>>> from arsc import clones
>>> ct = clones.clone_table(arsc.datasets.toy().obs, "clone_id", sample_cols=("patient", "timepoint"))
>>> list(ct.columns)
['patient', 'timepoint', 'clone', 'n', 'freq']
```

### `arsc.clones.clone_tissue_entropy(mat: 'pd.DataFrame') -> 'pd.DataFrame'`

```text
Per-clone tissue distribution: entropy, normalized entropy, breadth and dominant tissue.

Low entropy = tissue-restricted clone, high = disseminated.

Parameters
----------
mat : clone x tissue count matrix from :func:`clone_tissue_matrix`.

Returns
-------
DataFrame indexed by clone: ``n_cells``, ``n_tissues``, ``entropy`` (bits), ``entropy_norm``
(entropy / log2(number of tissues)), ``dominant``, ``dominant_frac``.

Examples
--------
>>> import arsc
>>> from arsc import clones
>>> m = clones.clone_tissue_matrix(arsc.datasets.toy().obs, tissue_col="cell_type", min_cells=5)
>>> list(clones.clone_tissue_entropy(m).columns)
['n_cells', 'n_tissues', 'entropy', 'entropy_norm', 'dominant', 'dominant_frac']
```

### `arsc.clones.clone_tissue_matrix(meta, clone_col='clone_id', tissue_col='tissue', min_cells=1) -> 'pd.DataFrame'`

```text
Clone x tissue cell-count matrix (clones with fewer than ``min_cells`` cells dropped).

Parameters
----------
meta : cell-level metadata.
clone_col : clone id column.
tissue_col : tissue (or any compartment) column.
min_cells : minimum total cells per clone (default 1).

Returns
-------
DataFrame, clones x tissues, integer cell counts.

Examples
--------
>>> import arsc
>>> from arsc import clones
>>> m = clones.clone_tissue_matrix(arsc.datasets.toy().obs, tissue_col="cell_type", min_cells=5)
>>> list(m.columns)
['CD4 T', 'CD8 T']
```

### `arsc.clones.Cyclone(clonestats: 'pd.DataFrame', assignment: 'pd.DataFrame', counts: 'pd.DataFrame', mat: 'pd.DataFrame', mat_cor: 'pd.DataFrame', linkage: 'np.ndarray | None', cell_k: 'pd.Series', params: 'dict' = <factory>) -> None`

```text
Result of :func:`cyclone` (mirrors the R ``ar_cyclone`` list).

Attributes
----------
clonestats : clone x patient x timepoint long table: ``perc`` (% of the patient's clonotyped cells),
    consecutive deltas (``delta_fc``, ``delta_c``), ``pass`` (passes the dynamics filter) and ``k``.
assignment : one row per clone kept by the filter: patient, group columns, ``clone_id``, ``k``
    (``"Traj 1"``...; ``"Other"`` after the outlier rule), ``n_max``.
counts : patient x trajectory x timepoint: ``n_clones`` and summed ``perc``.
mat : timepoint x clone matrix of ``perc`` (missing -> 0).
mat_cor : clone x clone Pearson correlation of the standardized matrix.
linkage : scipy linkage matrix (ward == R ward.D2); ``None`` for ``method="pam"``/``"absolute"``.
cell_k : trajectory per row of the input ``meta`` (``"Other"`` = not clustered, ``None`` = no clone).
params : the arguments used.

Examples
--------
>>> import arsc
>>> from arsc import clones
>>> obs = arsc.datasets.toy().obs
>>> cy = clones.cyclone(obs, clone_col="clone_id", patient_col="patient", time_col="timepoint",
...                     n_clust=2, n_max_min=3, verbose=False)
>>> cy.params["timepoints"]
['pre', 'post']
>>> list(cy.assignment.columns)
['patient', 'response', 'clone_id', 'k', 'n_max']
```

### `arsc.clones.cyclone(meta: 'pd.DataFrame', clone_col='clone_id', patient_col='patient_alias', time_col='timepoint', timepoints=None, n_clust=6, min_fc=2.0, min_c=0.25, n_max_min=5, method='hclust', scale='timepoint', custom_order=None, outlier_sd=None, reg=0.0, group_cols=('treatment', 'response'), drop_incomplete=True, min_cells_sample=1, stats_only=False, verbose=True)`

```text
Cyclone: cluster expanding/contracting clones by their temporal trajectory.

Exact port of the published ``cyclone_find_trajectories`` algorithm (ARI 1.0 against the published
assignments; ``CLONES_PARITY.md``): per patient x timepoint ``perc`` -> consecutive deltas (fold change and
absolute) -> keep clones with any increasing (fc > ``min_fc`` & delta > ``min_c`` & peak not at t0) or
decreasing (fc < 1/``min_fc`` & delta < -``min_c`` & peak at t0) step and ``n_max >= n_max_min`` -> timepoint x
clone matrix standardized -> Pearson correlation between clones -> Euclidean distance on the correlation
matrix -> Ward.D2 -> ``cutree(n_clust)``, numbered by first appearance in sorted(clone_id) order.

Parameters
----------
meta : cell-level metadata, one row per cell (e.g. ``Store.obs``).
clone_col : column with the clone id (NaN / "" = no TCR; those cells are ignored).
patient_col : patient column (default ``"patient_alias"``; pass yours explicitly).
time_col : timepoint column.
timepoints : ordered timepoint labels to use (default: categorical order, else sorted).
n_clust : number of trajectories (clusters) to cut.
min_fc : fold-change threshold of a step (default 2).
min_c : absolute change threshold of a step, in percentage points (default 0.25).
n_max_min : minimum peak cell count of a clone at any timepoint (default 5).
method : ``"hclust"`` (published default), ``"pam"`` (PAM on the correlation matrix) or ``"absolute"``
    (trajectory = timepoint index of the maximum ``perc``).
scale : ``"timepoint"`` (faithful default: z-score per timepoint across clones), ``"clone"`` (per-clone z)
    or ``"none"``.
custom_order : list of cutree ids giving the order to rename them ``Traj 1..n``.
outlier_sd : among rows with perc > ``min_c`` and n_max >= ``n_max_min``, clones with any perc >=
    mean + ``outlier_sd`` * sd become ``"Other"`` (default ``None`` = off).
reg : pseudocount added to ``perc`` in the fold change (default 0, as published).
group_cols : patient-level columns carried into the outputs when present (e.g. treatment, response).
drop_incomplete : drop patients missing any of the timepoints (default True).
min_cells_sample : minimum clonotyped cells per patient x timepoint.
stats_only : return only the clone x patient x timepoint change table (with logical ``pass``).
verbose : print filter counts.

Returns
-------
:class:`Cyclone` (or a DataFrame with ``stats_only=True``).

Examples
--------
>>> import arsc
>>> from arsc import clones
>>> obs = arsc.datasets.toy().obs
>>> cy = clones.cyclone(obs, clone_col="clone_id", patient_col="patient", time_col="timepoint",
...                     n_clust=2, n_max_min=3, verbose=False)
>>> sorted(cy.assignment["k"].unique())
['Traj 1', 'Traj 2']
```

### `arsc.clones.cyclone_patient_table(cy: 'Cyclone', value='n_clones', at=None) -> 'pd.DataFrame'`

```text
Per-patient trajectory table: clones per trajectory (or summed perc at timepoint ``at``), wide.

Parameters
----------
cy : a :class:`Cyclone` result.
value : ``"n_clones"`` (number of clones per trajectory) or ``"perc"`` (summed % of clonotyped cells).
at : timepoint label at which ``perc`` is reported (``value="perc"`` only; default all timepoints summed).

Returns
-------
DataFrame, one row per patient (plus group columns), one column per trajectory.

Examples
--------
>>> import arsc
>>> from arsc import clones
>>> cy = clones.cyclone(arsc.datasets.toy().obs, patient_col="patient", n_clust=2, n_max_min=3, verbose=False)
>>> clones.cyclone_patient_table(cy, "perc", at="post").shape[0]
6
```

### `arsc.clones.cyclone_states(cy: 'Cyclone', meta: 'pd.DataFrame', state_col: 'str', delta='fc', reg=2.0, mki67=None, group_col='treatment') -> 'pd.DataFrame'`

```text
Per-patient cell-state composition of trajectory clones over time (``ar_cyclone_states``).

Value ``n`` = % of the patient's clonotyped cells at that timepoint that are in trajectory k AND state s.

Parameters
----------
cy : a :class:`Cyclone` result computed on the same ``meta`` rows.
meta : the cell-level metadata passed to :func:`cyclone`.
state_col : column with the cell state (cluster / cell type).
delta : ``"fc"`` = log2((n + reg) / (n_t0 + reg)), ``"abs"`` = n - n_t0, ``"none"`` = n.
reg : pseudocount for ``delta="fc"`` (default 2).
mki67 : optional per-cell values (e.g. MKI67 counts); adds a ``cycling`` column (% > 0).
group_col : patient-level grouping column (default ``"treatment"``; ``"all"`` if absent).

Returns
-------
DataFrame, one row per patient x timepoint x trajectory x state: ``cells``, ``count``, ``n``, ``n0``,
``value`` (+ ``cycling``).

Examples
--------
>>> import arsc
>>> from arsc import clones
>>> obs = arsc.datasets.toy().obs
>>> cy = clones.cyclone(obs, patient_col="patient", n_clust=2, n_max_min=3, verbose=False)
>>> st = clones.cyclone_states(cy, obs, state_col="cell_type", group_col="response")
>>> "value" in st.columns
True
```

### `arsc.clones.fisher_exact_2x2(a, b, N1, N2) -> 'np.ndarray'`

```text
Two-sided Fisher exact p-values for many 2 x 2 tables ``[[a, N1-a], [b, N2-b]]`` (== R ``fisher.test``).

Uses R's relative tolerance (1 + 1e-7) for "as extreme" tables; unique tables are evaluated once.

Parameters
----------
a, b : integer arrays, e.g. a clone's cell count at timepoint 1 and timepoint 2.
N1, N2 : totals (scalars or arrays broadcastable to ``a``), e.g. clonotyped cells per timepoint.

Returns
-------
numpy array of two-sided p-values, one per table.

Examples
--------
>>> from arsc import clones
>>> p = clones.fisher_exact_2x2([5, 0], [20, 1], 100, 100)
>>> bool(p[0] < 0.01), bool(p[1] == 1.0)
(True, True)
```

### `arsc.clones.p_adjust_bh(p) -> 'np.ndarray'`

```text
Benjamini-Hochberg adjustment (== R ``p.adjust(p, "BH")``; NaN preserved).

Parameters
----------
p : array-like of p-values (NaN allowed; they stay NaN and are not counted).

Returns
-------
numpy array of adjusted p-values, same shape as ``p``.

Examples
--------
>>> from arsc import clones
>>> clones.p_adjust_bh([0.01, 0.02, 0.03]).round(3).tolist()
[0.03, 0.03, 0.03]
```

### `arsc.clones.pam(D: 'np.ndarray', k: 'int', max_iter: 'int' = 100)`

```text
Partitioning around medoids (BUILD + SWAP) on a dissimilarity matrix (``cluster::pam`` equivalent).

Parameters
----------
D : (n, n) symmetric dissimilarity matrix.
k : number of clusters.
max_iter : maximum number of SWAP iterations (default 100).

Returns
-------
tuple ``(labels, medoids)``: labels 0..k-1 numbered in medoid order, and the medoid row indices.

Examples
--------
>>> import numpy as np
>>> from arsc import clones
>>> D = np.abs(np.subtract.outer(np.arange(6.0), np.arange(6.0)))
>>> labels, medoids = clones.pam(D, 2)
>>> labels.tolist()
[0, 0, 0, 1, 1, 1]
```

### `arsc.clones.read_10x_clonotypes(files, require_paired=True) -> 'pd.DataFrame'`

```text
Read 10x ``clonotypes.csv`` files named ``"<donor> <timepoint> clonotypes.csv"`` into one row per cell.

The clone id is TRA nt + " " + TRB nt (multiple chains comma-joined), the paired definition used by
:func:`clone_expansion_null` for healthy-donor references.

Parameters
----------
files : iterable of paths; donor and timepoint are parsed from each file name.
require_paired : keep only clonotypes with both a TRA and a TRB chain (default True).

Returns
-------
DataFrame with columns ``patient_alias``, ``timepoint``, ``clone_id``, one row per cell
(clonotypes repeated by their ``frequency``).

Examples
--------
>>> import os, tempfile
>>> import pandas as pd
>>> from arsc import clones
>>> d = tempfile.mkdtemp()
>>> f = os.path.join(d, "HD1 T1 clonotypes.csv")
>>> pd.DataFrame({"cdr3s_nt": ["TRA:AAA;TRB:CCC", "TRB:GGG"], "frequency": [3, 1]}).to_csv(f, index=False)
>>> clones.read_10x_clonotypes([f])["clone_id"].tolist()
['AAA CCC', 'AAA CCC', 'AAA CCC']
```

### `arsc.clones.read_10x_contigs(path, *, sample=None, chains=('TRA', 'TRB'), productive_only: 'bool' = True, high_confidence: 'bool' = True, key: 'str' = 'cdr3_nt', require_paired: 'bool' = False) -> 'pd.DataFrame'`

```text
Read a CellRanger VDJ ``filtered_contig_annotations.csv`` into one row per cell barcode.

Contigs are kept when their ``chain`` is in ``chains`` and (by default) they are productive, high-confidence
and (if the column exists) ``is_cell``. Per chain type, several contigs of one cell are ordered by ``umis``
(highest first, ties by sequence) and comma-joined in the ``<chain>_*`` columns. The clone id joins, per
chain type in ``chains`` order, the cell's ``key`` sequences **sorted alphabetically** (so the id does not
depend on UMI counts) with ``","`` inside a type and ``"|"`` between types, e.g. ``"TGTGCC|TGTGCA"``.

Parameters
----------
path : path of ``filtered_contig_annotations.csv`` (or ``all_contig_annotations.csv``).
sample : optional sample label written to a ``sample`` column (the barcode is left unchanged; prefix it
    yourself if barcodes must be unique across samples).
chains : chain types to use, in clone-id order (default ``("TRA", "TRB")``; e.g. ``("TRG", "TRD")`` for
    gamma-delta T cells, ``("IGH", "IGK", "IGL")`` for B cells).
productive_only : keep only ``productive`` contigs (default True).
high_confidence : keep only ``high_confidence`` contigs (default True).
key : contig column that defines the clone (default ``"cdr3_nt"``; ``"cdr3"`` = amino acids).
require_paired : ``clone_id`` is NaN unless every chain type in ``chains`` is present (default False:
    NaN only when no chain passes).

Returns
-------
DataFrame, one row per barcode with at least one passing contig: ``barcode``, ``sample``, then per chain
type ``<chain>_cdr3``, ``<chain>_cdr3_nt``, ``<chain>_v_gene``, ``<chain>_j_gene``, ``n_<chain>`` (number
of contigs), and ``clone_id``.

Examples
--------
>>> import os, tempfile
>>> import pandas as pd
>>> from arsc import clones
>>> f = os.path.join(tempfile.mkdtemp(), "filtered_contig_annotations.csv")
>>> pd.DataFrame({
...     "barcode": ["AAA-1", "AAA-1", "CCC-1", "CCC-1", "GGG-1", "GGG-1"],
...     "is_cell": [True] * 6, "high_confidence": [True] * 6,
...     "chain": ["TRA", "TRB", "TRA", "TRB", "TRB", "TRB"],
...     "productive": [True, True, True, True, True, False],
...     "cdr3": ["CAV", "CAS", "CAV", "CAS", "CSA", "CXX"],
...     "cdr3_nt": ["TGTGCA", "TGTGCC", "TGTGCA", "TGTGCC", "TGCAGT", "TGCNNN"],
...     "v_gene": ["TRAV1", "TRBV2", "TRAV1", "TRBV2", "TRBV5", "TRBV6"],
...     "j_gene": ["TRAJ1", "TRBJ1", "TRAJ1", "TRBJ1", "TRBJ2", "TRBJ2"],
...     "umis": [3, 5, 2, 7, 4, 1]}).to_csv(f, index=False)
>>> tcr = clones.read_10x_contigs(f, sample="P1_pre")
>>> tcr[["barcode", "n_TRA", "n_TRB", "clone_id"]].values.tolist()
[['AAA-1', 1, 1, 'TGTGCA|TGTGCC'], ['CCC-1', 1, 1, 'TGTGCA|TGTGCC'], ['GGG-1', 0, 1, '|TGCAGT']]
>>> clones.read_10x_contigs(f, require_paired=True)["clone_id"].isna().tolist()
[False, False, True]
```

### `arsc.clones.startrac(meta, clone_col='clone_id', state_col='metaclusters_exh', group_cols='patient_alias', loc_col=None, min_cells=20, downsample=None, seed=1)`

```text
STARTRAC-style clonal indices (Zhang et al. 2018) computed per patient (or sample).

expa(state) = 1 - H(clone sizes in state) / log2(n cells in state); tran(state) = sum_c w_c H_c(states);
migr(state) = the same with ``loc_col``; pairwise tran(A, B) over pairs of states. Entropies in bits,
unnormalized.

Parameters
----------
meta : cell-level metadata.
clone_col : clone id column (cells without a clone are dropped).
state_col : cell-state column (cluster / cell type). Default is a project-specific name; pass yours.
group_cols : column(s) defining the unit (default ``"patient_alias"``; pass yours, e.g. ``"patient"``).
loc_col : optional tissue/location column for the migration index.
min_cells : states (and state pairs) with fewer cells in a group are dropped (default 20).
downsample : optional number of cells per group x state to sample first.
seed : integer seed for ``downsample`` (default 1; unused without it).

Returns
-------
tuple ``(cluster, pairwise)`` DataFrames: per group x state ``n``, ``expa``, ``tran`` (+ ``migr``), and per
group x state pair ``tran`` and ``n_shared_clones``.

Examples
--------
>>> import arsc
>>> from arsc import clones
>>> obs = arsc.datasets.toy().obs
>>> per_state, pairs = clones.startrac(obs, clone_col="clone_id", state_col="cell_type", group_cols="patient")
>>> list(per_state.columns)
['patient', 'state', 'n', 'expa', 'tran']
```

## arsc.clonotrace

CloneTrace (clonal fate).

### `arsc.clonotrace.bin_filter(mass, time, thresh: 'float' = 5, binsize: 'float' = 0.005) -> 'np.ndarray'`

```text
Cells in pseudotime bins where both target and control mass exceed ``thresh`` (R ``bin_filter_profile_mass``).

Mirrors R's ``cut(time, seq(min, max, by))``: right-closed bins. Times at the minimum, or above the last
break, fall into an NA bin, which is kept only if that NA bin itself passes.

Parameters
----------
mass : cells x 2 array (target mass, control mass).
time : pseudotime per cell.
thresh : minimum summed mass of each column per bin (default 5).
binsize : bin width in pseudotime units (default 0.005).

Returns
-------
boolean numpy array, one per cell (True = kept).

Examples
--------
>>> import numpy as np
>>> from arsc import clonotrace as ct
>>> keep = ct.bin_filter(np.ones((100, 2)), np.linspace(0, 1, 100), thresh=2, binsize=0.1)
>>> keep.dtype.name, int(keep.sum())
('bool', 99)
```

### `arsc.clonotrace.cell_profiles(Q: 'pd.DataFrame', clone_cluster: 'pd.Series', min_prob: 'float' = 0.5)`

```text
Cell x profile probabilities ``Q P`` (P = one-hot clone -> profile) and a hard label per cell.

Parameters
----------
Q : cells x clones probabilities.
clone_cluster : profile per clone (Series indexed by clone name, e.g. :func:`cluster_clones`).
min_prob : the hard label is the argmax profile when its probability is ``> min_prob``, else NA
    (default 0.5).

Returns
-------
tuple ``(prob DataFrame cells x profiles, label Series)``.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> Q = ct.filter_and_sparsify(cp)
>>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
>>> D = ct.clone_distance(Xq, Q)
>>> prob, label = ct.cell_profiles(Q, ct.cluster_clones(D, k=5))
>>> prob.shape[0] == Q.shape[0]
True
```

### `arsc.clonotrace.cell_transition(embedding, k: 'int' = 30) -> 'sp.csr_matrix'`

```text
Double-normalized transition matrix ``T = Z D^-1 K D^-1 Z`` of the kNN kernel (R ``compute_transition``).

Parameters
----------
embedding : cells x dims array (e.g. 15 PCs).
k : neighbors per cell for :func:`knn_connectivity` (default 30).

Returns
-------
scipy.sparse.csr_matrix, cells x cells (CloneTrace step 1).

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> T = ct.cell_transition(pcs, k=15)
>>> T.shape
(1597, 1597)
```

### `arsc.clonotrace.clone_distance(embedding, Q, *, method: 'str' = 'nn', graph_k: 'int' = 10, prob_thresh: 'float | None' = None, k: 'int' = 2, graph: 'sp.csr_matrix | None' = None, square: 'bool' = True)`

```text
Clone x clone distance on the SNN cost graph (R ``clone_disance`` with ``revised=FALSE``).

The graph is :func:`snn_cost_graph` ``(embedding, graph_k)``. ``method="nn"`` (default; ``graph_clone_nn``):
cells with ``Q >= prob_thresh`` (default 0.1) belong to the clone. The distance between clones *i* and *j*
is the average of (mean over *i*'s cells of the mean of the ``k`` smallest path lengths to *j*'s cells)
and the same from *j* to *i*; a single-cell clone uses the plain mean. ``method="exact"``: the
Wasserstein-1 distance between the thresholded (default 0.05), renormalized clone distributions, with
shortest-path ground costs (``clone_2_ot``); solved with POT (``ot.emd2``) when installed, else HiGHS. It is
slow for many clones.

Parameters
----------
embedding : cells x dims, rows aligned to ``Q``.
Q : cells x clones (DataFrame keeps the clone names), e.g. :func:`filter_and_sparsify`.
method : ``"nn"`` (default) or ``"exact"``.
graph_k : neighbors of the SNN cost graph (default 10).
prob_thresh : membership threshold (default 0.1 for ``"nn"``, 0.05 for ``"exact"``).
k : number of smallest path lengths averaged (``"nn"``; default 2).
graph : a precomputed cost graph (skips :func:`snn_cost_graph`).
square : return a square DataFrame (default) or the long table.

Returns
-------
square clone x clone DataFrame, or the long table ``group1, group2, dis`` with 1-based clone indices
like R.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> Q = ct.filter_and_sparsify(cp)
>>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
>>> D = ct.clone_distance(Xq, Q)
>>> D.shape
(21, 21)
```

### `arsc.clonotrace.clone_dpt(clone_embedding, root, *, k: 'int' = 10, dpt_k: 'int' = 30) -> 'pd.Series'`

```text
Diffusion pseudotime of clones (R ``clone_dpt`` / ``dpt``) on a kNN graph of the clone embedding.

``T`` = row-normalized :func:`knn_connectivity` ``(clone_embedding, k)``. The top ``dpt_k`` eigenpairs by
magnitude are taken, and those with ``lambda >= 0.9999`` dropped. The kept eigenvectors are scaled by
``lambda / (1 - lambda)``; pseudotime is the Euclidean distance to the root, divided by its maximum.

Parameters
----------
clone_embedding : clones x dims DataFrame (e.g. :func:`cmdscale` / :func:`isomds`).
root : clone name or row position of the root.
k : neighbors of the clone kNN graph (default 10).
dpt_k : number of eigenpairs (default 30; capped by the number of clones).

Returns
-------
pandas.Series of pseudotime in [0, 1] indexed by clone (0 at the root).

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> Q = ct.filter_and_sparsify(cp)
>>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
>>> D = ct.clone_distance(Xq, Q)
>>> E = ct.cmdscale(D, 4)
>>> tt = ct.clone_dpt(E, E.index[0], k=5, dpt_k=4)
>>> float(tt.iloc[0])
0.0
```

### `arsc.clonotrace.clone_root(clone_names: 'Sequence[str]', cell_clones, cell_clusters, start_cluster) -> 'str'`

```text
Root clone for pseudotime: highest fraction of its cells in ``start_cluster``, ties broken by size.

Port of R ``clone_root``.

Parameters
----------
clone_names : candidate clones (e.g. ``D.index``).
cell_clones : clone label per cell.
cell_clusters : cell cluster / type per cell.
start_cluster : the cluster where trajectories start (e.g. naive cells).

Returns
-------
str, the root clone name.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> Q = ct.filter_and_sparsify(cp)
>>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
>>> D = ct.clone_distance(Xq, Q)
>>> root = ct.clone_root(list(D.index), clones, t.obs["cell_type"].astype(str).to_numpy(), "CD8 T")
>>> root in D.index
True
```

### `arsc.clonotrace.clone_table(ct: 'CloneTrace', cyclone=None, *, clone_col: 'str | None' = None) -> 'pd.DataFrame'`

```text
One row per CloneTrace clone: mass, cell count, profile, pseudotime (+ Cyclone trajectory).

Pass a :class:`arsc.clones.Cyclone` result to add its trajectory assignment (``cyclone_k``) and patient for
clones whose id matches. Typically the CloneTrace clone is the TCR clonotype; when it is clonotype x
timepoint, map it to the clonotype before joining.

Parameters
----------
ct : a :class:`CloneTrace` result.
cyclone : optional :class:`arsc.clones.Cyclone` result.
clone_col : Cyclone clone column (default: the Cyclone ``clone_col``).

Returns
-------
DataFrame: ``clone, mass, n_cells, profile, clone_t`` (+ ``cyclone_k`` and the patient column).

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> res = ct.run(pcs, clones, cells=t.obs.index, cell_clusters=t.obs["cell_type"].astype(str).to_numpy(),
...              start_cluster="CD8 T", min_size=5, n_boot=8, n_mds=5, cluster_k=5, dpt_k=10, dpt_nn=5)
>>> cy = arsc.clones.cyclone(ds.obs, patient_col="patient", n_clust=2, n_max_min=3, verbose=False)
>>> list(ct.clone_table(res, cy).columns)
['clone', 'mass', 'n_cells', 'profile', 'clone_t', 'cyclone_k', 'patient']
```

### `arsc.clonotrace.CloneProb(prob: 'np.ndarray', deviance: 'np.ndarray', clones: 'list', cells: 'pd.Index') -> None`

```text
Output of :func:`propagate_clones`.

Attributes
----------
prob : cells x clones array of propagated clone probabilities (rows sum to 1).
deviance : per-cell bootstrap deviance (mean L1 distance from the full-label reference; NaN without
    bootstraps).
clones : clone names (columns of ``prob``).
cells : cell names (rows of ``prob``).

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> cp.frame().shape == (len(cp.cells), len(cp.clones))
True
```

### `arsc.clonotrace.CloneTrace(Q: 'pd.DataFrame', deviance: 'pd.Series', distance: 'pd.DataFrame', raw_distance: 'pd.DataFrame', clusters: 'pd.Series', embedding: 'pd.DataFrame', clone_t: 'pd.Series', cell_t: 'pd.Series', cell_profile_prob: 'pd.DataFrame', cell_profile: 'pd.Series', params: 'dict' = <factory>) -> None`

```text
Result of :func:`run`.

Attributes
----------
Q : cells x clones propagated probabilities (after the deviance filter and sparsification).
deviance : per-cell bootstrap deviance.
distance : clone x clone distance used downstream (revised when ``revised=True``).
raw_distance : the unrevised :func:`clone_distance`.
clusters : fate profile per clone (:func:`cluster_clones`).
embedding : clone MDS embedding.
clone_t : pseudotime per clone (:func:`clone_dpt`).
cell_t : pseudotime per cell (Q-weighted clone pseudotime).
cell_profile_prob : cells x profiles probabilities; cell_profile : hard profile per cell.
params : the arguments used.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> res = ct.run(pcs, clones, cells=t.obs.index, cell_clusters=t.obs["cell_type"].astype(str).to_numpy(),
...              start_cluster="CD8 T", min_size=5, n_boot=8, n_mds=5, cluster_k=5, dpt_k=10, dpt_nn=5)
>>> res.Q.shape[1] == res.distance.shape[0]
True
```

### `arsc.clonotrace.cluster_clones(D, k: 'int' = 10, resolution: 'float' = 1.0, *, prune: 'float' = 0.0, weight: 'str' = 'jaccard', seed: 'int' = 0, n_iterations: 'int' = 2) -> 'pd.Series'`

```text
Leiden communities of clones from their distance matrix (R ``leiden_dis``, ``if_umap=FALSE``).

Each clone gets its ``k`` nearest clones (ties go to the lower index). Edge weight = Jaccard of the two kNN
sets, ``shared / (2k - shared)``, or the distance with ``weight="dis"``. The graph is undirected and keeps
both directions as parallel edges, as igraph does in R. Leiden optimizes modularity (igraph
``community_leiden``, beta 0.01). The RNG differs from R's, so expect a high ARI, not identical labels.
A private ``random.Random(seed)`` is installed as igraph's RNG for the call and igraph's default restored
afterwards (Python's global ``random`` state is not touched).

Parameters
----------
D : clone x clone distance DataFrame.
k : nearest clones per clone (default 10).
resolution : Leiden resolution (default 1.0).
prune : drop edges with weight below this (default 0).
weight : ``"jaccard"`` (default) or ``"dis"``.
seed : integer seed (default 0).
n_iterations : Leiden iterations (default 2).

Returns
-------
pandas.Series of 1-based cluster labels indexed by clone (the clone "fate profiles").

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> Q = ct.filter_and_sparsify(cp)
>>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
>>> D = ct.clone_distance(Xq, Q)
>>> prof = ct.cluster_clones(D, k=5)
>>> int(prof.min())
1
```

### `arsc.clonotrace.cmdscale(D, k: 'int' = 2) -> 'pd.DataFrame'`

```text
Classical (Torgerson) MDS, as R ``cmdscale`` (column signs may differ).

Parameters
----------
D : square distance DataFrame (or array).
k : number of dimensions (default 2).

Returns
-------
DataFrame, clones x k, columns ``mds_1..mds_k``.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> Q = ct.filter_and_sparsify(cp)
>>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
>>> D = ct.clone_distance(Xq, Q)
>>> list(ct.cmdscale(D, 2).columns)
['mds_1', 'mds_2']
```

### `arsc.clonotrace.coembed_graph(cell_embedding, clone_embedding, Q, *, cell_k: 'int' = 30, clone_k: 'int' = 15, min_weight: 'float' = 0.1, n_keep: 'int' = 20) -> 'sp.csr_matrix'`

```text
Clone-weighted cell kNN distances (R ``cell_clone_coembed``), for UMAP with a precomputed metric.

For each cell kNN pair (self included), ``w_ij = Q_i V Q_j^T``, where V is the binary clone kNN graph
(``clone_k``, self included). Pairs with ``w > min_weight`` get distance ``d_ij / w_ij``. Each cell keeps
its ``n_keep`` smallest (ties kept), then the graph is symmetrized and the diagonal dropped.

Parameters
----------
cell_embedding : cells x dims, rows aligned to ``Q``.
clone_embedding : clones x dims, rows aligned to ``Q``'s columns.
Q : cells x clones probabilities.
cell_k : cell neighbors (default 30).
clone_k : clone neighbors (default 15).
min_weight : minimum clone weight of a kept pair (default 0.1).
n_keep : distances kept per cell (default 20).

Returns
-------
scipy.sparse.csr_matrix of distances, cells x cells.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> Q = ct.filter_and_sparsify(cp)
>>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
>>> D = ct.clone_distance(Xq, Q)
>>> G = ct.coembed_graph(Xq, ct.cmdscale(D, 2), Q, cell_k=10, clone_k=3)
>>> G.shape == (len(Q), len(Q))
True
```

### `arsc.clonotrace.encode_clones(clones, levels: 'Sequence[str] | None' = None) -> 'tuple[np.ndarray, list]'`

```text
Clone labels -> integer codes (``-1`` = missing) and the level order.

R's ``as.factor`` orders levels in the session locale. Pass ``levels`` to reproduce a specific R run
exactly; the default is code-point (C locale) order.

Parameters
----------
clones : one label per cell (``None`` / NaN = no clone).
levels : explicit level order (labels not in it become ``-1``).

Returns
-------
tuple ``(codes, levels)``: int64 array per cell and the list of levels.

Examples
--------
>>> from arsc import clonotrace as ct
>>> codes, levels = ct.encode_clones(["b", None, "a", "b"])
>>> codes.tolist(), levels
([1, -1, 0, 1], ['a', 'b'])
```

### `arsc.clonotrace.fate_genes(profile, cluster, X, genes: 'Sequence[str]', cell_cluster, cell_t, cell_profile_prob: 'pd.DataFrame', *, n_perm: 'int' = 50, seed: 'int' = 0, min_cells: 'int' = 30, mass_thresh: 'float' = 3, binsize: 'float' = 0.005) -> 'pd.DataFrame | None'`

```text
Fate-associated genes of ``profile`` inside cell ``cluster`` (R ``profile_cluster_DEG``).

Target mass = summed probability of ``profile`` (one or several); control = the rest. Steps: bin filter
(:func:`bin_filter`); genes with ``> min_cells`` non-zero cells; :func:`soft_cluster_fit` with ``df=1``;
calibrate each p-value against all genes' p-values under ``n_perm`` label permutations (rows shuffled,
then shuffled within each row); then BH.

Parameters
----------
profile : profile label (or list of labels) of interest.
cluster : cell cluster in which genes are tested.
X : cells x genes expression aligned to ``cell_profile_prob`` rows.
genes : gene names (columns of ``X``).
cell_cluster : cluster per cell (same order).
cell_t : pseudotime per cell (same order).
cell_profile_prob : cells x profiles DataFrame (:func:`cell_profiles`).
n_perm : number of permutations (default 50).
seed : integer seed of a local ``np.random.default_rng`` (default 0).
min_cells : minimum non-zero cells per gene (default 30).
mass_thresh : ``thresh`` of :func:`bin_filter` (default 3).
binsize : ``binsize`` of :func:`bin_filter` (default 0.005).

Returns
-------
DataFrame per gene (``stat, pval, mean_diff, cohen, pcali, padj``), or None if fewer than 5 cells pass, as
R does.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> Q = ct.filter_and_sparsify(cp)
>>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
>>> D = ct.clone_distance(Xq, Q)
>>> prob, label = ct.cell_profiles(Q, ct.cluster_clones(D, k=5))
>>> cell_type = t.obs["cell_type"].astype(str).reindex(Q.index).to_numpy()
>>> pos = t.obs.index.get_indexer(Q.index)
>>> fg = ct.fate_genes(prob.columns[0], "CD8 T", L[pos], list(t.features("RNA")), cell_type,
...                    np.linspace(0, 1, len(pos)), prob, n_perm=5, mass_thresh=1, binsize=0.1)
>>> list(fg.columns)
['stat', 'pval', 'mean_diff', 'cohen', 'pcali', 'padj']
```

### `arsc.clonotrace.filter_and_sparsify(cp: 'CloneProb', max_deviance: 'float' = 0.3, row_mass: 'float' = 0.9, col_mass: 'float' = 0.9) -> 'pd.DataFrame'`

```text
Vignette post-processing of :func:`propagate_clones`: deviance filter, renormalize, sparsify, renormalize.

Parameters
----------
cp : a :class:`CloneProb`.
max_deviance : drop cells with deviance ``>= max_deviance`` (default 0.3).
row_mass, col_mass : passed to :func:`sparsify` (default 0.9 each).

Returns
-------
``Q``: cells x clones DataFrame, rows summing to 1; empty rows and columns are dropped.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> Q = ct.filter_and_sparsify(cp)
>>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
>>> bool(np.allclose(Q.sum(1), 1))
True
```

### `arsc.clonotrace.isomds(D, k: 'int' = 2, *, maxit: 'int' = 200, tol: 'float' = 1e-06, init=None) -> 'tuple[pd.DataFrame, float]'`

```text
Kruskal non-metric MDS initialized at :func:`cmdscale` (like ``MASS::isoMDS``).

Minimizes Kruskal stress-1 with L-BFGS, taking the isotonic regression of configuration distances on the
dissimilarity order. The optimizer is not the one ``MASS`` uses, so the configurations differ slightly from
R; the stress (in percent, as R reports it) is close.

Parameters
----------
D : square distance DataFrame (or array).
k : number of dimensions (default 2).
maxit : maximum optimizer iterations (default 200).
tol : convergence tolerance (default 1e-6).
init : starting configuration (default :func:`cmdscale`).

Returns
-------
tuple ``(points DataFrame, stress in percent)``.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> Q = ct.filter_and_sparsify(cp)
>>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
>>> D = ct.clone_distance(Xq, Q)
>>> pts, stress = ct.isomds(D, 2)
>>> pts.shape
(21, 2)
```

### `arsc.clonotrace.knn_connectivity(embedding, k: 'int' = 30, *, self_loops: 'bool' = False, mode: 'str' = 'connectivity') -> 'sp.csr_matrix'`

```text
Symmetrized kNN graph with Gaussian kernel weights (R ``embedding2knn``).

Edges are the union of the kNN relations. With ``sigma_i`` the mean distance over node *i*'s edges, the
weight is ``K_ij = exp(-d_ij^2 / (sigma_i sigma_j))``.

Parameters
----------
embedding : cells x dims array.
k : neighbors per cell (default 30).
self_loops : add ``(i, i, 0)`` edges after symmetrization (R ``if_self``).
mode : ``"connectivity"`` (kernel weights, default) or ``"dist"`` (the distances).

Returns
-------
scipy.sparse.csr_matrix, cells x cells, symmetric.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> ct.knn_connectivity(pcs, k=15).shape
(1597, 1597)
```

### `arsc.clonotrace.label_spreading(adj, labels: 'np.ndarray', n_classes: 'int | None' = None, *, alpha: 'float' = 0.9, max_iter: 'int' = 100, tol: 'float' = 0.001, block_size: 'int' = 128, epsilon: 'float' = 0.0, _P: 'sp.csr_matrix | None' = None) -> 'np.ndarray'`

```text
Label spreading ``F <- alpha P F + (1 - alpha) Y`` from ``F = 0`` (R ``label_spreading_blocked``).

``P`` is ``adj`` row-normalized. ``Y[i, c] = log2(n_c) / n_c`` for labeled cells, ``epsilon / C`` for
unlabeled cells. Classes are processed in blocks of ``block_size``; each block iterates until the max-abs
change is ``< tol``, exactly as in R (so the block size changes the result slightly).

Parameters
----------
adj : cells x cells graph (ignored when ``_P`` is given).
labels : integer class codes per cell, ``-1`` = unlabeled (see :func:`encode_clones`).
n_classes : number of classes (default ``labels.max() + 1``).
alpha : propagation weight (default 0.9; CloneTrace uses 0.6).
max_iter : iterations per block (default 100).
tol : convergence tolerance on the max-abs change (default 1e-3).
block_size : classes per block (default 128).
epsilon : prior mass of unlabeled cells (default 0).
_P : internal: a precomputed row-normalized ``adj``.

Returns
-------
dense numpy array, cells x classes (unnormalized).

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> codes, levels = ct.encode_clones(clones)
>>> F = ct.label_spreading(ct.cell_transition(pcs, 15), codes, alpha=0.6)
>>> F.shape == (len(clones), len(levels))
True
```

### `arsc.clonotrace.ns_basis(x, df: 'int') -> 'np.ndarray'`

```text
Natural cubic spline basis without intercept (R ``splines::ns(x, df)``; columns up to sign/rotation).

The ridge fits in :func:`soft_cluster_fit` are invariant to the orthogonal ambiguity of the constraint
projection.

Parameters
----------
x : 1-D array (e.g. pseudotime).
df : degrees of freedom (number of columns).

Returns
-------
numpy array, len(x) x df.

Examples
--------
>>> import numpy as np
>>> from arsc import clonotrace as ct
>>> ct.ns_basis(np.linspace(0, 1, 50), 3).shape
(50, 3)
```

### `arsc.clonotrace.profile_enrichment(cell_profile_prob: 'pd.DataFrame', cluster_label, n_perm: 'int' = 300, seed: 'int' = 0)`

```text
Cluster x profile mass and a one-sided permutation p-value (R ``cluster_profile_enrich``).

The cluster labels are permuted ``n_perm`` times; ``p = mean(perm >= observed)``, with no ``+1``, as in R.

Parameters
----------
cell_profile_prob : cells x profiles DataFrame.
cluster_label : cell cluster per row (same order; no missing values).
n_perm : number of permutations (default 300).
seed : integer seed of a local ``np.random.default_rng`` (default 0).

Returns
-------
tuple ``(mass, pval)`` DataFrames, clusters x profiles.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> Q = ct.filter_and_sparsify(cp)
>>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
>>> D = ct.clone_distance(Xq, Q)
>>> prob, label = ct.cell_profiles(Q, ct.cluster_clones(D, k=5))
>>> cell_type = t.obs["cell_type"].astype(str).reindex(Q.index).to_numpy()
>>> mass, pval = ct.profile_enrichment(prob, cell_type, n_perm=50)
>>> mass.shape == pval.shape
True
```

### `arsc.clonotrace.profile_mass(cell_profile_prob: 'pd.DataFrame', cluster_label) -> 'pd.DataFrame'`

```text
Summed profile probability per cell cluster (R ``cluster_profile_mass``; rows sorted).

Parameters
----------
cell_profile_prob : cells x profiles DataFrame (from :func:`cell_profiles`).
cluster_label : cell cluster per row of ``cell_profile_prob`` (same order; no missing values).

Returns
-------
DataFrame, clusters x profiles.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> Q = ct.filter_and_sparsify(cp)
>>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
>>> D = ct.clone_distance(Xq, Q)
>>> prob, label = ct.cell_profiles(Q, ct.cluster_clones(D, k=5))
>>> cell_type = t.obs["cell_type"].astype(str).reindex(Q.index).to_numpy()
>>> sorted(ct.profile_mass(prob, cell_type).index)
['CD4 T', 'CD8 T']
```

### `arsc.clonotrace.propagate_clones(adj, clones, *, min_size: 'int' = 10, alpha: 'float' = 0.6, sample_rate: 'float' = 0.8, n_boot: 'int' = 48, block_size: 'int' = 128, tol: 'float' = 0.001, max_iter: 'int' = 100, epsilon: 'float' = 0.0, seed: 'int' = 0, levels: 'Sequence[str] | None' = None, cells=None) -> 'CloneProb'`

```text
Spread expanded-clone labels over the cell graph, with a bootstrap deviance (CloneTrace step 2).

Clones with fewer than ``min_size`` cells are treated as unlabeled. Defaults follow the published vignette:
``alpha=0.6``, 48 bootstraps at ``sample_rate=0.8``. For each bootstrap, a random ``round(0.8 n_labeled)``
subset keeps its labels and the spreading is rerun. The per-cell deviance is the mean L1 distance from the
full-label reference, over the bootstraps where the cell's labeled/unlabeled status is unchanged. Bootstrap
draws use numpy's RNG, not R's, so the deviances agree with R only statistically; the spreading itself is
deterministic, so same input + seed -> identical deviances.

Parameters
----------
adj : cell graph, normally :func:`cell_transition`.
clones : one label per cell (``None`` / NaN = no clone).
min_size : minimum cells of a clone to be propagated (default 10, as the vignette).
alpha : propagation weight (default 0.6).
sample_rate : fraction of labeled cells kept per bootstrap (default 0.8).
n_boot : number of bootstraps (default 48; 0 = no deviance).
block_size, tol, max_iter, epsilon : passed to :func:`label_spreading`.
seed : integer seed of a local ``np.random.default_rng`` (default 0).
levels : explicit clone level order (see :func:`encode_clones`).
cells : cell names for the rows (default ``0..n-1``); pass them to join results back to ``obs``.

Returns
-------
:class:`CloneProb`.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> len(cp.clones)
21
```

### `arsc.clonotrace.revise_distance(D, embedding, Q, *, clusters: 'pd.DataFrame | np.ndarray | None' = None, sample=None, local_rescale: 'bool' = False, local_k: 'int' = 7, resolutions=(0.3, 0.5, 0.8, 1.2), cluster_k: 'int' = 15, mean_weight: 'float | None' = None, s_grid=None, seed: 'int' = 0)`

```text
Blend the CloneTrace distance with a centroid distance (R ``revise_clone_distance``; default in 1.2.x).

Both distances are divided by their median over clone pairs (only same-``sample`` pairs when ``sample``
gives a group per clone). The blend weight ``s`` maximizes the Spearman correlation between blended pair
distances and the total-variation distance of the clones' cell-cluster compositions. It is chosen per
clustering, and the median over clusterings is used.

Parameters
----------
D : clone x clone distance from :func:`clone_distance`.
embedding : cells x dims, rows aligned to ``Q``.
Q : cells x clones probabilities.
clusters : cells x resolutions labels (give it for reproducible results). Default: a Leiden ladder on a
    UMAP kNN graph at ``resolutions``, like R (partitions are not identical to R's).
sample : optional group per clone; only same-group pairs set the medians.
local_rescale : rescale distances by local density (R ``local_rescale``).
local_k : neighbors for the local rescaling (default 7).
resolutions : Leiden resolutions of the default ladder.
cluster_k : neighbors of the default ladder's kNN graph (default 15).
mean_weight : fix the blend weight instead of selecting it.
s_grid : grid of candidate blend weights.
seed : leidenalg seed of the default ladder (default 0; unused with ``clusters`` or ``mean_weight``).

Returns
-------
tuple ``(D_revised DataFrame, info dict)`` with ``mean_weight``, ``picks``, ``selection_correlations``.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> Q = ct.filter_and_sparsify(cp)
>>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
>>> D = ct.clone_distance(Xq, Q)
>>> Dr, info = ct.revise_distance(D, Xq, Q, mean_weight=0.5)
>>> Dr.shape, sorted(info)
((21, 21), ['mean_weight', 'picks', 'selection_correlations'])
```

### `arsc.clonotrace.run(embedding, clones, *, cells=None, cell_clusters=None, start_cluster=None, k: 'int' = 30, min_size: 'int' = 10, alpha: 'float' = 0.6, n_boot: 'int' = 48, max_deviance: 'float' = 0.3, distance: 'str' = 'nn', graph_k: 'int' = 10, revised: 'bool' = False, revise_clusters=None, n_mds: 'int' = 30, cluster_k: 'int' = 20, resolution: 'float' = 0.5, dpt_k: 'int' = 30, dpt_nn: 'int' = 10, seed: 'int' = 0) -> 'CloneTrace'`

```text
Run CloneTrace steps 1-4 with the published vignette defaults.

``revised=False`` reproduces the published figures; ``True`` is the package's 1.2.x default. ``seed`` is the
single seed of the whole pipeline: it is passed unchanged to every stochastic step (bootstrap of
:func:`propagate_clones`, the Leiden ladder of :func:`revise_distance`, :func:`cluster_clones`); the other
steps are deterministic.

Parameters
----------
embedding : cells x dims (e.g. 15 PCs).
clones : clone label per cell (NaN = none).
cells : cell names (default: the DataFrame index or ``0..n-1``).
cell_clusters : cell cluster per cell; with ``start_cluster`` it picks the pseudotime root (default: the
    first clone).
start_cluster : the cluster where trajectories start.
k : neighbors of :func:`cell_transition` (default 30).
min_size : minimum cells per propagated clone (default 10).
alpha : propagation weight (default 0.6).
n_boot : bootstraps for the deviance (default 48).
max_deviance : deviance filter (default 0.3).
distance : ``"nn"`` (default) or ``"exact"`` (:func:`clone_distance`).
graph_k : neighbors of the SNN cost graph (default 10).
revised : apply :func:`revise_distance` (default False).
revise_clusters : cells x resolutions labels for :func:`revise_distance`.
n_mds : dimensions of the clone MDS (default 30; capped by the number of clones).
cluster_k : neighbors for :func:`cluster_clones` (default 20).
resolution : Leiden resolution for :func:`cluster_clones` (default 0.5).
dpt_k : eigenpairs for :func:`clone_dpt` (default 30).
dpt_nn : neighbors for :func:`clone_dpt` (default 10).
seed : integer seed of the whole pipeline (default 0).

Returns
-------
:class:`CloneTrace`.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> res = ct.run(pcs, clones, cells=t.obs.index, cell_clusters=t.obs["cell_type"].astype(str).to_numpy(),
...              start_cluster="CD8 T", min_size=5, n_boot=8, n_mds=5, cluster_k=5, dpt_k=10, dpt_nn=5)
>>> res.clusters.index.equals(res.distance.index)
True
```

### `arsc.clonotrace.shortest_paths(graph: 'sp.csr_matrix', sources) -> 'np.ndarray'`

```text
Weighted shortest-path distances from ``sources`` to all nodes (numba Dijkstra, parallel).

Parameters
----------
graph : cells x cells sparse graph of non-negative edge costs (e.g. :func:`snn_cost_graph`).
sources : node indices to start from.

Returns
-------
numpy array, len(sources) x n_nodes (``inf`` = unreachable).

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> d = ct.shortest_paths(ct.snn_cost_graph(pcs, 10), [0, 1])
>>> d.shape, float(d[0, 0])
((2, 1597), 0.0)
```

### `arsc.clonotrace.snn_cost_graph(embedding, k: 'int' = 10) -> 'sp.csr_matrix'`

```text
Shared-nearest-neighbor cost graph (``bluster::makeSNNGraph(type="number")`` + ``exp(-w)``).

``w_ij = |({i} + kNN_i) & ({j} + kNN_j)|``, i.e. the off-diagonal of ``(I + A)(I + A)^T``. There is an edge
for every pair with ``w >= 1``, and its cost is ``exp(-w_ij)``.

Parameters
----------
embedding : cells x dims array.
k : neighbors per cell (default 10).

Returns
-------
scipy.sparse.csr_matrix of edge costs, cells x cells.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> ct.snn_cost_graph(pcs, k=10).shape
(1597, 1597)
```

### `arsc.clonotrace.soft_cluster_fit(G, t, P, *, df: 'int' = 5, lam: 'float' = 0.0001, test: 'str' = 'F') -> 'pd.DataFrame'`

```text
Test whether expression along pseudotime differs between soft groups (R ``soft_cluster_gam_fit``).

The null is a shared ``[1, ns(t, df)]`` curve; the full model has one curve per group (design
``B * P[:, k]``). Both are ridge fits (``lam``). ``test="F"`` gives
``F = ((RSS0 - RSS1) / df1) / (RSS1 / df2)``; ``"LRT"`` gives the Gaussian likelihood ratio. Cells are the
observations here: use it to rank genes, and confirm across patients before claiming an effect.

Parameters
----------
G : cells x genes expression (dense or sparse).
t : pseudotime per cell.
P : cells x m soft assignment (two columns, target and control, for Cohen's d).
df : spline degrees of freedom (default 5).
lam : ridge penalty (default 1e-4).
test : ``"F"`` (default) or ``"LRT"``.

Returns
-------
DataFrame per gene: ``stat``, ``pval``, ``mean_diff``, ``cohen``.

Examples
--------
>>> import arsc, numpy as np
>>> from arsc import clonotrace as ct
>>> ds = arsc.datasets.toy()
>>> t = ds.subset(ds.obs["cell_type"].isin(["CD8 T", "CD4 T"]).to_numpy())
>>> L = arsc.pp.log_normalize(t.X("RNA", "counts"))
>>> pcs = arsc.pp.pca(arsc.pp.scale(L.toarray())[0], n_comps=10).embedding
>>> clones = t.obs["clone_id"].to_numpy()
>>> cp = ct.propagate_clones(ct.cell_transition(pcs, 15), clones, min_size=5, n_boot=8, cells=t.obs.index)
>>> Q = ct.filter_and_sparsify(cp)
>>> Xq = pcs[t.obs.index.get_indexer(Q.index)]
>>> D = ct.clone_distance(Xq, Q)
>>> prob, label = ct.cell_profiles(Q, ct.cluster_clones(D, k=5))
>>> cell_type = t.obs["cell_type"].astype(str).reindex(Q.index).to_numpy()
>>> pos = t.obs.index.get_indexer(Q.index)
>>> w = prob.iloc[:, 0].to_numpy()
>>> fit = ct.soft_cluster_fit(L[pos][:, :10], np.linspace(0, 1, len(pos)), np.c_[w, 1 - w], df=3)
>>> list(fit.columns)
['stat', 'pval', 'mean_diff', 'cohen']
```

### `arsc.clonotrace.sparsify(mat, row_mass: 'float' = 0.9, col_mass: 'float' = 0.9) -> 'np.ndarray'`

```text
Keep the top ``row_mass`` of each row, then of each column (R ``mat_sparsify``; dense in, dense out).

Per vector: sort the normalized masses in decreasing order. ``lwr`` is the smallest mass whose cumulative
sum is still ``< thresh``. Entries below ``lwr`` are zeroed, so ties are kept and the entry that crosses
``thresh`` is dropped. Rows are filtered first; columns are then filtered on the result.

Parameters
----------
mat : dense cells x clones array.
row_mass : mass kept per row (default 0.9).
col_mass : mass kept per column (default 0.9).

Returns
-------
dense numpy array of the same shape (not renormalized).

Examples
--------
>>> import numpy as np
>>> from arsc import clonotrace as ct
>>> ct.sparsify(np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])).tolist()
[[0.6, 0.0, 0.0], [0.0, 0.2, 0.6]]
```

## arsc.genesets

Gene sets.

### `arsc.genesets.cycling_fraction(meta, positive, sample_cols=('patient_alias', 'timepoint'), by=None, min_cells=20) -> 'pd.DataFrame'`

```text
Per-sample counts of positive cells by group, with a count-corrected logit for paired deltas.

``logit_cc = log((k + 0.5) / (n - k + 0.5))`` stays finite when ``k`` is 0 or ``n`` (sparse positives such as
MKI67+ cells). The sample (patient x timepoint) is the unit; compare samples, not cells.

Parameters
----------
meta : cell-level metadata.
positive : boolean per cell (e.g. ``MKI67 counts > 0``), row-aligned with ``meta``.
sample_cols : columns defining a sample (default ``("patient_alias", "timepoint")``; pass yours).
by : optional extra grouping column(s) (e.g. cell type).
min_cells : drop sample x group rows with fewer cells (default 20).

Returns
-------
DataFrame: ``sample_cols`` (+ ``by``), ``n``, ``k``, ``prop``, ``logit_cc``.

Examples
--------
>>> import arsc
>>> from arsc import genesets
>>> ds = arsc.datasets.toy()
>>> mki67 = ds.X("RNA", "counts")[:, list(ds.features("RNA")).index("MKI67")].toarray().ravel() > 0
>>> cf = genesets.cycling_fraction(ds.obs, mki67, sample_cols=("patient", "timepoint"), by="cell_type")
>>> list(cf.columns)
['patient', 'timepoint', 'cell_type', 'n', 'k', 'prop', 'logit_cc']
```

### `arsc.genesets.export_msigdb_gmt(collection='H', subcollection=None, species='Homo sapiens', out_dir=None) -> 'Path'`

```text
Write one MSigDB collection to a GMT file through R ``msigdbr`` (one-off; needs msigdbr + msigdbdf in R).

Extra R libraries come from ``$ARSC_RLIB`` and ``$AR_ROOT/.Rlib`` when set.

Parameters
----------
collection : MSigDB collection, e.g. ``"H"``.
subcollection : e.g. ``"CP:REACTOME"``.
species : msigdbr species name (default ``"Homo sapiens"``).
out_dir : output directory (default: the first folder :func:`msigdb` searches).

Returns
-------
pathlib.Path of the written ``.gmt``.

Examples
--------
>>> from arsc import genesets
>>> f = genesets.export_msigdb_gmt("H")  # doctest: +SKIP
>>> hallmark = genesets.msigdb("H")  # doctest: +SKIP
```

### `arsc.genesets.filter_genesets(gs: 'dict', universe, min_size=1) -> 'dict'`

```text
Restrict gene sets to genes in ``universe`` (e.g. the dataset's features) and drop small sets.

Parameters
----------
gs : dict ``{set name: [genes]}``.
universe : iterable of genes to keep (e.g. ``ds.features("RNA")``).
min_size : drop sets with fewer than this many genes left (default 1).

Returns
-------
dict ``{set name: [genes in universe]}``.

Examples
--------
>>> import arsc
>>> from arsc import genesets
>>> ds = arsc.datasets.toy()
>>> genesets.filter_genesets({"T": ["CD3E", "CD8A", "NOT_A_GENE"]}, ds.features("RNA"))
{'T': ['CD3E', 'CD8A']}
```

### `arsc.genesets.load_v3(name: 'str', min_size=1) -> 'dict'`

```text
Load a named gene-set collection from the folder given by the ``ARSC_DATA_V3`` environment variable.

The names and file layout are listed in ``V3_SETS`` (JSON pathway files, hotspot modules, CellPhoneDB
tables, cell-cycle lists). Raises a RuntimeError that explains the fix when ``ARSC_DATA_V3`` is not set.

Parameters
----------
name : one of ``sorted(arsc.genesets.V3_SETS)``.
min_size : drop sets with fewer genes (default 1).

Returns
-------
dict ``{set name: [genes]}``.

Examples
--------
>>> from arsc import genesets
>>> "pathways" in genesets.V3_SETS
True
>>> gs = genesets.load_v3("pathways")  # doctest: +SKIP
```

### `arsc.genesets.msigdb(collection='H', subcollection=None, species='Homo sapiens', cache_dir=None) -> 'dict'`

```text
MSigDB gene sets from a local GMT cache (no network).

Looks for ``<collection>[.<subcollection>].gmt`` (case-insensitive, ``:`` replaced by ``_``) or any GMT
whose name follows the MSigDB convention (``h.all*``, ``c2.cp.reactome*``) in ``cache_dir``, else in
``$ARSC_MSIGDB_DIR``, ``<cache>/msigdb`` (``$ARSC_CACHE``, ``$AR_ROOT/.cache`` or ``~/.cache/arsc``) and, when
``AR_ROOT`` is set, ``$AR_ROOT/data/msigdb``. Raises FileNotFoundError
listing the searched folders when absent (fill them with :func:`export_msigdb_gmt` or GMTs downloaded from
gsea-msigdb.org). For the R-msigdbr route with a csv cache see :func:`arsc.de.msigdb`.

Parameters
----------
collection : MSigDB collection, e.g. ``"H"`` (hallmark), ``"C2"``, ``"C7"``.
subcollection : e.g. ``"CP:REACTOME"``; ``None`` for the whole collection.
species : kept for API symmetry with :func:`export_msigdb_gmt` (the GMT file decides the species).
cache_dir : search only this directory.

Returns
-------
dict ``{set name: [genes]}``.

Examples
--------
>>> import os, tempfile
>>> from arsc import genesets
>>> d = tempfile.mkdtemp()
>>> _ = open(os.path.join(d, "h.all.v2024.1.Hs.symbols.gmt"), "w").write("HALLMARK_X\tna\tA\tB\n")
>>> genesets.msigdb("H", cache_dir=d)
{'HALLMARK_X': ['A', 'B']}
```

### `arsc.genesets.read_cpdb(cpdb_csv, names_from=None) -> 'dict'`

```text
Read a CellPhoneDB pathway table (columns ``pathway``, ``hgnc_symbol_ids`` comma-separated).

The first row of a duplicated pathway name wins, as in the original R helper.

Parameters
----------
cpdb_csv : path of the pathway CSV.
names_from : optional ``.txt`` file, or directory of ``.txt`` files, listing the pathway names to keep
    (in that order).

Returns
-------
dict ``{pathway: [genes]}``.

Examples
--------
>>> import os, tempfile
>>> import pandas as pd
>>> from arsc import genesets
>>> f = os.path.join(tempfile.mkdtemp(), "cpdb.csv")
>>> pd.DataFrame({"pathway": ["IFN", "TNF"], "hgnc_symbol_ids": ["IFNG,IFNGR1", "TNF,TNFRSF1A"]}).to_csv(f, index=False)
>>> genesets.read_cpdb(f)
{'IFN': ['IFNG', 'IFNGR1'], 'TNF': ['TNF', 'TNFRSF1A']}
```

### `arsc.genesets.read_genesets(path=None, cpdb_csv=None, min_size=1) -> 'dict'`

```text
Read gene sets from any supported file into ``{name: [genes]}`` (Python port of ``ar_read_genesets``).

Dispatches on the path: ``.json`` (:func:`read_json_sets`), ``.gmt`` (:func:`read_gmt`), ``.csv``
(:func:`read_module_csv`), a directory of ``.txt`` files (one set per file, named by the file stem), or a
single ``.txt`` file (one gene per line). Genes are de-duplicated in order and empty entries dropped.

Parameters
----------
path : file or directory (for ``cpdb_csv``: optional list of pathway names to keep, see :func:`read_cpdb`).
cpdb_csv : path of a CellPhoneDB pathway table (then ``path`` restricts the pathways).
min_size : drop sets with fewer genes (default 1).

Returns
-------
dict ``{set name: [genes]}``.

Examples
--------
>>> import os, tempfile
>>> from arsc import genesets
>>> f = os.path.join(tempfile.mkdtemp(), "sets.gmt")
>>> _ = open(f, "w").write("TCELL\tna\tCD3E\tCD8A\nBCELL\tna\tMS4A1\tCD79A\n")
>>> gs = genesets.read_genesets(f, min_size=2)
>>> sorted(gs)
['BCELL', 'TCELL']
```

### `arsc.genesets.read_gmt(path) -> 'dict'`

```text
Read a GMT file (``name <TAB> description <TAB> gene1 <TAB> gene2 ...``, one set per line).

Parameters
----------
path : path of the ``.gmt`` file.

Returns
-------
dict ``{set name: [genes]}``; lines with no genes are skipped.

Examples
--------
>>> import os, tempfile
>>> from arsc import genesets
>>> f = os.path.join(tempfile.mkdtemp(), "sets.gmt")
>>> _ = open(f, "w").write("TCELL\tna\tCD3E\tCD8A\nBCELL\tna\tMS4A1\tCD79A\n")
>>> genesets.read_gmt(f)
{'TCELL': ['CD3E', 'CD8A'], 'BCELL': ['MS4A1', 'CD79A']}
```

### `arsc.genesets.read_json_sets(path) -> 'dict'`

```text
Read gene sets from JSON: MSigDB-style ``{name: {"geneSymbols": [...]}}`` or plain ``{name: [genes]}``.

Parameters
----------
path : path of the ``.json`` file.

Returns
-------
dict ``{set name: [gene symbols]}`` (not de-duplicated; :func:`read_genesets` cleans).

Examples
--------
>>> import json, os, tempfile
>>> from arsc import genesets
>>> f = os.path.join(tempfile.mkdtemp(), "sets.json")
>>> json.dump({"TCELL": {"geneSymbols": ["CD3E", "CD8A"]}, "BCELL": ["MS4A1"]}, open(f, "w"))
>>> genesets.read_json_sets(f)
{'TCELL': ['CD3E', 'CD8A'], 'BCELL': ['MS4A1']}
```

### `arsc.genesets.read_module_csv(path) -> 'dict'`

```text
Read gene modules from a CSV: long (gene + module/set/pathway column) or wide (one column per set).

Long format (e.g. hotspot ``module_genes``): module ``-1`` (unassigned) is dropped and numeric modules are
named ``module<k>``. Wide format: every column is a set (index-like columns are ignored).

Parameters
----------
path : path of the ``.csv`` file.

Returns
-------
dict ``{set name: [genes]}``.

Examples
--------
>>> import os, tempfile
>>> import pandas as pd
>>> from arsc import genesets
>>> f = os.path.join(tempfile.mkdtemp(), "modules.csv")
>>> pd.DataFrame({"gene": ["CD3E", "CD8A", "LYZ"], "module": [1, 1, -1]}).to_csv(f, index=False)
>>> genesets.read_module_csv(f)
{'module1': ['CD3E', 'CD8A']}
```

## arsc.bam

BAM reading, per-cell allele counts, genotype checks.

### `arsc.bam.AlleleCounts(sites: 'pd.DataFrame', barcodes: 'list', counts: 'dict', AD: 'sp.csr_matrix | None', DP: 'sp.csr_matrix | None', OTH: 'sp.csr_matrix | None', stats: 'dict') -> None`

```text
Per-cell allele counts at SNPs (the output of :func:`pileup_cells`).

Attributes
----------
sites : the input sites (rows in input order), with columns chrom, pos (1-based), ref, alt when given.
barcodes : list of cell barcodes (columns of the matrices; ``["pooled"]`` in pooled mode).
counts : dict base -> sparse CSR ``sites x cells`` matrix of UMI counts for ``"A"``, ``"C"``, ``"G"``, ``"T"``,
    ``"N"`` (N = any non-ACGT base).
AD, DP, OTH : CSR ``sites x cells`` (cellsnp-lite layout): ALT UMIs, REF + ALT UMIs, other-allele UMIs; ``None``
    when the sites have no ref / alt columns.
stats : dict of read counters (reads overlapping a site before filters, missing cell tag, barcode not in the
    list, missing UMI tag, bases below the quality cut, UMIs dropped for an allele tie).

Examples
--------
>>> import os, tempfile, arsc
>>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
>>> ac = arsc.bam.pileup_cells(toy["path"], toy["sites"], toy["barcodes"])
>>> int(ac.DP.sum()) == ac.stats["umis"] - int(ac.OTH.sum())
True
```

### `arsc.bam.BamFile(path, index=None, threads: 'int | None' = None, inflater: 'str | None' = None, chunk_mb: 'float' = 32)`

```text
A coordinate-sorted BAM file, read natively (the parts of ``pysam.AlignmentFile`` genotype checks use).

Parameters
----------
path : the ``.bam`` file.
index : the ``.bai`` path (default: ``path + ".bai"``, else ``path`` with ``.bai`` replacing ``.bam``); needed
    only for region queries.
threads : inflating threads (default: arsc's thread setting, ``AR_THREADS``).
inflater : ``None`` (fastest installed) or ``"libdeflate"``, ``"isal"``, ``"zlib"``.
chunk_mb : compressed bytes read per sequential read (default 32 MB; large reads suit hard disks).

Attributes
----------
header_text : the SAM header text.
references, lengths : reference names and lengths.

Examples
--------
>>> import os, tempfile, arsc
>>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
>>> with arsc.bam.BamFile(toy["path"]) as bam:
...     n = bam.count()
>>> n > 0
True
```

### `arsc.bam.discover_snps(bam, *, regions=None, barcodes=None, cell_tag: 'str' = 'CB', min_depth: 'int' = 20, min_alt: 'int' = 4, min_maf: 'float' = 0.1, min_top2: 'float' = 0.0, min_mapq: 'int' = 20, min_baseq: 'int' = 20, min_len: 'int' = 30, exclude_flags: 'int' = 1796, threads: 'int | None' = None, window: 'int' = 8388608, inflater: 'str | None' = None) -> 'pd.DataFrame'`

```text
Find candidate SNPs de novo: pooled pileup of every covered position (cellsnp-lite mode 2 style).

Counts reads (not UMIs: duplicates are excluded by flag, which in Cell Ranger BAMs leaves one read per UMI)
with A / C / G / T / N at every position, over all reads or only those of listed barcodes, and keeps
positions with depth (A + C + G + T) >= ``min_depth`` whose second most frequent base has >= ``min_alt`` reads
and >= ``min_maf`` of the depth. Without a reference genome the alleles are the pooled major (``ref``) and
minor (``alt``) bases, not the genome's reference allele. Feed the result to :func:`pileup_cells`.

Parameters
----------
bam : path of a coordinate-sorted BAM (or an open :class:`BamFile`).
regions : ``None`` (the whole file) or a list of reference names / ``(contig, start, stop)`` regions.
barcodes : count only reads whose ``cell_tag`` is in this list (``None``: all reads).
cell_tag : the cell barcode tag used with ``barcodes``.
min_depth, min_alt, min_maf : site thresholds (defaults 20, 4, 0.1).
min_top2 : the two top bases must hold at least this fraction of the depth (e.g. 0.95; default 0, off).
min_mapq, min_baseq, min_len, exclude_flags : read and base filters as in :func:`pileup_cells`.
threads : inflating threads.
window : positions held in the pileup ring buffer (default 8.4 Mb); bases of a read more than this beyond the
    read start (very long introns) are dropped and counted in ``df.attrs["dropped_bases"]``.
inflater : DEFLATE engine.

Returns
-------
pandas.DataFrame: chrom, pos (1-based), ref (major), alt (minor), A, C, G, T, N, depth, maf.

Examples
--------
>>> import os, tempfile, arsc
>>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
>>> snps = arsc.bam.discover_snps(toy["path"], min_depth=10)
>>> len(snps) > 0
True
```

### `arsc.bam.genotype_concordance(table: 'pd.DataFrame', *, group: 'str' = 'group', min_depth: 'int' = 10, het: 'tuple' = (0.1, 0.9)) -> 'pd.DataFrame'`

```text
Pairwise genotype agreement between samples at shared sites (same donor or not).

Each sample's ALT fraction at a site with depth >= ``min_depth`` is called 0 (< ``het[0]``), 1 (het) or 2
(> ``het[1]``). For each pair: the sites both samples cover, the informative ones among them (at least one
sample not homozygous REF: 0 / 0 sites agree for any two people), the fraction of identical calls on both, the
opposite-homozygous sites (0 vs 2: impossible for one donor short of errors) and the Pearson correlation of
the fractions. Same donor: concordance near 1, no opposite homozygotes, correlation near 1; two unrelated
donors: informative concordance around 0.3-0.5 and correlation near 0.2-0.4.

Parameters
----------
table : long table with ``group``, chrom, pos, AD, DP (see :func:`sample_allele_fractions`).
group : the sample column.
min_depth : minimum REF + ALT UMIs per sample and site.
het : allele-fraction bounds of a heterozygous call.

Returns
-------
pandas.DataFrame, one row per sample pair: sample_a, sample_b, n_shared, n_informative, concordance (all shared
sites), concordance_informative, n_opposite_hom, af_pearson.

Examples
--------
>>> import os, tempfile, arsc
>>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
>>> ac = arsc.bam.pileup_cells(toy["path"], toy["sites"], toy["barcodes"])
>>> agg = ac.aggregate(toy["donor"].str.cat(toy["half"], sep="_"))
>>> pairs = arsc.bam.genotype_concordance(agg, min_depth=5)
>>> same = pairs[pairs.sample_a.str[:2] == pairs.sample_b.str[:2]]
>>> bool((same.concordance_informative > 0.9).all())
True
```

### `arsc.bam.inflater(name: 'str | None' = None) -> 'str'`

```text
Name of the raw-DEFLATE engine BGZF blocks are inflated with.

Parameters
----------
name : ``None`` (the fastest installed: ``"libdeflate"`` from the ``deflate`` wheel, else ``"isal"``, else the
    standard library ``"zlib"``) or one of those names (``KeyError`` if it is not installed).

Returns
-------
str, the engine name.

Examples
--------
>>> import arsc
>>> arsc.bam.inflater("zlib")
'zlib'
```

### `arsc.bam.pileup_cells(bam, sites: 'pd.DataFrame', barcodes=None, *, cell_tag: 'str | None' = 'CB', umi_tag: 'str | None' = 'UB', min_mapq: 'int' = 20, min_baseq: 'int' = 20, min_len: 'int' = 30, exclude_flags: 'int | None' = None, umi_rule: 'str' = 'majority', threads: 'int | None' = None, gap_kb: 'float' = 1024, inflater: 'str | None' = None) -> 'AlleleCounts'`

```text
Count alleles per cell barcode at SNP positions, one count per UMI (cellsnp-lite mode 1 style).

For every read covering a site: the read must pass the flag / MAPQ / aligned-length filters, the base at the
site must have quality >= ``min_baseq``, the site must fall in an M / = / X block of the CIGAR (a site in a
deletion ``D`` or intron ``N`` is not covered; insertions and soft clips shift the read position), the read
must carry ``cell_tag`` with a barcode in ``barcodes`` and ``umi_tag``. Reads are then collapsed per
(site, cell, UMI): ``umi_rule="majority"`` keeps the most frequent base of the UMI's reads (a UMI whose reads
tie between bases is dropped and counted in ``stats["umi_ties"]``), ``"first"`` keeps the first read's base.

Parameters
----------
bam : path of a coordinate-sorted, indexed BAM, or an open :class:`BamFile`.
sites : DataFrame with ``chrom`` and 1-based ``pos`` (VCF convention); optional ``ref`` and ``alt`` give
    AD / DP / OTH. Sites on references absent from the BAM get zero counts.
barcodes : cell barcodes to count (e.g. the filtered barcodes of Cell Ranger). ``None`` pools all reads that
    carry ``cell_tag`` into one column (``"pooled"``), UMIs then keyed by cell barcode + UMI.
cell_tag : cell barcode tag (``None``: ignore barcodes, pool all reads).
umi_tag : UMI tag (``None``: count reads, no UMI collapsing).
min_mapq : minimum MAPQ (default 20, as cellsnp-lite; STAR / Cell Ranger unique reads have 255).
min_baseq : minimum base quality at the site (default 20).
min_len : minimum aligned length (bases in M / = / X; default 30, as cellsnp-lite).
exclude_flags : reads with any of these flag bits are skipped. Default: unmapped, secondary, QC-fail
    (``0x304``) with a UMI tag (duplicates are collapsed by UMI, as cellsnp-lite) and also duplicates
    (``0x704``) without one.
umi_rule : ``"majority"`` (default) or ``"first"``.
threads : inflating threads (default ``AR_THREADS``).
gap_kb : index chunks closer than this many compressed kilobytes are read in one go (default 1 MB: on a hard
    disk reading 1 MB is cheaper than a seek). Dense site lists end up reading the BAM front to back.
inflater : DEFLATE engine (see :func:`inflater`).

Returns
-------
:class:`AlleleCounts` (sparse ``sites x cells`` matrices AD / DP / OTH and per-base counts; ``.long()``,
``.aggregate(groups)``).

Examples
--------
>>> import os, tempfile, arsc
>>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
>>> ac = arsc.bam.pileup_cells(toy["path"], toy["sites"], toy["barcodes"])
>>> ac.AD.shape == (len(toy["sites"]), len(toy["barcodes"]))
True
```

### `arsc.bam.read_bai(path) -> 'list'`

```text
Read a BAM index (``.bai``).

Parameters
----------
path : the ``.bai`` file.

Returns
-------
list with one entry per reference: an object with ``bins`` (dict bin number -> ``(n, 2)`` uint64 array of
chunk virtual offsets ``[begin, end)``, the pseudo-bin 37450 removed) and ``ioffsets`` (linear index, uint64
virtual offset of the first record overlapping each 16 kb window).

Examples
--------
>>> import os, tempfile, arsc
>>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
>>> idx = arsc.bam.read_bai(toy["path"] + ".bai")
>>> len(idx)
2
```

### `arsc.bam.Records(ref_id: 'np.ndarray', pos: 'np.ndarray', end: 'np.ndarray', mapq: 'np.ndarray', flag: 'np.ndarray', next_ref_id: 'np.ndarray', next_pos: 'np.ndarray', tlen: 'np.ndarray', name: 'np.ndarray', cigar_ops: 'np.ndarray', cigar_lens: 'np.ndarray', cigar_off: 'np.ndarray', seq: 'np.ndarray', qual: 'np.ndarray', seq_off: 'np.ndarray', tags: 'dict' = <factory>) -> None`

```text
Columnar BAM records (what pysam's ``AlignedSegment`` fields hold, for many reads at once).

Attributes
----------
ref_id, pos, end : int32 arrays: reference index (-1 unplaced), 0-based leftmost position, 0-based exclusive
    end on the reference (``pos + 1`` for unmapped reads or reads without reference-consuming CIGAR, as htslib's
    ``bam_endpos``).
mapq, flag : uint8 / uint16 arrays.
next_ref_id, next_pos, tlen : int32 arrays (mate reference, mate position, template length).
name : object array of read names.
cigar_ops, cigar_lens, cigar_off : flat CIGAR (op codes 0-8 = ``MIDNSHP=X``, lengths) and per-record offsets
    (record i owns ``[cigar_off[i], cigar_off[i+1])``).
seq, qual, seq_off : flat ASCII sequence and base qualities (uint8, Phred; 255 = absent) with per-record offsets.
tags : dict tag -> array: int64 when every record has the tag as an integer, else an object array (str for
    Z / H / A tags, int, float, list for B arrays; ``None`` where the tag is absent).

Examples
--------
>>> import os, tempfile, arsc
>>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
>>> recs = arsc.bam.BamFile(toy["path"]).fetch("chr2", tags=("CB",))
>>> len(recs) == len(recs.tags["CB"])
True
```

### `arsc.bam.sample_allele_fractions(table: 'pd.DataFrame', *, group: 'str' = 'group', min_depth: 'int' = 10) -> 'pd.DataFrame'`

```text
Wide table of ALT allele fractions (sites x samples), NaN where a sample's depth is below ``min_depth``.

Parameters
----------
table : long table with columns ``group`` (sample), chrom, pos, AD, DP (e.g. :meth:`AlleleCounts.aggregate`,
    or several samples' pooled :func:`pileup_cells` results concatenated with a group column).
group : the sample column.
min_depth : minimum REF + ALT UMIs for a sample's fraction to count.

Returns
-------
pandas.DataFrame indexed by ``chrom:pos``, one column per sample.

Examples
--------
>>> import os, tempfile, arsc
>>> toy = arsc.datasets.toy_bam(os.path.join(tempfile.mkdtemp(), "toy.bam"))
>>> ac = arsc.bam.pileup_cells(toy["path"], toy["sites"], toy["barcodes"])
>>> af = arsc.bam.sample_allele_fractions(ac.aggregate(toy["donor"]), min_depth=5)
>>> list(af.columns)
['D1', 'D2', 'D3']
```

### `arsc.bam.write_bam(path, references, records, *, header_text: 'str | None' = None, level: 'int' = 6, block_bytes: 'int' = 4096, index: 'bool' = True) -> 'str'`

```text
Write a small coordinate-sorted BAM (and its ``.bai``) from Python dictionaries.

Meant for tests, documentation and synthetic data, not for large outputs.

Parameters
----------
path : output ``.bam`` path.
references : list of ``(name, length)``.
records : iterable of dicts sorted by (reference, pos) with keys name, flag, chrom, pos (0-based), mapq,
    cigar (SAM string), seq, qual (list of Phred values, optional), next_chrom, next_pos, tlen and tags (dict
    tag -> str / int / float; a one-letter ``RE`` value is written as type ``A``). Unplaced records
    (``chrom=None``) go last.
header_text : SAM header text (default: ``@HD`` + one ``@SQ`` line per reference).
level : zlib compression level.
block_bytes : uncompressed bytes per BGZF block (small blocks exercise multi-block reading).
index : also write ``path + ".bai"``.

Returns
-------
str, the BAM path.

Examples
--------
>>> import os, tempfile, arsc
>>> p = os.path.join(tempfile.mkdtemp(), "one.bam")
>>> rec = dict(name="r1", chrom="chr1", pos=99, mapq=255, cigar="10M", seq="ACGTACGTAC", tags={"CB": "AAAC-1"})
>>> _ = arsc.bam.write_bam(p, [("chr1", 1000)], [rec])
>>> arsc.bam.BamFile(p).fetch("chr1", 100, 101).tags["CB"].tolist()
['AAAC-1']
```

## arsc.seurat_extra

Seurat parity.

### `arsc.seurat_extra.add_metadata(obs: 'pd.DataFrame', metadata, col_name: 'str | None' = None) -> 'pd.DataFrame'`

```text
Seurat ``AddMetaData``: align ``metadata`` (DataFrame / Series indexed by cell barcode, or a dict) to
``obs`` by index and return a new ``obs`` with the columns added (missing cells -> NA). Existing columns are
overwritten, like Seurat.

Parameters
----------
obs : cell metadata.
metadata : DataFrame / Series indexed by barcode, or a dict of columns.
col_name : column name for a Series or single-column frame.

Returns
-------
DataFrame, a new ``obs`` with the added columns.

Examples
--------
>>> import arsc, pandas as pd
>>> ds = arsc.datasets.toy()
>>> obs = arsc.add_metadata(ds.obs, pd.Series(1.0, index=ds.cells[:10]), col_name="flag")
>>> int(obs["flag"].notna().sum())
10
```

### `arsc.seurat_extra.average_expression(X, obs: 'pd.DataFrame | None' = None, group_by=None, *, features=None, layer: 'str' = 'data', method: 'str' = 'average', transpose: 'bool' = False, sep: 'str' = '_') -> 'pd.DataFrame'`

```text
Seurat ``AverageExpression`` (``method='average'``) / ``AggregateExpression``-style sums (``'aggregate'``).

``layer='data'`` averages ``expm1(data)`` (non-log values, as Seurat); ``'counts'`` / ``'scale.data'`` use
the values as given. ``group_by`` is a column name, list of columns (labels joined with ``sep``, first
column varying slowest -- Seurat's ``"a_b"`` naming) or a per-cell label vector. Cells with NA in a
grouping variable are dropped; grouping variables with a single level are ignored (Seurat); if none
remain the single group ``"all"`` is returned.

Parameters
----------
X : cells x features matrix.
obs : cell metadata (needed when ``group_by`` names columns).
group_by : column name, list of column names, or per-cell labels.
features : feature names.
layer : ``"data"`` (average ``expm1``), ``"counts"`` or ``"scale.data"``.
method : ``"average"`` or ``"aggregate"`` (sums).
transpose : return groups x features.
sep : separator of combined group labels.

Returns
-------
DataFrame, features x groups (Seurat orientation); ``transpose=True`` gives groups x features.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> av = arsc.average_expression(L, ds.obs, "cell_type", features=ds.features("RNA"))
>>> av.shape
(300, 5)
>>> av.loc["MS4A1"].idxmax()
'B'
```

### `arsc.seurat_extra.cell_cycle_scoring(X, features, s_features=None, g2m_features=None, *, ctrl: 'int | None' = None, nbin: 'int' = 24, seed: 'int' = 1, cells=None, pool=None) -> 'pd.DataFrame'`

```text
Seurat ``CellCycleScoring``: ``S.Score`` / ``G2M.Score`` (``AddModuleScore``) and ``Phase``.

``X`` is the log-normalized (cells x features) matrix; default gene sets are ``Seurat::cc.genes``.
``ctrl`` defaults to the size of the smaller input set (as Seurat). Phase: both scores < 0 -> ``G1``;
equal maxima -> ``Undecided``; else the larger of ``S`` / ``G2M``.
``seed`` (default 1, as Seurat) is the ``AddModuleScore`` R seed; both scores share one RNG stream.

Parameters
----------
X : cells x features log-normalized matrix.
features : feature names.
s_features, g2m_features : S and G2M gene lists (default ``cc_genes``).
ctrl : control genes per signature gene (default: the smaller set size, as Seurat).
nbin : expression bins.
seed : R seed.
cells : index of the output.
pool : control-gene pool.

Returns
-------
DataFrame with ``S.Score``, ``G2M.Score`` and ``Phase`` (``G1`` / ``S`` / ``G2M`` / ``Undecided``).

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> cc = arsc.cell_cycle_scoring(L, ds.features("RNA"), ctrl=5, cells=ds.cells)
>>> list(cc.columns)
['S.Score', 'G2M.Score', 'Phase']
>>> set(cc.Phase) <= {"G1", "S", "G2M", "Undecided"}
True
```

### `arsc.seurat_extra.find_all_markers(X, groups, *, features=None, test_use: 'str' = 'wilcox', logfc_threshold: 'float' = 0.25, min_pct: 'float' = 0.1, min_diff_pct: 'float' = -inf, only_pos: 'bool' = False, return_thresh: 'float' = 0.01, max_cells_per_ident: 'float' = inf, random_seed: 'int' = 1, latent_vars=None, min_cells_group: 'int' = 3, pseudocount_use: 'float' = 1.0, base: 'float' = 2.0, norm_method: 'str | None' = 'LogNormalize', slot: 'str' = 'data', levels=None) -> 'pd.DataFrame'`

```text
Seurat ``FindAllMarkers``: each identity vs all other cells.

Clusters are visited in ``levels`` order (default: categorical order, else sorted). Note that
``Idents(obj) <- "meta_column"`` in Seurat creates the factor with levels in order of FIRST APPEARANCE in the
object, so pass ``levels=pd.unique(labels)`` to reproduce the row order of such an R run.

Returns the concatenation of per-cluster :func:`find_markers` tables, keeping ``p_val < return_thresh``
(raw p-value, as Seurat), with ``cluster`` and ``gene`` columns. Clusters with fewer than
``min_cells_group`` cells are skipped (Seurat reports an error message and continues).
``random_seed`` is passed unchanged to every per-cluster :func:`find_markers` call (as Seurat).
For ``test_use="wilcox"`` without cell downsampling every cluster is tested from a single rank pass.

Parameters
----------
X : cells x features matrix (Seurat ``data`` slot).
groups : identity per cell (e.g. cluster labels).
features : feature names for ``X`` columns.
test_use, logfc_threshold, min_pct, min_diff_pct, only_pos, max_cells_per_ident, random_seed, latent_vars,
    min_cells_group, pseudocount_use, base, norm_method, slot : as in :func:`find_markers`.
return_thresh : keep rows with raw ``p_val < return_thresh``.
levels : order in which identities are visited (and reported).

Returns
-------
DataFrame with the :func:`find_markers` columns plus ``cluster`` and ``gene``.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> allm = arsc.find_all_markers(L, ds.obs["cell_type"], features=ds.features("RNA"), only_pos=True)
>>> sorted(allm.cluster.unique())
['B', 'CD4 T', 'CD8 T', 'Mono', 'NK']
>>> allm.groupby("cluster", observed=True).head(1).set_index("cluster").gene["Mono"] in ("LYZ", "CD14", "S100A8", "FCGR3A")
True
```

### `arsc.seurat_extra.find_markers(X, groups=None, ident_1=None, ident_2=None, *, features=None, cells_1=None, cells_2=None, test_features=None, test_use: 'str' = 'wilcox', logfc_threshold: 'float' = 0.25, min_pct: 'float' = 0.1, min_diff_pct: 'float' = -inf, only_pos: 'bool' = False, max_cells_per_ident: 'float' = inf, random_seed: 'int' = 1, latent_vars: 'pd.DataFrame | None' = None, min_cells_group: 'int' = 3, pseudocount_use: 'float' = 1.0, base: 'float' = 2.0, norm_method: 'str | None' = 'LogNormalize', slot: 'str' = 'data', n_features_total: 'int | None' = None) -> 'pd.DataFrame'`

```text
Seurat ``FindMarkers`` (``ident.1`` vs ``ident.2`` or vs all other cells).

Parameters
----------
X : cells x features matrix (Seurat ``data`` slot: log-normalized RNA, CLR ADT, ...).
groups, ident_1, ident_2 : identity per cell and the two identities (``ident_1`` may be a list;
    ``ident_2=None`` = all other cells).
cells_1, cells_2 : alternatively, boolean masks of the two groups (``cells_2=None`` = all other cells).
features : feature names for ``X`` columns.
test_features : restricts the tested set (Seurat ``features=``).
test_use : ``"wilcox"`` (limma rank-sum, as Seurat), ``"LR"`` (needs ``latent_vars``), ``"t"``.
    (``MAST``, ``DESeq2``, ``negbinom``, ``poisson``, ``roc``, ``bimod`` are not ported; use ``arsc.pseudobulk``
    + an external DE tool for sample-level DESeq2.)
norm_method : ``"LogNormalize"`` => fold change from ``log(mean(expm1(x)) + 1)``; anything else (``"CLR"``)
    => ``log(mean(x) + 1)`` (what Seurat does for ADT).
slot : what ``X`` holds: ``"data"`` (default), ``"counts"`` or ``"scale.data"``.
logfc_threshold, min_pct, min_diff_pct, only_pos : Seurat's gene pre-filters.
max_cells_per_ident : downsample each group to at most this many cells.
latent_vars : per-cell covariates (DataFrame) for ``test_use="LR"``.
min_cells_group : minimum cells per group.
pseudocount_use, base : fold-change pseudocount and log base.
n_features_total : number of features in the assay (Bonferroni multiplier; default ``X.shape[1]``).
random_seed : R ``set.seed`` value (default 1 = Seurat ``random.seed = 1``) for the
    ``max_cells_per_ident`` downsampling, drawn with the R-compatible RNG (identical cells to R).
    Unused when ``max_cells_per_ident`` is infinite (the default).

Cell-level p-values rank markers; they are not evidence for a difference between conditions
(for that, pseudobulk per patient and use :func:`arsc.de.voom` / :func:`arsc.de.deseq2`).

Returns
-------
DataFrame indexed by feature: ``p_val, avg_log2FC, pct.1, pct.2, p_val_adj``, ordered by ``p_val`` with
ties in feature order, like Seurat's ``FindMarkers`` (``find_all_markers`` re-sorts by ``(p_val, -avg_log2FC)``
like ``FindAllMarkers``).

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> mk = arsc.find_markers(L, ds.obs["cell_type"], ident_1="B", features=ds.features("RNA"), only_pos=True)
>>> list(mk.columns)
['p_val', 'avg_log2FC', 'pct.1', 'pct.2', 'p_val_adj']
>>> mk.index[0] in ("MS4A1", "CD79A", "CD19")
True
```

### `arsc.seurat_extra.fold_change(X, mask1, mask2, *, norm_method: 'str | None' = 'LogNormalize', slot: 'str' = 'data', pseudocount: 'float' = 1.0, base: 'float' = 2.0, features=None) -> 'pd.DataFrame'`

```text
Seurat ``FoldChange``: ``avg_log{base}FC`` (group 1 - group 2), ``pct.1``, ``pct.2``.

``slot='data'`` with ``norm_method='LogNormalize'`` averages ``expm1(x)`` before the log (Seurat's default
for log-normalized data); any other ``norm_method`` (e.g. ``'CLR'``) or ``slot='counts'`` averages the raw
values; ``slot='scale.data'`` returns the plain mean difference (``avg_diff``). ``pct`` is rounded to 3
digits like Seurat.

Parameters
----------
X : cells x features matrix (``slot`` says what it holds).
mask1, mask2 : boolean masks of the two cell groups.
norm_method : ``"LogNormalize"`` (average ``expm1(x)``) or another method name (average ``x``).
slot : ``"data"``, ``"counts"`` or ``"scale.data"``.
pseudocount : added before the log.
base : log base (2 -> ``avg_log2FC``).
features : feature names (index of the result).

Returns
-------
DataFrame indexed by feature: ``avg_log2FC`` (or ``avg_diff`` / ``avg_log{base}FC``), ``pct.1``, ``pct.2``.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> b = (ds.obs["cell_type"] == "B").to_numpy()
>>> fc = arsc.fold_change(L, b, ~b, features=ds.features("RNA"))
>>> list(fc.columns), fc["avg_log2FC"].idxmax() in ("MS4A1", "CD79A", "CD19")
(['avg_log2FC', 'pct.1', 'pct.2'], True)
```

### `arsc.seurat_extra.normalize_data(X, method: 'str' = 'LogNormalize', scale_factor: 'float' = 10000.0, margin: 'int' = 1)`

```text
Seurat ``NormalizeData``: ``LogNormalize``, ``CLR`` or ``RC`` (relative counts * ``scale_factor``).

``margin`` follows Seurat's CLR convention: ``margin=1`` normalizes each *feature* across cells (Seurat's
default), ``margin=2`` each *cell* across features (the recommended setting for ADT / HTO).

Parameters
----------
X : cells x features counts.
method : ``"LogNormalize"``, ``"CLR"`` or ``"RC"``.
scale_factor : scale factor (LogNormalize, RC).
margin : CLR margin (see above).

Returns
-------
scipy.sparse.csr_matrix, cells x features.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> adt = arsc.normalize_data(ds.X("ADT", "counts"), method="CLR", margin=2)
>>> adt.shape
(3000, 6)
```

### `arsc.seurat_extra.rename_idents(labels, mapping: 'dict') -> 'pd.Series'`

```text
Seurat ``RenameIdents``: map identity labels (unmapped labels kept; several old levels may merge).

Parameters
----------
labels : per-cell labels.
mapping : ``{old: new}``.

Returns
-------
pandas.Series (categorical) of renamed labels.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> lab = arsc.rename_idents(ds.obs["cell_type"], {"CD8 T": "T", "CD4 T": "T"})
>>> sorted(lab.cat.categories)
['B', 'Mono', 'NK', 'T']
```

### `arsc.seurat_extra.scale_data(X, features=None, *, select=None, vars_to_regress=None, latent_data: 'pd.DataFrame | None' = None, split_by=None, model_use: 'str' = 'linear', do_scale: 'bool' = True, do_center: 'bool' = True, scale_max: 'float' = 10.0, block_size: 'int' = 1000, dtype=<class 'numpy.float64'>)`

```text
Seurat ``ScaleData`` for dense output (cells x selected features).

Parameters
----------
X : cells x features log-normalized matrix.
features : feature names of ``X``.
select : names to scale (Seurat ``features=``; default all; unknown names are ignored).
vars_to_regress : columns of ``latent_data`` and / or feature names in ``X`` regressed out per gene with
    ``lm`` (``model_use='linear'``; residuals are then centered/scaled). ``poisson`` / ``negbinom`` are
    not ported.
latent_data : per-cell covariates (row-aligned with ``X``).
split_by : per-cell labels; regression and scaling are done separately within each split.
model_use : only ``"linear"``.
do_scale, do_center : divide by s.d. / subtract the mean.
scale_max : upper clip.
block_size : genes per block.
dtype : output dtype.

Returns
-------
tuple ``(Z, feature_names)``: dense cells x selected-features array (clipping at ``scale_max`` is
upper-tail only, like Seurat) and the names of its columns.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> L = arsc.pp.log_normalize(ds.X("RNA", "counts"))
>>> obs = ds.obs.assign(nUMI=ds.X("RNA", "counts").sum(axis=1).A1)
>>> Z, names = arsc.scale_data(L, ds.features("RNA"), select=["CD8A", "GZMK"],
...                            vars_to_regress=["nUMI"], latent_data=obs)
>>> Z.shape, list(names)
((3000, 2), ['CD8A', 'GZMK'])
```

### `arsc.seurat_extra.split_object(ds, by: 'str') -> 'dict'`

```text
Seurat ``SplitObject``: ``{level: Dataset}`` split by an ``obs`` column (uses ``Dataset.subset``).

Parameters
----------
ds : arsc.Dataset or arsc.Store.
by : ``obs`` column to split by.

Returns
-------
dict ``{level: Dataset}`` (levels without cells are omitted).

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> parts = arsc.split_object(ds, "timepoint")
>>> list(parts), parts["pre"].n_cells + parts["post"].n_cells
(['pre', 'post'], 3000)
```

### `arsc.seurat_extra.which_cells(obs: 'pd.DataFrame', expression: 'str | None' = None, *, idents=None, ident_col: 'str | None' = None, cells=None, invert: 'bool' = False) -> 'np.ndarray'`

```text
Seurat ``WhichCells`` / the row selection of ``subset``: boolean mask over ``obs``.

``expression`` is a pandas ``eval`` string on ``obs`` columns (e.g. ``"timepoint == 'pre' and
nCount_RNA > 1000"``; use backticks for names with dots, e.g. ``"`percent.mt` < 10"``). ``idents`` selects
values of ``ident_col``; ``cells`` restricts to barcodes. Conditions are AND-ed.

Parameters
----------
obs : cell metadata.
expression : pandas ``eval`` condition on ``obs`` columns.
idents : identities to keep (values of ``ident_col``).
ident_col : column holding the identities.
cells : barcodes to keep.
invert : return the complement.

Returns
-------
numpy.ndarray of bool, one per row of ``obs``.

Examples
--------
>>> import arsc
>>> ds = arsc.datasets.toy()
>>> m = arsc.which_cells(ds.obs, "timepoint == 'post'", idents=["CD8 T"], ident_col="cell_type")
>>> sub = ds.subset(m)
>>> sorted(sub.obs.cell_type.unique())
['CD8 T']
```

## arsc.scanpy_extra

scanpy parity.

### `arsc.scanpy_extra.dendrogram(rep, groups, *, cor_method: 'str' = 'pearson', linkage_method: 'str' = 'complete', optimal_ordering: 'bool' = False) -> 'dict'`

```text
``sc.tl.dendrogram``: hierarchical clustering of group means of a representation (PCA by default).

Parameters
----------
rep : cells x d array (e.g. ``X_pca``) or a cells x genes DataFrame (``var_names`` mode).
groups : label per cell.
cor_method : correlation between group means: ``"pearson"`` (default), ``"spearman"``, ``"kendall"``.
linkage_method : scipy linkage method (default ``"complete"``).
optimal_ordering : scipy optimal leaf ordering.

Returns
-------
dict with ``linkage``, ``categories_ordered``, ``categories_idx_ordered``, ``correlation_matrix`` and
``dendrogram_info`` (scipy ``dendrogram`` output, no plotting).

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> Z, _, _ = arsc.pp.scale(L.toarray())
>>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
>>> nn = sx.neighbors(pcs, n_neighbors=15)
>>> dn = sx.dendrogram(pcs, ds.obs["cell_type"])
>>> sorted(dn["categories_ordered"])
['B', 'CD4 T', 'CD8 T', 'Mono', 'NK']
```

### `arsc.scanpy_extra.diffmap(connectivities, n_comps: 'int' = 15, *, random_state: 'int' = 0) -> 'dict'`

```text
``sc.tl.diffmap`` on a scanpy connectivities graph (symmetric ``transitions_sym`` eigendecomposition).

Parameters
----------
connectivities : cells x cells graph from :func:`neighbors`.
n_comps : number of diffusion components (default 15).
random_state : integer seed (default 0) for the ARPACK start vector (local ``RandomState``).

Returns
-------
dict with ``evals`` (n_comps), ``X_diffmap`` (cells x n_comps; column 0 is the stationary component, as
scanpy) and the transition matrix pieces (``Tsym``, ``q``) needed by :func:`dpt`.

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> Z, _, _ = arsc.pp.scale(L.toarray())
>>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
>>> nn = sx.neighbors(pcs, n_neighbors=15)
>>> dm = sx.diffmap(nn["connectivities"], n_comps=10)
>>> dm["X_diffmap"].shape
(600, 10)
```

### `arsc.scanpy_extra.dpt(dm: 'dict', root: 'int', n_dcs: 'int' = 10, *, thresh: 'float' = 0.9994) -> 'np.ndarray'`

```text
``sc.tl.dpt`` diffusion pseudotime (``n_branchings=0``) from :func:`diffmap` output.

Distance to the root in diffusion space (Haghverdi 2016 with the Wolf 2019 stationary-state term): sum over
the first ``n_dcs`` components of ``(l / (1 - l) (psi_i - psi_root))^2`` for l < ``thresh``, plus the plain
squared difference for l >= ``thresh``; then sqrt and divide by the maximum. A connected graph is assumed.

Parameters
----------
dm : output of :func:`diffmap`.
root : index (position) of the root cell.
n_dcs : number of diffusion components used (default 10).
thresh : eigenvalue threshold of the stationary-state term (default 0.9994).

Returns
-------
numpy array, pseudotime per cell in [0, 1] (0 at the root).

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> Z, _, _ = arsc.pp.scale(L.toarray())
>>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
>>> nn = sx.neighbors(pcs, n_neighbors=15)
>>> dm = sx.diffmap(nn["connectivities"], n_comps=10)
>>> t = sx.dpt(dm, root=0)
>>> float(t[0])
0.0
```

### `arsc.scanpy_extra.highly_variable_genes(X, features=None, *, flavor: 'str' = 'seurat', n_top_genes: 'int | None' = None, min_mean: 'float' = 0.0125, max_mean: 'float' = 3.0, min_disp: 'float' = 0.5, max_disp: 'float' = inf, n_bins: 'int' = 20, span: 'float' = 0.3, log1p_base: 'float | None' = None, batch_key=None) -> 'pd.DataFrame'`

```text
``sc.pp.highly_variable_genes`` for ``flavor`` in ``seurat`` | ``cell_ranger`` | ``seurat_v3``.

seurat / cell_ranger: ``X`` = log1p-normalized cells x genes; columns ``means, dispersions,
dispersions_norm, highly_variable`` (exactly scanpy's). With ``n_top_genes`` the mean/dispersion cutoffs
are ignored. seurat_v3: ``X`` = RAW counts; delegates to :func:`arsc.pp.hvg_vst` (Seurat vst, R-validated;
scanpy's skmisc loess is the same model); columns ``means, variances, variances_norm,
highly_variable_rank, highly_variable_nbatches, highly_variable``.

Parameters
----------
X : cells x genes (log1p-normalized for seurat/cell_ranger; raw counts for seurat_v3).
features : gene names (index of the result).
flavor : ``"seurat"`` (default), ``"cell_ranger"``, ``"seurat_v3"`` (or ``"seurat_v3_paper"``).
n_top_genes : number of genes to flag (required for seurat_v3).
min_mean, max_mean, min_disp, max_disp : cutoffs for seurat/cell_ranger when ``n_top_genes`` is None.
n_bins : number of mean bins for dispersion normalization (default 20).
span : loess span for seurat_v3 (default 0.3).
log1p_base : base used for log1p, if not natural.
batch_key : array of batch labels (seurat_v3 only): per-batch ranking combined by median rank, as scanpy.

Returns
-------
DataFrame indexed by gene with the scanpy columns listed above.

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> hv = sx.highly_variable_genes(L, genes, n_top_genes=50)
>>> int(hv["highly_variable"].sum())
50
>>> v3 = sx.highly_variable_genes(ds.X("RNA", "counts"), genes, flavor="seurat_v3", n_top_genes=50)
>>> "variances_norm" in v3.columns
True
```

### `arsc.scanpy_extra.leiden(connectivities, resolution: 'float' = 1.0, *, random_state: 'int' = 0, n_iterations: 'int' = -1, directed: 'bool' = True, use_weights: 'bool' = True, flavor: 'str' = 'leidenalg', engine: 'str' = 'scanpy') -> 'np.ndarray'`

```text
``sc.tl.leiden`` on a connectivities graph (scanpy 1.11 defaults); same graph + seed -> identical labels.

Default settings: leidenalg, directed graph, ``n_iterations=-1``, ``RBConfigurationVertexPartition``.
Labels are integers exactly as leidenalg numbers them (scanpy stores them as strings). For
``flavor="igraph"`` a private ``numpy.random.RandomState(random_state)`` is installed as igraph's RNG for
the call, as scanpy does (same seed -> scanpy's labels), and igraph's default RNG restored afterwards (Python's global ``random`` state is never touched).

Parameters
----------
connectivities : cells x cells weighted graph (``neighbors(...)["connectivities"]``).
resolution : resolution parameter (default 1.0; higher = more clusters).
random_state : integer seed (default 0).
n_iterations : Leiden iterations (-1 = until stable, both flavors, as scanpy; scanpy recommends 2 with
    ``flavor="igraph"``).
directed : build a directed igraph graph (scanpy default True for leidenalg).
use_weights : use edge weights.
flavor : ``"leidenalg"`` (default) or ``"igraph"`` (modularity, undirected).
engine : ``"scanpy"`` (default; leidenalg / igraph called as scanpy calls them: same graph + seed ->
    scanpy's labels) or ``"arsc"`` (``flavor="igraph"`` only: arsc's own Leiden with igraph's
    refinement, :func:`arsc.cluster` ``refinement="igraph"``; about 4x faster at 100,000 cells, a seed
    gives a different but equally close partition: on 100,000 PBMCs its ARI against scanpy's clustering
    equals scanpy's own seed-to-seed ARI, ``SCANPY_PARITY.md``). Labels are numbered by decreasing
    cluster size.

Returns
-------
numpy integer array of cluster labels, one per cell.

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> Z, _, _ = arsc.pp.scale(L.toarray())
>>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
>>> nn = sx.neighbors(pcs, n_neighbors=15)
>>> labels = sx.leiden(nn["connectivities"], resolution=0.5)
>>> labels.shape
(600,)
```

### `arsc.scanpy_extra.magic_impute(X, T, n_steps: 'int' = 3, clip_threshold: 'float' = 0.01, dtype=<class 'numpy.float32'>, chunk: 'int' = 500)`

```text
``palantir.utils.run_magic_imputation``: ``T^n_steps @ X``, values below ``clip_threshold`` set to 0.

Applies ``T`` ``n_steps`` times to dense gene chunks instead of forming the denser ``T^n_steps`` (same
result up to float32 rounding, ~1e-6 relative). Imputed values are smoothed over neighbors: use them for
visualization, not as independent observations in tests.

Parameters
----------
X : cells x genes (sparse or dense), e.g. log-normalized data of the genes of interest.
T : cells x cells row-stochastic transition matrix (``palantir_diffusion_maps(...)["T"]``).
n_steps : diffusion steps (default 3).
clip_threshold : values below this are set to 0 (default 0.01).
dtype : computation dtype (default float32).
chunk : genes per dense chunk (default 500).

Returns
-------
scipy.sparse.csr_matrix, cells x genes.

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> Z, _, _ = arsc.pp.scale(L.toarray())
>>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
>>> nn = sx.neighbors(pcs, n_neighbors=15)
>>> dm = sx.palantir_diffusion_maps(pcs, n_components=10)
>>> sx.magic_impute(L[:, :20], dm["T"]).shape
(600, 20)
```

### `arsc.scanpy_extra.neighbors(emb, n_neighbors: 'int' = 15, *, metric: 'str' = 'euclidean', method: 'str' = 'auto', random_state: 'int' = 0) -> 'dict'`

```text
``sc.pp.neighbors`` (``method="umap"``): kNN + UMAP fuzzy-simplicial-set connectivities.

Uses :func:`arsc.knn` with scanpy's switch (``method="auto"``: exact up to 4,096 cells, arsc's approximate
search above; ``"exact"`` forces exact search). As
scanpy, ``n_neighbors`` counts the cell itself (n_neighbors - 1 real neighbors).

Parameters
----------
emb : cells x d embedding (e.g. PCA or Harmony).
n_neighbors : neighborhood size including the cell itself (default 15).
metric : ``"euclidean"`` or ``"cosine"``.
method : ``"auto"`` (default), ``"exact"`` or ``"approximate"`` (see :func:`arsc.knn`).
random_state : integer seed (default 0) of the approximate search's partition (unused by exact search).

Returns
-------
dict ``{"connectivities", "distances", "indices"}`` (sparse cells x cells graphs and the kNN index array).

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> Z, _, _ = arsc.pp.scale(L.toarray())
>>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
>>> nn = sx.neighbors(pcs, n_neighbors=15)
>>> sorted(nn)
['connectivities', 'distances', 'indices']
```

### `arsc.scanpy_extra.normalize_total_log1p(counts, target_sum: 'float' = 10000.0, dtype=<class 'numpy.float64'>) -> 'sp.csr_matrix'`

```text
``sc.pp.normalize_total(target_sum)`` followed by ``sc.pp.log1p`` (bitwise equal to scanpy).

Differs from :func:`arsc.pp.log_normalize` (Seurat: ``x / s * sf``) only at the 1-ulp level, but that
changes tie structure, which matters for rank tests (Wilcoxon): use this one to reproduce scanpy
``rank_genes_groups`` p-values exactly. Also equals ``sc.pp.normalize_per_cell(counts_per_cell_after=)``
+ log1p. Cells with zero counts stay zero.

Parameters
----------
counts : cells x genes raw counts (sparse or dense).
target_sum : counts per cell after scaling (default 1e4).
dtype : output dtype (default float64).

Returns
-------
scipy.sparse.csr_matrix of log1p-normalized values, cells x genes.

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> L.shape
(600, 300)
```

### `arsc.scanpy_extra.palantir_diffusion_maps(emb, n_components: 'int' = 10, knn: 'int' = 30, alpha: 'float' = 0.0, seed: 'int' = 0, method: 'str' = 'auto') -> 'dict'`

```text
``palantir.utils.run_diffusion_maps`` (adaptive anisotropic kernel, Setty 2019).

kNN (self excluded, ``knn - 1`` neighbors as scanpy's distance matrix), adaptive bandwidth = distance to
the ``floor(knn / 3)``-th neighbor, kernel ``exp(-d / sigma_i)`` symmetrized as ``W + W^T``, row-normalized
transition ``T``, ``eigs`` with palantir's tolerance and start vector (``default_rng(seed).random``). Exact
kNN here vs scanpy's approximate kNN for > 4,096 cells in palantir: identical below that size.

Parameters
----------
emb : cells x d embedding (e.g. PCA / Harmony).
n_components : number of eigenvectors (default 10).
knn : neighborhood size (default 30).
alpha : anisotropy (default 0).
seed : integer seed (default 0, as palantir) for the ARPACK start vector and the kNN partition.
method : kNN method, ``"auto"`` (default: exact up to 4,096 cells), ``"exact"`` or ``"approximate"``.

Returns
-------
dict with ``kernel``, ``T`` (sparse cells x cells), ``EigenVectors`` (cells x n_components, unit norm) and
``EigenValues`` (descending).

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> Z, _, _ = arsc.pp.scale(L.toarray())
>>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
>>> nn = sx.neighbors(pcs, n_neighbors=15)
>>> dm = sx.palantir_diffusion_maps(pcs, n_components=10)
>>> dm["EigenVectors"].shape
(600, 10)
```

### `arsc.scanpy_extra.palantir_multiscale_space(dm: 'dict', n_eigs: 'int | None' = None) -> 'np.ndarray'`

```text
``palantir.utils.determine_multiscale_space``: eigenvectors 1..n_eigs-1 scaled by ``l / (1 - l)``.

Parameters
----------
dm : output of :func:`palantir_diffusion_maps`.
n_eigs : number of eigenvectors to use (default: palantir's eigen-gap rule, at least 3).

Returns
-------
numpy array, cells x (n_eigs - 1) multiscale space.

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> Z, _, _ = arsc.pp.scale(L.toarray())
>>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
>>> nn = sx.neighbors(pcs, n_neighbors=15)
>>> dm = sx.palantir_diffusion_maps(pcs, n_components=10)
>>> sx.palantir_multiscale_space(dm, n_eigs=4).shape
(600, 3)
```

### `arsc.scanpy_extra.qc_metrics(counts, features=None, qc_vars: 'dict | None' = None, percent_top=(20,), log1p: 'bool' = True) -> 'pd.DataFrame'`

```text
``sc.pp.calculate_qc_metrics`` (cell-level columns) on a raw-counts matrix.

Parameters
----------
counts : cells x genes raw counts.
features : gene names (needed when ``qc_vars`` values are gene lists).
qc_vars : ``{"mt": boolean mask or list of genes, ...}``.
percent_top : top-N gene counts for ``pct_counts_in_top_<N>_genes`` (default ``(20,)``).
log1p : add ``log1p_`` versions (default True).

Returns
-------
DataFrame, one row per cell: ``n_genes_by_counts, total_counts, pct_counts_in_top_<n>_genes,
total_counts_<v>, pct_counts_<v>`` (+ ``log1p_*``), as scanpy.

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> qc = sx.qc_metrics(ds.X("RNA", "counts"), genes, qc_vars={"mt": genes.str.startswith("MT-")})
>>> keep = (qc["n_genes_by_counts"] >= 20) & (qc["pct_counts_mt"] < 20)
>>> "pct_counts_mt" in qc.columns
True
```

### `arsc.scanpy_extra.rank_genes_groups(X, groups, features=None, *, method: 'str' = 'wilcoxon', reference: 'str' = 'rest', groups_subset=None, tie_correct: 'bool' = False, corr_method: 'str' = 'benjamini-hochberg', pts: 'bool' = False, n_genes: 'int | None' = None, rankby_abs: 'bool' = False, log1p_base: 'float | None' = None) -> 'pd.DataFrame'`

```text
``sc.tl.rank_genes_groups`` as a tidy table (equal to ``sc.get.rank_genes_groups_df(adata, group=None)``).

Cell-level marker ranking: use it to describe clusters, not as evidence for differences between
conditions (cells are not independent replicates; for conditions use :func:`arsc.pseudobulk` + DE).

Parameters
----------
X : cells x genes; expected log1p-normalized (scanpy's ``logfoldchanges`` use ``expm1``).
groups : label per cell (e.g. ``ds.obs["leiden"]``).
features : gene names (default ``0..p-1``).
method : ``"wilcoxon"`` (normal approximation, no continuity correction), ``"t-test"`` (Welch) or
    ``"t-test_overestim_var"``. ``logreg`` is not implemented.
reference : ``"rest"`` or one group label (ranks are then computed within group + reference only).
groups_subset : restrict the output to these groups (scanpy ``groups=``).
tie_correct : tie correction for wilcoxon (scanpy default False).
corr_method : ``"benjamini-hochberg"`` or ``"bonferroni"``.
pts : add ``pct_nz_group`` / ``pct_nz_reference``.
n_genes : keep the top n genes per group (by |score| when ``rankby_abs``).
rankby_abs : rank by absolute score.
log1p_base : base used for log1p, if not natural (as ``adata.uns["log1p"]["base"]``).

Returns
-------
DataFrame with ``group, names, scores, logfoldchanges, pvals, pvals_adj`` (+ ``pct_nz_*`` with ``pts``),
sorted by decreasing score within group.

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> rg = sx.rank_genes_groups(L, ds.obs["cell_type"], genes, n_genes=3)
>>> sorted(rg[rg.group == "B"]["names"])
['CD19', 'CD79A', 'MS4A1']
```

### `arsc.scanpy_extra.regress_out(X, covariates, *, categorical: 'bool' = False) -> 'np.ndarray'`

```text
``sc.pp.regress_out``: residuals of every gene after OLS on ``covariates`` (+ intercept).

Parameters
----------
X : cells x genes (sparse is densified, as scanpy; subset genes first for large data).
covariates : cells x k numeric array / DataFrame, or (``categorical=True``) one label vector.
categorical : treat ``covariates`` as a single categorical key: each gene is centered per category.

Returns
-------
dense numpy array of residuals, cells x genes.

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> import numpy as np
>>> depth = np.asarray(ds.X("RNA", "counts").sum(1)).ravel()
>>> sx.regress_out(L[:, :20], depth[:, None]).shape
(600, 20)
```

### `arsc.scanpy_extra.score_genes(X, features, gene_list, *, ctrl_size: 'int' = 50, gene_pool=None, n_bins: 'int' = 25, random_state: 'int' = 0, ctrl_as_ref: 'bool' = True, return_controls: 'bool' = False)`

```text
``sc.tl.score_genes``: module score with expression-matched control genes (scanpy-exact).

``random_state`` (default 0, as scanpy) seeds the control-gene sampling. scanpy calls
``np.random.seed(random_state)`` and then ``Series.sample`` on the global RNG; arsc draws the identical
stream from a *local* ``np.random.RandomState(random_state)`` (same MT19937 sequence), so the control set,
and hence the score, equals scanpy's, without touching numpy's global RNG. ``random_state=None`` (scanpy:
"do not reseed") is rejected.

Parameters
----------
X : cells x genes, whatever ``adata.X`` / ``.raw.X`` would be (log-normalized or scaled; sparse or dense).
features : gene names of the columns of ``X``.
gene_list : genes of the signature (genes absent from ``features`` are ignored, as scanpy).
ctrl_size : control genes drawn per expression bin (default 50).
gene_pool : genes to draw controls from (default all).
n_bins : number of expression bins (default 25).
random_state : integer seed (default 0).
ctrl_as_ref : include the signature genes in the control pool (scanpy >= 1.10 default True).
return_controls : also return the control genes.

Returns
-------
numpy array, one score per cell (plus the control gene Index when ``return_controls``).

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> s = sx.score_genes(L, genes, ["CD8A", "CD8B", "GZMK"])
>>> s.shape
(600,)
```

### `arsc.scanpy_extra.umap_from_connectivities(connectivities, *, min_dist: 'float' = 0.5, spread: 'float' = 1.0, n_components: 'int' = 2, maxiter: 'int | None' = None, alpha: 'float' = 1.0, gamma: 'float' = 1.0, negative_sample_rate: 'int' = 5, init_pos: 'str' = 'spectral', random_state: 'int' = 0) -> 'np.ndarray'`

```text
``sc.tl.umap``: UMAP layout optimization on the scanpy connectivities graph (not on raw X).

Same call path as scanpy (``umap.umap_.simplicial_set_embedding`` with ``a, b = find_ab_params``).
``random_state`` seeds the spectral-init noise and the SGD layout through a local ``RandomState``;
umap-learn then runs the layout single-threaded, so same graph + seed -> bit-identical coordinates.

Parameters
----------
connectivities : cells x cells graph from :func:`neighbors`.
min_dist : minimum distance (scanpy default 0.5).
spread : effective scale of embedded points (default 1.0).
n_components : output dimensions (default 2).
maxiter : number of epochs (default: 500 up to 10,000 cells, else 200, as scanpy).
alpha : initial learning rate.
gamma : repulsion strength.
negative_sample_rate : negative samples per positive edge.
init_pos : ``"spectral"`` (default) or ``"random"``.
random_state : integer seed (default 0); ``None`` is rejected.

Returns
-------
numpy array, cells x ``n_components`` coordinates.

Examples
--------
>>> import arsc
>>> from arsc import scanpy_extra as sx
>>> ds = arsc.datasets.toy(n_cells=600)
>>> genes = ds.features("RNA")
>>> L = sx.normalize_total_log1p(ds.X("RNA", "counts"))
>>> Z, _, _ = arsc.pp.scale(L.toarray())
>>> pcs = arsc.pp.pca(Z, n_comps=10).embedding
>>> nn = sx.neighbors(pcs, n_neighbors=15)
>>> sx.umap_from_connectivities(nn["connectivities"], maxiter=50).shape
(600, 2)
```
