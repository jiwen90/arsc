"""arsc -- fast, R-validated single-cell compute engine.

Start here: ``arsc.help()`` prints the task -> function map, ``arsc.recipes()`` the runnable recipes
(``arsc.recipes("compare responders and non-responders")`` finds one from a plain-English question), and
``arsc.datasets.toy()`` gives a small synthetic dataset to try them on. Agents: read AGENTS.md.

Quick map (see README.md):
    arsc.read(path[, store=dir])                      any format -> Dataset (memory) or Store (mmap)
    arsc.read_h5ad / read_10x_h5 / read_10x_mtx / read_seurat   format-specific readers
    arsc.write_h5ad / Store.to_anndata / Store.write_h5ad       hand-off to scanpy / anndata
    arsc.open / read_seurat_export / write_table      I/O bridge (memory-mapped store)
    arsc.pseudobulk                                   per-sample aggregation
    arsc.pp.log_normalize / clr / hvg_vst / scale / pca
    arsc.run_harmony                                  harmony 1.2 port
    arsc.symphony (build_reference / map_query / transfer_labels / mapping_score)   Symphony reference mapping
    arsc.wilcoxauc                                    presto port
    arsc.score_ucell / add_module_score               UCell / Seurat AddModuleScore
    arsc.neighbors / wnn / cluster / umap             graph tools
    arsc.clones.cyclone / clone_expansion_null / startrac / clone_*   clonal methods (Cyclone exact)
    arsc.clonotrace.run / propagate_clones / clone_distance / clone_dpt / fate_genes   CloneTrace port (CLONOTRACE_PARITY.md)
    arsc.genesets.load_v3 / read_genesets / cycling_fraction          gene-set readers
    arsc.stats (composition / paired_test / perm_test / glmm_prop / mde)  patient-level stats
    arsc.find_markers / find_all_markers / average_expression / cell_cycle_scoring / scale_data
                                                      Seurat 4.3 parity (seurat_extra; SEURAT_PARITY.md)
    arsc.rank_genes_groups / score_genes / highly_variable_genes / scanpy_extra.*   scanpy parity (SCANPY_PARITY.md)
    arsc.de (deseq2 / voom / fgsea / msigdb)                           pseudobulk DE + GSEA
    arsc.seeds_used / reset_seeds_used                seeds used by stochastic calls (SEEDS.md)
"""
from ._threads import preconfigure_env as _pre

_pre()  # map AR_THREADS (default 4, never the core count) onto numba/BLAS env vars before they load

from . import preprocessing as pp  # noqa: E402
from ._threads import get_threads, set_threads, default_threads as _default_threads  # noqa: E402
from .harmony import run_harmony  # noqa: E402
from .io import (Assay, Dataset, Reduction, Store, export_seurat, exporter_script, from_anndata, open,  # noqa: E402
                 read_seurat_export, write_table)
from .formats import (ArscFormatError, StoreBuilder, detect_format, make_unique, read, read_10x_h5,  # noqa: E402
                      read_10x_mtx, read_h5ad, read_seurat, to_anndata, write_h5ad)
from .markers import top_markers, wilcoxauc  # noqa: E402
from .pseudobulk import Pseudobulk, pseudobulk  # noqa: E402
from .scores import add_module_score, score_ucell  # noqa: E402
from .neighbors import find_neighbors, knn, snn_graph, wnn  # noqa: E402
from .cluster import cluster, umap  # noqa: E402
from ._umap import fuzzy_simplicial_set  # noqa: E402
from . import clones, clonotrace, genesets, symphony  # noqa: E402
from . import stats, de, datasets  # noqa: E402
from ._help import help, recipes  # noqa: E402,A004  (task map / runnable recipes)
from ._seedlog import reset_seeds_used, seeds_used  # noqa: E402  (seed of every stochastic call, SEEDS.md)
from .stats import composition, paired_test, perm_test, glmm_prop  # noqa: E402
from .de import deseq2, voom, fgsea, msigdb  # noqa: E402
from . import scanpy_extra  # noqa: E402  (scanpy parity; neighbors/leiden live here as scanpy_extra.*)
from .scanpy_extra import (normalize_total_log1p, score_genes, rank_genes_groups, highly_variable_genes,  # noqa: E402
                           regress_out, qc_metrics, diffmap, dpt, magic_impute)
from . import seurat_extra as se  # noqa: E402
from .seurat_extra import (find_markers, find_all_markers, fold_change, average_expression,  # noqa: E402
                           cell_cycle_scoring, scale_data, normalize_data, add_metadata, which_cells,
                           split_object, rename_idents, cc_genes, cc_genes_updated_2019)

set_threads(_default_threads())  # also caps BLAS pools that were already loaded (threadpoolctl)

__version__ = "0.3.0"
