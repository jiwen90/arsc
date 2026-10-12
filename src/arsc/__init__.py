"""arsc -- fast, R-validated single-cell compute engine.

Quick map (see README.md):
    arsc.open / read_seurat_export / write_table      I/O bridge (memory-mapped store)
    arsc.pseudobulk                                   per-sample aggregation
    arsc.pp.log_normalize / clr / hvg_vst / scale / pca
    arsc.run_harmony                                  harmony 1.2 port
    arsc.wilcoxauc                                    presto port
    arsc.score_ucell / add_module_score               UCell / Seurat AddModuleScore
    arsc.neighbors / wnn / cluster / umap             graph tools
    arsc.clones.cyclone / clone_expansion_null / startrac / clone_*   clonal methods (Cyclone exact)
    arsc.genesets.load_v3 / read_genesets / cycling_fraction          gene-set readers
    arsc.stats (composition / paired_test / perm_test / glmm_prop / mde)  patient-level stats
    arsc.find_markers / find_all_markers / average_expression / cell_cycle_scoring / scale_data
                                                      Seurat 4.3 parity (seurat_extra; SEURAT_PARITY.md)
    arsc.de (deseq2 / voom / fgsea / msigdb)                           pseudobulk DE + GSEA
"""
from ._threads import preconfigure_env as _pre

_pre()  # map AR_THREADS onto numba/BLAS env vars before they load

from . import preprocessing as pp  # noqa: E402
from ._threads import get_threads, set_threads, env_threads as _env_threads  # noqa: E402
from .harmony import run_harmony  # noqa: E402
from .io import (Assay, Dataset, Reduction, Store, export_seurat, exporter_script, from_anndata, open,  # noqa: E402
                 read_seurat_export, write_table)
from .markers import top_markers, wilcoxauc  # noqa: E402
from .pseudobulk import Pseudobulk, pseudobulk  # noqa: E402
from .scores import add_module_score, score_ucell  # noqa: E402
from .neighbors import find_neighbors, knn, snn_graph, wnn  # noqa: E402
from .cluster import cluster, umap  # noqa: E402
from . import clones, genesets  # noqa: E402
from . import stats, de  # noqa: E402
from .stats import composition, paired_test, perm_test, glmm_prop  # noqa: E402
from .de import deseq2, voom, fgsea, msigdb  # noqa: E402
from . import seurat_extra as se  # noqa: E402
from .seurat_extra import (find_markers, find_all_markers, fold_change, average_expression,  # noqa: E402
                           cell_cycle_scoring, scale_data, normalize_data, add_metadata, which_cells,
                           split_object, rename_idents, cc_genes, cc_genes_updated_2019)

if _env_threads() is not None:
    set_threads(_env_threads())

__version__ = "0.1.0"
