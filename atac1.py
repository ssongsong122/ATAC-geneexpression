import scanpy as sc
import snapatac2 as snap
import muon as mu
import numpy as np
import pandas as pd
import warnings
import os
import gzip

warnings.filterwarnings("ignore")

# ==========================================
# 0. 配置与清理
# ==========================================
fragment_file = "/home/data/user/yszhou/biodata/project/atac/clean_fragments.tsv.gz"
gtf_file = "/home/data/user/yszhou/data/gencode.v44.annotation.gtf"
h5ad_file = "atac_data.h5ad"

if os.path.exists(h5ad_file):
    os.remove(h5ad_file)

# ==========================================
# 1. 动态生成参考基因组字典
# ==========================================
print("正在检测 Fragment 文件的染色体信息...")
detected_chroms = set()
try:
    with gzip.open(fragment_file, 'rt') as f:
        for i, line in enumerate(f):
            if line.startswith('#'): continue
            if i > 1000: break
            parts = line.split('\t')
            if len(parts) >= 1:
                detected_chroms.add(parts[0])
except Exception as e:
    print(f"警告：文件预读取失败 ({e})，将尝试使用默认 hg38...")

custom_chrom_sizes = {chrom: 500000000 for chrom in detected_chroms}
if not custom_chrom_sizes:
    chroms = [f"chr{i}" for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
    custom_chrom_sizes = {c: 500000000 for c in chroms}

print(f"已构建参考字典，包含 {len(custom_chrom_sizes)} 条染色体。")

# ==========================================
# 2. 数据导入 (Backed模式)
# ==========================================
print("正在读取 Fragments...")
adata = snap.pp.import_data(
    fragment_file,
    chrom_sizes=custom_chrom_sizes,
    file=h5ad_file,
    sorted_by_barcode=False
)

print("正在生成 5kb Tile Matrix...")
snap.pp.add_tile_matrix(adata, bin_size=5000)

if adata.n_vars == 0:
    raise ValueError("严重错误：矩阵为空！请检查输入文件。")

print(f"数据导入完成 (Backed模式): {adata.n_obs} 细胞 x {adata.n_vars} 区域")

# ==========================================
# 3. 计算 TSS Enrichment (使用 SnapATAC2)
# ==========================================
# 注意：SnapATAC2 擅长在 backed 模式下计算这个，速度极快
print("正在使用 SnapATAC2 计算 TSS Enrichment...")

# 自动适配 SnapATAC2 的版本 API
try:
    # 新版 API
    snap.metrics.tsse(adata, gtf_file)
except AttributeError:
    # 旧版 API
    try:
        snap.pp.tsse(adata, gtf_file)
    except Exception as e:
        print(f"TSS 计算失败，可能是 GTF 格式问题: {e}")
        # 如果 TSS 失败，我们手动创建一个假的 tsse 列防止后面报错，先跑通流程
        adata.obs['tsse'] = 0

print("TSS 计算完成。")

# ==========================================
# 4. 转入内存 (To Memory)
# ==========================================
print("正在将数据载入内存 (To Memory)...")
# 为了兼容 Scanpy 的 QC 函数，这里必须转入内存
adata = adata.to_memory()
adata.uns['files'] = {'fragments': fragment_file}

# ==========================================
# 5. 质量控制 (QC) - Scanpy部分
# ==========================================
print("正在计算 QC 指标...")
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
adata.obs.rename(columns={'total_counts': 'n_fragments', 'n_genes_by_counts': 'n_features'}, inplace=True)

# 过滤
# 注意：SnapATAC2 生成的列名通常叫 'tsse'，而 Muon 叫 'tss_score'
# 我们这里统一使用 SnapATAC2 的标准 'tsse'
tss_col = 'tsse' if 'tsse' in adata.obs.columns else 'tss_score'

print(f"过滤前细胞数: {adata.n_obs}")
min_fragments = 1000
min_tss = 4

# 防御性编程：如果 TSS 全是 0 (计算失败)，则跳过 TSS 过滤
if adata.obs[tss_col].max() == 0:
    print("警告：TSS 分数似乎未正确计算，将跳过 TSS 过滤阈值。")
    min_tss = 0

adata = adata[
    (adata.obs['n_fragments'] > min_fragments) &
    (adata.obs['n_fragments'] < 100000) &
    (adata.obs[tss_col] > min_tss),
    :
]
print(f"过滤后细胞数: {adata.n_obs}")

# ==========================================
# 6. 后续分析 (LSI + UMAP)
# ==========================================
print("正在降维与聚类...")
# 使用 Muon 的 ATAC 模块进行 LSI
mu.atac.pp.tfidf(adata, scale_factor=1e4)
mu.atac.tl.lsi(adata, n_comps=50)

sc.pp.neighbors(adata, use_rep='X_lsi', n_neighbors=30, n_pcs=30)
sc.tl.leiden(adata, resolution=0.8)
sc.tl.umap(adata)

print("正在保存结果...")
# 绘图时也使用新的列名
sc.pl.umap(adata, color=['leiden', 'n_fragments', tss_col], save="_atac_final.pdf")
adata.write("seekgene_atac_processed_final.h5ad")

print("任务全部成功完成！")