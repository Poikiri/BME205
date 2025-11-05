#!/usr/bin/env python3
import os, sys, argparse, math, gzip, csv
from collections import defaultdict, namedtuple
import pandas as pd
import numpy as np
import pysam
from scipy.stats import mannwhitneyu

Gene = namedtuple("Gene", ["seqid", "start", "end", "strand", "gene_id", "length"])

# ----------------------------
# GFF parsing
# ----------------------------
def parse_gff(gff_path):
    """
    Returns:
      genes_by_seq: dict(seqid -> list[Gene]) (sorted by start)
      gene_lengths: dict(gene_id -> length_bp)
      gene_ids: list of all gene_ids in file order
    """
    opener = gzip.open if gff_path.endswith(".gz") else open
    genes_by_seq = defaultdict(list)
    gene_ids = []

    with opener(gff_path, "rt") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            seqid, source, feature, start, end, score, strand, phase, attrs = parts
            if feature.lower() != "gene":
                # If your file uses "CDS" or "locus", you can broaden here if needed.
                continue
            start, end = int(start), int(end)
            # Parse attributes to get an identifier
            gene_id = None
            attr_pairs = {}
            for kv in attrs.split(";"):
                if not kv:
                    continue
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    attr_pairs[k.strip()] = v.strip()
                elif " " in kv:
                    k, v = kv.split(" ", 1)
                    attr_pairs[k.strip()] = v.strip().strip('"')
            for key in ("ID", "gene_id", "locus_tag", "Name"):
                if key in attr_pairs:
                    gene_id = attr_pairs[key]
                    break
            if gene_id is None:
                # fallback to seqid:start-end
                gene_id = f"{seqid}:{start}-{end}"
            length = end - start + 1
            g = Gene(seqid=seqid, start=start, end=end, strand=strand, gene_id=gene_id, length=length)
            genes_by_seq[seqid].append(g)
            gene_ids.append(gene_id)

    # sort by start per contig
    for seq in genes_by_seq:
        genes_by_seq[seq].sort(key=lambda g: g.start)

    gene_lengths = {g.gene_id: g.length for seq in genes_by_seq for g in genes_by_seq[seq]}
    return genes_by_seq, gene_lengths, gene_ids

# ----------------------------
# Interval helpers
# ----------------------------
def interval_overlap_len(a_start, a_end, b_start, b_end):
    """Inclusive endpoints for genes; BAM coordinates are 0-based half-open.
       We'll treat inputs as 0-based half-open for overlap (end exclusive).
    """
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    return max(0, e - s)

def blocks_overlap_len(blocks, start, end):
    """Sum overlap between list of (b_start, b_end) and interval [start, end)."""
    total = 0
    for bs, be in blocks:
        total += interval_overlap_len(bs, be, start, end)
    return total

# ----------------------------
# Gene lookup per contig
# ----------------------------
def find_candidate_genes(genes_sorted, qstart, qend):
    """Binary-search window to reduce checks: return indices possibly overlapping [qstart,qend)."""
    # simple linear scan fallback (E. coli small) – still efficient enough
    # If needed, implement bisect on starts.
    out = []
    for g in genes_sorted:
        if g.end < qstart:
            continue
        if g.start > qend:
            break
        out.append(g)
    return out

# ----------------------------
# Count fragments
# ----------------------------
def get_read_blocks(aln):
    """Return alignment blocks as 0-based half-open intervals on reference."""
    # pysam returns list of (start, end) zero-based, end-exclusive for aligned blocks (matches/mismatches)
    return aln.get_blocks()

def estimate_pair_blocks(read1, bamfile):
    """
    FAST: approximate fragment as the template span using TLEN if available.
    Falls back to read1's own aligned blocks.
    """
    # If paired, same contig, and TLEN is nonzero, use the template span
    try:
        same_contig = (read1.reference_id == read1.next_reference_id) and (read1.next_reference_id != -1)
    except AttributeError:
        same_contig = False

    if read1.is_paired and same_contig and read1.template_length != 0:
        start = min(read1.reference_start, read1.next_reference_start)
        end   = start + abs(read1.template_length)
        if end > start:
            return [(start, end)]

    # Fallback: just this read's aligned blocks
    return read1.get_blocks()


def aligned_len(blocks):
    return sum(be - bs for bs, be in blocks)

def count_fragments_for_bam(bam_path, genes_by_seq, mapq_min=10, overlap_frac=0.5):
    """
    Returns dict: gene_id -> count (float, due to proportional assignment)
    """
    counts = defaultdict(float)
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for aln in bam.fetch(until_eof=True):
            # primary, mapped, properly paired, use only read1 to count each template once
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue
            if aln.mapping_quality < mapq_min:
                continue

            if aln.is_paired:
                if not aln.is_proper_pair:
                    continue
                if not aln.is_read1:
                    # count only from read1 to avoid double-count
                    continue
                # Build fragment blocks (union of both mates if possible)
                frag_blocks = estimate_pair_blocks(aln, bam)
                frag_len = aligned_len(frag_blocks)
                if frag_len == 0:
                    continue
                seqid = bam.get_reference_name(aln.reference_id)
                if seqid not in genes_by_seq:
                    continue
                # Find overlapping genes; compute per-gene overlap length
                candidates = find_candidate_genes(genes_by_seq[seqid], frag_blocks[0][0], frag_blocks[-1][1])
                overlaps = []
                for g in candidates:
                    olap_bp = blocks_overlap_len(frag_blocks, g.start-1, g.end)  # gene as 0-based half-open
                    if olap_bp > 0:
                        overlaps.append((g.gene_id, olap_bp))
                if not overlaps:
                    continue
                # ≥ 50% overlap criterion (relative to fragment aligned length)
                total_olap = sum(o for _, o in overlaps)
                if total_olap < overlap_frac * frag_len:
                    continue
                # Proportional assignment
                for gid, ol in overlaps:
                    counts[gid] += ol / total_olap
            else:
                # Single-end fallback (or unpaired read)
                blocks = get_read_blocks(aln)
                rlen = aligned_len(blocks)
                if rlen == 0:
                    continue
                seqid = bam.get_reference_name(aln.reference_id)
                if seqid not in genes_by_seq:
                    continue
                candidates = find_candidate_genes(genes_by_seq[seqid], blocks[0][0], blocks[-1][1])
                overlaps = []
                for g in candidates:
                    olap_bp = blocks_overlap_len(blocks, g.start-1, g.end)
                    if olap_bp > 0:
                        overlaps.append((g.gene_id, olap_bp))
                if not overlaps:
                    continue
                total_olap = sum(o for _, o in overlaps)
                if total_olap < overlap_frac * rlen:
                    continue
                for gid, ol in overlaps:
                    counts[gid] += ol / total_olap

    return counts

# ----------------------------
# TPM
# ----------------------------
def counts_to_tpm(count_series, gene_lengths_bp):
    """
    count_series: pd.Series indexed by gene_id for one sample
    Returns pd.Series TPM for that sample.
    """
    # RPK = count / (length_kb)
    lengths_kb = pd.Series(gene_lengths_bp, dtype=float) / 1000.0
    # align indices
    lengths_kb = lengths_kb.reindex(count_series.index)
    rpk = count_series.astype(float) / lengths_kb.replace(0, np.nan)
    rpk = rpk.fillna(0.0)
    total_rpk = rpk.sum()
    if total_rpk == 0:
        return pd.Series(0.0, index=count_series.index)
    scaling = total_rpk / 1_000_000.0
    tpm = rpk / scaling
    return tpm

# ----------------------------
# DE stats
# ----------------------------
def mann_whitney_safe(a, b):
    # Inputs are lists/arrays; return p-value
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 1.0
    # If all zeros on both, p=1
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return 1.0
    try:
        res = mannwhitneyu(a, b, alternative="two-sided", method="auto")
        return float(res.pvalue)
    except Exception:
        return 1.0

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="RNA-seq differential expression for E. coli (paired-end).")
    ap.add_argument("samples_tsv", help="TSV with columns: bam_path, sample_type (control|treatment)")
    ap.add_argument("annotation_gff", help="GFF3 with gene features")
    ap.add_argument("output_dir", help="Directory to write outputs")
    args = ap.parse_args()

    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    # Read samples.tsv
    samp = pd.read_csv(args.samples_tsv, sep="\t")
    required_cols = {"bam_path", "sample_type"}
    if not required_cols.issubset(set(samp.columns)):
        raise ValueError(f"samples.tsv must have columns: {required_cols}")
    # derive sample_name from bam filename (without extension)
    def sample_name_from_path(p):
        base = os.path.basename(p)
        if base.endswith(".bam"):
            base = base[:-4]
        return base
    samp["sample_name"] = samp["bam_path"].apply(sample_name_from_path)
    # Validate sample types
    samp["sample_type"] = samp["sample_type"].str.strip().str.lower()
    if not set(samp["sample_type"]).issubset({"control", "treatment"}):
        raise ValueError("sample_type must be either 'control' or 'treatment'")

    # Parse GFF
    genes_by_seq, gene_lengths_bp, gene_ids_in_gff = parse_gff(args.annotation_gff)
    all_gene_ids = list(gene_lengths_bp.keys())

    # Count per BAM
    counts_matrix = pd.DataFrame(0.0, index=all_gene_ids, columns=samp["sample_name"])
    for _, row in samp.iterrows():
        bam_path = row["bam_path"]
        if not os.path.exists(bam_path):
            raise FileNotFoundError(f"BAM not found: {bam_path}")
        cdict = count_fragments_for_bam(bam_path, genes_by_seq, mapq_min=10, overlap_frac=0.5)
        # fill column
        col = pd.Series(cdict, dtype=float)
        counts_matrix.loc[col.index, row["sample_name"]] = col

    # Ensure all genes present, keep original GFF ordering
    counts_matrix = counts_matrix.reindex(all_gene_ids).fillna(0.0)

    # Write gene_counts.tsv (keep up to 3 decimals for proportional splits)
    counts_out = counts_matrix.copy()
    counts_out.insert(0, "gene_id", counts_out.index)
    counts_out.to_csv(os.path.join(outdir, "gene_counts.tsv"), sep="\t", index=False, float_format="%.3f")

    # TPM per sample
    tpm_df = pd.DataFrame(index=counts_matrix.index)
    for s in counts_matrix.columns:
        tpm_df[s] = counts_to_tpm(counts_matrix[s], gene_lengths_bp)

    # Filter low-expression genes using RAW counts (sum across all samples)
    keep_mask = counts_matrix.sum(axis=1) >= 10.0
    filt_genes = counts_matrix.index[keep_mask]

    # Split groups
    ctrl_samples = samp.loc[samp["sample_type"] == "control", "sample_name"].tolist()
    trt_samples  = samp.loc[samp["sample_type"] == "treatment", "sample_name"].tolist()

    # Compute stats
    rows = []
    for gid in filt_genes:
        ctrl_vals = tpm_df.loc[gid, ctrl_samples].values if ctrl_samples else np.array([])
        trt_vals  = tpm_df.loc[gid, trt_samples].values if trt_samples else np.array([])

        mean_ctrl   = float(np.mean(ctrl_vals)) if ctrl_vals.size else 0.0
        median_ctrl = float(np.median(ctrl_vals)) if ctrl_vals.size else 0.0
        mean_trt    = float(np.mean(trt_vals)) if trt_vals.size else 0.0
        median_trt  = float(np.median(trt_vals)) if trt_vals.size else 0.0

        # log2 fold change with small pseudocount to avoid div-by-zero
        log2fc = math.log2((mean_trt + 1.0) / (mean_ctrl + 1.0)) if (mean_trt + mean_ctrl) >= 0 else 0.0

        pval = mann_whitney_safe(ctrl_vals, trt_vals)

        rows.append({
            "gene_id": gid,
            "mean_control": mean_ctrl,
            "median_control": median_ctrl,
            "mean_treatment": mean_trt,
            "median_treatment": median_trt,
            "log2_fold_change": log2fc,
            "p_value": pval
        })

    de_df = pd.DataFrame(rows, columns=[
        "gene_id","mean_control","median_control","mean_treatment","median_treatment","log2_fold_change","p_value"
    ])
    de_df.to_csv(os.path.join(outdir, "differential_expression.tsv"), sep="\t", index=False, float_format="%.6g")

if __name__ == "__main__":
    main()
