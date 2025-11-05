#!/usr/bin/env python3

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import time

class IntervalTree:
    """Efficient interval tree for genomic overlap queries using numpy arrays."""
    
    def __init__(self):
        self.intervals = {}  # chrom -> (starts, ends) numpy arrays
    
    def add_from_dataframe(self, df: pd.DataFrame):
        """Add intervals from a DataFrame efficiently."""
        for chrom in df['chrom'].unique():
            chrom_data = df[df['chrom'] == chrom]
            starts = chrom_data['start'].values
            ends = chrom_data['end'].values
            # Sort by start position
            sort_idx = np.argsort(starts)
            self.intervals[chrom] = (starts[sort_idx], ends[sort_idx])
    
    def query_overlap(self, chrom: str, start: int, end: int) -> np.ndarray:
        """
        Query all intervals that overlap with the given range.
        Returns indices of overlapping intervals.
        """
        if chrom not in self.intervals:
            return np.array([])
        
        starts, ends = self.intervals[chrom]
        # Intervals overlap if: interval_start < query_end AND interval_end > query_start
        overlaps = (starts < end) & (ends > start)
        return np.where(overlaps)[0]


def load_bed_file(filepath: str) -> pd.DataFrame:
    """
    Load a BED file into a DataFrame.
    
    Args:
        filepath: Path to BED file
        
    Returns:
        DataFrame with columns: chrom, start, end
    """
    df = pd.read_csv(filepath, sep='\t', header=None, names=['chrom', 'start', 'end'])
    return df


def load_genome_index(filepath: str) -> Dict[str, int]:
    """
    Load chromosome lengths from FASTA index file.
    
    Args:
        filepath: Path to .fai file
        
    Returns:
        Dictionary mapping chromosome name to length
    """
    chrom_lengths = {}
    with open(filepath, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            chrom_lengths[fields[0]] = int(fields[1])
    return chrom_lengths


def calculate_overlap_bp(start1: int, end1: int, start2: int, end2: int) -> int:
    """
    Calculate overlapping base pairs between two intervals.
    
    Args:
        start1, end1: First interval [start1, end1)
        start2, end2: Second interval [start2, end2)
        
    Returns:
        Number of overlapping base pairs
    """
    return max(0, min(end1, end2) - max(start1, start2))


def calculate_total_overlap(setA: pd.DataFrame, setB: pd.DataFrame, tree_B=None) -> int:
    """
    Calculate total overlapping base pairs between SetA and SetB.
    CRITICAL: Does NOT merge overlapping regions within sets.
    
    Args:
        setA: DataFrame with genomic regions (chrom, start, end)
        setB: DataFrame with genomic regions (chrom, start, end)
        tree_B: Pre-built interval tree for SetB (optional, for efficiency)
        
    Returns:
        Total weighted overlap in base pairs
    """
    # Build interval tree for SetB if not provided
    if tree_B is None:
        tree_B = IntervalTree()
        tree_B.add_from_dataframe(setB)
    
    total_overlap = 0
    
    # Group SetA by chromosome for efficient processing
    for chrom in setA['chrom'].unique():
        if chrom not in tree_B.intervals:
            continue
            
        setA_chrom = setA[setA['chrom'] == chrom]
        starts_a = setA_chrom['start'].values
        ends_a = setA_chrom['end'].values
        starts_b, ends_b = tree_B.intervals[chrom]
        
        # Vectorized overlap calculation for this chromosome
        for i in range(len(starts_a)):
            start_a = starts_a[i]
            end_a = ends_a[i]
            
            # Find overlapping intervals using vectorized operations
            overlaps = (starts_b < end_a) & (ends_b > start_a)
            if np.any(overlaps):
                # Calculate all overlaps at once
                overlap_starts = np.maximum(start_a, starts_b[overlaps])
                overlap_ends = np.minimum(end_a, ends_b[overlaps])
                overlap_lengths = overlap_ends - overlap_starts
                total_overlap += np.sum(overlap_lengths)
    
    return int(total_overlap)


def calculate_per_region_overlap(setA: pd.DataFrame, setB: pd.DataFrame, tree_A=None) -> List[int]:
    """
    Calculate overlap for each region in SetB with all regions in SetA.
    
    Args:
        setA: DataFrame with genomic regions
        setB: DataFrame with genomic regions
        tree_A: Pre-built interval tree for SetA (optional, for efficiency)
        
    Returns:
        List of overlap values for each SetB region (in same order as setB)
    """
    # Build interval tree for SetA if not provided
    if tree_A is None:
        tree_A = IntervalTree()
        tree_A.add_from_dataframe(setA)
    
    per_region_overlaps = []
    
    # Get SetB arrays
    starts_b = setB['start'].values
    ends_b = setB['end'].values
    chroms_b = setB['chrom'].values
    
    # For each region in SetB, calculate total overlap with SetA
    for i in range(len(setB)):
        chrom = chroms_b[i]
        start_b = starts_b[i]
        end_b = ends_b[i]
        
        region_overlap = 0
        
        if chrom in tree_A.intervals:
            starts_a, ends_a = tree_A.intervals[chrom]
            
            # Find overlapping intervals using vectorized operations
            overlaps = (starts_a < end_b) & (ends_a > start_b)
            if np.any(overlaps):
                # Calculate all overlaps at once
                overlap_starts = np.maximum(start_b, starts_a[overlaps])
                overlap_ends = np.minimum(end_b, ends_a[overlaps])
                overlap_lengths = overlap_ends - overlap_starts
                region_overlap = int(np.sum(overlap_lengths))
        
        per_region_overlaps.append(region_overlap)
    
    return per_region_overlaps


def permute_regions(setA: pd.DataFrame, chrom_lengths: Dict[str, int], seed: int = None) -> pd.DataFrame:
    """
    Generate a random permutation of SetA regions using vectorized operations.
    Preserves: region lengths, chromosome assignments.
    
    Args:
        setA: DataFrame with genomic regions
        chrom_lengths: Dictionary of chromosome lengths
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with permuted regions
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate region lengths
    region_lengths = setA['end'].values - setA['start'].values
    chroms = setA['chrom'].values
    
    # Pre-allocate arrays
    new_starts = np.zeros(len(setA), dtype=np.int32)
    
    # Permute regions by chromosome
    for chrom in setA['chrom'].unique():
        mask = chroms == chrom
        chrom_len = chrom_lengths[chrom]
        chrom_region_lengths = region_lengths[mask]
        
        # Calculate max starts for each region
        max_starts = chrom_len - chrom_region_lengths
        
        # Generate random starts
        random_starts = np.random.randint(0, max_starts + 1, size=len(chrom_region_lengths))
        new_starts[mask] = random_starts
    
    # Create new DataFrame
    permuted_df = pd.DataFrame({
        'chrom': chroms,
        'start': new_starts,
        'end': new_starts + region_lengths
    })
    
    return permuted_df


def run_permutation_test(setA: pd.DataFrame, setB: pd.DataFrame, 
                        chrom_lengths: Dict[str, int], 
                        num_permutations: int = 10000) -> Tuple[float, List[int]]:
    """
    Run permutation test to assess significance of observed overlap.
    
    Args:
        setA: DataFrame with genomic regions
        setB: DataFrame with genomic regions
        chrom_lengths: Dictionary of chromosome lengths
        num_permutations: Number of permutations to perform
        
    Returns:
        Tuple of (global_p_value, null_distribution)
    """
    start_time = time.time()
    
    # Pre-build interval tree for SetB (reuse across permutations)
    print(f"Building interval tree for SetB...", file=sys.stderr)
    tree_B = IntervalTree()
    tree_B.add_from_dataframe(setB)
    
    # Calculate observed overlap
    print(f"Calculating observed overlap...", file=sys.stderr)
    observed_overlap = calculate_total_overlap(setA, setB, tree_B)
    print(f"  Observed overlap: {observed_overlap} bp", file=sys.stderr)
    
    # Generate null distribution
    null_distribution = []
    
    print(f"\nRunning {num_permutations} permutations for global test...", file=sys.stderr)
    for i in range(num_permutations):
        iter_start = time.time()
        
        # Generate permuted SetA
        permuted_setA = permute_regions(setA, chrom_lengths, seed=i)
        
        # Calculate overlap for permuted data (reuse tree_B)
        permuted_overlap = calculate_total_overlap(permuted_setA, setB, tree_B)
        null_distribution.append(permuted_overlap)
        
        # Progress indicator with timing
        if (i + 1) % 200 == 0 or i == 0 or i == num_permutations - 1:
            elapsed = time.time() - start_time
            iter_time = time.time() - iter_start
            avg_time_per_perm = elapsed / (i + 1)
            estimated_remaining = avg_time_per_perm * (num_permutations - i - 1)
            print(f"  [{i + 1}/{num_permutations}] Elapsed: {elapsed:.1f}s | "
                  f"Avg: {avg_time_per_perm:.3f}s/perm | "
                  f"Est. remaining: {estimated_remaining:.1f}s", file=sys.stderr)
    
    # Calculate p-value (one-tailed test)
    num_greater_or_equal = sum(1 for perm_overlap in null_distribution 
                               if perm_overlap >= observed_overlap)
    p_value = (num_greater_or_equal + 1) / (num_permutations + 1)
    
    total_time = time.time() - start_time
    print(f"\nGlobal permutation test completed in {total_time:.1f}s", file=sys.stderr)
    
    return p_value, null_distribution


def run_per_region_permutation_test(setA: pd.DataFrame, setB: pd.DataFrame,
                                    chrom_lengths: Dict[str, int],
                                    num_permutations: int = 10000) -> Tuple[List[float], List[int]]:
    """
    Run permutation test for each region in SetB.
    
    Args:
        setA: DataFrame with genomic regions
        setB: DataFrame with genomic regions
        chrom_lengths: Dictionary of chromosome lengths
        num_permutations: Number of permutations
        
    Returns:
        Tuple of (p_values, observed_per_region)
    """
    start_time = time.time()
    
    # Calculate observed overlap for each SetB region
    print(f"\nCalculating observed per-region overlaps...", file=sys.stderr)
    observed_per_region = calculate_per_region_overlap(setA, setB)
    
    # Initialize null distributions for each SetB region
    null_distributions = [[] for _ in range(len(setB))]
    
    print(f"Running {num_permutations} permutations for per-region test...", file=sys.stderr)
    
    for i in range(num_permutations):
        iter_start = time.time()
        
        # Generate permuted SetA
        permuted_setA = permute_regions(setA, chrom_lengths, seed=i + 100000)
        
        # Build interval tree for this permutation
        tree_A = IntervalTree()
        tree_A.add_from_dataframe(permuted_setA)
        
        # Calculate per-region overlaps for permuted data (reuse tree)
        permuted_per_region = calculate_per_region_overlap(permuted_setA, setB, tree_A)
        
        # Add to null distributions
        for j, overlap in enumerate(permuted_per_region):
            null_distributions[j].append(overlap)
        
        # Progress indicator with timing (every 100)
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - start_time
            iter_time = time.time() - iter_start
            avg_time_per_perm = elapsed / (i + 1)
            estimated_remaining = avg_time_per_perm * (num_permutations - i - 1)
            print(f"  [{i + 1}/{num_permutations}] Elapsed: {elapsed:.1f}s | "
                  f"Avg: {avg_time_per_perm:.3f}s/perm | "
                  f"Est. remaining: {estimated_remaining:.1f}s", file=sys.stderr)
    
    # Calculate p-values for each region
    p_values = []
    for observed, null_dist in zip(observed_per_region, null_distributions):
        num_greater_or_equal = sum(1 for perm_overlap in null_dist 
                                   if perm_overlap >= observed)
        p_value = (num_greater_or_equal + 1) / (num_permutations + 1)
        p_values.append(p_value)
    
    total_time = time.time() - start_time
    print(f"\nPer-region permutation test completed in {total_time:.1f}s", file=sys.stderr)
    
    return p_values, observed_per_region


def main():
    parser = argparse.ArgumentParser(
        description='Genomic region overlap analysis with permutation testing'
    )
    parser.add_argument('setA_bed', help='Path to SetA BED file')
    parser.add_argument('setB_bed', help='Path to SetB BED file')
    parser.add_argument('genome_fai', help='Path to genome FASTA index file')
    parser.add_argument('output_dir', help='Output directory for results')
    parser.add_argument('num_permutations', nargs='?', type=int, default=10000,
                       help='Number of permutations (default: 10000)')
    
    args = parser.parse_args()
    
    overall_start = time.time()
    
    # Load data
    print(f"="*70, file=sys.stderr)
    print(f"GENOMIC PERMUTATION TESTING ANALYSIS", file=sys.stderr)
    print(f"="*70, file=sys.stderr)
    print(f"\nLoading SetA from {args.setA_bed}...", file=sys.stderr)
    setA = load_bed_file(args.setA_bed)
    
    print(f"Loading SetB from {args.setB_bed}...", file=sys.stderr)
    setB = load_bed_file(args.setB_bed)
    
    print(f"Loading genome index from {args.genome_fai}...", file=sys.stderr)
    chrom_lengths = load_genome_index(args.genome_fai)
    
    # Calculate basic statistics
    num_setA_regions = len(setA)
    num_setB_regions = len(setB)
    setA_total_bases = (setA['end'] - setA['start']).sum()
    setB_total_bases = (setB['end'] - setB['start']).sum()
    
    print(f"\n" + "-"*70, file=sys.stderr)
    print(f"DATASET STATISTICS", file=sys.stderr)
    print(f"-"*70, file=sys.stderr)
    print(f"  SetA regions: {num_setA_regions:,}", file=sys.stderr)
    print(f"  SetB regions: {num_setB_regions:,}", file=sys.stderr)
    print(f"  SetA total bases: {setA_total_bases:,}", file=sys.stderr)
    print(f"  SetB total bases: {setB_total_bases:,}", file=sys.stderr)
    print(f"  Chromosomes: {', '.join(sorted(setA['chrom'].unique()))}", file=sys.stderr)
    print(f"-"*70, file=sys.stderr)
    
    # Run global permutation test
    print(f"\n" + "="*70, file=sys.stderr)
    print(f"GLOBAL PERMUTATION TEST", file=sys.stderr)
    print(f"="*70, file=sys.stderr)
    global_p_value, null_distribution = run_permutation_test(
        setA, setB, chrom_lengths, args.num_permutations
    )
    print(f"  Global p-value: {global_p_value:.6f}", file=sys.stderr)
    
    # Run per-region permutation test
    print(f"\n" + "="*70, file=sys.stderr)
    print(f"PER-REGION PERMUTATION TEST", file=sys.stderr)
    print(f"="*70, file=sys.stderr)
    per_region_p_values, observed_per_region = run_per_region_permutation_test(
        setA, setB, chrom_lengths, args.num_permutations
    )
    
    # Calculate Bonferroni correction
    bonferroni_threshold = 0.05 / num_setB_regions
    significant_regions = sum(1 for p in per_region_p_values if p < bonferroni_threshold)
    
    print(f"\n" + "-"*70, file=sys.stderr)
    print(f"SIGNIFICANCE SUMMARY", file=sys.stderr)
    print(f"-"*70, file=sys.stderr)
    print(f"  Bonferroni threshold: {bonferroni_threshold:.2e}", file=sys.stderr)
    print(f"  Significant regions: {significant_regions}/{num_setB_regions}", file=sys.stderr)
    print(f"-"*70, file=sys.stderr)
    
    # Create output directory if needed
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write summary results
    summary_file = output_path / 'results.tsv'
    print(f"\nWriting summary results to {summary_file}...", file=sys.stderr)
    
    # Get observed overlap from the global test
    observed_overlap = calculate_total_overlap(setA, setB)
    
    summary_data = {
        'metric': [
            'observed_overlap',
            'global_p_value',
            'num_permutations',
            'setA_regions',
            'setB_regions',
            'setA_total_bases',
            'setB_total_bases',
            'bonferroni_threshold',
            'significant_regions_bonferroni'
        ],
        'value': [
            observed_overlap,
            global_p_value,
            args.num_permutations,
            num_setA_regions,
            num_setB_regions,
            setA_total_bases,
            setB_total_bases,
            bonferroni_threshold,
            significant_regions
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, sep='\t', index=False)
    
    # Write per-region results
    per_region_file = output_path / 'results_per_region.tsv'
    print(f"Writing per-region results to {per_region_file}...", file=sys.stderr)
    
    per_region_df = setB.copy()
    per_region_df['observed_overlap'] = observed_per_region
    per_region_df['p_value'] = per_region_p_values
    per_region_df['significant_bonferroni'] = [p < bonferroni_threshold for p in per_region_p_values]
    
    # Sort by chromosome and start position
    per_region_df = per_region_df.sort_values(['chrom', 'start'])
    per_region_df.to_csv(per_region_file, sep='\t', index=False)
    
    total_time = time.time() - overall_start
    print(f"\n" + "="*70, file=sys.stderr)
    print(f"ANALYSIS COMPLETE!", file=sys.stderr)
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.2f} minutes)", file=sys.stderr)
    print(f"="*70, file=sys.stderr)


if __name__ == '__main__':
    main()
