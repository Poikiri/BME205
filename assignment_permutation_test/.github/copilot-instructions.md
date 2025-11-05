# Copilot Instructions: Genomic Permutation Testing Assignment

## Project Overview

This is a computational biology assignment implementing statistical permutation testing to analyze genomic region overlap between transcription factor binding sites (SetA) and active chromatin regions (SetB). The goal is to determine if co-localization patterns are statistically significant.

## Key Architecture & Data Flow

### Core Algorithm Requirements
- **Overlap Calculation**: Calculate base-pair overlaps between genomic intervals WITHOUT merging overlapping regions within sets
- **Permutation Testing**: Randomly redistribute SetA regions while preserving region sizes and chromosome constraints
- **Statistical Analysis**: Generate global p-values and per-region significance with Bonferroni correction

### Critical Implementation Details

**Overlap Logic (CRITICAL)**:
```python
# Each region must be processed independently - DO NOT merge overlapping intervals
# Example: SetA has overlapping regions chr1:100-200 and chr1:150-250
# Both must contribute separately to overlap calculations
for region_A in SetA:
    for region_B in SetB:
        if same_chromosome:
            overlap_bp = max(0, min(region_A.end, region_B.end) - max(region_A.start, region_B.start))
```

**Permutation Strategy**:
- Preserve original region lengths exactly
- Keep regions on same chromosome
- Ensure regions don't exceed chromosome boundaries (use `genome.fa.fai` for chromosome lengths)
- Use efficient random positioning within valid chromosome ranges

## Required File Structure & I/O

### Input Files
- `data/SetA.bed` - TF binding sites (3-column BED: chr, start, end)
- `data/SetB.bed` - Chromatin regions (3-column BED: chr, start, end) 
- `data/genome.fa.fai` - Chromosome lengths (use columns 1,2: name, length)

### Expected Output Format
1. `results.tsv` - Summary metrics (observed_overlap, global_p_value, etc.)
2. `results_per_region.tsv` - Per-SetB-region analysis with Bonferroni correction

### Command Line Interface
```bash
python firstname_lastname_permutation_test.py <setA_bed> <setB_bed> <genome_fai> <output_dir> [num_permutations]
```

## Development Environment

**Conda Environment**: Use `permutation_test.yaml`
- Python ≥3.10, pandas ≥2.0, numpy ≥1.24, scipy ≥1.10
- **Restriction**: No specialized bioinformatics libraries (pybedtools, pysam) for core logic
- Standard library + numpy/pandas only for main algorithms

## Performance Requirements

**Efficiency Constraints**:
- Must handle 57K+ regions in SetA, 10K+ in SetB within reasonable time
- Implement interval trees or similar efficient data structures
- Naive nested loops will exceed time limits for 10K permutations

**Memory Management**:
- Process large BED files efficiently
- Generate null distributions without excessive memory usage

## Key Patterns & Conventions

### Statistical Calculations
- Use one-tailed p-values: `(greater_or_equal + 1) / (permutations + 1)`
- Bonferroni threshold: `0.05 / num_setB_regions`
- Include ALL SetB regions in per-region output, even zero-overlap regions

### Data Handling
- BED coordinates are 0-based, half-open intervals [start, end)
- Sort output by chromosome, then start position
- Handle multiple chromosomes present in datasets

### Validation Strategy
- Test overlap calculations against small manual examples first
- Can validate against `bedtools intersect -wo` for development
- Start with small permutation counts, scale up after verification

## Common Pitfalls to Avoid

1. **Merging overlapping regions** within SetA/SetB (breaks biological interpretation)
2. **Using inappropriate random distributions** for permutation positioning
3. **Boundary violations** when placing permuted regions
4. **Inefficient algorithms** that can't handle dataset scale
5. **Missing Bonferroni correction** or incorrect multiple testing handling