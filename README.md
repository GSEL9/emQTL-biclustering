# emQTL-clustering
Comparing the performance of biclustering algorithms applied to a data matrix containing of Bonferroni corrected p-values. The p-values are calculated from Pearson's correlations between DNA methylated CpG data and gene expression data. The aim of the clustering is to detect relations between genes and CpG methylation level to elucidate biological relevance of emQTL.

## Todo
* Rerun benchmarking.
  * Check if opt. hparams of R wrapped models are same as default params (indicates hparams not being updated in wrappers during GS).
* Verify result model selection procedure by applying discarded models to reference data and compare the results to the outcome of applying the selected models to the reference data.
* Bicluster method for saving predictions to disk.

## Algorithms
* sklearn SpectralCoclustering:
  * supports only sparse matrices if nonnegative.
* sklearn SpectralBiclustering:
  * Assumes checkerboard structure.
* R Plaid:
* R Cheng Church:
* R Bimax:
* R Quest:
* R Xmotifs:
* R Spectral:
  * Assumes checkerboard structure.

## Score metrics
* Recovery:
* Relevance:
* Jaccard Index:
* Average Spearman's Rho:
* Transposed Virtual Error:
* Mean Squared Residue:
* Scaled Mean Squared Residue:

## Evaluation framework

Synthetic data, experiemntal data

### Model selection

Jaccard Index

### Bicluster quality

Recovery, relevance,
