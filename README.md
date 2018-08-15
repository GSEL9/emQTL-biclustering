# emQTL-clustering
Comparing the performance of biclustering algorithms applied to a data matrix containing of Bonferroni corrected p-values. The p-values are calculated from Pearson's correlations between DNA methylated CpG data and gene expression data. The aim of the clustering is to detect relations between genes and CpG methylation level to elucidate biological relevance of emQTL.

## Todo
* Checkout [paper](http://www.scitepress.org/Papers/2018/66625/66625.pdf) for ref on relevance scores.
* Rerun benchmarking. Check if opt. hparams of R wrapped models are same as default params (indicates hparams not being updated in wrappers during GS). NB: heatmapBC() potential to replace drawHeatmap3?
* Verify result model selection procedure by applying discarded models to reference data and compare the results to the outcome of applying the selected models to the reference data.
* Bicluster method for saving predictions to disk.
* Automatically remove empty biclusters when creating bicluster instances for model results?
* Average Spearman’s Rho evaluation measure is reported to be the most effective criteria to improve bicluster relevance
* Adjust test data towards improved resemblance?

# Notes to algorithms
* sklearn SpectralCoclustering:
  * supports only sparse matrices if nonnegative.
* sklearn SpectralBiclustering:
  * Assumes checkerboard structure.
* R Plaid:
* R Cheng Church:
* R Bimax:
* R Quest:
* R

## Observations
