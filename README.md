# emQTL-clustering
Comparing the performance of biclustering algorithms applied to a data matrix containing of Bonferroni corrected p-values. The p-values are calculated from Pearson's correlations between DNA methylated CpG data and gene expression data. The aim of the clustering is to detect relations between genes and CpG methylation level to elucidate biological relevance of emQTL.


## Todo
* Rerun benchmarking. Check if opt. hparams of R wrapped models are same as default params (indicates hparams not being updated in wrappers during GS).
* ERROR: R plotting function displays single green cluster only. Check R alg performance and apply best perfom. alg. to corr. dataset to confirm function error.
* Adjust test data towards improved resemblance?
* Verify result model selection procedure by applying discarded models to reference data and compare the results to the outcome of applying the selected models to the reference data.
* Bicluster method for saving predictions to disk.
* Automatically remove empty biclusters when creating bicluster instances for model results?


## Observations
