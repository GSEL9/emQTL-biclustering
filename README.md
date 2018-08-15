# emQTL-clustering
Comparing the performance of biclustering algorithms applied to a data matrix containing of Bonferroni corrected p-values. The p-values are calculated from Pearson's correlations between DNA methylated CpG data and gene expression data. The aim of the clustering is to detect relations between genes and CpG methylation level to elucidate biological relevance of emQTL.


## Todo
* ERROR: R plotting function displays single green cluster only. Check R alg performance and apply best perfom. alg. to corr. dataset to confirm function error.
* Adjust test data towards improved resemblance?
* Ensure model selection procedure by applying discarded models to reference data and compare the results to the outcome of applying the selected models to the reference data.
* Increase size of param grid for extended grid search.
* Bicluster method for saving predictions to disk.


## Observations
* Spectral (sklearn) models unable to handle sparse data?
  * Typ. results in one single large cluster.
  * The *sel_pvalues* data captures both ref. cluster one and two in a single cluster.
  * The *sel_pcc* desults in ref. samples spread across clusters.
* Running $n \in {2, 3, 4}$:
  * SpectralCoclustering(applied to tot./orig. p-values data:
* Running $n=5$:
  * SpectralCoclustering(applied to tot./orig. p-values data:
    * Approx. $90-90 \%$ of ref. cluster 1 contents stored in single cluster.
    * Cluster four and five captures together approx $90 \%$ of the contents in ref. cluster 2.
      * According to paper, ref. cluster 2 was divided into
* All interesting results typ. corr. to $< 50 \%$ relevance.
* Better to cluster $-\log(p-values)$ because these typically results in larger values than the original p-values and PCCs?
