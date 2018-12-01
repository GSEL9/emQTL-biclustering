# emQTL-clustering
A research projecd supervised by Thomas Fleicher Oslo Radiom Hospital summer 2018, funded by [UiO: Life Science](https://www.uio.no/english/research/strategic-research-areas/life-science/). The project aimed at benchmarking biclustering algorithms wrt. Bonferroni corrected p-values relating gene expression CpG methylation. 

## Algorithms
* SpectralCoclustering (scikit-learn)
* sklearn SpectralBiclustering (scikit-learn)
* Plaid (R)
* Cheng Church (R)
* Bimax (R)
* Quest (R)
* Xmotifs (R)
* Spectral (R)

## Score metrics
* Recovery (custom implementaiton)
* Relevance (custom implementaiton)
* Jaccard Index (scikit-learn)
* Average Spearman's Rho (custom implementaiton)
* Transposed Virtual Error (custom implementaiton)
* Mean Squared Residue (custom implementaiton)
* Scaled Mean Squared Residue (custom implementaiton)

## Evaluation framework
Synthetic data, experiemntal data

### Model selection
Jaccard Index

### Bicluster quality
Recovery, relevance,
