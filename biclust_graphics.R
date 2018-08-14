# Runs R biclust algorithms with determined optimal parameter settings
# in order to generate heatmaps displaying the biclsutering results.

library('biclust')

# Reading data into memory and converting to matrix.
#data <- read.csv(file='./../data/train/sel_pvalues_prep.csv', sep=',')
frame <- read.csv(file='./dummy.csv', sep=',')
x = data.matrix(frame)

# Setup
n_clusters = 2
figure_name = './../dummy_hm.png'

# Performing biclustering
model <- biclust(
  x, method='BCCC',
  delta=1.5, alpha=0.1, number=n_clusters
)


# Plotting
grapher <- dget('drawHeatmap3.R')

png(figure_name)
grapher(x=x, bicResult=model)
dev.off()
