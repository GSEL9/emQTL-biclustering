# -*- coding: utf-8 -*-
#
# biclust_graphics.R
#
# Runs R biclust algorithms with determined optimal parameter settings
# in order to generate heatmaps displaying the biclsutering results.
#


library('biclust')

print('Loading data...')

# Read data into memory.
frame <- read.csv(file='./../data/train/sel_pvalues_prep.csv', sep=',')
#frame <- read.csv(file='./dummy.csv', sep=',')
mat = data.matrix(frame)

print('Loading data complete')

# Setup.
N_CLUSTERS = 2
figure_name = './../predictions/imgs/r_sel_pvalues_prep.png'

print('Model training...')

# Perform biclustering.
model <- biclust(
  mat, method='BCCC', delta=1.5, alpha=0.1, number=N_CLUSTERS
)

print('Model training complete')

# Gen graphics.
grapher <- dget('drawHeatmap3.R')

print('Generating graphics...')

png(figure_name)
grapher(mat, bicResult=model)
dev.off()

print('Graphics complete')
