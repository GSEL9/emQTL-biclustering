# -*- coding: utf-8 -*-
#
# drawHeatmap3.R
#


drawHeatmap3 = function(x, bicResult=NULL) {
  # The function builds on refactoring and modification of the R biclust
  # package drawHeatmap2 function.

  # Setup
  nrows = nrow(x)
  ncols = ncol(x)
  row_x_num = bicResult@RowxNumber
  num_x_col = bicResult@NumberxCol

  # Graphics color specs.
  numColores = 255 * 2
  gvect = c(array(255:0), array(0, dim=255))
  rvect = c(array(0, dim=255), array(0:255))
  bvect = array(0, dim=numColores)
  paleta = rgb(rvect, gvect, bvect, 255, maxColorValue=255)
  oldmai = par('mai')
  oldmar = par('mar')
  par(mai=c(0, 0, 0, 0), mar=c(0, 0, 0, 0))

  bicRows = row(matrix(row_x_num[, 1]))[row_x_num[, 1] == T]
  bicCols = row(matrix(num_x_col[1, ]))[num_x_col[1, ] == T]
  rowlength <- 1:bicResult@Number
  rowlength[1] <- length(bicRows)
  collength <- 1:bicResult@Number
  collength[1] <- length(bicCols)

  if (bicResult@Number >= 2) {
     for (i in 2:bicResult@Number) {
       bicRows = c(bicRows, row(matrix(row_x_num[, i]))[row_x_num[, i] == T])
       rowlength[i] <- length(bicRows)

       bicCols = c(bicCols, row(matrix(num_x_col[i, ]))[num_x_col[i, ] == T])
       collength[i] <- length(bicCols)
     }
  }
  # NOTE: Handling overlapping biclusters.
  bicRows = unique(bicRows)
  bicCols = unique(bicCols)
  # QUESTION: Replaced with R which()?
  col_uniques = c(setdiff(c(1:nrows), bicRows), bicRows)
  row_uniques = c(bicCols, setdiff(c(1:ncols), bicCols))

  image(
    1:ncols, 1:nrows, t(x[col_uniques, row_uniques]), col=paleta, axes=FALSE
  )

  desp = (nrows - rowlength[1]) / nrows
  grid.lines(
    x=unit(c(0, 1), 'npc'),
    y=unit(c(desp, desp), 'npc'),
    gp=gpar(col='yellow')
  )

  desp = (collength[1]) / ncols
  grid.lines(
    y=unit(c(0, 1), 'npc'),
    x=unit(c(desp, desp), 'npc'),
    gp=gpar(col='yellow')
  )
  for (i in 2:bicResult@Number) {
     desp = (nrows - rowlength[i]) / nrows
     grid.lines(
       x=unit(c(0, 1), 'npc'),
       y=unit(c(desp, desp), 'npc'),
       gp=gpar(col='yellow')
     )
     desp = (collength[i] - collength[i - 1]) / ncols
     grid.lines(
       y=unit(c(0, 1), 'npc'),
       x=unit(c(desp, desp), 'npc'),
       gp=gpar(col='yellow')
     )
  }
  par(mai = oldmai, mar = oldmar)
}
