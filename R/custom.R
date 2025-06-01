#' @name
#' customLHD
#'
#' @title
#' Generate a Latin-hypercube design (LHD) based on a custom criterion.
#'
#' @description
#' This function generates a LHD by minimizing a user-specified design criterion.
#'
#' @details
#' \code{customLHD} generates a LHD by minimizing a user-specified design criterion.
#' \itemize{
#'  \item If \code{method='sa'}, then a simulated annealing algorithm is used to optimize the LHD. To custom the optimization process, you can change the default values for \code{max.sa.iter, temp, decay, no.update.iter.max}. In this optimization step, two design points are randomly chosen and their coordinate along one dimension are swaped. If the new design improves the criterion, then it is accepted; otherwise, it is accepted with some probability.
#'  \item If \code{method='deterministic'}, then a deterministic swap algorithm is used to optimize the LHD. To custom the optimization process, you can change the default values for \code{num.passes, max.det.iter}. In this optimization step, we swap the coordinates of all pairs of design points (start with design point 1 with design point 2, then 1 with 3, ... 1 with n, then 2 with 3 until n-1 with n). Only accept the change if the swap leads to an improvement.
#'  \item If \code{method='full'}, then optimization goes through the above two stages.
#' }
#'
#' @param compute.distance.matrix a function to calculate pairwise distance
#' @param compute.criterion a function to calculate the criterion based on the pairwise distance
#' @param update.distance.matrix a function to update the distance matrix after swapping one column of two design points
#' @param n design size.
#' @param p design dimension.
#' @param design an initial LHD. If design=NULL, a random LHD is generated.
#' @param max.sa.iter maximum number of swapping involved in the simulated annealing (SA) algorithm.
#' @param temp initial temperature of the simulated annealing algorithm. If temp=0, it will be automatically determined.
#' @param decay the temperature decay rate of simulated annealing.
#' @param no.update.iter.max the maximum number of iterations where there is no update to the global optimum before SA stops.
#' @param num.passes the maximum number of passes of the whole design matrix if deterministic swapping is used.
#' @param max.det.iter maximum number of swapping involved in the deterministic swapping algorithm.
#' @param method choice of "deterministic", "sa", or "full". See details for the description of each choice.
#' @param scaled whether the design is scaled to unit hypercube. If scaled=FALSE, the design is represented by integer numbers from 1 to design size. Leave it as TRUE when no initial design is provided.
#'
#' @return
#' \item{design}{optimized LHD.}
#' \item{total.iter}{total number of swaps in the optimization.}
#' \item{criterion}{optimized criterion.}
#' \item{crit.hist}{criterion history during the optimization process.}
#'
#' @examples
#' # Below is an example showing how to create functions needed to generate
#' # MaxPro LHD manually by customLHD without using the maxproLHD function in
#' # the package.
#' compute.distance.matrix <- function(A){
#'    s = 2
#'    log_prod_metric = function(x, y) s * sum(log(abs(x-y)))
#'    return (c(proxy::dist(A, log_prod_metric)))
#' }
#' compute.criterion <- function(n, p, d) {
#'   s = 2
#'   dim <- as.integer(n * (n - 1) / 2)
#'   # Find the minimum distance
#'   Dmin <- min(d)
#'   # Compute the exponential summation
#'   avgdist <- sum(exp(Dmin - d))
#'   # Apply the logarithmic transformation and scaling
#'   avgdist <- log(avgdist) - Dmin
#'   avgdist <- exp((avgdist - log(dim)) * (p * s) ^ (-1))
#'   return(avgdist)
#' }
#'
#' update.distance.matrix <- function(A, col, selrow1, selrow2, d) {
#'   s = 2
#'   n = nrow(A)
#'   # transform from c++ idx to r idx
#'   selrow1 = selrow1 + 1
#'   selrow2 = selrow2 + 1
#'   col = col + 1
#'   # A is the updated matrix
#'   row1 <- min(selrow1, selrow2)
#'   row2 <- max(selrow1, selrow2)
#'
#'   compute_position <- function(row, h, n) {
#'     n*(h-1) - h*(h-1)/2 + row-h
#'   }
#'
#'   # Update for rows less than row1
#'   if (row1 > 1) {
#'     for (h in 1:(row1-1)) {
#'       position1 <- compute_position(row1, h, n)
#'       position2 <- compute_position(row2, h, n)
#'       d[position1] <- d[position1] + s * log(abs(A[row1, col] - A[h, col])) -
#'           s * log(abs(A[row2, col] - A[h, col]))
#'       d[position2] <- d[position2] + s * log(abs(A[row2, col] - A[h, col])) -
#'           s * log(abs(A[row1, col] - A[h, col]))
#'     }
#'   }
#'
#'   # Update for rows between row1 and row2
#'   if ((row2-row1) > 1){
#'     for (h in (row1+1):(row2-1)) {
#'       position1 <- compute_position(h, row1, n)
#'       position2 <- compute_position(row2, h, n)
#'       d[position1] <- d[position1] + s * log(abs(A[row1, col] - A[h, col])) -
#'           s * log(abs(A[row2, col] - A[h, col]))
#'       d[position2] <- d[position2] + s * log(abs(A[row2, col] - A[h, col])) -
#'           s * log(abs(A[row1, col] - A[h, col]))
#'     }
#'   }
#'
#'   # Update for rows greater than row2
#'   if (row2 < n) {
#'     for (h in (row2+1):n) {
#'       position1 <- compute_position(h, row1, n)
#'       position2 <- compute_position(h, row2, n)
#'       d[position1] <- d[position1] + s * log(abs(A[row1, col] - A[h, col])) -
#'           s * log(abs(A[row2, col] - A[h, col]))
#'       d[position2] <- d[position2] + s * log(abs(A[row2, col] - A[h, col])) -
#'           s * log(abs(A[row1, col] - A[h, col]))
#'     }
#'   }
#'   return (d)
#' }
#'
#' n = 6
#' p = 2
#' # Find an appropriate initial temperature
#' crit1 = 1 / (n-1)
#' crit2 = (1 / ((n-1)^(p-1) * (n-2))) ^ (1/p)
#' delta = crit2 - crit1
#' temp = - delta / log(0.99)
#' result_custom = customLHD(compute.distance.matrix,
#' function(d) compute.criterion(n, p, d),
#' update.distance.matrix, n, p, temp = temp)
#'
customLHD = function(compute.distance.matrix,
                     compute.criterion,
                     update.distance.matrix,
                     n, p, design=NULL,
                     max.sa.iter = 1e6, temp = 0, decay = 0.95, no.update.iter.max = 400,
                     num.passes = 10, max.det.iter = 1e6,
                     method = "full", scaled=TRUE) {
  if (is.null(design)){
    design = randomLHD(n, p)
  }
  if (!scaled){
    design = (apply(design, 2, rank) - 0.5)/n
  }

  if (method == "deterministic"){
    result = customLHDOptimizer_cpp(compute.distance.matrix,
                                     compute.criterion,
                                     update.distance.matrix,
                                     design, num.passes, max.det.iter,
                                     temp, decay, no.update.iter.max,
                                     method)
  }else if(method == "sa"){
    result = customLHDOptimizer_cpp(compute.distance.matrix,
                                     compute.criterion,
                                     update.distance.matrix,
                                     design, num.passes, max.sa.iter,
                                     temp, decay, no.update.iter.max,
                                     method)
  }else if (method == "full"){
    result = customLHDOptimizer_cpp(compute.distance.matrix,
                                     compute.criterion,
                                     update.distance.matrix,
                                     design, num.passes, max.sa.iter,
                                     temp, decay, no.update.iter.max,
                                     "sa")
    crit_hist = result$crit_hist
    result = customLHDOptimizer_cpp(compute.distance.matrix,
                                     compute.criterion,
                                     update.distance.matrix,
                                     result$design, num.passes, max.det.iter,
                                     temp, decay, no.update.iter.max,
                                     "deterministic")
    result$crit_hist = c(crit_hist, result$crit_hist)
  }
  return (result)
}
