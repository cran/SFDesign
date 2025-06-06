% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/continuousOptim.R
\name{continuous.optim}
\alias{continuous.optim}
\title{Continuous optimization of a design}
\usage{
continuous.optim(
  D.ini,
  objective,
  gradient = NULL,
  iteration = 10,
  sa = FALSE,
  sa.objective = NULL
)
}
\arguments{
\item{D.ini}{initial design matrix.}

\item{objective}{the criterion to minimize for the design. It can also return gradient information at the same time in a list with elements "objective" and "gradient".}

\item{gradient}{the gradient of the objective with respect to the design.}

\item{iteration}{number iterations for LBFGS.}

\item{sa}{whether to use simulated annealing. If the final criterion is different from the objective function specified above, simulated annealing can be useful. Use this option only when the design size and dimension are not large.}

\item{sa.objective}{the criterion to minimize for the simulated annealing.}
}
\value{
the optimized design.
}
\description{
This function does continuous optimization of an existing design based on a specified criterion. It has an option to run simulated annealing after the continuous optimization.
}
\details{
\code{continuous.optim} optimizes an existing design based on a specified criterion. It is a wrapper for the L-BFGS-B function from the nloptr packakge (Johnson 2008) and/or GenSA function in GenSA package (Xiang, Gubian, Suomela and Hoeng 2013).
}
\examples{
# Below is an example showing how to create functions needed to generate MaxPro design manually by
# continuous.optim without using the maxpro.optim function in the package.
compute.distance.matrix <- function(A){
   log_prod_metric = function(x, y) 2 * sum(log(abs(x-y)))
   return (c(proxy::dist(A, log_prod_metric)))
}
optim.obj = function(x){
  D = matrix(x, nrow=n, ncol=p)
  d = exp(compute.distance.matrix(D))
  d_matrix = matrix(0, n, n)
  d_matrix[lower.tri(d_matrix)] = d
  d_matrix = d_matrix + t(d_matrix)
  fn = sum(1/d)
  lfn = log(fn)
  I = diag(n)
  diag(d_matrix) = rep(1,n)
  A = B = D
  for(j in 1:p)
  {
    A = t(outer(D[,j], D[,j], "-"))
    diag(A) = rep(1, n)
    B[, j] = diag((1/A - I) \%*\% (1/d_matrix - I))
  }
  grad = 2 * B / fn
  return(list("objective"=lfn, "gradient"=grad))
}
n = 20
p = 3
D.ini = maxproLHD(n, p)$design
D = continuous.optim(D.ini, optim.obj)


}
\references{
Johnson, S. G. (2008), The NLopt nonlinear-optimization package, available at https://github.com/stevengj/nlopt.
Xiang Y, Gubian S, Suomela B, Hoeng (2013). "Generalized Simulated Annealing for Efficient Global Optimization: the GenSA Package for R". The R Journal Volume 5/1, June 2013.
}
