#' @name
#' continuous.optim
#'
#' @title
#' Continuous optimization of a design
#'
#' @description
#' This function does continuous optimization of an existing design based on a specified criterion. It has an option to run simulated annealing after the continuous optimization.
#'
#' @details
#' \code{continuous.optim} optimizes an existing design based on a specified criterion. It is a wrapper for the L-BFGS-B function from the nloptr packakge (Johnson 2008) and/or GenSA function in GenSA package (Xiang, Gubian, Suomela and Hoeng 2013).
#'
#' @param D.ini initial design matrix.
#' @param objective the criterion to minimize for the design. It can also return gradient information at the same time in a list with elements "objective" and "gradient".
#' @param gradient the gradient of the objective with respect to the design.
#' @param iteration number iterations for LBFGS.
#' @param sa whether to use simulated annealing. If the final criterion is different from the objective function specified above, simulated annealing can be useful. Use this option only when the design size and dimension are not large.
#' @param sa.objective the criterion to minimize for the simulated annealing.
#'
#' @return
#' the optimized design.
#'
#' @references
#' Johnson, S. G. (2008), The NLopt nonlinear-optimization package, available at https://github.com/stevengj/nlopt.
#' Xiang Y, Gubian S, Suomela B, Hoeng (2013). "Generalized Simulated Annealing for Efficient Global Optimization: the GenSA Package for R". The R Journal Volume 5/1, June 2013.
#'
#' @examples
#' # Below is an example showing how to create functions needed to generate MaxPro design manually by
#' # continuous.optim without using the maxpro.optim function in the package.
#' compute.distance.matrix <- function(A){
#'    log_prod_metric = function(x, y) 2 * sum(log(abs(x-y)))
#'    return (c(proxy::dist(A, log_prod_metric)))
#' }
#' optim.obj = function(x){
#'   D = matrix(x, nrow=n, ncol=p)
#'   d = exp(compute.distance.matrix(D))
#'   d_matrix = matrix(0, n, n)
#'   d_matrix[lower.tri(d_matrix)] = d
#'   d_matrix = d_matrix + t(d_matrix)
#'   fn = sum(1/d)
#'   lfn = log(fn)
#'   I = diag(n)
#'   diag(d_matrix) = rep(1,n)
#'   A = B = D
#'   for(j in 1:p)
#'   {
#'     A = t(outer(D[,j], D[,j], "-"))
#'     diag(A) = rep(1, n)
#'     B[, j] = diag((1/A - I) %*% (1/d_matrix - I))
#'   }
#'   grad = 2 * B / fn
#'   return(list("objective"=lfn, "gradient"=grad))
#' }
#' n = 20
#' p = 3
#' D.ini = maxproLHD(n, p)$design
#' D = continuous.optim(D.ini, optim.obj)
#'
#'
continuous.optim = function(D.ini, objective, gradient=NULL, iteration=10,
                            sa=FALSE, sa.objective=NULL){
  D0 = as.matrix(D.ini)
  n = nrow(D0)
  p = ncol(D0)

  objective.list.flag = is.list(objective(D.ini))
  if (is.null(gradient) & (!objective.list.flag)){
    optim.obj = objective
    D1 = D0
    for(i in 1:iteration)
    {
      a = stats::optim(D1, optim.obj, method='L-BFGS-B', lower=rep(0, n*p), upper=rep(1, n*p))
      D1 = matrix(a$par, nrow=n, ncol=p)
    }
  }else{
    if (objective.list.flag){
      optim.obj = objective
    }else{
      optim.obj = function(x){
        return(list("objective"=objective(x), "gradient"=gradient(x)))
      }
    }
    D1 = D0
    for(i in 1:iteration)
    {
      a = nloptr::nloptr(c(D1), optim.obj, opts=list("algorithm"="NLOPT_LD_LBFGS", "maxeval"=100),
                         lb=rep(0, n*p), ub=rep(1, n*p))
      D1 = matrix(a$sol, nrow=n, ncol=p)
    }
  }

  if (sa){
    if (is.null(sa.objective)){
      warning("No sa objective is provided!")
    }
    result = GenSA::GenSA(D1, function(x) sa.objective(x),
                          lower = rep(0, n*p), upper = rep(1, n*p))
    D1 = matrix(result$par, ncol=p)
  }

  return (D1)
}
