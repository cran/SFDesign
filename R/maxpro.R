#' @name
#' maxpro.crit
#'
#' @title
#' Maximum projection (MaxPro) criterion
#'
#' @description
#' This function calculates the MaxPro criterion of a design.
#'
#' @details
#' \code{maxpro.crit} calculates the MaxPro criterion of a design. The MaxPro criterion for a design \eqn{D=[\bm x_1, \dots, \bm x_n]^T} is defined as \deqn{\left\{\frac{1}{{n\choose 2}}\sum_{i=1}^{n-1}\sum_{j=i+1}^{n}\frac{1}{\prod_{l=1}^p(x_{il}-x_{jl})^2+ \delta}\right\}^{1/p},} where \eqn{p} is the dimension of the design (Joseph, V. R., Gul, E., & Ba, S. 2015).
#'
#' @param design the design matrix.
#' @param delta a small value added to the denominator of the maximum projection criterion. By default it is set as zero.
#'
#' @return
#' the MaxPro criterion of the design.
#' @references
#' Joseph, V. R., Gul, E., & Ba, S. (2015). Maximum projection designs for computer experiments. Biometrika, 102(2), 371-380.
#'
#' @examples
#' n = 20
#' p = 3
#' D = randomLHD(n, p)
#' maxpro.crit(D)
#'
maxpro.crit = function(design, delta = 0){
  return (maxproCrit(design, 2, delta))
}
#' @name
#' maxproLHD
#'
#' @title
#' Generate a MaxPro Latin-hypercube design
#'
#' @description
#' This function generates a MaxPro Latin-hypercube design.
#'
#' @details
#' \code{maxproLHD} generates a MaxPro Latin-hypercube design (Joseph, V. R., Gul, E., & Ba, S. 2015). The major difference with the \code{MaxPro} packages is that we have a deterministic swap algorithm, which can be enabled by setting \code{method="deterministic"} or \code{method="full"}. For optimization details, see the detail section in \code{\link{customLHD}}.
#'
#' @param n design size.
#' @param p design dimension.
#' @param design an initial LHD. If design=NULL, a random LHD is generated.
#' @param max.sa.iter maximum number of swapping involved in the simulated annealing (SA) algorithm.
#' @param temp initial temperature of the simulated annealing algorithm. If temp=0, it will be automatically determined.
#' @param decay the temperature decay rate of simulated annealing.
#' @param no.update.iter.max the maximum number of iterations where there is no update to the global optimum before SA stops.
#' @param num.passes the maximum number of passes of the whole design matrix if deterministic swapping is used.
#' @param max.det.iter maximum number of swapping involved in the deterministic swapping algorithm.
#' @param method choice of "deterministic", "sa", or "full". If the method="full", the design is first optimized by SA and then deterministic swapping.
#' @param scaled whether the design is scaled to unit hypercube. If scaled=FALSE, the design is represented by integer numbers from 1 to design size. Leave it as TRUE when no initial design is provided.
#'
#' @return
#' \item{design}{final design points.}
#' \item{total.iter}{total number of swaps in the optimization.}
#' \item{criterion}{final optimized criterion.}
#' \item{crit.hist}{criterion history during the optimization process.}
#'
#' @references
#' Joseph, V. R., Gul, E., & Ba, S. (2015). Maximum projection designs for computer experiments. Biometrika, 102(2), 371-380.
#'
#' @examples
#' n = 20
#' p = 3
#' D = maxproLHD(n, p)
#'
maxproLHD = function(n, p, design=NULL,
                     max.sa.iter = 1e6, temp = 0, decay = 0.95, no.update.iter.max = 400,
                     num.passes = 10, max.det.iter = 1e6,
                     method = "full", scaled=TRUE) {
  s = 2
  if (is.null(design)){
    design = randomLHD(n, p)
  }
  if (scaled){
    design = design * n + 0.5
  }
  if (temp == 0){
    crit1 = 1 / (n-1)
    crit2 = (1 / ((n-1)^(p-1) * (n-2))) ^ (1/p)
    delta = crit2 - crit1
    temp = - delta / log(0.99)
  }
  if (method == "deterministic"){
    result = maxproLHDOptimizer_cpp(design, s, num.passes, max.det.iter,
                                    temp, decay, no.update.iter.max,
                                    method)
  }else if (method == "sa"){
    result = maxproLHDOptimizer_cpp(design, s, num.passes, max.sa.iter,
                                    temp, decay, no.update.iter.max,
                                    method)
  }else if (method == "full"){
    result = maxproLHDOptimizer_cpp(design, s, num.passes, max.sa.iter,
                                    temp, decay, no.update.iter.max,
                                    "sa")
    crit_hist = result$crit_hist
    result = maxproLHDOptimizer_cpp(result$design, s, num.passes, max.det.iter,
                                    temp, decay, no.update.iter.max,
                                    "deterministic")
    result$crit_hist = c(crit_hist, result$crit_hist)
  }
  result$design <- (apply(result$design, 2, rank) - 0.5)/n

  return (list(design = result$design,
               total.iter = result$total_iter,
               criterion = result$criterion,
               crit.hist = result$crit_hist))
}

#' @name
#' maxpro.remove
#'
#' @title
#' Sequentially remove design points from a design while maintaining low maximum projection criterion as possible
#'
#' @description
#' This function sequentially removes design points one-at-a-time from a design while maintaining low maximum projection criterion as possible.
#'
#' @details
#' \code{maxpro.remove} sequentially removes design points from a design while maintaining low maximum projection criterion (see \code{\link{maxpro.crit}}) as possible. The maximum projection criterion is modified to include a small delta term: \deqn{\phi_{\text{maxpro}}(\bm{D}_n) = \left\{\frac{1}{{n\choose 2}}\sum_{i=1}^{n-1}\sum_{j=i+1}^{n}\frac{1}{\prod_{l=1}^p(x_{il}-x_{jl})^2 +\delta}\right\}^{1/p}.} The index of the point to remove is \eqn{k^* =\arg\min_k \sum_{i \neq k}\frac{1}{\prod_{l=1}^p(x_{il}-x_{kl})^2 +\delta}}.
#'
#' @param D design
#' @param n.remove number of design points to remove
#' @param delta a small value added to the denominator of the maximum projection criterion. By default it is set as zero.
#'
#' @return
#' the updated design.
#'
#' @examples
#' n = 20
#' p = 3
#' n.remove =  5
#' D = maxproLHD(n, p)$design
#' D = maxpro.remove(D, n.remove)
maxpro.remove = function(D, n.remove, delta=0){
  if (n.remove <= 0){
    return (D)
  }
  n = nrow(D)
  p = ncol(D)

  d = 1 / (exp(distmatrix.maxpro(D)) ^ 2 + delta)
  dist.matrix = matrix(0, n, n)
  dist.matrix[lower.tri(dist.matrix)] = d
  dist.matrix = dist.matrix + t(dist.matrix)

  dist.vec = apply(dist.matrix, 1, sum)
  idx.left = 1:n
  for(k in 1:(n.remove)){
    idx = which.max(dist.vec)
    dist.matrix = dist.matrix[-idx, -idx]
    idx.left = idx.left[-idx]
    dist.vec = apply(dist.matrix, 1, sum)
  }
  return(D[idx.left,])
}
#'
#' @name
#' maxpro.optim
#'
#' @title
#' Optimize a design based on the maximum projection criterion
#'
#' @description
#' This function optimizes a design by continuous optimization based on maximum projection criterion (Joseph, V. R., Gul, E., & Ba, S. 2015).
#'
#' @details
#' \code{maxpro.optim} optimizes a design by L-BFGS-B algorithm (Liu and Nocedal 1989) based on the maximum projection criterion \code{\link{maxpro.crit}}.
#'
#' @param D.ini the initial design.
#' @param iteration number iterations for L-BFGS-B.
#'
#' @return
#' \item{design}{optimized design.}
#' \item{D.ini}{initial design.}
#'
#' @references
#' Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. Mathematical programming, 45(1), 503-528.
#'
#' Joseph, V. R., Gul, E., & Ba, S. (2015). Maximum projection designs for computer experiments. Biometrika, 102(2), 371-380.
#'
#' @examples
#' n = 20
#' p = 3
#' D = maxproLHD(n, p)$design
#' D = maxpro.optim(D)$design
#'
maxpro.optim = function(D.ini, iteration=10){
  sa = FALSE
  n = nrow(D.ini)
  p = ncol(D.ini)
  s = 2
  optim.obj = function(x){
    D = matrix(x, nrow=n, ncol=p)
    d = exp(distmatrix.maxpro(D))
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
      B[, j] = diag((1/A - I) %*% (1/d_matrix - I))
    }
    grad = s * B / fn
    return(list("objective"=lfn, "gradient"=grad))
  }

  sa.objective = function(x){
    D = matrix(x, nrow=n)
    return (maxpro.crit(D))
  }

  design = continuous.optim(D.ini, optim.obj, NULL, iteration, sa, sa.objective)
  return (list(design = design, D.ini = D.ini))
}
