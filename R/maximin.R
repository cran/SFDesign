#' @name
#' maximin.crit
#'
#' @title
#' Maximin criterion
#'
#' @description
#' This function calculates the maximin distance or the average reciprocal distance of a design.
#'
#' @details
#' \code{maximin.crit} calculates the maximin distance or the average reciprocal distance of a design. The maximin distance for a design \eqn{D=[\bm x_1, \dots, \bm x_n]^T} is defined as \eqn{\phi_{Mm} = \min_{i\neq j} \|\bm x_i- \bm x_j\|_2}. In optimization, the average reciprocal distance is usually used (Morris and Mitchell, 1995): \deqn{\phi_{\text{rec}} = \left(\frac{2}{n(n-1)} \sum_{i\neq j}\frac{1}{\|\bm{x}_i-\bm{x}_j\|_2^r}\right)^{1/r}.} The \eqn{r} is a power parameter and when it is large enough, the reciprocal distance is similar to the exact maximin distance.
#'
#' @param design the design matrix.
#' @param r the power used in the reciprocal distance objective function. The default value is set as twice the dimension of the design.
#' @param surrogate whether to return the surrogate average reciprocal distance objective function or the maximin distance. If setting surrogate=TRUE, then the average reciprocal distance is returned.
#'
#' @return
#' the maximin distance or reciprocal distance of the design.
#'
#' @references
#' Morris, M. D. and Mitchell, T. J. (1995), “Exploratory designs for computational experiments,” Journal of statistical planning and inference, 43, 381–402.
#'
#' @examples
#' n = 20
#' p = 3
#' D = randomLHD(n, p)
#' maximin.crit(D)
#'
maximin.crit = function(design, r = 2*ncol(design), surrogate = FALSE){
  if (surrogate){
    return (maximinObj(design, r))
  }else{
    return (maximinCrit(design))
  }
}
#'
#' @name
#' maximinLHD
#'
#' @title
#' Generate a maximin Latin-hypercube design (LHD)
#'
#' @description
#' This function generates a LHD with large maximin distance.
#'
#' @details
#' \code{maximinLHD} generates a LHD or optimize an existing LHD to achieve large maximin distance by optimizing the reciprocal distance (see \code{\link{maximin.crit}}). The optimization details can be found in \code{\link{customLHD}}.
#'
#' @param n design size.
#' @param p design dimension.
#' @param design an initial LHD. If design=NULL, a random LHD is generated.
#' @param power the power used in the maximin objective function.
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
#' @examples
#' # We show three different ways to use this function.
#' n = 20
#' p = 3
#' D.random = randomLHD(n, p)
#' # optimize over a random LHD by SA
#' D = maximinLHD(n, p, D.random, method='sa')
#' # optimize over a random LHD by deterministic swapping
#' D = maximinLHD(n, p, D.random, method='deterministic')
#' # Directly generate an optimized LHD for maximin criterion which goes
#' # through the above two optimization stages.
#' D = maximinLHD(n, p)
#'
maximinLHD = function (n, p, design=NULL, power = 2*p,
                       max.sa.iter = 1e6, temp = 0, decay = 0.95, no.update.iter.max = 100,
                       num.passes = 10,  max.det.iter = 1e6,
                       method = "full", scaled=TRUE) {
  if (is.null(design)){
    design = randomLHD(n, p)
  }
  if (scaled){
    design = design * n + 0.5
  }
  if (temp == 0){
    avg = p * n * (n+1) / 6
    delta = (avg - p) ^ (-0.5) - (avg) ^ (-0.5)
    temp = -delta / log(0.99)
  }
  if (method == "deterministic"){
    result = maximinLHDOptimizer_cpp(design, power, num.passes, max.det.iter,
                                     temp, decay, no.update.iter.max,
                                     method)
  }else if(method == "sa"){
    result = maximinLHDOptimizer_cpp(design, power, num.passes, max.sa.iter,
                                     temp, decay, no.update.iter.max,
                                     method)
  }else if (method == "full"){
    result = maximinLHDOptimizer_cpp(design, power, num.passes, max.sa.iter,
                                     temp, decay, no.update.iter.max,
                                     "sa")
    crit_hist = result$crit_hist
    result = maximinLHDOptimizer_cpp(result$design, power, num.passes, max.det.iter,
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
#' maximin.augment
#'
#' @title
#' Augment a design by adding new design points that minimize the reciprocal distance criterion greedily
#'
#' @description
#' This function augments a design by adding new design points one-at-a-time that minimize the reciprocal distance criterion.
#'
#' @details
#' \code{maximin.augment} augments a design by adding new design points that minimize the reciprocal distance criterion (see \code{\link{maximin.crit}}) greedily. In each iteration, the new design points is selected as the one from the candidate points that has the smallest sum of reciprocal distance to the existing design, that is, \eqn{\bm x_{k+1} = \arg\min_{\bm x} \sum_{i=1}^k \frac{1}{\|\bm x - \bm x_i\|^r}}.
#'
#' @param n the size of the final design.
#' @param p design dimension.
#' @param D.ini initial design.
#' @param candidate candidate points to choose from. The default candidates are Sobol points of size 100n.
#' @param r the power parameter in the \code{\link{maximin.crit}}. By default it is set as \code{2p}.
#'
#' @return
#' the augmented design.
#'
#' @examples
#' n.ini = 10
#' n = 20
#' p = 3
#' D.ini = maximinLHD(n.ini, p)$design
#' D = maximin.augment(n, p, D.ini)
maximin.augment = function(n, p, D.ini, candidate = NULL, r = 2*p){
  # reciprocal
  if (n<=nrow(D.ini)){
    return (D.ini)
  }
  if (is.null(candidate)){
    candidate = spacefillr::generate_sobol_set(100*n, p, seed = sample(10^6,1))
  }
  D = D.ini
  # reciprocal distance criterion
  dist.matrix = as.matrix(proxy::dist(candidate, D))
  dist.matrix = 1/(dist.matrix)^r
  candidate.dist.sum = apply(dist.matrix, 1, sum)
  idx.selected = c()
  n.new = n - nrow(D.ini)
  for(i in 1:n.new){
    idx = which.min(candidate.dist.sum)
    idx.selected = c(idx.selected, idx)
    D = rbind(D, candidate[idx,])
    dist.update = 1/apply((candidate - rep(1, nrow(candidate)) %*% t(candidate[idx,]))^2, 1, sum)^(r/2)
    candidate.dist.sum = candidate.dist.sum + dist.update
  }
  return(D)
}
#' @name
#' maximin.remove
#'
#' @title
#' Sequentially remove design points from a design while maintaining low reciprocal distance criterion as possible
#'
#' @description
#' This function sequentially removes design points one-at-a-time from a design while maintaining low reciprocal distance criterion as possible.
#'
#' @details
#' \code{maximin.remove} sequentially removes design points from a design in a greedy way while maintaining low reciprocal distance criterion (see \code{\link{maximin.crit}}) as possible. In each iteration, the design point with the largest sum of reciprocal distances with the other design points is removed, that is, \eqn{k^* = \arg\max_k \sum_{i\neq k}\frac{1}{\|\bm x_k - \bm x_i\|^r}}.
#'
#' @param D the design matrix.
#' @param n.remove number of design points to remove.
#' @param r the power parameter in the \code{\link{maximin.crit}}. By default it is set as \code{2p}.
#'
#' @return
#' the updated design.
#' @examples
#' n = 20
#' p = 3
#' n.remove =  5
#' D = maximinLHD(n, p)$design
#' D = maximin.remove(D, n.remove)
maximin.remove = function(D, n.remove, r = 2*p){
  if (n.remove<=0){
    return (D)
  }
  n = nrow(D)
  p = ncol(D)
  dist.matrix = as.matrix(dist(D))
  dist.matrix = 1/(dist.matrix+diag(n))^r-diag(n)
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
#' @keywords internal
#' @noRd
#' @name
#' maximin.ini
#'
#' @title
#' Generate a proper initial design for maximin design.
#'
#' @description
#' This function generates an initial design for maximin design without the LHD constraints.
#'
#' @details
#' \code{maximin.ini} generates an initial design for maximin design. It will first find the closest full factorial design and then either augment it or reduce it to the ideal design size.
#'
#' @param n design size
#' @param p design dimension
#' @param factorial whether use factorial design as candidate points
#'
#' @return
#' the updated design.
#' @examples
#' n = 20
#' p = 3
#' D = maximin.ini(n, p)
maximin.ini = function(n, p, factorial=TRUE){
  level.decimal = n^(1/p)
  level = floor(level.decimal)
  if (n > ((level+1)^p+level^p)/2){
    # remove from a larger design
    level = level + 1
    D.ini = full.factorial(p, level)
    D = maximin.remove(D.ini, n.remove=nrow(D.ini)-n)
  }else{
    # augment from a smaller design
    D.ini = full.factorial(p, level)
    if (factorial){
      candidate = full.factorial(p, level+1)
      D = maximin.augment(n, p, D.ini, candidate)
    }else{
      D = maximin.augment(n, p, D.ini)
    }
  }
  return (D)
}

#' @name
#' maximin.optim
#'
#' @title
#' Optimize a design based on maximin or reciprocal distance criterion.
#'
#' @description
#' This function optimizes a design by continuous optimization based on reciprocal distance criterion. A simulated annealing step can be enabled in the end to directly optimize the maximin distance criterion.
#'
#' @details
#' \code{maximin.optim} optimizes a design by L-BFGS-B algorithm (Liu and Nocedal 1989) based on the reciprocal distance criterion. A simulated annealing step can be enabled in the end to directly optimize the maximin distance criterion. Optimization detail can be found in \code{\link{continuous.optim}}. We also provide the option to try other initial designs generated internally besides the \code{D.ini} provided by the user (see argument \code{find.best.ini}).
#'
#' @param D.ini the initial design.
#' @param iteration number iterations for L-BFGS-B algorithm.
#' @param sa whether to use simulated annealing in the end. If sa=TRUE, continuous optimization is first used to optimize the reciprocal distance criterion and then SA is performed to optimize the maximin criterion.
#' @param find.best.ini whether to generate other initial designs. If find.best.ini=TRUE, it will first find the closest full factorial design in terms of size to \code{D.ini}. If the size of the full factorial design is larger than \code{D.ini}, design points will be removed by the \code{maximin.remove} function. If the the size of the full factorial design is smaller than \code{D.ini}, then we will augment the design by \code{maximin.augment}. In this case, we have two different candidate set to choose from: one is a full factorial design with level+1 and another is Sobol points. All initial designs are optimized and the best is returned.
#'
#' @return
#' \item{design}{the optimized design.}
#' \item{D.ini}{initial designs. If find.best.ini=TRUE, a list will be returned containing all the initial designs considered.}
#'
#' @references
#' Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. Mathematical programming, 45(1), 503-528.
#'
#' @examples
#' n = 20
#' p = 3
#' D = maximinLHD(n, p)$design
#' D = maximin.optim(D, sa=FALSE)$design
#' # D = maximin.optim(D, sa=TRUE)$design # Let sa=TRUE only when the n and p is not large.
maximin.optim = function(D.ini, iteration=10, sa=FALSE, find.best.ini=FALSE){
  n = nrow(D.ini)
  p = ncol(D.ini)
  power = 2 * p
  optim.obj = function(x){
    D = matrix(x, nrow=n, ncol=p)
    d = distmatrix.maximin(D)
    d_matrix = matrix(0, n, n)
    d_matrix[lower.tri(d_matrix)] = d
    d_matrix = d_matrix + t(d_matrix)
    # obj
    log_d = log(d)
    Dmin <- min(log_d)
    fn <- sum(exp(power * (Dmin - log_d)))
    lfn <- log(fn) - power * Dmin
    fn <- exp(lfn)
    # grad
    grad =  D
    for(row in 1:n){
      A = sweep(D, 2, D[row, ], '-')
      A = A[-row, ]
      grad[row, ] = apply(A/d_matrix[row, -row]^(power+2), 2, sum)
    }
    grad = power * grad / fn
    return(list("objective"=lfn, "gradient"=c(grad)))
  }

  sa.objective = function(x){
    D = matrix(x, nrow=n)
    return (-maximin.crit(D))
  }

  if (find.best.ini){
    D.ini.fac = maximin.ini(n, p)
    D.ini.sobol = maximin.ini(n, p, FALSE)
    if (identical(D.ini.fac, D.ini.sobol)){
      D.fac = continuous.optim(D.ini.fac, optim.obj, NULL, iteration, sa, sa.objective)
      D.sobol = D.fac
    }else{
      D.fac = continuous.optim(D.ini.fac, optim.obj, NULL, iteration, sa, sa.objective)
      D.sobol = continuous.optim(D.ini.sobol, optim.obj, NULL, iteration, sa, sa.objective)
    }
    D.user = continuous.optim(D.ini, optim.obj, NULL, iteration, sa, sa.objective)
    fac.crit = maximin.crit(D.fac)
    sobol.crit = maximin.crit(D.sobol)
    user.crit = maximin.crit(D.user)
    max.idx = which.max(c(fac.crit, sobol.crit, user.crit))
    if (max.idx==1){
      design = D.fac
    }else if (max.idx==2){
      design = D.sobol
    }else{
      design = D.user
    }
    result = list(design=design, D.ini = list(D.ini, D.ini.fac, D.ini.sobol))
  }else{
    design = continuous.optim(D.ini, optim.obj, NULL, iteration, sa, sa.objective)
    result = list(design = design, D.ini = D.ini)
  }

  return (result)
}

