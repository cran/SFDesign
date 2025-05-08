#' @name
#' uniform.crit
#'
#' @title
#' Uniform criterion
#'
#' @description
#' This function calculates the wrap-around discrepancy of a design.
#'
#' @details
#' \code{uniform.crit} calculates the wrap-around discrepancy of a design. The wrap-around discrepancy for a design \eqn{D=[\bm x_1, \dots, \bm x_n]^T} is defined as (Hickernell, 1998):
#' \deqn{\phi_{wa} = -\left(\frac{4}{3}\right)^p + \frac{1}{n^2}\sum_{i,j=1}^n\prod_{k=1}^p\left[\frac{3}{2} - |x_{ik}-x_{jk}|(1-|x_{ik}-x_{jk}|)\right].}
#'
#' @param design a design matrix.
#'
#' @return
#' wrap-around discrepancy of the design
#'
#' @references
#' Hickernell, F. (1998), “A generalized discrepancy and quadrature error bound,” Mathematics of computation, 67, 299–322.
#'
#' @examples
#' n = 20
#' p = 3
#' D = randomLHD(n, p)
#' uniform.crit(D)
#'
uniform.crit = function(design){
  return (uniformCrit(design))
}
#'
#' @name
#' uniformLHD
#'
#' @title
#' Generate a uniform Latin-hypercube design (LHD)
#'
#' @description
#' This function generates a uniform LHD by minimizing the wrap-around discrepancy.
#'
#' @details
#' \code{uniformLHD} generates a uniform LHD minimizing wrap-around discrepancy (see \code{\link{uniform.crit}}). The optimization details can be found in \code{\link{customLHD}}.
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
#' @examples
#' n = 20
#' p = 3
#' D = uniformLHD(n, p)
#'
uniformLHD = function(n, p, design = NULL,
                      max.sa.iter = 1e6, temp = 0, decay = 0.95, no.update.iter.max = 400,
                      num.passes = 10, max.det.iter = 1e6,
                      method = "full", scaled = TRUE) {
  if (is.null(design)){
    design = randomLHD(n, p)
  }
  if (!scaled){
    design = (apply(design, 2, rank) - 0.5)/n
  }
  if (temp == 0){
    crit1 = sqrt(-(4/3)^p + (3/2)^p*(n^2-2)/n^2 + (5/4)^p*2/n^2)
    crit2 = sqrt(-(4/3)^p + (3/2)^p)
    delta = crit2 - crit1
    temp = - delta / log(0.99)
  }
  if (method == "deterministic"){
    result = uniformLHDOptimizer_cpp(design, num.passes, max.det.iter,
                                     temp, decay, no.update.iter.max,
                                     method)
  }else if(method == "sa"){
    result = uniformLHDOptimizer_cpp(design, num.passes, max.sa.iter,
                                  temp, decay, no.update.iter.max,
                                  method)
  }else if (method == "full"){
    result = uniformLHDOptimizer_cpp(design, num.passes, max.sa.iter,
                                  temp, decay, no.update.iter.max,
                                  "sa")
    crit_hist = result$crit_hist
    result = uniformLHDOptimizer_cpp(result$design, num.passes, max.det.iter,
                                  temp, decay, no.update.iter.max,
                                  "deterministic")
    result$crit_hist = c(crit_hist, result$crit_hist)
  }
  return (list(design = result$design,
               total.iter = result$total_iter,
               criterion = result$criterion,
               crit.hist = result$crit_hist))
}
#' @name
#' uniform.optim
#'
#' @title
#' Optimize a design based on the wrap-around discrepancy
#'
#' @description
#' This function optimizes a design through continuous optimization of the wrap-around discrepancy.
#'
#' @details
#' \code{uniform.optim} optimizes a design through continuous optimization of the wrap-around discrepancy (see \code{\link{uniform.crit}}) by L-BFGS-B algorithm (Liu and Nocedal 1989).
#'
#' @param D.ini the initial design.
#' @param iteration number iterations for LBFGS.
#'
#' @return
#' \item{design}{optimized design.}
#' \item{D.ini}{initial design.}
#' @references
#' Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. Mathematical programming, 45(1), 503-528.
#'
#' @examples
#' n = 20
#' p = 3
#' D = uniformLHD(n, p)$design
#' D = uniform.optim(D)$design
#'
uniform.optim = function(D.ini, iteration = 10){
  sa = FALSE
  n = nrow(D.ini)
  p = ncol(D.ini)
  optim.obj = function(x){
    D = matrix(x, nrow=n, ncol=p)
    d = exp(distmatrix.uniform(D))
    d_matrix = matrix(0, n, n)
    d_matrix[lower.tri(d_matrix)] = d
    d_matrix = d_matrix + t(d_matrix)
    # obj
    fn <- sum(d)
    # grad
    grad = D
    for(col in 1:p){
      diff = outer(D[, col], D[, col], "-")
      diff1 = 3/2 - abs(diff) + diff^2
      grad[, col] = apply(d_matrix / diff1 *(-1*sign(diff) + 2*diff), 1, sum)
    }
    return(list("objective"=fn, "gradient"=c(grad)))
  }

  sa.objective = function(x){
    D = matrix(x, nrow=n)
    return (uniform.crit(D))
  }

  design = continuous.optim(D.ini, optim.obj, NULL, iteration, sa, sa.objective)
  return (list(design = design, D.ini = D.ini))
}
#'
#' @name
#' uniform.discrete
#'
#' @title
#' Generate a uniform design for discrete factors with different number of levels
#'
#' @description
#' This function generates a uniform design for discrete factors with different number of levels.
#'
#' @details
#' \code{uniform.discrete} generates a uniform design of discrete factors with different number of levels by minimizing the wrap-around discrepancy criterion (see \code{\link{uniform.crit}}).
#'
#' @param t multiple of the least common multiple of the levels.
#' @param p design dimension.
#' @param levels a vector of the number of levels for each dimension.
#' @param design an initial design. If design=NULL, a random design is generated.
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
#' \item{design.int}{design transformed to integer numbers for each dimenion}
#' \item{total.iter}{total number of swaps in the optimization.}
#' \item{criterion}{final optimized criterion.}
#' \item{crit.hist}{criterion history during the optimization process.}
#'
#' @examples
#' p = 5
#' levels = c(3, 4, 6, 2, 3)
#' t = 1
#' D = uniform.discrete(t, p, levels)
uniform.discrete = function(t, p, levels, design=NULL,
                         max.sa.iter = 1e6, temp = 0, decay = 0.95, no.update.iter.max = 400,
                         num.passes = 10, max.det.iter = 1e6,
                         method = "full", scaled = TRUE) {
  lcm = primes::Rscm(levels)
  n = t * lcm
  if (is.null(design)){
    design = matrix(0, nrow = n, ncol = p)
    for (j in 1:p) {
      s = (rep(seq(from=1, to=levels[j]), n/levels[j]) - 0.5)/levels[j]
      # Randomly shuffle the sequence
      design[, j] <- sample(s, n)
    }
  }
  if (!scaled){
    dense_rank <- function(x) {
      match(x, sort(unique(x)))
    }
    design = t(t(apply(design, 2, dense_rank) - 0.5)/levels)
  }
  if (temp == 0){
    crit1 = sqrt(-(4/3)^p + (3/2)^p*(n^2-2)/n^2 + (5/4)^p*2/n^2)
    crit2 = sqrt(-(4/3)^p + (3/2)^p)
    delta = crit2 - crit1
    temp = - delta / log(0.99)
  }
  if (method == "deterministic"){
    result = uniformLHDOptimizer_cpp(design, num.passes, max.det.iter,
                                     temp, decay, no.update.iter.max,
                                     method)
  }else if(method == "sa"){
    result = uniformLHDOptimizer_cpp(design, num.passes, max.sa.iter,
                                     temp, decay, no.update.iter.max,
                                     method)
  }else if (method == "full"){
    result = uniformLHDOptimizer_cpp(design, num.passes, max.sa.iter,
                                     temp, decay, no.update.iter.max,
                                     "sa")
    crit_hist = result$crit_hist
    result = uniformLHDOptimizer_cpp(result$design, num.passes, max.det.iter,
                                     temp, decay, no.update.iter.max,
                                     "deterministic")
    result$crit_hist = c(crit_hist, result$crit_hist)
  }
  result$design.int = t(t(result$design) * levels) + 0.5
  return (result)
}
