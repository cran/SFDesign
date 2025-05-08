#' @name
#' randomLHD
#'
#' @title
#' Random Latin hypercube design
#'
#' @description
#' This function generates a random Latin hypercube design.
#'
#' @details
#' \code{randomLHD} generates a random Latin hypercube design.
#'
#' @param n design size.
#' @param p design dimension.
#'
#' @return
#' a random Latin hypercube design.
#'
#' @examples
#' n = 20
#' p = 3
#' D = randomLHD(n, p)
#'
randomLHD = function(n, p) {
  D = matrix(0, nrow = n, ncol = p)
  # Generate each column of the Latin Hypercube Design
  for (j in 1:p) {
    # Generate a sequence from 1 to n, center intervals, and scale to (0, 1)
    seq <- (seq_len(n) - 0.5) / n
    # Randomly shuffle the sequence
    D[, j] <- sample(seq, n)
  }
  return(D)
}

#' @name
#' full.factorial
#'
#' @title
#' Full factorial design.
#'
#' @description
#' This function generates a full factorial design.
#'
#' @details
#' \code{full.factorial} generates a p dimensional full factorial design.
#'
#' @param p design dimension.
#' @param level an integer specifying the number of levels.
#'
#' @return
#' a full factorial design matrix (scaled to [0, 1])
#'
#' @examples
#' p = 3
#' level = 3
#' D = full.factorial(p, level)
#'
full.factorial = function(p, level) {
  if (level==1){
    return (matrix(rep(0, p), ncol=p))
  }
  level_list = replicate(p, seq_len(level), simplify = FALSE)
  # Generate full factorial design
  design = as.matrix(expand.grid(level_list)-1)/(level-1)
  colnames(design) = NULL
  return(design)
}

# compute distance matrix
distmatrix.maxpro = function(D, s = 2){
  return (computeDistanceMatrixMaxPro(D, s))
}
distmatrix.maximin = function(D){
  return (computeDistanceMatrixMaximin(D))
}
distmatrix.uniform = function(D){
  return (computeDistanceMatrixUniform(D))
}
