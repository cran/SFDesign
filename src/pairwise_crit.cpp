// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>
#include <string>
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// Maximin criterion functions ---------------------------------------------------------------------------
// [[Rcpp::export]]
arma::vec computeDistanceMatrixMaximin(const arma::mat& A) {
  int n = A.n_rows;
  int dim = n * (n - 1) / 2;
  arma::vec d(dim, fill::zeros);
  int count = 0;
  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      d(count) = norm(A.row(i) - A.row(j));
      count++;
    }
  }
  return d;
}

double computeCriterionMaximin(const arma::vec& d, int power) {
  int dim = d.n_elem;
  double avg = 0;
  for (int i = 0; i < dim; i++) {
    avg += pow(d[i], -power);
  }
  avg /= double(dim);
  return pow(avg, 1.0 / power);
}

// [[Rcpp::export]]
double maximinObj(const arma::mat& A, int power){
  return computeCriterionMaximin(computeDistanceMatrixMaximin(A), power);
}
// [[Rcpp::export]]
double maximinCrit(const arma::mat& A){
  return arma::min(computeDistanceMatrixMaximin(A));
}


// MaxPro criterion functions ---------------------------------------------------------------------------
// [[Rcpp::export]]
arma::vec computeDistanceMatrixMaxPro(const arma::mat& A, int s = 2, double delta = 0) {
  int p = A.n_cols;
  int n = A.n_rows;
  int dim = n * (n - 1) / 2;
  arma::vec d = arma::zeros(dim);
  int count = 0;
  for (int row1=0; row1<(n - 1); row1++) {
    for (int row2=row1+1; row2<n; row2++) {
      for (int col=0; col<p; col++) {
        d(count) += log(pow(A(row1, col) - A(row2, col), s) + delta);
      }
      count++;
    }
  }
  return d;
}

double computeCriterionMaxPro(const arma::vec& d, int p, int s = 2) {
  int dim = d.n_elem;
  double avg = 0;
  double Dmin = d.min();
  avg = sum(exp(Dmin - d));
  avg = log(avg) - Dmin;
  avg = exp((avg - log(dim)) / (p * s));

  return avg;
}

// [[Rcpp::export]]
double maxproObj(const arma::mat& A, int s = 2, double delta = 0){
  int p = A.n_cols;
  return std::pow(computeCriterionMaxPro(computeDistanceMatrixMaxPro(A, s, delta), p, s), s);
}
// [[Rcpp::export]]
double maxproCrit(const arma::mat& A, int s = 2, double delta = 0){
  return maxproObj(A, s, delta);
}

// Uniform criterion functions ---------------------------------------------------------------------------
// [[Rcpp::export]]
arma::vec computeDistanceMatrixUniform(const arma::mat& A) {
  // A needs to be scaled into [0, 1]
  int n = A.n_rows;
  int p = A.n_cols;
  int dim = n * (n - 1) / 2;

  arma::vec d = arma::zeros(dim);
  int count = 0;
  for (int row1 = 0; row1 < (n - 1); row1++) {
    for (int row2 = row1+1; row2 < n; row2++) {
      for (int col = 0; col < p; col++) {
        double diff = fabs(A(row1, col) - A(row2, col));
        d(count) += log(1.5 - diff * (1 - diff));
      }
      count++;
    }
  }
  return d;
}

double computeCriterionUniform(const arma::vec& d, int n, int p) {
  double avgdist = (sum(exp(d)) * 2 +  n * std::pow(1.5, p)) / std::pow(n, 2) - std::pow(4./3., p);
  return std::sqrt(avgdist);
}

// [[Rcpp::export]]
double uniformObj(const arma::mat& A){
  int n = A.n_rows;
  int p = A.n_cols;
  return computeCriterionUniform(computeDistanceMatrixUniform(A), n, p);
}
// [[Rcpp::export]]
double uniformCrit(const arma::mat& A, int s=2){
  return uniformObj(A);
}
