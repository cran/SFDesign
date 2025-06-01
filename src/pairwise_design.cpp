// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>
#include <string>
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

//---------------------------------------------------------------------------
// Base class for pairwise design optimization.
//---------------------------------------------------------------------------
class LHDPairDesignOptimizer {
protected:
  arma::mat X;         // Design matrix.
  arma::vec d;         // Distance vector.
  int n;               // Number of rows (points).
  int p;               // Number of columns (factors).
  int num_passes;      // Number of passes in the algorithm.
  int max_iter;        // Maximum allowed iterations.
  int total_iter;      // Iteration counter.
  double temp;         // Initial temperature for the SA algorithm.
  double decay;        // Decay rate for the temperature.
  int no_update_iter_max; // Maximum iterations allowed for no updates in SA algorithm.

  // A member variable to choose between different strategies.
  std::string optimizationMethod;

  // Helper: compute index in distance vector for a pair (row, h).
  int computePosition(int row, int h, int n) {
    return (int) (row + 1 - pow((double)(h + 1), 2) * 0.5 + (n - 0.5) * (h + 1) - n - 1);
  }

public:
  // Constructor: takes the design matrix and parameters.
  LHDPairDesignOptimizer(const arma::mat& design, int num_passes, int max_iter,
                      double temp, double decay = 0.95, int no_update_iter_max = 400,
                      const std::string &method = "deterministic")
    : X(design), num_passes(num_passes), max_iter(max_iter), total_iter(0),
      temp(temp), decay(decay),
      optimizationMethod(method) {
    n = design.n_rows;
    p = design.n_cols;
    this->no_update_iter_max = std::min(no_update_iter_max, 5 * n * (n - 1) * p);
  }
  virtual ~LHDPairDesignOptimizer() {}

  // Set the optimization method.
  void setOptimizationMethod(const std::string &method) {
    optimizationMethod = method;
  }

  // Pure virtual functions to be provided by the derived optimizer classes.
  virtual arma::vec computeDistanceMatrix(const arma::mat &A) = 0;
  virtual double computeCriterion(const arma::vec &d) = 0;
  virtual arma::vec updateDistanceMatrix(arma::mat &A, int col, int selrow1, int selrow2, arma::vec d) = 0;

  // Strategy 1: deterministic swapping.
  List optimizeDet() {
    d = computeDistanceMatrix(X);
    double critbest = computeCriterion(d);
    std::vector<double> xcrit_hist; // history of current criterion
    for (int pass = 0; pass < num_passes; pass++) {
      bool changed = false;
      for (int row1 = 0; row1 < n - 1; row1++) {
        for (int row2 = row1 + 1; row2 < n; row2++) {
          for (int col = 0; col < p; col++) {
            total_iter++;
            if (total_iter > max_iter) goto endloop;
            // Try swapping the two entries in column col.
            arma::mat X_try = X;
            std::swap(X_try(row1, col), X_try(row2, col));
            arma::vec d_try = updateDistanceMatrix(X_try, col, row1, row2, d);
            double crit_try = computeCriterion(d_try);

            if (crit_try < critbest) {
              X(row1, col) = X_try(row1, col);
              X(row2, col) = X_try(row2, col);
              critbest = crit_try;
              d = d_try;
              changed = true;
            }
            xcrit_hist.push_back(critbest);
          }
        }
      }
      if (!changed) break;
    }
    endloop:
      Rcpp::NumericVector crit_hist_R(xcrit_hist.begin(), xcrit_hist.end());
      return List::create(Named("design") = X,
                          Named("total_iter") = total_iter,
                          Named("criterion") = critbest,
                          Named("crit_hist") = crit_hist_R);
  }

  // Strategy 2: Simulated Annealing.
  List optimizeSA() {
    d = computeDistanceMatrix(X);
    arma::mat X_best = X;
    double critbest = computeCriterion(d);
    double xcrit = critbest;
    arma::mat X_try = X;
    int ipert = 1;
    bool ichange = true;
    std::vector<double> xcrit_hist; // history of current criterion
    while (ichange) {
      ichange = false;
      ipert = 1;
      while (ipert < no_update_iter_max) {
        if (total_iter > max_iter) break;
        total_iter++;
        // Randomly choose a column and two distinct rows.
        int col = arma::randi(distr_param(0, p-1));
        int row1 = arma::randi(distr_param(0, n-1));
        int row2 = arma::randi(distr_param(0, n-2));
        if (row2 >= row1) {
          row2 ++;
        }
        // Create candidate design by swapping the two entries in the chosen column.
        X_try = X;
        std::swap(X_try(row1, col), X_try(row2, col));

        // Update distance vector based on the swap.
        arma::vec d_try = updateDistanceMatrix(X_try, col, row1, row2, d);
        double crit_try = computeCriterion(d_try);

        // Acceptance rules.
        if (crit_try < critbest) {
          // New overall best design found.
          ichange = true;
          X_best = X_try;
          critbest = crit_try;
          // Also update current design.
          X(row1, col) = X_try(row1, col);
          X(row2, col) = X_try(row2, col);
          xcrit = crit_try;
          d = d_try;
          ipert = 1; // reset inner counter
        } else {
          ipert = ipert + 1;
          if (crit_try < xcrit) {
            // Improvement on current design.
            X(row1, col) = X_try(row1, col);
            X(row2, col) = X_try(row2, col);
            d = d_try;
            xcrit = crit_try;
            ichange = true;
          } else if (arma::randu() < exp(- (crit_try - xcrit) / temp)) {
            X(row1, col) = X_try(row1, col);
            X(row2, col) = X_try(row2, col);
            d = d_try;
            xcrit = crit_try;
          }
        }
        xcrit_hist.push_back(xcrit);
      }
      // Decay temperature.
      temp = temp * decay;
    }

    Rcpp::NumericVector xcrit_hist_R(xcrit_hist.begin(), xcrit_hist.end());
    return List::create(Named("design") = X_best,
                        Named("total_iter") = total_iter,
                        Named("crit_hist") = xcrit_hist_R,
                        Named("criterion") = critbest);
  }

  // Default optimize() method: selects strategy based on the optimizationMethod member.
  List optimize() {
    if (optimizationMethod == "deterministic") {
      return optimizeDet();
    } else if (optimizationMethod == "sa") {
      return optimizeSA();
    } else {
      Rcpp::Rcout << "Unknown optimization method: " << optimizationMethod
                  << ". Using deterministic." << std::endl;
      return optimizeDet();
    }
  }
};

//---------------------------------------------------------------------------
// Derived class: MaximinLHDOptimizer.
// Implements the maximin criterion using Euclidean distances and a power parameter.
//---------------------------------------------------------------------------
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

class MaximinLHDOptimizer : public LHDPairDesignOptimizer {
private:
  int power; // Parameter for the maximin criterion.
public:
  MaximinLHDOptimizer(const arma::mat &design, int power, int num_passes, int max_iter,
                       double temp = 0, double decay = 0.95, int no_update_iter_max = 400,
                       const std::string &method = "deterministic")
    : LHDPairDesignOptimizer(design, num_passes, max_iter,
              temp, decay, no_update_iter_max, method), power(power) {}

  // Compute pairwise Euclidean distances.
  arma::vec computeDistanceMatrix(const arma::mat &A) {
    return computeDistanceMatrixMaximin(A);
  }

  // Compute the maximin criterion.
  double computeCriterion(const arma::vec &d) {
    return computeCriterionMaximin(d, power);
  }

  // Update the distance vector after a swap for maximin.
  arma::vec updateDistanceMatrix(arma::mat &A, int col, int selrow1, int selrow2, arma::vec d) {
    int row1 = std::min(selrow1, selrow2);
    int row2 = std::max(selrow1, selrow2);
    int pos1, pos2;
    double s = 0;
    if (row1 > 0) {
      for (int h = 0; h < row1; h++) {
        s = pow(A(row2, col) - A(h, col), 2) - pow(A(row1, col) - A(h, col), 2);
        pos1 = computePosition(row1, h, n);
        pos2 = computePosition(row2, h, n);
        d(pos1) = sqrt(pow(d(pos1), 2) - s);
        d(pos2) = sqrt(pow(d(pos2), 2) + s);
      }
    }
    for (int h = row1 + 1; h < row2; h++) {
      s = pow(A(row2, col) - A(h, col), 2) - pow(A(row1, col) - A(h, col), 2);
      pos1 = computePosition(h, row1, n);
      pos2 = computePosition(row2, h, n);
      d(pos1) = sqrt(pow(d(pos1), 2) - s);
      d(pos2) = sqrt(pow(d(pos2), 2) + s);
    }
    if (row2 < n - 1) {
      for (int h = row2 + 1; h < n; h++) {
        s = pow(A(row2, col) - A(h, col), 2) - pow(A(row1, col) - A(h, col), 2);
        pos1 = computePosition(h, row1, n);
        pos2 = computePosition(h, row2, n);
        d(pos1) = sqrt(pow(d(pos1), 2) - s);
        d(pos2) = sqrt(pow(d(pos2), 2) + s);
      }
    }
    return d;
  }
};

//---------------------------------------------------------------------------
// Derived class: MaxProLHDOptimizer.
// Implements the maxpro criterion using a log-based distance and a scaling parameter.
//---------------------------------------------------------------------------
// [[Rcpp::export]]
arma::vec computeDistanceMatrixMaxPro(const arma::mat& A, int s = 2) {
  int p = A.n_cols;
  int n = A.n_rows;
  int dim = n * (n - 1) / 2;
  arma::vec d = arma::zeros(dim);
  int count = 0;
  for (int row1=0; row1<(n - 1); row1++) {
    for (int row2=row1+1; row2<n; row2++) {
      for (int col=0; col<p; col++) {
        d(count) += s * log(fabs(A(row1, col) - A(row2, col)));
      }
      count++;
    }
  }
  return d;
}

double computeCriterionMaxPro(const arma::vec& d, int p, int s = 2, double delta = 0) {
  int dim = d.n_elem;
  double avg = 0;
  if (delta == 0){
    double Dmin = d.min();
    avg = sum(exp(Dmin - d));
    avg = log(avg) - Dmin;
    avg = exp((avg - log(dim)) / (p * s));
  }else{
    avg = std::pow(arma::mean(arma::pow(exp(d) + delta, -1)), 1.0 / (p * s));
  }
  return avg;
}

class MaxProLHDOptimizer : public LHDPairDesignOptimizer {
private:
  double s; // Power parameter for the maxpro criterion.
public:
  MaxProLHDOptimizer(const arma::mat &design, double s, int num_passes, int max_iter,
                      double temp = 0, double decay = 0.95, int no_update_iter_max = 400,
                      const std::string &method = "deterministic")
    : LHDPairDesignOptimizer(design, num_passes, max_iter,
                          temp, decay, no_update_iter_max, method), s(s) {}

  // Compute the distance vector using a log-based measure.
  arma::vec computeDistanceMatrix(const arma::mat &A) {
    return computeDistanceMatrixMaxPro(A, s);
  }

  // Compute the maxpro criterion.
  double computeCriterion(const arma::vec &d) {
    return computeCriterionMaxPro(d, p, s);
  }

  // Update the distance vector after a swap for maxpro.
  arma::vec updateDistanceMatrix(arma::mat &A, int col, int selrow1, int selrow2, arma::vec d) {
    int row1 = std::min(selrow1, selrow2);
    int row2 = std::max(selrow1, selrow2);
    int pos1, pos2;
    if (row1 > 0) {
      for (int h = 0; h < row1; h++) {
        pos1 = computePosition(row1, h, n);
        pos2 = computePosition(row2, h, n);
        d(pos1) += s * log(fabs(A(row1, col) - A(h, col))) -
          s * log(fabs(A(row2, col) - A(h, col)));
        d(pos2) += s * log(fabs(A(row2, col) - A(h, col))) -
          s * log(fabs(A(row1, col) - A(h, col)));
      }
    }
    for (int h = row1 + 1; h < row2; h++) {
      pos1 = computePosition(h, row1, n);
      pos2 = computePosition(row2, h, n);
      d(pos1) += s * log(fabs(A(row1, col) - A(h, col))) -
        s * log(fabs(A(row2, col) - A(h, col)));
      d(pos2) += s * log(fabs(A(row2, col) - A(h, col))) -
        s * log(fabs(A(row1, col) - A(h, col)));
    }
    if (row2 < n - 1) {
      for (int h = row2 + 1; h < n; h++) {
        pos1 = computePosition(h, row1, n);
        pos2 = computePosition(h, row2, n);
        d(pos1) += s * log(fabs(A(row1, col) - A(h, col))) -
          s * log(fabs(A(row2, col) - A(h, col)));
        d(pos2) += s * log(fabs(A(row2, col) - A(h, col))) -
          s * log(fabs(A(row1, col) - A(h, col)));
      }
    }
    return d;
  }
};

//---------------------------------------------------------------------------
// Derived class: UniformLHDOptimizer.
// Implements the maximin criterion using Euclidean distances and a power parameter.
//---------------------------------------------------------------------------
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

class UniformLHDOptimizer : public LHDPairDesignOptimizer {
private:
public:
  UniformLHDOptimizer(const arma::mat &design, int num_passes, int max_iter,
                       double temp = 0, double decay = 0.95, int no_update_iter_max = 400,
                       const std::string &method = "deterministic")
    : LHDPairDesignOptimizer(design, num_passes, max_iter,
                          temp, decay, no_update_iter_max, method){}

  // Compute pairwise wraparound discrepancy.
  arma::vec computeDistanceMatrix(const arma::mat &A) {
    return computeDistanceMatrixUniform(A);
  }

  // Compute the wraparound criterion.
  double computeCriterion(const arma::vec &d) {
    return computeCriterionUniform(d, n, p);
  }

  // Update the distance vector after a swap.
  arma::vec updateDistanceMatrix(arma::mat &A, int col, int selrow1, int selrow2, arma::vec d) {
    int n = A.n_rows;

    int row1 = std::min(selrow1, selrow2);
    int row2 = std::max(selrow1, selrow2);

    int position1, position2;
    double diff1, diff2;

    if (row1 > 0){
      for (int h = 0; h < row1; h++) {
        position1 = computePosition(row1, h, n);
        position2 = computePosition(row2, h, n);
        diff1 = fabs(A(row1, col) - A(h, col));
        diff2 = fabs(A(row2, col) - A(h, col));
        d(position1) += log(1.5 - diff1 * (1 - diff1)) - log(1.5 - diff2 * (1 - diff2));
        d(position2) += log(1.5 - diff2 * (1 - diff2)) - log(1.5 - diff1 * (1 - diff1));
      }
    }
    for (int h = row1+1; h < row2; h++) {
      position1 = computePosition(h, row1, n);
      position2 = computePosition(row2, h, n);
      diff1 = fabs(A(row1, col) - A(h, col));
      diff2 = fabs(A(row2, col) - A(h, col));
      d(position1) += log(1.5 - diff1 * (1 - diff1)) - log(1.5 - diff2 * (1 - diff2));
      d(position2) += log(1.5 - diff2 * (1 - diff2)) - log(1.5 - diff1 * (1 - diff1));
    }
    if (row2 < (n - 1)){
      for (int h=row2+1; h < n; h++) {
        position1 = computePosition(h, row1, n);
        position2 = computePosition(h, row2, n);
        diff1 = fabs(A(row1, col) - A(h, col));
        diff2 = fabs(A(row2, col) - A(h, col));
        d(position1) += log(1.5 - diff1 * (1 - diff1)) - log(1.5 - diff2 * (1 - diff2));
        d(position2) += log(1.5 - diff2 * (1 - diff2)) - log(1.5 - diff1 * (1 - diff1));
      }
    }
    return d;
  }

};

//---------------------------------------------------------------------------
// Derived class: CustomLHDOptimizer.
// Implements the custom criterion using Euclidean distances and a power parameter.
//---------------------------------------------------------------------------
class CustomLHDOptimizer : public LHDPairDesignOptimizer {
private:
  // User-supplied R functions wrapped as std::function objects.
  std::function<arma::vec(const arma::mat&)> user_computeDistanceMatrix;
  std::function<double(const arma::vec&)> user_computeCriterion;
  std::function<arma::vec(arma::mat&, int, int, int, arma::vec)> user_updateDistanceMatrix;

public:
  // Constructor: accept Rcpp::Function objects and wrap them.
  CustomLHDOptimizer(Rcpp::Function r_computeDistanceMatrix,
                     Rcpp::Function r_computeCriterion,
                     Rcpp::Function r_updateDistanceMatrix,
                     const arma::mat &design, int num_passes, int max_iter,
                     double temp = 0, double decay = 0.95, int no_update_iter_max = 400,
                     const std::string &method = "deterministic")
    : LHDPairDesignOptimizer(design, num_passes, max_iter,
      temp, decay, no_update_iter_max, method)
  {
    user_computeDistanceMatrix = [r_computeDistanceMatrix](const arma::mat &A) -> arma::vec {
      Rcpp::NumericMatrix A_rcpp = Rcpp::wrap(A);
      Rcpp::NumericVector result = r_computeDistanceMatrix(A_rcpp);
      return Rcpp::as<arma::vec>(result);
    };
    user_computeCriterion = [r_computeCriterion](const arma::vec &d) -> double {
      Rcpp::NumericVector d_rcpp = Rcpp::wrap(d);
      Rcpp::NumericVector result = r_computeCriterion(d_rcpp);
      return result[0];
    };
    user_updateDistanceMatrix = [r_updateDistanceMatrix](arma::mat &A, int col, int selrow1, int selrow2, arma::vec d) -> arma::vec {
      Rcpp::NumericMatrix A_rcpp = Rcpp::wrap(A);
      Rcpp::NumericVector d_rcpp = Rcpp::wrap(d);
      Rcpp::NumericVector result = r_updateDistanceMatrix(A_rcpp, col, selrow1, selrow2, d_rcpp);
      return Rcpp::as<arma::vec>(result);
    };
  }
  // Override virtual functions to call the user-supplied functions.
  arma::vec computeDistanceMatrix(const arma::mat &A) override {
    return user_computeDistanceMatrix(A);
  }
  double computeCriterion(const arma::vec &d) override {
    return user_computeCriterion(d);
  }
  arma::vec updateDistanceMatrix(arma::mat &A, int col, int selrow1, int selrow2, arma::vec d) override {
    return user_updateDistanceMatrix(A, col, selrow1, selrow2, d);
  }
};

//---------------------------------------------------------------------------
// Exported functions (accessible from R).
// These functions take a design matrix and parameters, then call the respective optimizer.
//---------------------------------------------------------------------------
// Maximin
// [[Rcpp::export]]
double maximinObj(const arma::mat& A, int power){
  return computeCriterionMaximin(computeDistanceMatrixMaximin(A), power);
}
// [[Rcpp::export]]
double maximinCrit(const arma::mat& A){
  return arma::min(computeDistanceMatrixMaximin(A));
}
// [[Rcpp::export]]
List maximinLHDOptimizer_cpp(arma::mat design, int power = 15, int num_passes = 10, int max_iter = 1e6,
                             double temp = 0, double decay = 0.95, int no_update_iter_max = 400,
                             std::string method = "deterministic") {
  MaximinLHDOptimizer optimizer(design, power, num_passes, max_iter,
                                 temp, decay, no_update_iter_max,
                                 method);
  return optimizer.optimize();
}

// MaxPro
// [[Rcpp::export]]
double maxproObj(const arma::mat& A, int s=2, double delta = 0){
  int p = A.n_cols;
  return std::pow(computeCriterionMaxPro(computeDistanceMatrixMaxPro(A, s), p, s, delta), s);
}
// [[Rcpp::export]]
double maxproCrit(const arma::mat& A, int s = 2, double delta = 0){
  return maxproObj(A, s, delta);
}
// [[Rcpp::export]]
List maxproLHDOptimizer_cpp(arma::mat design, double s = 2, int num_passes = 10, int max_iter = 1e6,
                         double temp = 0, double decay = 0.95, int no_update_iter_max = 400,
                         std::string method = "deterministic") {
  MaxProLHDOptimizer optimizer(design, s, num_passes, max_iter,
                                temp, decay, no_update_iter_max,
                                method);
  return optimizer.optimize();
};

// Uniform
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
// [[Rcpp::export]]
List uniformLHDOptimizer_cpp(arma::mat design, int num_passes = 10, int max_iter = 1e6,
                          double temp = 0, double decay = 0.95, int no_update_iter_max = 400,
                          std::string method = "deterministic") {
  UniformLHDOptimizer optimizer(design, num_passes, max_iter,
                                 temp, decay, no_update_iter_max,
                                 method);
  return optimizer.optimize();
}
// Custom
// [[Rcpp::export]]
List customLHDOptimizer_cpp(Rcpp::Function r_computeDistanceMatrix,
                             Rcpp::Function r_computeCriterion,
                             Rcpp::Function r_updateDistanceMatrix,
                             arma::mat design, int num_passes = 10, int max_iter = 1e6,
                             double temp = 0, double decay = 0.95, int no_update_iter_max = 400,
                             std::string method = "deterministic") {
  CustomLHDOptimizer optimizer(r_computeDistanceMatrix,
                               r_computeCriterion,
                               r_updateDistanceMatrix,
                               design, num_passes, max_iter,
                               temp, decay, no_update_iter_max,
                               method);
  return optimizer.optimize();
}
