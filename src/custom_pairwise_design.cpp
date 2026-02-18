// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>
#include <string>
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

//---------------------------------------------------------------------------
// Base class for pairwise design optimization.
//---------------------------------------------------------------------------
class LHDPairDesignOptimizer_custom {
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
  LHDPairDesignOptimizer_custom(const arma::mat& design, int num_passes, int max_iter,
                      double temp, double decay = 0.95, int no_update_iter_max = 400,
                      const std::string &method = "deterministic")
    : X(design), num_passes(num_passes), max_iter(max_iter), total_iter(0),
      temp(temp), decay(decay),
      optimizationMethod(method) {
    n = design.n_rows;
    p = design.n_cols;
    this->no_update_iter_max = std::min(no_update_iter_max, 5 * n * (n - 1) * p);
  }
  virtual ~LHDPairDesignOptimizer_custom() {}

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
// Derived class: CustomLHDOptimizer.
// Implements the custom criterion using Euclidean distances and a power parameter.
//---------------------------------------------------------------------------
class CustomLHDOptimizer : public LHDPairDesignOptimizer_custom {
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
    : LHDPairDesignOptimizer_custom(design, num_passes, max_iter,
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
