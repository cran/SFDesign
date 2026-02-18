#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <Rcpp.h>
using namespace Rcpp;

// Random number utilities (R's built-in RNG)
inline int rc(int n) {
  double u = unif_rand();
  return (int)(n * u);
}

inline int rc2(int n, int del) {
  int rctwo = rc(n - 1);
  if (rctwo >= del) rctwo++;
  return rctwo;
}

//---------------------------------------------------------------------------
// Base class for pairwise design optimization.
//---------------------------------------------------------------------------
class LHDPairDesignOptimizer {
protected:
  double **X;          // Design matrix (n x p).
  double *d;           // Distance vector.
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
  int computePosition(int row, int h) {
    return (int)(row + 1 - pow((double)(h + 1), 2) * 0.5 + (n - 0.5) * (h + 1) - n - 1);
  }

public:
  // Constructor: takes the design matrix and parameters.
  LHDPairDesignOptimizer(double **design, int n_rows, int n_cols, int num_passes, int max_iter,
                         double temp, double decay = 0.95, int no_update_iter_max = 400,
                         const std::string &method = "deterministic")
    : n(n_rows), p(n_cols), num_passes(num_passes), max_iter(max_iter), total_iter(0),
      temp(temp), decay(decay),
      optimizationMethod(method) {
    
    this->no_update_iter_max = std::min(no_update_iter_max, 5 * n * (n - 1) * p);
    
    // Allocate and copy design matrix
    X = new double*[p];
    for (int i = 0; i < p; i++) {
      X[i] = new double[n];
      for (int j = 0; j < n; j++) {
        X[i][j] = design[i][j];
      }
    }
    
    // Allocate distance vector
    int dim = (int)(n * (n - 1) * 0.5);
    d = new double[dim];
  }

  virtual ~LHDPairDesignOptimizer() {
    if (X != NULL) {
      for (int i = 0; i < p; i++) {
        delete[] X[i];
      }
      delete[] X;
    }
    if (d != NULL) {
      delete[] d;
    }
  }

  // Set the optimization method.
  void setOptimizationMethod(const std::string &method) {
    optimizationMethod = method;
  }

  // Pure virtual functions to be provided by the derived optimizer classes.
  virtual void computeDistanceMatrix() = 0;
  virtual double computeCriterion() = 0;
  virtual void updateDistanceMatrix(int col, int selrow1, int selrow2, double *d_old) = 0;

  // Revert distance matrix: restore only affected positions (common to all derived classes).
  void revertDistanceMatrix(int selrow1, int selrow2, double *d_old) {
    int row1 = std::min(selrow1, selrow2);
    int row2 = std::max(selrow1, selrow2);
    int pos1, pos2;

    if (row1 > 0) {
      for (int h = 0; h < row1; h++) {
        pos1 = computePosition(row1, h);
        pos2 = computePosition(row2, h);
        d[pos1] = d_old[pos1];
        d[pos2] = d_old[pos2];
      }
    }

    for (int h = row1 + 1; h < row2; h++) {
      pos1 = computePosition(h, row1);
      pos2 = computePosition(row2, h);
      d[pos1] = d_old[pos1];
      d[pos2] = d_old[pos2];
    }

    if (row2 < n - 1) {
      for (int h = row2 + 1; h < n; h++) {
        pos1 = computePosition(h, row1);
        pos2 = computePosition(h, row2);
        d[pos1] = d_old[pos1];
        d[pos2] = d_old[pos2];
      }
    }
  }

  // Strategy 1: deterministic swapping.
  List optimizeDet() {
    computeDistanceMatrix();
    double critbest = computeCriterion();
    std::vector<double> xcrit_hist;

    int dim = (int)(n * (n - 1) * 0.5);
    double *d_old = new double[dim];
    
    for (int pass = 0; pass < num_passes; pass++) {
      bool changed = false;
      for (int row1 = 0; row1 < n - 1; row1++) {
        for (int row2 = row1 + 1; row2 < n; row2++) {
          for (int col = 0; col < p; col++) {
            total_iter++;
            if (total_iter > max_iter) goto endloop;
            
            // Try swapping the two entries in column col.
            std::swap(X[col][row1], X[col][row2]);
            updateDistanceMatrix(col, row1, row2, d_old);
            double crit_try = computeCriterion();

            if (crit_try < critbest) {
              critbest = crit_try;
              changed = true;
            } else {
              // Revert the swap and distance
              std::swap(X[col][row1], X[col][row2]);
              revertDistanceMatrix(row1, row2, d_old);
            }
            
            xcrit_hist.push_back(critbest);
          }
        }
      }
      if (!changed) break;
    }
    
    endloop:
    delete[] d_old;
    
    // Convert design matrix back to Rcpp format
    NumericMatrix design_R(n, p);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
        design_R(i, j) = X[j][i];
      }
    }
    
    NumericVector crit_hist_R(xcrit_hist.begin(), xcrit_hist.end());
    return List::create(Named("design") = design_R,
                        Named("total_iter") = total_iter,
                        Named("criterion") = critbest,
                        Named("crit_hist") = crit_hist_R);
  }

  // Strategy 2: Simulated Annealing.
  List optimizeSA() {
    GetRNGstate();
    
    int dim = (int)(n * (n - 1) * 0.5);
    double **X_best = new double*[p];
    for (int i = 0; i < p; i++) {
      X_best[i] = new double[n];
      for (int j = 0; j < n; j++) {
        X_best[i][j] = X[i][j];
      }
    }
    
    computeDistanceMatrix();
    double critbest = computeCriterion();
    double xcrit = critbest;
    double *d_old = new double[dim];
    std::vector<double> xcrit_hist;
    
    int ipert = 1;
    bool ichange = true;
    double current_temp = temp;
    
    while (ichange) {
      ichange = false;
      ipert = 1;
      
      while (ipert < no_update_iter_max) {
        if (total_iter > max_iter) break;
        total_iter++;
        
        // Randomly choose a column and two distinct rows.
        int col = rc(p);
        int row1 = rc(n);
        int row2 = rc(n - 1);
        if (row2 >= row1) {
          row2++;
        }
        
        // Create candidate design by swapping the two entries in the chosen column.
        std::swap(X[col][row1], X[col][row2]);
        updateDistanceMatrix(col, row1, row2, d_old);
        double crit_try = computeCriterion();

        // Acceptance rules.
        if (crit_try < critbest) {
          // New overall best design found.
          ichange = true;
          for (int i = 0; i < p; i++) {
            for (int j = 0; j < n; j++) {
              X_best[i][j] = X[i][j];
            }
          }
          critbest = crit_try;
          xcrit = crit_try;
          ipert = 1; // reset inner counter
        } else {
          ipert = ipert + 1;
          if (crit_try < xcrit) {
            // Improvement on current design.
            xcrit = crit_try;
            ichange = true;
          } else if (unif_rand() < exp(-(crit_try - xcrit) / current_temp)) {
            xcrit = crit_try;
          } else {
            // Reject: revert the swap
            std::swap(X[col][row1], X[col][row2]);
            revertDistanceMatrix(row1, row2, d_old);
          }
        }
        xcrit_hist.push_back(xcrit);
      }
      // Decay temperature.
      current_temp = current_temp * decay;
    }
    
    PutRNGstate();
    
    // Copy best design back to X
    for (int i = 0; i < p; i++) {
      for (int j = 0; j < n; j++) {
        X[i][j] = X_best[i][j];
      }
    }
    
    // Convert design matrix back to Rcpp format
    NumericMatrix design_R(n, p);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
        design_R(i, j) = X[j][i];
      }
    }
    
    NumericVector xcrit_hist_R(xcrit_hist.begin(), xcrit_hist.end());
    
    // Cleanup
    for (int i = 0; i < p; i++) {
      delete[] X_best[i];
    }
    delete[] X_best;
    delete[] d_old;
    
    return List::create(Named("design") = design_R,
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
class MaximinLHDOptimizer : public LHDPairDesignOptimizer {
private:
  int power; // Parameter for the maximin criterion.

public:
  MaximinLHDOptimizer(double **design, int n_rows, int n_cols, int power,
                      int num_passes, int max_iter,
                      double temp = 0, double decay = 0.95, int no_update_iter_max = 400,
                      const std::string &method = "deterministic")
    : LHDPairDesignOptimizer(design, n_rows, n_cols, num_passes, max_iter,
                            temp, decay, no_update_iter_max, method), power(power) {}

  // Compute pairwise Euclidean distances.
  void computeDistanceMatrix() {
    int dim = (int)(n * (n - 1) * 0.5);
    for (int i = 0; i < dim; i++) {
      d[i] = 0;
    }
    int count = 0;
    for (int i = 0; i < n - 1; i++) {
      for (int j = i + 1; j < n; j++) {
        double dist_sq = 0;
        for (int col = 0; col < p; col++) {
          double diff = X[col][i] - X[col][j];
          dist_sq += diff * diff;
        }
        d[count] = std::sqrt(dist_sq);
        count++;
      }
    }
  }

  // Compute the maximin criterion.
  double computeCriterion() {
    int dim = (int)(n * (n - 1) * 0.5);
    double avg = 0;
    for (int i = 0; i < dim; i++) {
      avg += pow(d[i], -power);
    }
    avg /= double(dim);
    return pow(avg, 1.0 / power);
  }

  // Update the distance vector after a swap for maximin.
  // Uses incremental update: d_new = std::sqrt(d_old^2 ± delta_col)
  void updateDistanceMatrix(int col, int selrow1, int selrow2, double *d_old) {
    int row1 = std::min(selrow1, selrow2);
    int row2 = std::max(selrow1, selrow2);
    int pos1, pos2;
    
    // Incremental update based on changes in the swapped column only
    if (row1 > 0) {
      for (int h = 0; h < row1; h++) {
        pos1 = computePosition(row1, h);
        pos2 = computePosition(row2, h);
        
        // Save old values before updating
        d_old[pos1] = d[pos1];
        d_old[pos2] = d[pos2];
        
        // Calculate delta for the changed column
        double delta = pow(X[col][row2] - X[col][h], 2) - pow(X[col][row1] - X[col][h], 2);
        
        // Update distances incrementally
        d[pos1] = std::sqrt(pow(d[pos1], 2) - delta);
        d[pos2] = std::sqrt(pow(d[pos2], 2) + delta);
      }
    }
    
    for (int h = row1 + 1; h < row2; h++) {
      pos1 = computePosition(h, row1);
      pos2 = computePosition(row2, h);
      
      // Save old values before updating
      d_old[pos1] = d[pos1];
      d_old[pos2] = d[pos2];
      
      // Calculate delta for the changed column
      double delta = pow(X[col][row2] - X[col][h], 2) - pow(X[col][row1] - X[col][h], 2);
      
      // Update distances incrementally
      d[pos1] = std::sqrt(pow(d[pos1], 2) - delta);
      d[pos2] = std::sqrt(pow(d[pos2], 2) + delta);
    }
    
    if (row2 < n - 1) {
      for (int h = row2 + 1; h < n; h++) {
        pos1 = computePosition(h, row1);
        pos2 = computePosition(h, row2);
        
        // Save old values before updating
        d_old[pos1] = d[pos1];
        d_old[pos2] = d[pos2];
        
        // Calculate delta for the changed column
        double delta = pow(X[col][row2] - X[col][h], 2) - pow(X[col][row1] - X[col][h], 2);
        
        // Update distances incrementally
        d[pos1] = std::sqrt(pow(d[pos1], 2) - delta);
        d[pos2] = std::sqrt(pow(d[pos2], 2) + delta);
      }
    }
  }
};

//---------------------------------------------------------------------------
// Derived class: MaxProLHDOptimizer.
// Implements the maxpro criterion using a log-based distance and a scaling parameter.
//---------------------------------------------------------------------------
class MaxProLHDOptimizer : public LHDPairDesignOptimizer {
private:
  int s; // Power parameter for the MaxPro criterion.

public:
  MaxProLHDOptimizer(double **design, int n_rows, int n_cols, int s,
                    int num_passes, int max_iter,
                    double temp = 0, double decay = 0.95, int no_update_iter_max = 400,
                    const std::string &method = "deterministic")
    : LHDPairDesignOptimizer(design, n_rows, n_cols, num_passes, max_iter,
                            temp, decay, no_update_iter_max, method), s(s) {}

  // Compute the distance vector using a log-based measure.
  void computeDistanceMatrix() {
    int dim = (int)(n * (n - 1) * 0.5);
    for (int i = 0; i < dim; i++) {
      d[i] = 0;
    }
    int count = 0;
    for (int k1 = 0; k1 < n - 1; k1++) {
      for (int k2 = k1 + 1; k2 < n; k2++) {
        for (int col = 0; col < p; col++) {
          d[count] += s * log(fabs(X[col][k1] - X[col][k2]));
        }
        count++;
      }
    }
  }

  // Compute the maxpro criterion.
  double computeCriterion() {
    int dim = (int)(n * (n - 1) * 0.5);
    double Dmin = d[0];
    for (int i = 1; i < dim; i++) {
      if (d[i] < Dmin) Dmin = d[i];
    }
    double avgdist = 0;
    for (int i = 0; i < dim; i++) {
      avgdist += exp(Dmin - d[i]);
    }
    avgdist = log(avgdist) - Dmin;
    avgdist = exp((avgdist - log((double)dim)) / (double)(p * s));
    return avgdist;
  }

  // Update the distance vector after a swap for maxpro (efficient incremental update).
  void updateDistanceMatrix(int col, int selrow1, int selrow2, double *d_old) {
    int row1 = std::min(selrow1, selrow2);
    int row2 = std::max(selrow1, selrow2);
    int pos1, pos2;

    if (row1 > 0) {
      for (int h = 0; h < row1; h++) {
        pos1 = computePosition(row1, h);
        pos2 = computePosition(row2, h);
        // Save old values before updating
        d_old[pos1] = d[pos1];
        d_old[pos2] = d[pos2];
        // Update distances incrementally
        d[pos1] = d[pos1] + s * log(fabs(X[col][row1] - X[col][h])) - 
                       s * log(fabs(X[col][row2] - X[col][h]));
        d[pos2] = d[pos2] + s * log(fabs(X[col][row2] - X[col][h])) - 
                       s * log(fabs(X[col][row1] - X[col][h]));
      }
    }

    for (int h = row1 + 1; h < row2; h++) {
      pos1 = computePosition(h, row1);
      pos2 = computePosition(row2, h);
      // Save old values before updating
      d_old[pos1] = d[pos1];
      d_old[pos2] = d[pos2];
      // Update distances incrementally
      d[pos1] = d[pos1] + s * log(fabs(X[col][row1] - X[col][h])) - 
                     s * log(fabs(X[col][row2] - X[col][h]));
      d[pos2] = d[pos2] + s * log(fabs(X[col][row2] - X[col][h])) - 
                     s * log(fabs(X[col][row1] - X[col][h]));
    }

    if (row2 < n - 1) {
      for (int h = row2 + 1; h < n; h++) {
        pos1 = computePosition(h, row1);
        pos2 = computePosition(h, row2);
        // Save old values before updating
        d_old[pos1] = d[pos1];
        d_old[pos2] = d[pos2];
        // Update distances incrementally
        d[pos1] = d[pos1] + s * log(fabs(X[col][row1] - X[col][h])) - 
                       s * log(fabs(X[col][row2] - X[col][h]));
        d[pos2] = d[pos2] + s * log(fabs(X[col][row2] - X[col][h])) - 
                       s * log(fabs(X[col][row1] - X[col][h]));
      }
    }
  }
};

//---------------------------------------------------------------------------
// Derived class: UniformLHDOptimizer.
// Implements the uniform criterion using wraparound discrepancy.
//---------------------------------------------------------------------------
class UniformLHDOptimizer : public LHDPairDesignOptimizer {
public:
  UniformLHDOptimizer(double **design, int n_rows, int n_cols,
                     int num_passes, int max_iter,
                     double temp = 0, double decay = 0.95, int no_update_iter_max = 400,
                     const std::string &method = "deterministic")
    : LHDPairDesignOptimizer(design, n_rows, n_cols, num_passes, max_iter,
                            temp, decay, no_update_iter_max, method) {}

  // Compute pairwise wraparound discrepancy.
  void computeDistanceMatrix() {
    int dim = (int)(n * (n - 1) * 0.5);
    for (int i = 0; i < dim; i++) {
      d[i] = 0;
    }
    int count = 0;
    for (int row1 = 0; row1 < n - 1; row1++) {
      for (int row2 = row1 + 1; row2 < n; row2++) {
        for (int col = 0; col < p; col++) {
          double diff = fabs(X[col][row1] - X[col][row2]);
          d[count] += log(1.5 - diff * (1.0 - diff));
        }
        count++;
      }
    }
  }

  // Compute the wraparound criterion.
  double computeCriterion() {
    int dim = (int)(n * (n - 1) * 0.5);
    double avgdist = 0;
    for (int i = 0; i < dim; i++) {
      avgdist += exp(d[i]);
    }
    avgdist = (avgdist * 2.0 + n * pow(1.5, p)) / pow((double)n, 2.0) - pow(4.0 / 3.0, p);
    return std::sqrt(avgdist);
  }

  // Update the distance vector after a swap.
  void updateDistanceMatrix(int col, int selrow1, int selrow2, double *d_old) {
    int row1 = std::min(selrow1, selrow2);
    int row2 = std::max(selrow1, selrow2);
    int pos1, pos2;
    double diff1, diff2;

    if (row1 > 0) {
      for (int h = 0; h < row1; h++) {
        pos1 = computePosition(row1, h);
        pos2 = computePosition(row2, h);
        // Save old values before updating
        d_old[pos1] = d[pos1];
        d_old[pos2] = d[pos2];
        // Update distances incrementally
        diff1 = fabs(X[col][row1] - X[col][h]);
        diff2 = fabs(X[col][row2] - X[col][h]);
        d[pos1] += log(1.5 - diff1 * (1.0 - diff1)) - log(1.5 - diff2 * (1.0 - diff2));
        d[pos2] += log(1.5 - diff2 * (1.0 - diff2)) - log(1.5 - diff1 * (1.0 - diff1));
      }
    }
    
    for (int h = row1 + 1; h < row2; h++) {
      pos1 = computePosition(h, row1);
      pos2 = computePosition(row2, h);
      // Save old values before updating
      d_old[pos1] = d[pos1];
      d_old[pos2] = d[pos2];
      // Update distances incrementally
      diff1 = fabs(X[col][row1] - X[col][h]);
      diff2 = fabs(X[col][row2] - X[col][h]);
      d[pos1] += log(1.5 - diff1 * (1.0 - diff1)) - log(1.5 - diff2 * (1.0 - diff2));
      d[pos2] += log(1.5 - diff2 * (1.0 - diff2)) - log(1.5 - diff1 * (1.0 - diff1));
    }
    
    if (row2 < n - 1) {
      for (int h = row2 + 1; h < n; h++) {
        pos1 = computePosition(h, row1);
        pos2 = computePosition(h, row2);
        // Save old values before updating
        d_old[pos1] = d[pos1];
        d_old[pos2] = d[pos2];
        // Update distances incrementally
        diff1 = fabs(X[col][row1] - X[col][h]);
        diff2 = fabs(X[col][row2] - X[col][h]);
        d[pos1] += log(1.5 - diff1 * (1.0 - diff1)) - log(1.5 - diff2 * (1.0 - diff2));
        d[pos2] += log(1.5 - diff2 * (1.0 - diff2)) - log(1.5 - diff1 * (1.0 - diff1));
      }
    }
  }
};


//---------------------------------------------------------------------------
// Exported functions (accessible from R).
// These functions take a design matrix and parameters, then call the respective optimizer.
//---------------------------------------------------------------------------
// Helper function to convert Rcpp NumericMatrix to native C++ double**
double** rcppMatrixToNative(const NumericMatrix &design_R) {
  int nrows = design_R.nrow();
  int ncols = design_R.ncol();
  double **design = new double*[ncols];
  for (int j = 0; j < ncols; j++) {
    design[j] = new double[nrows];
    for (int i = 0; i < nrows; i++) {
      design[j][i] = design_R(i, j);
    }
  }
  return design;
}

void deleteNativeMatrix(double **design, int ncols) {
  for (int j = 0; j < ncols; j++) {
    delete[] design[j];
  }
  delete[] design;
}

// Maximin
// [[Rcpp::export]]
List maximinLHDOptimizer_cpp(NumericMatrix design, int power = 15, int num_passes = 10, int max_iter = 1e6,
                             double temp = 0, double decay = 0.95, int no_update_iter_max = 400,
                             std::string method = "deterministic") {
  int nrows = design.nrow();
  int ncols = design.ncol();
  
  double **design_native = rcppMatrixToNative(design);
  
  MaximinLHDOptimizer optimizer(design_native, nrows, ncols, power, num_passes, (int)max_iter,
                                temp, decay, no_update_iter_max, method);
  List result = optimizer.optimize();
  
  deleteNativeMatrix(design_native, ncols);
  
  return result;
}

// MaxPro
// [[Rcpp::export]]
List maxproLHDOptimizer_cpp(NumericMatrix design, double s = 2,
                            int num_passes = 10, int max_iter = 1e6,
                            double temp = 0, double decay = 0.95, int no_update_iter_max = 400,
                            std::string method = "deterministic") {
  int nrows = design.nrow();
  int ncols = design.ncol();
  
  double **design_native = rcppMatrixToNative(design);
  
  MaxProLHDOptimizer optimizer(design_native, nrows, ncols, (int)s, num_passes, (int)max_iter,
                               temp, decay, no_update_iter_max, method);
  List result = optimizer.optimize();
  
  deleteNativeMatrix(design_native, ncols);
  
  return result;
}

// Uniform
// [[Rcpp::export]]
List uniformLHDOptimizer_cpp(NumericMatrix design, int num_passes = 10, int max_iter = 1e6,
                             double temp = 0, double decay = 0.95, int no_update_iter_max = 400,
                             std::string method = "deterministic") {
  int nrows = design.nrow();
  int ncols = design.ncol();
  
  double **design_native = rcppMatrixToNative(design);
  
  UniformLHDOptimizer optimizer(design_native, nrows, ncols, num_passes, (int)max_iter,
                                temp, decay, no_update_iter_max, method);
  List result = optimizer.optimize();
  
  deleteNativeMatrix(design_native, ncols);
  
  return result;
}
