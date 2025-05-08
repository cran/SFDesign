// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <vector>
#include <cstdlib>
#include <cmath>
using namespace Rcpp;
using namespace arma;

//--------------------------------------------------------------
// Class: ClusterDesign
// Implements a Lloyd‐style clustering design update.
//--------------------------------------------------------------
class ClusterDesign {
private:
  arma::mat X;       // Data matrix.
  arma::mat centers; // Cluster centers.
  double alpha;
  int Lloyd_iter_max;
  double Lloyd_tol;
  int cen_iter_max;
  double cen_tol;

public:
  arma::uvec cluster_assign;  // Cluster assignment (1-indexed).
  arma::vec distances;        // For each data point, distance to its assigned center.

  // Constructor.
  ClusterDesign(const arma::mat &data, const arma::mat &initialCenters, double alpha,
                int Lloyd_iter_max, double Lloyd_tol,
                int cen_iter_max, double cen_tol)
    : X(data), centers(initialCenters), alpha(alpha),
      Lloyd_iter_max(Lloyd_iter_max), Lloyd_tol(Lloyd_tol),
      cen_iter_max(cen_iter_max), cen_tol(cen_tol)
  {
    cluster_assign.set_size(X.n_rows);
    distances.set_size(X.n_rows);
  }

  // Assign each data point to the closest center.
  void assignClusters() {
    int N = X.n_rows;
    for (int j = 0; j < N; j++) {
      rowvec xj = X.row(j);
      vec dist = sqrt(sum(square(centers.each_row() - xj), 1));
      distances(j) = min(dist);
      cluster_assign(j) = dist.index_min() + 1;  // 1-based indexing for R
    }
  }

  // Static method: Weiszfeld algorithm
  static arma::rowvec Weiszfeld(const arma::mat &clusterPoints, arma::rowvec center,
                                double alpha, int iter_max=10, double tol=1e-4) {
    double obj0 = datum::inf;
    if (clusterPoints.n_rows == 1) {
      return clusterPoints.row(0);
    }
    for (int j = 0; j < iter_max; j++) {
      vec dists = sqrt(sum(square(clusterPoints.each_row() - center), 1));
      double obj = accu(pow(dists, alpha));
      vec w;
      if (alpha < 2) {
        w = 1.0 / (pow(dists, 2.0 - alpha) + 1e-16);
      } else {
        w = pow(dists, alpha - 2.0);
      }
      w = w / sum(w);
      rowvec newCenter = (w.t() * clusterPoints).eval();
      if ((obj0 - obj) / obj0 < tol) break;
      center = newCenter;
      obj0 = obj;
    }
    return center;
  }

  // Static method: Accelerated gradient descent for center update.
  static arma::rowvec cqAGD(const arma::mat &clusterPoints, double alpha,
                            int iter_max=1000, double tol=1e-4) {
    int n = clusterPoints.n_rows;
    int p = clusterPoints.n_cols;
    int it_num = 0;
    double beta = alpha - 1;
    double diff = 0.0;
    rowvec curz = mean(clusterPoints, 0);
    rowvec curu = curz;
    double lambdat = 1.0, lambdatp1 = 1.0;
    rowvec prevu(p);
    double gammat = (1.0 - lambdat) / lambdatp1;
    while (it_num < iter_max) {
      lambdat = lambdatp1;
      lambdatp1 = (1.0 + sqrt(1.0 + 4.0 * lambdat * lambdat)) / 2.0;
      gammat = (1.0 - lambdat) / lambdatp1;
      prevu = curu;
      curu = curz;
      vec wts(n, fill::zeros);
      for (int i = 0; i < n; i++){
        double dist = norm(curz - clusterPoints.row(i), 2);
        wts(i) = (1.0 / n) * (1.0 / beta) * pow(dist, alpha - 2.0);
      }
      for (int i = 0; i < n; i++){
        curu -= wts(i) * (curz - clusterPoints.row(i));
      }
      for (int j = 0; j < p; j++){
        curz(j) = (1 - gammat) * curu(j) + gammat * prevu(j);
      }
      diff = norm(curu - prevu, 2);
      if (diff < tol) break;
      it_num++;
    }
    return curz;
  }

  // Run the Lloyd algorithm: update centers and assignments until convergence.
  List runLloyd() {
    int N = X.n_rows;
    assignClusters();
    double obj0 = accu(pow(distances, alpha)) / N;
    double obj = obj0;
    int i;
    for (i = 0; i < Lloyd_iter_max; i++) {
      int k = centers.n_rows;
      for (int cl = 0; cl < k; cl++) {
        uvec idx = find(cluster_assign == (cl + 1));
        if (idx.n_elem > 0) {
          mat clusterPoints = X.rows(idx);
          if (alpha <= 2) {
            centers.row(cl) = Weiszfeld(clusterPoints, centers.row(cl), alpha, cen_iter_max, cen_tol);
          } else {
            centers.row(cl) = cqAGD(clusterPoints, alpha, cen_iter_max, cen_tol);
          }
        }
      }
      assignClusters();
      obj = accu(pow(distances, alpha)) / N;
      if ((obj0 - obj) > 0 && (obj0 - obj) / obj0 < Lloyd_tol) break;
      obj0 = obj;
    }
    return List::create(
      Named("design") = centers,
      Named("cluster") = cluster_assign,
      Named("cluster_error") = obj,
      Named("total_iter") = i
    );
  }
};

//--------------------------------------------------------------
// Class: KMeansPSO
// Implements a particle swarm optimization (PSO) approach to clustering.
//--------------------------------------------------------------
// class KMeansPSO {
// private:
//   arma::mat X;         // Clustering data.
//   arma::mat initCenters; // Initial cluster centers for all particles (dimensions: cluster_num x (p * part_num)).
//   double alpha;
//   double w, c1, c2;
//   int pso_iter_max, pso_iter_same_lim;
//   double pso_iter_tol;
//   int Lloyd_iter_max, cen_iter_max;
//   double Lloyd_tol, cen_tol;
//   int num_proc;
//
//   // Derived dimensions.
//   int p, point_num, cluster_num, part_num;
//
//   // Particle state.
//   arma::cube cluster_center;  // cluster_num x p x part_num.
//   arma::cube cluster_vel;     // Velocities.
//   arma::cube cluster_lbes;    // Local best positions.
//   arma::mat lbes_obj;         // Local best objective (1 x part_num).
//   arma::mat cluster_gbes;     // Global best cluster centers (cluster_num x p).
//   double gbes_obj;
//   arma::rowvec assign_gbes;
//   arma::rowvec energy_gbes;
//
// public:
//   // Constructor.
//   KMeansPSO(const arma::mat &X, const arma::mat &cluster_center_mat,
//             double alpha, double w, double c1, double c2,
//             int pso_iter_max, int pso_iter_same_lim, double pso_iter_tol,
//             int Lloyd_iter_max, double Lloyd_tol, int cen_iter_max, double cen_tol,
//             int num_proc)
//     : X(X), initCenters(cluster_center_mat), alpha(alpha), w(w), c1(c1), c2(c2),
//       pso_iter_max(pso_iter_max), pso_iter_same_lim(pso_iter_same_lim),
//       pso_iter_tol(pso_iter_tol), Lloyd_iter_max(Lloyd_iter_max), cen_iter_max(cen_iter_max),
//       Lloyd_tol(Lloyd_tol), cen_tol(cen_tol), num_proc(num_proc)
//   {
//     p = X.n_cols;
//     point_num = X.n_rows;
//     cluster_num = initCenters.n_rows;
//     part_num = initCenters.n_cols / p;
//     cluster_center = cube(initCenters.memptr(), cluster_num, p, part_num);
//     cluster_vel = zeros<cube>(cluster_num, p, part_num);
//     cluster_lbes = zeros<cube>(cluster_num, p, part_num);
//     lbes_obj = arma::ones<mat>(1, part_num) * DBL_MAX;
//     cluster_gbes = zeros<mat>(cluster_num, p);
//     gbes_obj = DBL_MAX;
//     assign_gbes = arma::zeros<rowvec>(point_num);
//     energy_gbes = arma::zeros<rowvec>(cluster_num);
//   }
//
//   // Run the PSO algorithm.
//   List runPSO() {
//     int it_num = 0;
//     int it_same = 0;
//     arma::mat cluster_dist = zeros<mat>(part_num, point_num);
//     arma::mat cluster_tot = zeros<mat>(part_num, 1);
//     arma::mat cluster_assign = -1 * ones<mat>(part_num, point_num);
//     arma::mat Dtmp(cluster_num, p);
// #ifdef _OPENMP
//     omp_set_num_threads(num_proc);
// #endif
//     Rcout << "Minimax clustering ..." << std::endl;
//     while ((it_num < pso_iter_max) && (it_same < pso_iter_same_lim)) {
//       // Update clustering for each particle in parallel.
// #pragma omp parallel for
//       for (int i = 0; i < part_num; i++) {
//         // For each particle, run a Lloyd clustering update.
//         arma::mat particleCenters = cluster_center.slice(i);
//         ClusterDesign cd(X, particleCenters, alpha, Lloyd_iter_max, Lloyd_tol, cen_iter_max, cen_tol);
//         List res = cd.runLloyd();
//         cluster_center.slice(i) = as<arma::mat>(res["centers"]);
//         // For simplicity, we store the mean error as the particle's objective.
//         double error = as<double>(res["mean_error"]);
//         cluster_dist.row(i).fill(error);
//         cluster_assign.row(i) = as<arma::rowvec>(res["cluster"]);
//       }
//       // Update local and global bests.
//       cluster_tot = sum(pow(cluster_dist, alpha), 1);
//       if (alpha >= 1) {
//         cluster_tot = pow(cluster_tot, 1 / alpha);
//       }
//       bool ch_flg = false;
//       for (int i = 0; i < part_num; i++) {
//         if (cluster_tot(i) < lbes_obj(i)) {
//           cluster_lbes.slice(i) = cluster_center.slice(i);
//           lbes_obj(i) = cluster_tot(i);
//         }
//         if (cluster_tot(i) < gbes_obj) {
//           cluster_gbes = cluster_center.slice(i);
//           gbes_obj = cluster_tot(i);
//           assign_gbes = cluster_assign.row(i);
//           energy_gbes = cluster_dist.row(i);
//           // For simplicity, flag improvement if global best is very low.
//           if (gbes_obj < 1e-9) {
//             ch_flg = true;
//           }
//         }
//       }
//       if (ch_flg) {
//         it_same = 0;
//       } else {
//         it_same++;
//       }
//       // Update velocities and positions.
//       for (int i = 0; i < part_num; i++) {
//         Dtmp = cluster_center.slice(i);
//         cluster_vel.slice(i) = w * cluster_vel.slice(i) +
//           c1 * (randu<mat>(cluster_num, p) % (cluster_lbes.slice(i) - Dtmp)) +
//           c2 * (randu<mat>(cluster_num, p) % (cluster_gbes - Dtmp));
//         cluster_center.slice(i) += cluster_vel.slice(i);
//       }
//       it_num++;
//     }
//     return List::create(
//       Named("centers") = cluster_gbes,
//       Named("cluster") = assign_gbes,
//       Named("gbes_obj") = gbes_obj,
//       Named("particles") = initCenters
//     );
//   }
// };

//--------------------------------------------------------------
// Exported functions callable from R.
//--------------------------------------------------------------
// Criterion
// [[Rcpp::export]]
double clusterError(const arma::mat& design, const arma::mat& X, double alpha=1.0) {
  int N = X.n_rows;
  vec min_dist(N);

  for(int i = 0; i < N; i++) {
    rowvec Xi = X.row(i);
    vec distances = sqrt(sum(square(design.each_row() - Xi), 1));
    min_dist(i) = min(distances);
  }

  return accu(pow(min_dist, alpha)) / N;
}

// Clustering design using Lloyd algorithm.
// [[Rcpp::export]]
List cluster_based_design_cpp(const arma::mat& X, arma::mat D_ini, double alpha=1.0,
                              int Lloyd_iter_max=100, double Lloyd_tol=1e-4,
                              int cen_iter_max=10, double cen_tol=1e-4) {
  ClusterDesign cd(X, D_ini, alpha, Lloyd_iter_max, Lloyd_tol, cen_iter_max, cen_tol);
  return cd.runLloyd();
}

// // In-place clustering design update.
// void cluster_based_design_inplace(const arma::mat& X, arma::mat& D,
//                                   arma::rowvec& cluster_assign, arma::rowvec& distances,
//                                   double alpha=1.0,
//                                   int Lloyd_iter_max=100,  int cen_iter_max=10,
//                                   double Lloyd_tol=1e-5, double cen_tol=1e-5) {
//   ClusterDesign cd(X, D, alpha, Lloyd_iter_max, Lloyd_tol, cen_iter_max, cen_tol);
//   List res = cd.runLloyd();
//   D = as<arma::mat>(res["centers"]);
//   // Convert uvec cluster assignments to rowvec.
//   arma::rowvec assign = conv_to<rowvec>::from(as<arma::uvec>(res["cluster"]));
//   cluster_assign = assign;
//   double me = as<double>(res["mean_error"]);
//   distances.fill(me);
// }
//
// // PSO clustering.
//
// // [[Rcpp::export]]
// List kmeanspso(const arma::mat& X, arma::mat& cluster_center_mat,
//                double alpha,
//                double w, double c1, double c2,
//                int pso_iter_max, int pso_iter_same_lim, double pso_iter_tol,
//                int Lloyd_iter_max, double Lloyd_tol,
//                int cen_iter_max, double cen_tol,
//                int num_proc)
// {
//   KMeansPSO pso(X, cluster_center_mat, alpha, w, c1, c2,
//                 pso_iter_max, pso_iter_same_lim, pso_iter_tol,
//                 Lloyd_iter_max, Lloyd_tol, cen_iter_max, cen_tol,
//                 num_proc);
//   return pso.runPSO();
// }
