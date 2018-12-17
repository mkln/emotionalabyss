#include "../inst/include/bmdataman.h"

//[[Rcpp::export]] 
arma::vec setdiff(const arma::vec& x, const arma::vec& y) {
  return bmdataman::bmms_setdiff(x, y);
}

//[[Rcpp::export]] 
arma::mat X2Dgrid(const arma::vec& x1, const arma::vec& x2){
  return bmdataman::X2Dgrid(x1, x2);
}

//[[Rcpp::export]] 
arma::mat X2Dgrid_alt(const arma::vec& x1, const arma::vec& x2){
  return bmdataman::X2Dgrid_alt(x1, x2);
}

//[[Rcpp::export]] 
arma::mat exclude(const arma::mat& test, const arma::vec& excl){
  return bmdataman::exclude(test, excl);
}

//[[Rcpp::export]] 
arma::vec nonzeromean(const arma::mat& mat_mcmc){
  return bmdataman::nonzeromean(mat_mcmc);
}

//[[Rcpp::export]] 
arma::vec col_eq_check(const arma::mat& A){
  return bmdataman::col_eq_check(A);
}

//[[Rcpp::export]] 
arma::vec col_sums(const arma::mat& matty){
  return bmdataman::col_sums(matty);
}

//[[Rcpp::export]] 
arma::mat hat_alt(const arma::mat& X){ // svd
  return bmdataman::hat_alt(X);
}

//[[Rcpp::export]] 
arma::mat hat(const arma::mat& X){ // inv_sympd
  return bmdataman::hat(X);
}

//[[Rcpp::export]] 
arma::mat cube_mean(const arma::cube& X, int dim){
  return bmdataman::cube_mean(X, dim);
}

//[[Rcpp::export]] 
arma::mat cube_sum(const arma::cube& X, int dim){
  return bmdataman::cube_sum(X, dim);
}

//[[Rcpp::export]] 
arma::mat cube_prod(const arma::cube& x, int dim){
  return bmdataman::cube_prod(x, dim);
}

// Vector index to matrix subscripts
// 
// Get matrix subscripts from corresponding vector indices (both start from 0).
// This is a utility function using Armadillo's ind2sub function.
// @param index a vector of indices
// @param m a matrix (only its size is important)
//[[Rcpp::export]]
arma::mat index_to_subscript(const arma::uvec& index, const arma::mat& m){
  return bmdataman::index_to_subscript(index, m);
}

//[[Rcpp::export]]
double gini(arma::vec& x, int p){
  return bmdataman::gini(x, p);
}

//[[Rcpp::export]] 
arma::vec find_avail(const arma::vec& tied, int n){
  return bmdataman::find_avail(tied, n);
}

//[[Rcpp::export]] 
arma::uvec find_first_unique(const arma::vec& x){
  return bmdataman::find_first_unique(x);
}

//[[Rcpp::export]] 
arma::vec find_ties(const arma::vec& x){
  return bmdataman::find_ties(x);
}

//[[Rcpp::export]] 
arma::vec pnorm01_vec(const arma::vec& x, int lower=1, int logged=0){
  return bmdataman::pnorm01_vec(x, lower, logged);
}

//[[Rcpp::export]] 
arma::vec qnorm01_vec(const arma::vec& x, int lower=1, int logged=0){
  return bmdataman::qnorm01_vec(x, lower, logged);
}

//[[Rcpp::export]] 
arma::vec log1p_vec(const arma::vec& x){
  return bmdataman::log1p_vec(x);
}

//[[Rcpp::export]] 
arma::uvec usetdiff(const arma::uvec& x, const arma::uvec& y) {
  return bmdataman::usetdiff(x, y);
}

