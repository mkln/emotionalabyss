#include "../inst/include/bmfuncs.h"

//[[Rcpp::export]] 
arma::vec setdiff(const arma::vec& x, const arma::vec& y) {
  return bmfuncs::bmms_setdiff(x, y);
}

//[[Rcpp::export]] 
arma::mat X2Dgrid(const arma::vec& x1, const arma::vec& x2){
  return bmfuncs::X2Dgrid(x1, x2);
}

//[[Rcpp::export]] 
arma::mat X2Dgrid_alt(const arma::vec& x1, const arma::vec& x2){
  return bmfuncs::X2Dgrid_alt(x1, x2);
}

//[[Rcpp::export]] 
arma::mat exclude(const arma::mat& test, const arma::vec& excl){
  return bmfuncs::exclude(test, excl);
}

//[[Rcpp::export]] 
arma::vec nonzeromean(const arma::mat& mat_mcmc){
  return bmfuncs::nonzeromean(mat_mcmc);
}

//[[Rcpp::export]] 
arma::vec col_eq_check(const arma::mat& A){
  return bmfuncs::col_eq_check(A);
}

//[[Rcpp::export]] 
arma::vec col_sums(const arma::mat& matty){
  return bmfuncs::col_sums(matty);
}

//[[Rcpp::export]] 
arma::mat hat_alt(const arma::mat& X){ // svd
  return bmfuncs::hat_alt(X);
}

//[[Rcpp::export]] 
arma::mat hat(const arma::mat& X){ // inv_sympd
  return bmfuncs::hat(X);
}

//[[Rcpp::export]] 
arma::mat cube_mean(const arma::cube& X, int dim){
  return bmfuncs::cube_mean(X, dim);
}

//[[Rcpp::export]] 
arma::mat cube_sum(const arma::cube& X, int dim){
  return bmfuncs::cube_sum(X, dim);
}

//[[Rcpp::export]] 
arma::mat cube_prod(const arma::cube& x, int dim){
  return bmfuncs::cube_prod(x, dim);
}

// Vector index to matrix subscripts
// 
// Get matrix subscripts from corresponding vector indices (both start from 0).
// This is a utility function using Armadillo's ind2sub function.
// @param index a vector of indices
// @param m a matrix (only its size is important)
//[[Rcpp::export]]
arma::mat index_to_subscript(const arma::uvec& index, const arma::mat& m){
  return bmfuncs::index_to_subscript(index, m);
}

//[[Rcpp::export]]
double gini(arma::vec& x, int p){
  return bmfuncs::gini(x, p);
}

//[[Rcpp::export]] 
arma::vec find_avail(const arma::vec& tied, int n){
  return bmfuncs::find_avail(tied, n);
}

//[[Rcpp::export]] 
arma::uvec find_first_unique(const arma::vec& x){
  return bmfuncs::find_first_unique(x);
}

//[[Rcpp::export]] 
arma::vec find_ties(const arma::vec& x){
  return bmfuncs::find_ties(x);
}

//[[Rcpp::export]] 
arma::vec pnorm01_vec(const arma::vec& x, int lower=1, int logged=0){
  return bmfuncs::pnorm01_vec(x, lower, logged);
}

//[[Rcpp::export]] 
arma::vec qnorm01_vec(const arma::vec& x, int lower=1, int logged=0){
  return bmfuncs::qnorm01_vec(x, lower, logged);
}

//[[Rcpp::export]] 
arma::vec log1p_vec(const arma::vec& x){
  return bmfuncs::log1p_vec(x);
}

//[[Rcpp::export]] 
arma::uvec usetdiff(const arma::uvec& x, const arma::uvec& y) {
  return bmfuncs::usetdiff(x, y);
}

//[[Rcpp::export]]
arma::mat single_split(const arma::mat& Jcoarse, int where, int p){
  return bmfuncs::single_split(Jcoarse, where, p);
}

//[[Rcpp::export]]
arma::mat single_split_new(const arma::mat& Jcoarse, int where, int p){
  return bmfuncs::single_split_new(Jcoarse, where, p);
}

//[[Rcpp::export]]
arma::mat multi_split(const arma::mat& Jcoarse, 
                      const arma::vec& where, int p){
  return bmfuncs::multi_split(Jcoarse, where, p);
}

//[[Rcpp::export]]
arma::mat multi_split_old(const arma::mat& Jcoarse, 
                      const arma::vec& where, int p){
  return bmfuncs::multi_split_old(Jcoarse, where, p);
}

//[[Rcpp::export]]
arma::mat multi_split_ones(const arma::vec& where, int p){
  return bmfuncs::multi_split_ones(where, p);
}

//[[Rcpp::export]]
arma::mat multi_split_ones_v2(const arma::vec& where, int p){
  return bmfuncs::multi_split_ones_v2(where, p);
}

/*
//[[Rcpp::export]] 
arma::vec break_ones(int pre, int thisbig, int where, int post){
  return bmfuncs::break_ones(pre, thisbig, where, post);
}

//[[Rcpp::export]]
arma::vec break_existing(arma::vec starting, int where){
  return bmfuncs::break_existing(starting, where);
}
  
//[[Rcpp::export]]
arma::mat decomp_identi(const arma::field<arma::vec>& where, int p){
  return bmfuncs::decomp_identi(where, p);
}
*/
//[[Rcpp::export]]
arma::vec split_fix(const arma::field<arma::vec>& in_splits, int stage){
  return bmfuncs::split_fix(in_splits, stage);
}

//[[Rcpp::export]]
arma::field<arma::vec> stage_fix(const arma::field<arma::vec>& in_splits){
  return bmfuncs::stage_fix(in_splits);
}

//[[Rcpp::export]]
arma::vec stretch_vec(const arma::mat& locations, const arma::vec& base, int dim){
  return bmfuncs::stretch_vec(locations, base, dim);
}

//[[Rcpp::export]]
arma::mat reshaper(const arma::field<arma::mat>& J_field, int s){
  return bmfuncs::reshaper(J_field, s);
}

//[[Rcpp::export]]
arma::field<arma::vec> splits_truncate(const arma::field<arma::vec>& splits, int k){
  return bmfuncs::splits_truncate(splits, k);
}

//[[Rcpp::export]]
arma::field<arma::mat> splits_augmentation(const arma::field<arma::mat>& splits){
  return bmfuncs::splits_augmentation(splits);
}

//[[Rcpp::export]]
arma::field<arma::mat> merge_splits(const arma::field<arma::mat>& old_splits, 
                                           const arma::field<arma::mat>& new_splits){
  return bmfuncs::merge_splits(old_splits, new_splits);
}

//[[Rcpp::export]]
arma::field<arma::mat> Ares(const arma::mat& L1, const arma::mat& L2, const arma::mat& L)
{
  return bmfuncs::Ares(L1, L2, L);
}

//[[Rcpp::export]]
arma::vec update_scale(const arma::mat& Di, 
                              const arma::mat& D1,
                              const arma::mat& D2,
                              const arma::field<arma::mat>& A,
                              const arma::mat& X1,
                              const arma::mat& X2,
                              const arma::vec& theta1,
                              const arma::vec& theta2){
  return bmfuncs::update_scale(Di, D1, D2, A, X1, X2, theta1, theta2);
}
