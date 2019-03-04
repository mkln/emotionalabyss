#include "../inst/include/bmfuncs.h"


//[[Rcpp::export]]
double split_struct_ratio(arma::vec prop_split, arma::vec orig_split, int p, double param){
  return bmfuncs::split_struct_ratio(prop_split, orig_split, p, param); 
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
