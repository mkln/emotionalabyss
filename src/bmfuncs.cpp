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
arma::mat multi_split_ones(const arma::vec& where, int p){
  return bmfuncs::multi_split_ones(where, p);
}

//[[Rcpp::export]]
arma::mat multi_split_ones_v2(const arma::vec& where, int p){
  return bmfuncs::multi_split_ones_v2(where, p);
}

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
