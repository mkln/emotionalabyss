#include "../inst/include/bm2d.h"


//[[Rcpp::export]]
arma::mat splitsub_to_splitmask(const arma::field<arma::mat>& splits, int p1, int p2){
return bm2d::splitsub_to_splitmask(splits, p1, p2);
}

//[[Rcpp::export]]
arma::field<arma::mat> splitmask_to_splitsub(const arma::mat& splitmask){
  return bm2d::splitmask_to_splitsub(splitmask);
}
  
//[[Rcpp::export]]
arma::mat mask_onesplit(const arma::mat& startmat, const arma::vec& onesplit, int seq){
  return bm2d::mask_onesplit(startmat, onesplit, seq);
}
 
//[[Rcpp::export]] 
arma::mat splitsub_to_groupmask_blocks(const arma::mat& splits, int p1, int p2){
  return bm2d::splitsub_to_groupmask_blocks(splits, p1, p2);
}

//[[Rcpp::export]]
arma::mat row_intersection(const arma::mat& mat1, const arma::mat& mat2){
  return bm2d::row_intersection(mat1, mat2);
}
  
//[[Rcpp::export]]
arma::mat row_difference(const arma::mat& mat1, const arma::mat& mat2){
  return bm2d::row_difference(mat1, mat2);
}
  
//[[Rcpp::export]]
arma::mat splitsub_to_groupmask(arma::field<arma::mat> splits, int p1, int p2){
  return bm2d::splitsub_to_groupmask(splits, p1, p2);
}

//[[Rcpp::export]]
arma::vec mat_unique(const arma::mat& A){
  return bm2d::mat_unique(A);
}

//[[Rcpp::export]]
arma::mat mask_oneval(const arma::mat& A, const arma::mat& mask, int val){
  return bm2d::mask_oneval(A, mask, val);
}

//[[Rcpp::export]]
double mask_oneval_sum(const arma::mat& A, const arma::mat& mask, int val){
  return bm2d::mask_oneval_sum(A, mask, val);
}

//[[Rcpp::export]]
double mask_cube_slice(const arma::cube& C, int slice, const arma::mat& mask, int val){
  return bm2d::mask_cube_slice(C, slice, mask, val);
}

//[[Rcpp::export]]
arma::vec mat_to_vec_by_region(const arma::mat& A, 
                               const arma::mat& mask, 
                               const arma::vec& unique_regions){
  return bm2d::mat_to_vec_by_region(A, mask, unique_regions);
}
  
//[[Rcpp::export]]
arma::mat cube_to_mat_by_region(const arma::cube& C, const arma::mat& mask, 
                                const arma::vec& unique_regions){
  return bm2d::cube_to_mat_by_region(C, mask, unique_regions);
}

//[[Rcpp::export]]
arma::mat unmask_vector(const arma::vec& beta, const arma::vec& regions, const arma::mat& mask){
  return bm2d::unmask_vector(beta, regions, mask);
}

//[[Rcpp::export]]
arma::mat splitmask_focus(const arma::mat& mask_of_splits, 
                          const arma::vec& onesplit, 
                          int radius_int=1){
  return bm2d::splitmask_focus(mask_of_splits, onesplit, radius_int);
}

//[[Rcpp::export]]
int number_availables(const arma::mat& splitmask_focus){
  return bm2d::number_availables(splitmask_focus);
}
  
  