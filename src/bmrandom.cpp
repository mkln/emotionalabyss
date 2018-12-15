#include "../inst/include/bmrandom.h"

//[[Rcpp::export]] 
arma::mat rndppll_mvnormal(int n, const arma::vec &mean, const arma::mat &sigma){
  return bmrandom::rndppll_mvnormal(n, mean, sigma);
}


//[[Rcpp::export]] 
arma::mat rndpp_mvnormal(int n, const arma::vec &mean, const arma::mat &sigma){
  return bmrandom::rndpp_mvnormal(n, mean, sigma);
}

//[[Rcpp::export]] 
arma::mat rndpp_mvnormal2(int n, const arma::vec& mu, const arma::mat& sigma) {
  return bmrandom::rndpp_mvnormal2(n, mu, sigma);
}

//[[Rcpp::export]] 
arma::mat rndpp_mvnormalnew(int n, const arma::vec &mean, const arma::mat &sigma){
  return bmrandom::rndpp_mvnormalnew(n, mean, sigma);
}
