#include "../inst/include/bmrandom.h"

//[[Rcpp::export]] 
arma::mat rndpp_mvnormal(int n, const arma::vec &mean, const arma::mat &sigma){
  return bmrandom::rndpp_mvnormal(n, mean, sigma);
}


//[[Rcpp::export]] 
arma::mat rndpp_mvnormal1(int n, const arma::vec &mean, const arma::mat &sigma){
  return bmrandom::rndpp_mvnormal1(n, mean, sigma);
}

//[[Rcpp::export]] 
arma::mat rndpp_mvnormal2(int n, const arma::vec& mu, const arma::mat& sigma) {
  return bmrandom::rndpp_mvnormal2(n, mu, sigma);
}

//[[Rcpp::export]] 
arma::mat rndpp_mvnormal3(int n, const arma::vec &mean, const arma::mat &sigma){
  return bmrandom::rndpp_mvnormal3(n, mean, sigma);
}

// sample n elements from 0:vsize-1 with no replacement
//[[Rcpp::export]] 
arma::uvec sample_index(const int& n, const int &vsize){
  return bmrandom::sample_index(n, vsize);
}

// sample 1 of 0:max-1 uniformly
//[[Rcpp::export]] 
int rndpp_unif_int(int maxi){
  return bmrandom::rndpp_unif_int(maxi);
}

// same as above
//[[Rcpp::export]] 
int sample_one_int(const int &vsize){
  return bmrandom::sample_one_int(vsize);
}

// bool valued bernoulli(p)
//[[Rcpp::export]] 
bool boolbern(double p){
  return bmrandom::boolbern(p);
}

// sample 1 from discrete on 0:length(probs)-1 with specified probs
//[[Rcpp::export]] 
int rndpp_discrete(const arma::vec& probs){
  return bmrandom::rndpp_discrete(probs);
}

// return a random element from those in fromvec (vector of integers)
//[[Rcpp::export]] 
int rndpp_sample1(const arma::vec& fromvec, const arma::vec& probs){
  return bmrandom::rndpp_sample1(fromvec, probs);
}

//[[Rcpp::export]] 
int rndpp_sample1_comp(const arma::vec& x, int p, int current_split, double decay=4.0){
  return bmrandom::rndpp_sample1_comp(x, p, current_split, decay);
}

//[[Rcpp::export]] 
arma::vec rndpp_shuffle(arma::vec x){
  return bmrandom::rndpp_shuffle(x);
}

//[[Rcpp::export]] 
double rndpp_bern(double p){
  return bmrandom::rndpp_bern(p);
}

//[[Rcpp::export]] 
double rndpp_gamma(const double& alpha, const double& beta) 
{
  return bmrandom::rndpp_gamma(alpha, beta);
}

//[[Rcpp::export]] 
double rndpp_normal(const double& mean, const double& sigma) 
{
  return bmrandom::rndpp_normal(mean, sigma);
}

//' Sample from Truncated Normal using Botev (2017)
//' 
//' @param mean A p-dimensional mean vector
//' @param l_in A p-dimensional vector of lower truncation limits (can be -Inf)
//' @param u_in A p-dimensional vector of upper truncation limits (can be Inf)
//' @param Sig A (p,p) covariance matrix.
//' @param n number of samples to extract
//' @export
//[[Rcpp::export]] 
arma::mat mvtruncnormal(const arma::vec& mean, 
                               const arma::vec& l_in, const arma::vec& u_in, 
                               const arma::mat& Sig, int n){
  return bmrandom::mvtruncnormal(mean, l_in, u_in, Sig, n);
}

//' Sample from Truncated and shifted Normal with Identity covariance
//' 
//' @param mean A p-dimensional mean vector
//' @param l_in A p-dimensional vector of lower truncation limits (can be -Inf)
//' @param u_in A p-dimensional vector of upper truncation limits (can be Inf)
//' @export
//[[Rcpp::export]] 
arma::mat mvtruncnormal_eye1(const arma::vec& mean, 
                                    const arma::vec& l_in, const arma::vec& u_in){
  return bmrandom::mvtruncnormal_eye1(mean, l_in, u_in);
}

//[[Rcpp::export]] 
arma::mat rndpp_stdmvnormal(int n, int dimension){
  return bmrandom::rndpp_stdmvnormal(n, dimension);
}

//[[Rcpp::export]] 
arma::mat rndpp_mvt(int n, const arma::vec &mu, const arma::mat &sigma, double df){
  return bmrandom::rndpp_mvt(n, mu, sigma, df);
}
