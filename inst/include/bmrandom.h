#ifndef RCPP_bmrandom
#define RCPP_bmrandom

#include "bmdataman.h"
#include "bmtruncn.h"
#include <omp.h>

using namespace std;
using namespace bmtruncn;

namespace bmrandom {

inline arma::mat rndppll_mvnormal(int n, const arma::vec &mean, const arma::mat &sigma){
  int dimension = arma::size(mean)(0);
  arma::vec xtemp = arma::zeros(dimension);
  arma::mat outmat = arma::zeros(n, dimension);
  arma::mat cholsigma = arma::chol(sigma, "lower");
#pragma omp parallel for num_threads(NUM_THREADS)
  for(int i=0; i<n; i++){
    //for(int j=0; j<dimension; j++){
    //  xtemp(j) = rndpp_normal(0.0, 1.0, mt);
    //}
    xtemp = arma::randn(dimension);
    //clog << arma::det(sigma) << endl;
    outmat.row(i) = (mean + cholsigma * xtemp).t();
  }
  return outmat;
}

// sample n elements from 0:vsize-1 with no replacement
inline arma::uvec sample_index(const int& n, const int &vsize){
  arma::uvec sequence = arma::linspace<arma::uvec>(0, vsize-1, vsize);
  //arma::uvec out = Rcpp::RcppArmadillo::sample(sequence, n, false);
  std::random_shuffle ( sequence.begin(), sequence.end() );
  return sequence.subvec(0, n-1);
}

// sample 1 of 0:max-1 uniformly
inline int rndpp_unif_int(int max){
  double uf = R::runif(0, max+1);
  return floor(uf);
}

// same as above
inline int sample_one_int(const int &vsize){
  //arma::uvec sequence = arma::linspace<arma::uvec>(0, vsize-1, vsize);
  //int out = (Rcpp::RcppArmadillo::sample(sequence, 1, false))(0);
  return rndpp_unif_int(vsize);
}

// bool valued bernoulli(p)
inline bool boolbern(double p){
  double run = arma::randu();
  return run < p;
}

// sample 1 from discrete on 0:length(probs)-1 with specified probs
inline int rndpp_discrete(const arma::vec& probs){
  Rcpp::IntegerVector xx = Rcpp::seq_len(probs.n_elem);
  return Rcpp::as<int>(Rcpp::sample(xx, 1, false, Rcpp::NumericVector(probs.begin(), probs.end()))) - 1;
}

// return a random element from those in fromvec (vector of integers)
inline int rndpp_sample1(const arma::vec& fromvec, const arma::vec& probs){
  return fromvec( rndpp_discrete(probs) );
}

inline int rndpp_sample1_comp(const arma::vec& x, int p, int current_split, double decay=4.0){
  /* vector x = current splits, p = how many in total
   * this returns 1 split out of the complement 
   * decay controls how far the proposed jump is going to be from the current
   * decay=1 corresponds to uniform prob on all availables
   * if current_split=-1 then pick uniformly
   */
  //double decay = 5.0;
  arma::vec all = arma::linspace(0, p-1, p);
  arma::vec avail = bmdataman::bmms_setdiff(all, x);
  //cout << avail << endl;
  arma::vec probweights;
  if(current_split == -1){
    probweights = arma::ones(arma::size(avail));
  } else {
    probweights = arma::exp(arma::abs(avail - current_split) * log(1.0/decay));
  }
  if(avail.n_elem > 0){
    int out = rndpp_sample1(avail, probweights); //Rcpp::RcppArmadillo::sample(avail, 1, true, probweights); 
    return out;//(0);
  } else {
    return -1;
  }
}

inline arma::vec rndpp_shuffle(arma::vec x){
  /* vector x = a vector
   output = reshuffled vector
   */
  //double decay = 5.0;
  //return Rcpp::RcppArmadillo::sample(x, x.n_elem, false); 
  std::random_shuffle ( x.begin(), x.end() );
  return x;
}

inline double rndpp_bern(double p){
  double run = arma::randu();
  if(run < p){ 
    return 1.0;
  } else {
    return 0.0;
  }
}

inline double rndpp_gamma(const double& alpha, const double& beta) 
{
  return R::rgamma(alpha, beta);
}

inline double rndpp_normal(const double& mean, const double& sigma) 
{
  return R::rnorm(mean, sigma);
}

inline arma::mat rndpp_mvnormal(int n, const arma::vec &mean, const arma::mat &sigma){
  int dimension = arma::size(mean)(0);
  arma::vec xtemp = arma::zeros(dimension);
  arma::mat outmat = arma::zeros(n, dimension);
  arma::mat cholsigma = arma::chol(sigma, "lower");
  for(int i=0; i<n; i++){
    //for(int j=0; j<dimension; j++){
    //  xtemp(j) = rndpp_normal(0.0, 1.0, mt);
    //}
    xtemp = Rcpp::rnorm(dimension, 0.0, 1.0);
    //clog << arma::det(sigma) << endl;
    outmat.row(i) = (mean + cholsigma * xtemp).t();
  }
  return outmat;
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
inline arma::mat mvtruncnormal(const arma::vec& mean, 
                               const arma::vec& l_in, const arma::vec& u_in, 
                               const arma::mat& Sig, int n){
  arma::mat meanmat = arma::zeros(mean.n_elem, n);
  arma::mat truncraw = mvrandn_cpp(l_in-mean, u_in-mean, Sig, n);
  for(unsigned int i=0; i<n; i++){
    meanmat.col(i) = mean + truncraw.col(i);
  }
  return meanmat;
}

//' Sample from Truncated and shifted Normal with Identity covariance
//' 
//' @param mean A p-dimensional mean vector
//' @param l_in A p-dimensional vector of lower truncation limits (can be -Inf)
//' @param u_in A p-dimensional vector of upper truncation limits (can be Inf)
//' @export
//[[Rcpp::export]]
inline arma::mat mvtruncnormal_eye1(const arma::vec& mean, 
                                    const arma::vec& l_in, const arma::vec& u_in){
  int n = 1;
  //arma::mat meanmat = arma::zeros(mean.n_elem, n);
  arma::vec truncraw = arma::zeros(mean.n_elem);
  truncraw = trandn_cpp(l_in - mean, u_in - mean);
  //meanmat.col(0) = mean + truncraw;
  
  return mean+truncraw;
}

inline arma::mat rndpp_mvnormal2(int n, const arma::vec& mu, const arma::mat& sigma) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  //Y.randn();
  return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}

inline arma::mat rndpp_mvnormalnew(int n, const arma::vec &mean, const arma::mat &sigma){
  int dimension = mean.n_elem;
  arma::mat outmat = arma::zeros(dimension, n);
  arma::mat cholsigma = arma::chol(sigma, "lower");
  arma::mat xtemp = (arma::randn(n, dimension)).t();
  arma::mat term2 = cholsigma * xtemp;
  for(int i=0; i<n; i++){
    outmat.col(i) = mean + term2.col(i);
  }
  return outmat.t();
}

inline arma::mat rndpp_stdmvnormal(int n, int dimension){
  arma::vec xtemp = arma::zeros(dimension);
  arma::mat outmat = arma::zeros(n, dimension);
  for(int i=0; i<n; i++){
    xtemp = Rcpp::rnorm(dimension, 0.0, 1.0);
    outmat.row(i) = xtemp.t();
  }
  return outmat;
}

inline arma::mat rndpp_mvt(int n, const arma::vec &mu, const arma::mat &sigma, double df){
  double w=1.0;
  arma::mat Z = rndpp_stdmvnormal(n, mu.n_elem);
  arma::mat cholsigma = arma::chol(sigma, "lower");
  arma::mat AZ = Z;
  for(int i=0; i<AZ.n_rows; i++){
    w = sqrt( df / R::rchisq(df) );
    AZ.row(i) = (mu + w * (cholsigma * Z.row(i).t())).t();
  }
  return AZ;
}

}

#endif