#ifndef RCPP_bmrng
#define RCPP_bmrng

#include <bmdataman.h>
//#include <omp.h>
#include "R.h"
#include "Rmath.h"
#include "RcppArmadillo.h"

using namespace std;

namespace bmrandom {

const double __PI = 3.141592653589793238462643383279502884197;
const double HALFPISQ = 0.5 * __PI * __PI;
const double FOURPISQ = 4 * __PI * __PI;
const double __TRUNC = 0.64;
const double __TRUNC_RECIP = 1.0 / __TRUNC;


inline arma::vec rndpp_rnormvec(int dim){
  Rcpp::RNGScope scope;
  return Rcpp::rnorm(dim, 0.0, 1.0);
}

inline arma::mat rndpp_mvnormal(int n, const arma::vec &mean, const arma::mat &sigma){
  int dimension = arma::size(mean)(0);
  arma::vec xtemp = arma::zeros(dimension);
  arma::mat outmat = arma::zeros(n, dimension);
  arma::mat cholsigma = arma::chol(sigma, "lower");
  //#pragma omp parallel for num_threads(NUM_THREADS)
  for(int i=0; i<n; i++){
    //for(int j=0; j<dimension; j++){
    //  xtemp(j) = rndpp_normal(0.0, 1.0, mt);
    //}
    xtemp = rndpp_rnormvec(dimension);
    //clog << arma::det(sigma) << endl;
    outmat.row(i) = (mean + cholsigma * xtemp).t();
  }
  return outmat;
}


inline arma::mat rndpp_mvnormal1(int n, const arma::vec &mean, const arma::mat &sigma){
  int dimension = arma::size(mean)(0);
  arma::vec xtemp = arma::zeros(dimension);
  arma::mat outmat = arma::zeros(n, dimension);
  arma::mat cholsigma = arma::chol(sigma, "lower");
  for(int i=0; i<n; i++){
    xtemp = rndpp_rnormvec(dimension);
    outmat.row(i) = (mean + cholsigma * xtemp).t();
  }
  return outmat;
}

inline arma::mat rndpp_mvnormal2(int n, const arma::vec& mu, const arma::mat& sigma) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::zeros(n, ncols);
  for(unsigned int j=0; j<ncols; j++){
    Y.col(j) = rndpp_rnormvec(n);
  }
  return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}

inline arma::mat rndpp_stdmvnormal(int n, int dimension){
  
  arma::vec xtemp = arma::zeros(dimension);
  arma::mat outmat = arma::zeros(dimension, n);
  for(int i=0; i<n; i++){
    outmat.col(i) = rndpp_rnormvec(dimension);
  }
  return outmat.t();
}

// sample 1 of 0:max-1 uniformly
inline int rndpp_unif_int(int max){
  //Rcpp::RNGScope scope;
  double uf = R::runif(0, max+1);
  return floor(uf);
}

inline arma::vec rndpp_runif(int n){
  Rcpp::RNGScope scope;
  return Rcpp::runif(n, 0, 1);
}

// same as above
inline int sample_one_int(const int &vsize){
  //arma::uvec sequence = arma::linspace<arma::uvec>(0, vsize-1, vsize);
  //int out = (Rcpp::RcppArmadillo::sample(sequence, 1, false))(0);
  return rndpp_unif_int(vsize);
}

// bool valued bernoulli(p)
inline bool boolbern(double p){
  Rcpp::RNGScope scope;
  double run = R::runif(0,1);
  return run < p;
}


// sample 1 from discrete on 0:length(probs)-1 with specified probs
inline int rndpp_discrete(const arma::vec& probs){
  Rcpp::RNGScope scope;
  Rcpp::IntegerVector xx = Rcpp::seq_len(probs.n_elem);
  return Rcpp::as<int>(Rcpp::sample(xx, 1, false, Rcpp::NumericVector(probs.begin(), probs.end()))) - 1;
}


inline arma::vec rndpp_sample(const arma::vec& x, int num){
  // no replacement
  arma::vec sol = arma::zeros(num);
  sol.fill(arma::datum::inf);
  arma::vec avails = x;
  for(int i=0; i<num; i++){
    arma::vec probs = arma::ones(avails.n_elem);
    sol(i) = avails(rndpp_discrete(probs));
    avails = bmdataman::bmms_setdiff(avails, sol);
  }
  return sol;
}

// sample n elements from 0:vsize-1 with no replacement
inline arma::uvec sample_index(const int& n, const int &vsize){
  arma::vec sequence = arma::regspace<arma::vec>(0, vsize-1);
  //arma::uvec out = Rcpp::RcppArmadillo::sample(sequence, n, false);
  //std::random_shuffle ( sequence.begin(), sequence.end() );
  //return sequence.subvec(0, n-1);
  arma::uvec result = arma::conv_to<arma::uvec>::from(rndpp_sample(sequence+1, n));
  return result-1;
}

// return a random element from those in fromvec (vector of integers)
inline int rndpp_sample1(const arma::vec& fromvec, const arma::vec& probs){
  return fromvec( rndpp_discrete(probs) );
}

inline int rndpp_sample1_comp_alt(const arma::vec& x, int p, int current_split, double decay=4.0){
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
    //probweights = arma::exp(arma::abs(avail - current_split) * log(1.0/decay));
    probweights = arma::exp(-decay*pow(avail - current_split, 2));
  }
  if(avail.n_elem > 0){
    int out = rndpp_sample1(avail, probweights); //Rcpp::RcppArmadillo::sample(avail, 1, true, probweights); 
    return out;//(0);
  } else {
    return -1;
  }
}

inline arma::vec pweight(const arma::vec& avail, int p, int current_split, int lev, int tot){
  double base = log(p+.0)/log(tot+.0);
  double spacing = pow((p+.0) / (tot * pow(base,lev+.0)), 2);
  arma::vec prob = arma::exp(-.5/spacing * pow(avail - current_split -.5, 2));
  return prob/arma::accu(prob);
}

inline int rndpp_sample1_comp(const arma::vec& x, int npossible, int current_split, int lev, double tot=4.0){
  /* vector x = current splits, npossible = how many in total
   * this returns 1 split out of the complement 
   * decay controls how far the proposed jump is going to be from the current
   * decay=1 corresponds to uniform prob on all availables
   * if current_split=-1 then pick uniformly
   */
  //double decay = 5.0;
  arma::vec all = arma::linspace(0, npossible-1, npossible);
  arma::vec avail = bmdataman::bmms_setdiff(all, x);
  //cout << avail << endl;
  arma::vec probweights;
  if(current_split == -1){
    probweights = arma::ones(arma::size(avail));
  } else {
    probweights = pweight(avail, npossible, current_split, lev+1, tot+1);
  }
  int out=-1;
  if(avail.n_elem > 0){
    out = rndpp_sample1(avail, probweights); 
    return out;//(0);
  } 
}

//inline arma::vec rndpp_shuffle(arma::vec x){
//  Rcpp::RNGScope scope;
//  return Rcpp::RcppArmadillo::sample(x, x.n_elem, false); 
//std::random_shuffle ( x.begin(), x.end() );
//return x;
//}

inline double rndpp_bern(double p){
  Rcpp::RNGScope scope;
  double run = R::runif(0,1);
  if(run < p){ 
    return 1.0;
  } else {
    return 0.0;
  }
}

inline double rndpp_gamma(const double& alpha, const double& beta) 
{
  Rcpp::RNGScope scope;
  return R::rgamma(alpha, beta);
}

inline double rndpp_normal(const double& mean, const double& sigma) 
{
  Rcpp::RNGScope scope;
  return R::rnorm(mean, sigma);
}


inline arma::mat rndpp_mvt(int n, const arma::vec &mu, const arma::mat &sigma, double df){
  Rcpp::RNGScope scope;
  double w=1.0;
  arma::mat Z = rndpp_stdmvnormal(n, mu.n_elem);
  arma::mat cholsigma = arma::chol(sigma, "lower");
  arma::mat AZ = Z;
  //#pragma omp parallel for num_threads(NUM_THREADS)
  for(int i=0; i<AZ.n_rows; i++){
    w = sqrt( df / R::rchisq(df) );
    AZ.row(i) = (mu + w * (cholsigma * Z.row(i).t())).t();
  }
  return AZ;
}

class PolyaGamma
{
  
public:
  
  // Default constructor.
  
  // Draw.
  double draw(int n, double z);
  double draw_like_devroye(double z);
  
  // Helper.
  double a(int n, double x);
  double pigauss(double x, double Z);
  double mass_texpon(double Z);
  double rtigauss(double Z);
  
};

inline double PolyaGamma::a(int n, double x)
{
  double K = (n + 0.5) * __PI;
  double y;
  if (x > __TRUNC) {
    y = K * exp( -0.5 * K*K * x );
  }
  else {
    double expnt = -1.5 * (log(0.5 * __PI)  + log(x)) + log(K) - 2.0 * (n+0.5)*(n+0.5) / x;
    y = exp(expnt);
  }
  return y;
}

inline double PolyaGamma::pigauss(double x, double Z)
{
  Rcpp::RNGScope scope;
  double b = sqrt(1.0 / x) * (x * Z - 1);
  double a = sqrt(1.0 / x) * (x * Z + 1) * -1.0;
  double y = R::pnorm(b, 0.0, 1.0, 1, 0) + exp(2 * Z) * R::pnorm(a, 0.0, 1.0, 1, 0);
  return y;
}

inline double PolyaGamma::mass_texpon(double Z)
{
  Rcpp::RNGScope scope;
  double t = __TRUNC;
  
  double fz = 0.125 * __PI*__PI + 0.5 * Z*Z;
  double b = sqrt(1.0 / t) * (t * Z - 1);
  double a = sqrt(1.0 / t) * (t * Z + 1) * -1.0;
  
  double x0 = log(fz) + fz * t;
  double xb = x0 - Z + R::pnorm(b, 0.0, 1.0, 1, 1);// RNG::p_norm(b, 1);
  double xa = x0 + Z + R::pnorm(b, 0.0, 1.0, 1, 1); //RNG::p_norm(a, 1);
  
  double qdivp = 4 / __PI * ( exp(xb) + exp(xa) );
  
  return 1.0 / (1.0 + qdivp);
}

inline double PolyaGamma::rtigauss(double Z)
{
  Rcpp::RNGScope scope;
  Z = fabs(Z);
  double t = __TRUNC;
  double X = t + 1.0;
  if (__TRUNC_RECIP > Z) { // mu > t
    double alpha = 0.0;
    while (R::runif(0,1) > alpha) {
      // X = t + 1.0;
      // while (X > t)
      // 	X = 1.0 / r.gamma_rate(0.5, 0.5);
      // Slightly faster to use truncated normal.
      double E1 = rndpp_gamma(1.0, 1.0); double E2 = rndpp_gamma(1.0, 1.0);
      while ( E1*E1 > 2 * E2 / t) {
        E1 = rndpp_gamma(1.0, 1.0); E2 = rndpp_gamma(1.0, 1.0);
      }
      X = 1 + E1 * t;
      X = t / (X * X);
      alpha = exp(-0.5 * Z*Z * X);
    }
  }
  else {
    double mu = 1.0 / Z;
    while (X > t) {
      double Y = R::rnorm(0,1); Y *= Y;
      double half_mu = 0.5 * mu;
      double mu_Y    = mu  * Y;
      X = mu + half_mu * mu_Y - half_mu * sqrt(4 * mu_Y + mu_Y * mu_Y);
      if (arma::randu() > mu / (mu + X))
        X = mu*mu / X;
    }
  }
  return X;
}

inline double PolyaGamma::draw(int n, double z)
{
  if (n < 1) throw std::invalid_argument("PolyaGamma::draw: n < 1.");
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    sum += draw_like_devroye(z);
  return sum;
}

inline double PolyaGamma::draw_like_devroye(double Z)
{
  // Change the parameter.
  Z = fabs(Z) * 0.5;
  
  // Now sample 0.25 * J^*(1, Z := Z/2).
  double fz = 0.125 * __PI*__PI + 0.5 * Z*Z;
  // ... Problems with large Z?  Try using q_over_p.
  // double p  = 0.5 * __PI * exp(-1.0 * fz * __TRUNC) / fz;
  // double q  = 2 * exp(-1.0 * Z) * pigauss(__TRUNC, Z);
  
  double X = 0.0;
  double S = 1.0;
  double Y = 0.0;
  
  while (true) {
    
    // if (r.unif() < p/(p+q))
    if ( R::runif(0, 1) < mass_texpon(Z) )
      X = __TRUNC + rndpp_gamma(1.0, 1.0) / fz;
    else
      X = rtigauss(Z);
    
    S = a(0, X);
    Y = R::runif(0, 1) * S;
    int n = 0;
    bool go = true;
    
    while (go) {
      ++n;
      if (n%2==1) {
        S = S - a(n, X);
        if ( Y<=S ) return 0.25 * X;
      }
      else {
        S = S + a(n, X);
        if ( Y>S ) go = false;
      }
    }
    
    // Need Y <= S in event that Y = S, e.g. when X = 0.
  }
}

inline arma::colvec rpg(const arma::colvec& shape, const arma::colvec& scale){
  PolyaGamma pg;
  int d = shape.n_elem;
  arma::colvec result(d);
  for(int i=0; i<d; i++) {
    result[i] = pg.draw(shape(i), scale(i));
  }
  return result;
}

}



#endif