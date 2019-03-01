#ifndef RCPP_bmmodels
#define RCPP_bmmodels

#include "bmrandom.h"
#include "bmdataman.h"
//#include "bmfuncs.h"

//using namespace bmdataman;
//using namespace bmrandom;
//using namespace bmfuncs;
using namespace std;

namespace bmmodels {

// log density of mvnormal mean 0
inline double m0mvnorm_dens(const arma::vec& x, const arma::mat& Si){
  int p = Si.n_cols;
  double normcore =  arma::conv_to<double>::from(x.t() * Si * x);
  double normconst = - p/2.0 * log(2*M_PI) + .5 * log(arma::det(Si));
  return normconst - 0.5 * (normcore);
}

// marglik of y ~ N(Xb, e In) with conjugate priors mean 0
// and gprior for b
inline double clm_marglik(const arma::vec& y, const arma::mat& Mi,
                          const arma::mat& Si, double muSimu, double a, double b){
  int p = Si.n_cols;
  int n = y.n_elem;
  double const1 = a * log(b) + lgamma(a + n/2.0) -  n/2.0 * log(2 * M_PI) - lgamma(a);
  double const2 = 0.5 * log(arma::det(Mi)) - 0.5 * log(arma::det(Si));
  
  double normcore = -(a+n/2.0) * log(b + 0.5 * arma::conv_to<double>::from(y.t() * y - muSimu));
  return const1 + const2 + normcore;
}

// log density of mvnormal mean 0 -- only useful in ratios with gpriors
inline double m0mvnorm_dens_gratio(double yy, double yPxy, double g, double p){
  return -0.5*p*log(g+1.0) + 0.5*g/(g+1.0) * yPxy - 0.5*yy;
}

// marglik of y ~ N(Xb, e In) with conjugate priors mean 0
//-- only useful in ratios with gpriors
// and gprior for b
inline double clm_marglik_gratio(double yy, double yPxy, double g, int n, double p, double a, double b){
  return -0.5*p*log(g+1.0) - (a+n/2.0)*log(b + 0.5*(yy - g/(g+1.0) * yPxy));
}

class BayesLM{
public:
  // data
  arma::vec y;
  arma::vec ycenter;
  arma::mat X;
  int n;
  int p;
  bool fix_sigma;
  
  // useful
  arma::mat XtX;
  arma::mat XtXi;
  double yty;
  
  // model: y = Xb + e
  // where e is Normal(0, sigmasq I_n)
  double icept;
  arma::vec b;
  double sigmasq;
  double lambda; // ridge
  
  // priors
  
  // inverse gamma for sigma
  double alpha;
  double beta;
  
  // mean and variance for Normal for b
  arma::vec m;
  arma::mat M;
  arma::mat Mi;
  double mtMim;
  
  arma::mat Ip;
  // posterior
  
  double alpha_n;
  double beta_n;
  arma::vec mu;
  arma::mat Sigma;
  double mutSimu;
  arma::mat Px;
  
  arma::mat inv_var_post;
  
  arma::vec reg_mean;
  arma::vec reg_mean_prior;
  
  void posterior();
  void beta_sample();
  void lambda_update(double);
  void chg_y(arma::vec&);
  void chg_data(arma::vec&, arma::mat&);
  
  BayesLM();
  BayesLM(const arma::vec&, const arma::mat&, bool);
  BayesLM(arma::vec, arma::mat, double);
  BayesLM(const arma::vec&, const arma::mat&, double, bool);
  BayesLM(const arma::vec&, const arma::mat&, double, bool, double);
  BayesLM(arma::vec, arma::mat, arma::mat);
};

class BayesLMg{
public:
  // data
  arma::vec y;
  arma::vec ycenter;
  arma::mat X;
  int n;
  int p;
  bool sampling_mcmc, fix_sigma;
  
  // useful
  arma::mat XtX;
  arma::mat Mi; // inverse of prior cov matrix. b ~ N(0, sigmasq * Mi^-1)
  double yty;
  arma::mat In;
  
  // model: y = Xb + e
  // where e is Normal(0, sigmasq I_n)
  double icept;
  arma::vec b;
  double sigmasq;
  double g; // ridge
  
  // priors
  // inverse gamma for sigma
  double alpha;
  double beta;
  
  // posterior
  
  double alpha_n;
  double beta_n;
  arma::vec mu;
  arma::mat Sigma;
  double mutSimu;
  
  arma::mat inv_var_post;
  
  arma::vec reg_mean;
  
  double yPxy;
  double marglik;
  double get_marglik(bool);
  
  void sample_sigmasq();
  void sample_beta();
  
  void change_X(const arma::mat&);
  
  BayesLMg();
  //BayesLMg(const arma::vec&, const arma::mat&, bool);
  //BayesLMg(arma::vec, arma::mat, double);
  BayesLMg(const arma::vec&, const arma::mat& , double, bool, bool);
  //BayesLMg(arma::vec, arma::mat, arma::mat);
};

class BayesSelect{
public:
  // data
  bool fix_sigma;
  int p, n;
  double icept, g, yty, yPxy, alpha, beta, marglik;
  arma::vec y, ycenter;
  
  void change_X(const arma::mat&);
  double get_marglik(bool);
  
  BayesSelect();
  //BayesLMg(const arma::vec&, const arma::mat&, bool);
  //BayesLMg(arma::vec, arma::mat, double);
  BayesSelect(const arma::vec&, const arma::mat&, double, bool);
  //BayesLMg(arma::vec, arma::mat, arma::mat);
};

class VarSelMCMC{
public:
  arma::vec y;
  arma::mat X;
  int n, p;
  
  arma::uvec gamma;
  arma::uvec gamma_proposal;
  BayesSelect model;
  BayesSelect model_proposal;
  BayesLMg sampled_model;
  
  //arma::vec gamma_start_prior; //prior prob of values to start from
  
  arma::vec sampling_order;
  
  int mcmc;
  
  void chain();
  
  arma::vec icept_stored;
  arma::mat gamma_stored;
  arma::mat beta_stored;
  arma::vec sigmasq_stored;
  
  VarSelMCMC(const arma::vec&, const arma::mat&, const arma::vec&, double, double, bool, int);
};

class ModularVS {
public:
  int K;
  int mcmc;
  
  //std::vector<VSModule> varsel_modules;
  
  arma::vec y;
  int n;
  arma::field<arma::mat> Xall;
  arma::vec resid;
  
  //ModularVS(const arma::vec&, const arma::field<arma::mat>&, int, double, arma::vec);
  ModularVS(const arma::vec&, const arma::field<arma::mat>&, 
            const arma::field<arma::vec>&,
            int, arma::vec, arma::vec, bool);
  
  arma::mat intercept;
  arma::field<arma::mat> beta_store;
  arma::field<arma::mat> gamma_store;
  arma::field<arma::vec> gamma_start;
};

inline BayesLM::BayesLM(){
  
}


inline BayesLM::BayesLM(const arma::vec& yy, const arma::mat& XX, bool fixs=false){
  fix_sigma = fixs;
  y = yy;
  X = XX;
  n = y.n_elem;
  p = X.n_cols;
  
  icept = arma::mean(y);
  ycenter = y - icept;
  
  lambda = 0.0;
  XtX = X.t() * X;
  
  yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
  
  m = arma::zeros(p);
  //M = n*XtXi;
  Mi = 1.0/log(1.0+n) * XtX + arma::eye(p,p) * lambda;
  mtMim = 0.0; //arma::conv_to<double>::from(m.t()*Mi*m);
  
  alpha = 0.0;
  beta = 0.0;
  
  inv_var_post = Mi + XtX;
  Sigma = arma::inv_sympd(inv_var_post); 
  mu = Sigma * (Mi*m + X.t()*ycenter);
  mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
  
  sigmasq = 1.0;
  Px = X * Sigma * X.t(); 
  b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  
  reg_mean = icept + X * b;
}

inline BayesLM::BayesLM(arma::vec yy, arma::mat XX, double lambda_in = 1){
  fix_sigma = false;
  y = yy;
  X = XX;
  n = y.n_elem;
  p = X.n_cols;
  //clog << arma::size(XX) << endl;
  lambda = lambda_in;
  
  icept = arma::mean(y);
  ycenter = y - icept;
  //clog << lambda << endl;
  
  XtX = X.t() * X;
  
  yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
  
  m = arma::zeros(p);
  //M = n*XtXi;
  Mi = 1.0/log(1.0+n) * XtX + arma::eye(p,p) * lambda;
  mtMim = 0.0; //arma::conv_to<double>::from(m.t()*Mi*m);
  
  alpha = 2.1; // parametrization: a = mean^2 / variance + 2
  beta = alpha-1;  //                  b = (mean^3 + mean*variance) / variance
  
  inv_var_post = Mi + XtX;
  Sigma = arma::inv_sympd(inv_var_post);
  //Px = X * Sigma * X.t(); 
  mu = Sigma * (Mi*m + X.t()*ycenter);
  mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
  
  alpha_n = alpha + n/2.0;
  beta_n = beta + 0.5*(mtMim - mutSimu + yty);
  
  sigmasq = 1.0/bmrandom::rndpp_gamma(alpha_n, 1.0/beta_n);
  b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  reg_mean = icept + X * b;
}


inline BayesLM::BayesLM(const arma::vec& yy, const arma::mat& XX, double lambda_in = 1, bool fixs=false){
  fix_sigma = fixs;
  y = yy;
  X = XX;
  n = y.n_elem;
  p = X.n_cols;
  //clog << arma::size(XX) << endl;
  lambda = lambda_in;
  XtX = X.t() * X;
  
  icept = arma::mean(y);
  ycenter = y - icept;
  
  Ip = arma::eye(p,p);
  m = arma::zeros(p);
  //M = n*XtXi;
  //clog << lambda << endl;
  Mi = 1.0/log(1.0+n) * XtX + Ip * lambda;
  mtMim = 0.0; //arma::conv_to<double>::from(m.t()*Mi*m);
  
  alpha = 2.1; // parametrization: a = mean^2 / variance + 2
  beta = alpha-1;  //                  b = (mean^3 + mean*variance) / variance
  
  inv_var_post = Mi + XtX;
  Sigma = arma::inv_sympd(inv_var_post);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  
  if(!fix_sigma) { 
    yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
    mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
    alpha_n = alpha + n/2.0;
    beta_n = beta + 0.5*(mtMim - mutSimu + yty);
    sigmasq = 1.0/bmrandom::rndpp_gamma(alpha_n, 1.0/beta_n);
  } else { 
    yty = 0.0;
    mutSimu = 0.0;
    alpha_n = 0.0;
    beta_n = 0.0;
    sigmasq = 1.0;
    Px = X * Sigma * X.t();
  }
  b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  reg_mean = icept + X * b;
}


inline BayesLM::BayesLM(const arma::vec& yy, const arma::mat& XX, 
                        double lambda_in = 1, bool fixs=false,
                        double sigmasqin=1.0){
  fix_sigma = fixs;
  y = yy;
  X = XX;
  n = y.n_elem;
  p = X.n_cols;
  //clog << arma::size(XX) << endl;
  lambda = lambda_in;
  XtX = X.t() * X;
  
  icept = arma::mean(y);
  ycenter = y - icept;
  
  Ip = arma::eye(p,p);
  m = arma::zeros(p);
  //M = n*XtXi;
  //clog << lambda << endl;
  Mi = 1.0/log(1.0+n) * XtX + Ip * lambda;
  mtMim = 0.0; //arma::conv_to<double>::from(m.t()*Mi*m);
  
  alpha = 2.1; // parametrization: a = mean^2 / variance + 2
  beta = alpha-1;  //                  b = (mean^3 + mean*variance) / variance
  
  inv_var_post = Mi + XtX;
  Sigma = arma::inv_sympd(inv_var_post);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  
  if(!fix_sigma) { 
    yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
    mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
    alpha_n = alpha + n/2.0;
    beta_n = beta + 0.5*(mtMim - mutSimu + yty);
    sigmasq = 1.0/bmrandom::rndpp_gamma(alpha_n, 1.0/beta_n);
  } else { 
    yty = 0.0;
    mutSimu = 0.0;
    alpha_n = 0.0;
    beta_n = 0.0;
    sigmasq = sigmasqin;
    Px = X * Sigma * X.t(); 
  }
  b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  reg_mean = icept + X * b;
}


inline BayesLM::BayesLM(arma::vec yy, arma::mat XX, arma::mat MM){
  // specifying prior for regression coefficient variance
  fix_sigma = false;
  y = yy;
  X = XX;
  n = y.n_elem;
  p = X.n_cols;
  
  lambda = 1.0;
  
  XtX = X.t() * X;
  
  icept = arma::mean(y);
  ycenter = y - icept;
  
  m = arma::zeros(p);
  M = MM;
  Mi = arma::inv_sympd(M);
  mtMim = arma::conv_to<double>::from(m.t()*Mi*m);
  
  alpha = 2.25; // parametrization: a = mean^2 / variance + 2
  beta = 0.625; //alpha-1;  //                  b = (mean^3 + mean*variance) / variance
  
  Sigma = arma::inv_sympd(Mi + XtX);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  
  if(!fix_sigma) { 
    yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
    mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
    alpha_n = alpha + n/2.0;
    beta_n = beta + 0.5*(mtMim - mutSimu + yty);
    sigmasq = 1.0/bmrandom::rndpp_gamma(alpha_n, 1.0/beta_n);
    b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
    //b = bmrandom::rndpp_mvt(1, mu, beta_n/alpha_n * Sigma/lambda, 2*alpha_n).t(); //(bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq/lambda)).row(0).t();
    
  } else { 
    yty = 0.0;
    mutSimu = 0.0;
    alpha_n = 0.0;
    beta_n = 0.0;
    sigmasq = 1.0;
    Px = X * Sigma * X.t(); 
    b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  }
  
  reg_mean = icept + X * b;
}

inline void BayesLM::lambda_update(double lambda_new){
  // specifying prior for regression coefficient variance
  
  lambda = lambda_new;
  //M = Ip * lambda_new;
  Mi = 1.0/log(1.0+n) * XtX + Ip * lambda_new;
  mtMim = arma::conv_to<double>::from(m.t()*Mi*m);
  
  //alpha = 2.25; // parametrization: a = mean^2 / variance + 2
  //beta = 0.625; //alpha-1;  //                  b = (mean^3 + mean*variance) / variance
  
  Sigma = arma::inv_sympd(Mi + XtX);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  
  if(!fix_sigma){
    mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
    
    alpha_n = alpha + n/2.0;
    beta_n = beta + 0.5*(mtMim - mutSimu + yty);
    
    sigmasq = 1.0/bmrandom::rndpp_gamma(alpha_n, 1.0/beta_n);
    b = bmrandom::rndpp_mvt(1, mu, beta_n/alpha_n * Sigma/lambda, 2*alpha_n).t();
  }
  else {
    Px = X * Sigma * X.t(); 
    b = (bmrandom::rndpp_mvnormal2(1, mu, Sigma*sigmasq/lambda)).row(0).t();
  }
  
  reg_mean = X * b;
}

inline void BayesLM::posterior(){
  sigmasq = 1.0/bmrandom::rndpp_gamma(alpha_n, 1.0/beta_n);
  b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq/lambda)).row(0).t();
}

inline void BayesLM::beta_sample(){
  b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq/lambda)).row(0).t();
  //b = bmrandom::rndpp_mvt(1, mu, beta_n/alpha_n * Sigma/lambda, 2*alpha_n).t(); //(bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq/lambda)).row(0).t();
}

inline void BayesLM::chg_y(arma::vec& yy){
  y = yy;
  icept = arma::mean(y);
  ycenter = y - icept;
  yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  
  if(!fix_sigma){
    mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
    beta_n = beta + 0.5*(mtMim - mutSimu + yty);
    
    sigmasq = 1.0/bmrandom::rndpp_gamma(alpha_n, 1.0/beta_n);
    b = bmrandom::rndpp_mvt(1, mu, beta_n/alpha_n * Sigma/lambda, 2*alpha_n).t(); 
  } else {
    b = (bmrandom::rndpp_mvnormal2(1, mu, Sigma*sigmasq/lambda)).row(0).t();
  }
  reg_mean = icept + X * b;
}

inline void BayesLM::chg_data(arma::vec& yy, arma::mat& XX){
  // specifying prior for regression coefficient variance
  y = yy;
  X = XX;
  n = y.n_elem;
  p = X.n_cols;
  
  XtX = X.t() * X;
  icept = arma::mean(y);
  ycenter = y - icept;
  
  Sigma = arma::inv_sympd(Mi + XtX);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  
  if(!fix_sigma){
    yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
    mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
    alpha_n = alpha + n/2.0;
    beta_n = beta + 0.5*(mtMim - mutSimu + yty);
    sigmasq = 1.0/bmrandom::rndpp_gamma(alpha_n, 1.0/beta_n);
    b = bmrandom::rndpp_mvt(1, mu, beta_n/alpha_n * Sigma/lambda, 2*alpha_n).t(); //(bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq/lambda)).row(0).t();
  } else {
    Px = X * Sigma * X.t(); 
    b = (bmrandom::rndpp_mvnormal2(1, mu, Sigma*sigmasq/lambda)).row(0).t();
  }
  reg_mean = icept + X * b;
}

inline BayesLMg::BayesLMg(){ 
  
}

inline BayesLMg::BayesLMg(const arma::vec& yy, const arma::mat& X, double gin, bool sampling=true, bool fixs = false){
  fix_sigma = fixs;
  sampling_mcmc = sampling;
  
  y = yy;
  n = y.n_elem;
  p = X.n_cols;
  
  icept = arma::mean(y);
  ycenter = y - icept;
  g = gin;
  
  yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
  
  if(sampling_mcmc){
    XtX = X.t() * X;
    
    In = arma::eye(n, n);
    Mi = 1.0/g * XtX + arma::eye(p,p)*1;
    
    inv_var_post = Mi + XtX;
    Sigma = arma::inv_sympd(inv_var_post);
    mu = Sigma * X.t() * ycenter;
    mutSimu = arma::conv_to<double>::from(mu.t()*inv_var_post*mu);
    if(fix_sigma){
      alpha = 0.0;
      beta = 0.0;
      sigmasq = 1.0;
      b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
      reg_mean = icept + X * b;
    } else {
      alpha = 2.1; // parametrization: a = mean^2 / variance + 2
      beta = alpha-1;  //                  b = (mean^3 + mean*variance) / variance
      alpha_n = alpha + n/2.0;
      beta_n = beta + 0.5*(-mutSimu + yty);
      
      sample_sigmasq();
      sample_beta();
      reg_mean = icept + X * b;
    }
  }
  //yPxy = arma::conv_to<double>::from(y.t() * X * arma::inv_sympd(XtX) * X.t() * y);
  yPxy = arma::conv_to<double>::from(ycenter.t() * bmdataman::hat(X) * ycenter);
  marglik = get_marglik(fixs);
};

inline void BayesLMg::change_X(const arma::mat& X){
  p = X.n_cols;
  
  if(sampling_mcmc){
    XtX = X.t() * X;
    Mi = 1.0/g * XtX + arma::eye(p,p)*1;
    inv_var_post = Mi + XtX;
    Sigma = arma::inv_sympd(inv_var_post);
    mu = Sigma * X.t() * ycenter;
    mutSimu = arma::conv_to<double>::from(mu.t()*inv_var_post*mu);
    if(fix_sigma){
      alpha = 0.0;
      beta = 0.0;
      sigmasq = 1.0;
      b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
      reg_mean = icept + X * b;
    } else {
      alpha = 2.1; // parametrization: a = mean^2 / variance + 2
      beta = alpha-1;  //                  b = (mean^3 + mean*variance) / variance
      alpha_n = alpha + n/2.0;
      beta_n = beta + 0.5*(-mutSimu + yty);
      
      sample_sigmasq();
      sample_beta();
      reg_mean = icept + X * b;
    }
  }
  //yPxy = arma::conv_to<double>::from(y.t() * X * arma::inv_sympd(XtX) * X.t() * y);
  yPxy = arma::conv_to<double>::from(ycenter.t() * bmdataman::hat(X) * ycenter);
  //clog << yPxy << endl;
  marglik = get_marglik(fix_sigma);
}

inline void BayesLMg::sample_beta(){
  b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
}

inline void BayesLMg::sample_sigmasq(){
  sigmasq = 1.0/bmrandom::rndpp_gamma(alpha_n, 1.0/beta_n);
}

inline double BayesLMg::get_marglik(bool fix_sigma=false){
  if(fix_sigma){
    return m0mvnorm_dens_gratio(yty, yPxy, g, p);
    //clog << marglik << endl;
  } else {
    return clm_marglik_gratio(yty, yPxy, g, n, p, alpha, beta);
    //clog << marglik << endl;
  }
}

inline BayesSelect::BayesSelect(){ 
  
}

inline double BayesSelect::get_marglik(bool fix_sigma=false){
  if(fix_sigma){
    return m0mvnorm_dens_gratio(yty, yPxy, g, p);
    //clog << marglik << endl;
  } else {
    return clm_marglik_gratio(yty, yPxy, g, n, p, alpha, beta);
    //clog << marglik << endl;
  }
}

inline BayesSelect::BayesSelect(const arma::vec& yy, const arma::mat& X, double gin, bool fixs = false){
  fix_sigma = fixs;
  
  alpha = 2.1;
  beta = 1.1;
  
  y = yy;
  n = y.n_elem;
  p = X.n_cols;
  
  icept = arma::mean(y);
  ycenter = y - icept;
  g = gin;
  
  yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
  
  //yPxy = arma::conv_to<double>::from(y.t() * X * arma::inv_sympd(XtX) * X.t() * y);
  yPxy = arma::conv_to<double>::from(ycenter.t() * bmdataman::hat(X) * ycenter);
  marglik = get_marglik(fix_sigma);
};

inline void BayesSelect::change_X(const arma::mat& X){
  p = X.n_cols;
  
  //yPxy = arma::conv_to<double>::from(y.t() * X * arma::inv_sympd(XtX) * X.t() * y);
  yPxy = arma::conv_to<double>::from(ycenter.t() * bmdataman::hat(X) * ycenter);
  //clog << yPxy << endl;
  marglik = get_marglik(fix_sigma);
}

inline VarSelMCMC::VarSelMCMC(const arma::vec& yy, const arma::mat& XX, const arma::vec& prior,
                       double gin=-1.0, double model_prior_par=1, bool fixsigma=false, int iter=1){
  //clog << "creating " << endl;
  y = yy;
  X = XX;
  mcmc = iter;
  
  p = X.n_cols;
  arma::vec p_indices = arma::linspace<arma::vec>(0, p-1, p);
  n = y.n_elem;
  
  icept_stored = arma::zeros(mcmc);
  beta_stored = arma::zeros(p, mcmc);
  gamma_stored = arma::zeros(p, mcmc);
  sigmasq_stored = arma::zeros(mcmc);
  
  gamma = arma::uvec(p);
  for(int j=0; j<p; j++){
    gamma(j) = 1*bmrandom::boolbern(prior(j));
  }
  
  arma::uvec gammaix = arma::find(gamma);
  //clog << "test  1" << endl;
  model = BayesSelect(y, X.cols(gammaix), gin, fixsigma);
  
  //clog << "test  2" << endl;
  for(int m=0; m<mcmc; m++){
    sampling_order = bmrandom::rndpp_shuffle(p_indices);
    for(int j=0; j<p; j++){
      int ix = sampling_order(j);
      gamma_proposal = gamma;
      gamma_proposal(ix) = 1-gamma(ix);
      arma::uvec gammaix_proposal = arma::find(gamma_proposal);
      //clog << "proposal gamma " << gamma_proposal << endl;
      model_proposal = model;
      model_proposal.change_X(X.cols(gammaix_proposal));
      //clog << "test  mcmc j " << j << endl;
      double accept_probability = exp(model_proposal.marglik - model.marglik) *
        exp(model_prior_par * (model.p - model_proposal.p));
      accept_probability = accept_probability > 1 ? 1.0 : accept_probability;
      
      int accepted = bmrandom::rndpp_bern(accept_probability);
      if(accepted == 1){
        //clog << "accepted." << endl;
        model = model_proposal;
        gamma = gamma_proposal;
        gammaix = gammaix_proposal;
      }
    }
    
    sampled_model = BayesLMg(y, X.cols(gammaix), gin, true, fixsigma);
    arma::vec beta_full = arma::zeros(p);
    beta_full.elem(gammaix) = sampled_model.b;
    
    icept_stored(m) = sampled_model.icept;
    beta_stored.col(m) = beta_full;
    gamma_stored.col(m) = arma::conv_to<arma::vec>::from(gamma);
    sigmasq_stored(m) = sampled_model.sigmasq;
  }
  //clog << selprob << endl;
}

inline ModularVS::ModularVS(const arma::vec& y_in, const arma::field<arma::mat>& Xall_in, 
                     const arma::field<arma::vec>& starting,
                     int mcmc_in,
                     arma::vec gg, 
                     arma::vec module_prior_par, bool binary=false){
  
  K = Xall_in.n_elem;
  //clog << K << endl;
  mcmc = mcmc_in;
  y = y_in;
  Xall = Xall_in;
  resid = y;
  
  n = y.n_elem;
  arma::vec z(n);
  arma::mat zsave(n, mcmc);
  
  z = (y-0.5)*2;
  
  arma::uvec yones = arma::find(y == 1);
  arma::uvec yzeros = arma::find(y == 0);
  
  // y=1 is truncated lower by 0. upper=inf
  // y=0 is truncated upper by 0, lower=-inf
  
  arma::vec trunc_lowerlim = arma::zeros(n);
  trunc_lowerlim.elem(yones).fill(0.0);
  trunc_lowerlim.elem(yzeros).fill(-arma::datum::inf);
  
  arma::vec trunc_upperlim = arma::zeros(n);
  trunc_upperlim.elem(yones).fill(arma::datum::inf);
  trunc_upperlim.elem(yzeros).fill(0.0);
  
  arma::mat In = arma::eye(n,n);
  
  intercept = arma::zeros(K, mcmc);
  beta_store = arma::field<arma::mat>(K);
  gamma_store = arma::field<arma::mat>(K);
  gamma_start = starting;
  
  for(int j=0; j<K; j++){
    gamma_start(j) = arma::zeros(Xall(j).n_cols);
    for(unsigned int h=0; h<Xall(j).n_cols; h++){
      gamma_start(j)(h) = bmrandom::rndpp_bern(0.1);
    }
    beta_store(j) = arma::zeros(Xall(j).n_cols, mcmc);
    gamma_store(j) = arma::zeros(Xall(j).n_cols, mcmc);
  }
  
  for(int m=0; m<mcmc; m++){
    //clog << "m: " << m << endl;
    if(binary){
      resid = z;
    } else {
      resid = y;
    }
    
    arma::vec xb_cumul = arma::zeros(n);
    for(int j=0; j<K; j++){
      //clog << " j: " << j << " " << arma::size(Xall(j)) << endl;
      //clog << "  module" << endl;
      double iboh = arma::mean(resid);
      
      //VSModule last_split_model = VSModule(y, X, gamma_start, MCMC, g, sprior, fixsigma, binary);
      //VarSelMCMC bvs_model(y, X, gamma_start, g, sprior, fixsigma, MCMC);
      
      //VSModule onemodule = VSModule(resid, Xall(j), gamma_start(j), 1, gg, ss(j), binary?true:false, false);
      
      VarSelMCMC onemodule(resid, Xall(j), gamma_start(j), gg(j), module_prior_par(j), binary?true:false, 1);
      
      //varsel_modules.push_back(onemodule);
      intercept(j, m) = onemodule.icept_stored(0);// onemodule.intercept;
      xb_cumul = xb_cumul + Xall(j) * onemodule.beta_stored.col(0) + onemodule.icept_stored(0);
      resid = (binary?z:y) - xb_cumul;
      //clog << "  beta store" << endl;
      beta_store(j).col(m) = onemodule.beta_stored.col(0);
      //clog << "  gamma store" << endl;
      gamma_store(j).col(m) = onemodule.gamma_stored.col(0);
      //clog << "  gamma start" << endl;
      gamma_start(j) = onemodule.gamma_stored.col(0);
      //clog << gamma_start(1) << endl;
    }
    
    if(binary){
      z = bmrandom::mvtruncnormal_eye1(xb_cumul, 
                             trunc_lowerlim, trunc_upperlim).col(0);
    }
    
    if(mcmc > 100){
      if(!(m % (mcmc / 10))){
        clog << m << " " << max(abs(z)) << endl;
      } 
    }
  }
}

// univariate model y ~ N(mu, sigmasq)
class NinvG_model{
public:
  // data
  arma::vec y; // univariate, n observations
  arma::vec x; // univariate, n observations
  arma::vec ystd;
  int n;
  
  // useful
  //double xssq;
  //double xssqi;
  double yssq;
  
  bool fix_sigma; 
  
  // model for each obs: y = theta + e
  // where e is Normal(0, sigmasq)
  double theta;
  double sigmasq;
  //double lambda; //ridge
  
  // priors
  
  // inverse gamma for sigma
  double alpha;
  double beta;
  
  // b ~ N(m, sigmasq * kappa)
  double m;
  double kappa;
  double msq_kappai;
  
  // posterior
  double alpha_n;
  double beta_n;
  double mu;
  double kappa_n;
  
  arma::vec iloglik;
  double loglik;
  
  void posterior();
  void ystdize();
  void sigmasq_sample();
  void theta_sample();
  void iloglik_calc();
  
  bool sigma_is_known;
  
  //void lambda_update(double);
  //void chg_y(arma::vec&);
  //void chg_data(arma::vec&, arma::vec&);
  
  // y
  NinvG_model(const arma::vec&, 
              const arma::vec&, 
              double, double, double);//, arma::vec);
  
  // y alpha, beta, kappa
  //NinvG_model(arma::vec, arma::vec, double, double, double); 
};

inline double uninormal_loglik(double x, double m, double ssq){
  return -.5 * log(2*M_PI*ssq) - .5/ssq * pow(x-m, 2);
}

inline double dinvgamma(double x, double a, double b){
  return a * log(b) + (-1-a)* log(x) -b/x - log(tgamma(a));
}

inline NinvG_model::NinvG_model(const arma::vec& yy, 
                         const arma::vec& xx,
                         double alpha_in=2.0001,
                         double beta_in=0.10001,
                         double sigmasqfixed=-1.0){
  y = yy;
  //x = xx;
  n = y.n_elem;
  
  if(sigmasqfixed==-1){
    sigma_is_known = false;
    sigmasq = 1.0;
  } else {
    sigma_is_known = true;
    sigmasq = sigmasqfixed;
  }
  
  
  iloglik = arma::zeros(n); // observation loglikelihood at sampled values theta and sigmasq
  //xssq = arma::conv_to<double>::from(x.t() * x);
  //xssqi = 1.0/xssq;
  
  yssq = arma::conv_to<double>::from(y.t()*y);
  
  m = 0.0;//arma::mean(y);
  kappa = n;//1.0;
  msq_kappai = 0.0; 
  
  // meanv = beta / (alpha-1)
  // variance = beta^2 / ((alpha-1)^2 * (alpha-2))
  alpha =  alpha_in; // parametrization: alpha = meanv^2 / variance + 2
  beta = beta_in; //alpha-1;  //       beta = (meanv^3 + meanv*variance) / variance
  
  kappa_n = 1.0/( 1.0/kappa + n);
  mu = kappa_n * (m / kappa + arma::accu(y));
  
  //sigmasq = 1.0/rndpp_gamma(alpha_n, 1.0/beta_n);
  //theta = mu + pow(sigmasq * kappa_n, 0.5) * arma::randn();
  posterior();
  ystdize();
}

inline void NinvG_model::iloglik_calc(){
  //clog << alpha << " " << beta << endl;
  for(unsigned int i=0; i<n; i++){
    iloglik(i) = uninormal_loglik(y(i), theta, sigmasq);
    //iloglik(i) = unistudent_marglik_reparam(y(i), kappa, alpha, beta);
  }
  loglik = arma::accu(iloglik) + 
    uninormal_loglik(theta, m, sigmasq * kappa);
  if(!sigma_is_known){
    loglik += dinvgamma(sigmasq, alpha, beta);
  } 
}

inline void NinvG_model::sigmasq_sample(){
  alpha_n = alpha + n/2.0;
  //beta_n = beta + 0.5*((pow(m, 2.0) - pow(mu, 2.0))/kappa + yssq - n*pow(mu, 2.0));
  beta_n = beta + 0.5*(yssq - pow(mu,2)/kappa_n);
  sigmasq = 1.0/bmrandom::rndpp_gamma(alpha_n, 1.0/beta_n);
  //iloglik_calc();
}

inline void NinvG_model::theta_sample(){
  theta = mu + pow(sigmasq * kappa_n, 0.5) * arma::randn();
  iloglik_calc();
}

inline void NinvG_model::posterior(){
  if(!sigma_is_known){
    sigmasq_sample(); 
  }
  theta_sample();
}

inline void NinvG_model::ystdize(){
  if(sigma_is_known){
    ystd = (y-theta);
  } else {
    ystd = (y-theta)/pow(sigmasq,.5);
  }
}


// "histogram regression with fixed bins" ?
class BayesHistReg{
public:
  arma::mat J_now;
  int pj;
  arma::vec y;
  arma::vec ystd;
  arma::vec x;
  
  arma::field<arma::vec> ygrouped;
  arma::field<arma::vec> ystd_group;
  arma::field<arma::vec> xgrouped;
  
  bool fixed_sigma;
  //std::vector<NinvG_model> nig_pieces;
  
  arma::vec theta;
  arma::vec sigmasq;
  bool fix_sigma;
  
  arma::vec mu;
  double alpha, beta;
  arma::vec kappa_n;
  arma::vec alpha_n;
  arma::vec beta_n;
  arma::vec sigmasq_postmean;
  
  arma::vec loglik_bycomp;
  double loglik;
  
  arma::vec resid;
  bool sigma_is_known;
  
  // y, x, Lj
  BayesHistReg();
  BayesHistReg(const arma::vec&, 
                  const arma::vec&,
                  const arma::mat&,
                  double, double, double);
};

inline BayesHistReg::BayesHistReg(const arma::vec& yy, 
                                 const arma::vec& xx,
                                 const arma::mat& Ljin, 
                                 double alpha_in=2.0001,
                                 double beta_in=0.10001,
                                 double sigmasqfixed=-1.0){
  J_now = Ljin;
  pj = Ljin.n_cols;
  y = yy;
  x = xx;
  ystd = arma::zeros(y.n_elem);
  
  if(sigmasqfixed==-1){
    sigma_is_known = false;
    sigmasq = arma::zeros(pj);
  } else {
    sigma_is_known = true;
    sigmasq = arma::zeros(pj) + sigmasqfixed;
  }
  
  // sampled
  theta = arma::zeros(pj);
  
  // posterior params
  mu = arma::zeros(pj);
  alpha = alpha_in;
  beta = beta_in;
  kappa_n = arma::zeros(pj);
  alpha_n = arma::zeros(pj);
  beta_n = arma::zeros(pj);
  sigmasq_postmean = arma::zeros(pj);
  loglik_bycomp = arma::zeros(pj);
  
  ygrouped = arma::field<arma::vec>(pj);
  xgrouped = arma::field<arma::vec>(pj);
  
  //ystd_group = arma::field<arma::vec>(pj);
  arma::uvec yselect;
  for(unsigned int i=0; i<pj; i++){
    //yselect = arma::find(J_now.col(i));
    //ygrouped(i) = y.elem(yselect);
    ygrouped(i) = y.subvec(J_now(0,i), J_now(1,i));
    xgrouped(i) = x.subvec(J_now(0,i), J_now(1,i));
    
    //clog << "piece " << endl;
    NinvG_model adding_piece(ygrouped(i), xgrouped(i), alpha, beta, sigmasqfixed);
    //ystd.elem(yselect) = adding_piece.ystd;
    ystd.subvec(J_now(0,i), J_now(1,i)) = adding_piece.ystd;
    //nig_pieces.push_back(adding_piece);
    
    sigmasq(i) = adding_piece.sigmasq;
    theta.subvec(i, i) = adding_piece.theta;
    
    //
    //mu(i) = adding_piece.mu;
    //kappa_n(i) = adding_piece.kappa_n;
    //alpha_n(i) = adding_piece.alpha_n;
    //beta_n(i) = adding_piece.beta_n;
    //sigmasq_postmean(i) = adding_piece.beta_n / (adding_piece.alpha_n - 1);
    loglik_bycomp(i) = adding_piece.loglik;
  }
  loglik = arma::accu(loglik_bycomp);
  resid = ystd; //y - J_now * theta;
  //clog << "NInvGMix module inited." << endl;
}


}

#endif