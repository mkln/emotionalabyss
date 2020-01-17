#ifndef RCPP_bmmodels
#define RCPP_bmmodels

#include "bmrandom.h"
#include "bmfuncs.h"

using namespace std;

namespace bmmodels {

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

inline double logdet(const arma::mat& X){
  double val, sign;
  arma::log_det(val, sign, X);
  return val;
}

// log density of mvnormal mean 0
inline double m0mvnorm_dens(const arma::vec& x, const arma::mat& Si){
  int p = Si.n_cols;
  double normcore =  arma::conv_to<double>::from(x.t() * Si * x);
  double normconst = - p/2.0 * log(2*M_PI) + .5 * logdet(Si);
  return normconst - 0.5 * (normcore);
}

// marglik of y ~ N(Xb, e In) with conjugate priors mean 0
// and gprior for b
inline double clm_marglik(const arma::vec& y, const arma::mat& Mi,
                          const arma::mat& Si, double muSimu, double a, double b){
  int p = Si.n_cols;
  int n = y.n_elem;
  double const1 = a * log(b) + lgamma(a + n/2.0) -  n/2.0 * log(2 * M_PI) - lgamma(a);
  double const2 = 0.5 * logdet(Mi) - 0.5 * logdet(Si);
  
  double normcore = -(a+n/2.0) * log(b + 0.5 * arma::conv_to<double>::from(y.t() * y - muSimu));
  return const1 + const2 + normcore;
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
  
  arma::vec reg_mean;
  
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
  double g; // gprior
  
  // priors
  
  // inverse gamma for sigma
  double alpha;
  double beta;
  
  // mean and variance for Normal for b
  arma::vec m;
  arma::mat M;
  arma::mat Mi;
  double mtMim;
  
  arma::sp_mat Ip;
  arma::sp_mat In;
  // posterior
  
  double alpha_n;
  double beta_n;
  arma::vec mu;
  arma::mat Sigma;
  double mutSimu;
  arma::mat Px;
  
  arma::mat inv_var_post;
  
  arma::vec linear_predictor;
  double logpost;
  
  void posterior();
  void beta_sample();
  void sigmasq_sample(bool);
  void lambda_update(double);
  void get_invgamma_post();
  void chg_y(const arma::vec&, bool);
  void chg_X(const arma::mat&, bool, double, bool);
  
  double calc_logpost();
  
  BayesLM();
  BayesLM(const arma::vec&, const arma::mat&, bool);
  BayesLM(arma::vec, arma::mat, double);
  BayesLM(const arma::vec&, const arma::mat&, double, bool);
  BayesLM(const arma::vec&, const arma::mat&, double, bool, double, double);
  BayesLM(arma::vec, arma::mat, arma::mat);
  BayesLM(const arma::vec&, const arma::mat&, 
               const arma::vec&, double, double, 
               double, bool, double, double);
  
};

inline BayesLM::BayesLM(){
  
}

inline void BayesLM::get_invgamma_post(){
  yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
  mutSimu = arma::conv_to<double>::from(mu.t()*(Mi + XtX)*mu);
  alpha_n = alpha + n/2.0;
  beta_n = beta + 0.5*(mtMim - mutSimu + yty);
}

inline void BayesLM::sigmasq_sample(bool calc_param=true){
  if(calc_param){
    get_invgamma_post();
  }
  sigmasq = 1.0/bmrandom::rndpp_gamma(alpha_n, 1.0/beta_n);
}

inline BayesLM::BayesLM(const arma::vec& yy, const arma::mat& XX, 
                                  double lambda_in = 1, bool fixs=false,
                                  double sigmasqin=1.0, double gin=-1){
  fix_sigma = fixs;
  y = yy;
  X = XX;
  n = y.n_elem;
  p = X.n_cols;
  //clog << arma::size(XX) << endl;
  lambda = lambda_in;
  XtX = X.t() * X;
  
  if(gin==-1){
    g = 1.0/log(1.0+n);
  } else {
    g = gin;
  }
  
  icept = arma::mean(y);
  ycenter = y - icept;
  
  In = arma::speye(n,n);
  Ip = arma::speye(p,p);
  m = arma::zeros(p);
  //M = n*XtXi;
  //clog << lambda << endl;
  Mi = 1.0/g * XtX + Ip * lambda;
  mtMim = 0.0; //arma::conv_to<double>::from(m.t()*Mi*m);
  
  alpha = 2.1; // parametrization: a = mean^2 / variance + 2
  beta = alpha-1;  //                  b = (mean^3 + mean*variance) / variance
  
  inv_var_post = Mi + XtX;
  Sigma = arma::inv_sympd(inv_var_post);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  
  if(!fix_sigma) { 
    sigmasq_sample(true);
  } else { 
    yty = 0.0;
    mutSimu = 0.0;
    alpha_n = alpha + n/2.0;
    beta_n = 0.0;
    sigmasq = sigmasqin;
    Px = X * Sigma * X.t(); 
  }
  b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  linear_predictor = icept + X * b;
}

inline BayesLM::BayesLM(const arma::vec& yy, const arma::mat& XX, 
                                  const arma::vec& mm, //prior mean
                                  double ain, double bin,
                                  double lambda_in = 1, bool fixs=false,
                                  double sigmasqin=1.0, double gin=-1){
  fix_sigma = fixs;
  y = yy;
  X = XX;
  n = y.n_elem;
  p = X.n_cols;
  
  lambda = lambda_in;
  XtX = X.t() * X;
  
  if(gin==-1){
    g = 1.0/log(1.0+n);
  } else {
    g = gin;
  }
  
  icept = arma::mean(y);
  ycenter = y - icept;
  
  Ip = arma::speye(p,p);
  In = arma::speye(n,n);
  m = mm;
  //M = n*XtXi;
  //clog << lambda << endl;
  Mi = 1.0/g * XtX + Ip * lambda;
  mtMim = arma::conv_to<double>::from(m.t()*Mi*m);
  
  alpha = ain; // parametrization: a = mean^2 / variance + 2
  beta = bin;  //                  b = (mean^3 + mean*variance) / variance
  alpha_n = alpha + n/2.0;
  
  inv_var_post = Mi + XtX;
  Sigma = arma::inv_sympd(inv_var_post);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  
  Px = X * Sigma * X.t(); 
  
  if(!fix_sigma) { 
    sigmasq_sample(true);
  } else { 
    yty = 0.0;
    mutSimu = 0.0;
    alpha_n = alpha + n/2.0;
    beta_n = 0.0;
    sigmasq = sigmasqin;
    
  }
  b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  
  linear_predictor = icept + X * b;
  logpost = calc_logpost();
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
  g = log(1.0+n);
  Mi = 1.0/g * XtX + arma::speye(p,p) * lambda;
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
  
  linear_predictor = icept + X * b;
  logpost = calc_logpost();
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
  g = log(1.0+n); 
  Mi = 1.0/g * XtX + arma::speye(p,p) * lambda;
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
  linear_predictor = icept + X * b;
  logpost = calc_logpost();
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
  
  Ip = arma::speye(p,p);
  m = arma::zeros(p);
  //M = n*XtXi;
  //clog << lambda << endl;
  g = log(1.0+n);
  Mi = 1.0/g * XtX + Ip * lambda;
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
    alpha_n = alpha + n/2.0;
    beta_n = 0.0;
    sigmasq = 1.0;
    Px = X * Sigma * X.t();
  }
  b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  linear_predictor = icept + X * b;
  logpost = calc_logpost();
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
  
  g = 0;
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
    alpha_n = alpha + n/2.0;
    beta_n = 0.0;
    sigmasq = 1.0;
    Px = X * Sigma * X.t(); 
    b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  }
  
  linear_predictor = icept + X * b;
  logpost = calc_logpost();
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
    b = (bmrandom::rndpp_mvnormal2(1, mu, Sigma*sigmasq)).row(0).t();
  }
  
  linear_predictor = X * b;
  logpost = calc_logpost();
}

inline void BayesLM::beta_sample(){
  b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  linear_predictor = icept + X * b;
  //b = bmrandom::rndpp_mvt(1, mu, beta_n/alpha_n * Sigma/lambda, 2*alpha_n).t(); //(bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq/lambda)).row(0).t();
}


inline void BayesLM::chg_y(arma::vec& yy, bool fix_sigma){
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

inline void BayesLM::chg_X(const arma::mat& XX, bool fixsigma, double sigmasqin=-1.0, bool sample_beta = false){
  X = XX;
  if(p != X.n_cols){
    clog << "Proposed a change in n. columns of X. Not implemented yet." << endl;
    throw 1;
  }
  if((fixsigma == true) & (sigmasqin == -1.0)){
    clog << "Input sigmasq=-1.0 (the default) but fixing sigma. " << endl;
  }
  
  XtX = X.t() * X;
  Mi = 1.0/g * XtX + Ip * lambda;
  mtMim = arma::conv_to<double>::from(m.t()*Mi*m);
  inv_var_post = Mi + XtX;
  Sigma = arma::inv_sympd(inv_var_post);
  mu = Sigma * (Mi*m + X.t()*ycenter);
  Px = X * Sigma * X.t(); 
  
  if(!fix_sigma) { 
    sigmasq_sample(true);
  } else { 
    yty = 0.0;
    mutSimu = 0.0;
    alpha_n = alpha + n/2.0;
    beta_n = 0.0;
    sigmasq = sigmasqin;
    
  }
  if(sample_beta){
    beta_sample();
  }
  
  logpost = calc_logpost();
}

inline double BayesLM::calc_logpost(){
  if(false){
    //if(fix_sigma){
    //clog << "calc. " << arma::size(X) << " " << arma::size(m) << "  " << arma::size(y) << endl;
    //clog << arma::size(y - X*m) << " " << logdet(Mi) << " " << logdet(inv_var_post) << " " << mutSimu << endl;
    logpost = m0mvnorm_dens(ycenter - X*m, 1.0/sigmasq * (In - Px)); // invgamma density cancels when taking ratios
    return logpost;
  } else {
    logpost = clm_marglik(ycenter - X*m, Mi, inv_var_post, mutSimu, alpha, beta);
    return logpost;
  }
}


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
  
  arma::vec xb;
  arma::vec linear_predictor;
  
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

inline BayesLMg::BayesLMg(){ 
  
}

inline BayesLMg::BayesLMg(const arma::vec& yy, const arma::mat& Xin, double gin, bool sampling=true, bool fixs = false){
  fix_sigma = fixs;
  sampling_mcmc = sampling;
  
  X = Xin;
  y = yy;
  n = y.n_elem;
  p = X.n_cols;
  
  
  ycenter = y - arma::mean(y);
  g = gin;
  
  yty = arma::conv_to<double>::from(ycenter.t()*ycenter);
  
  if(sampling_mcmc){
    XtX = X.t() * X;
    
    In = arma::speye(n, n);
    Mi = 1.0/g * XtX + arma::speye(p,p)*.01;
    
    inv_var_post = Mi + XtX;
    Sigma = arma::inv_sympd(inv_var_post);
    mu = Sigma * X.t() * ycenter;
    mutSimu = arma::conv_to<double>::from(mu.t()*inv_var_post*mu);
    if(fix_sigma){
      alpha = 0.0;
      beta = 0.0;
      sigmasq = 1.0;
      sample_beta();
      
    } else {
      alpha = 2.1; // parametrization: a = mean^2 / variance + 2
      beta = alpha-1;  //                  b = (mean^3 + mean*variance) / variance
      alpha_n = alpha + n/2.0;
      beta_n = beta + 0.5*(-mutSimu + yty);
      
      sample_sigmasq();
      sample_beta();
      
    }
  }
  //yPxy = arma::conv_to<double>::from(y.t() * X * arma::inv_sympd(XtX) * X.t() * y);
  yPxy = arma::conv_to<double>::from(ycenter.t() * bmfuncs::hat(X) * ycenter);
  marglik = get_marglik(fixs);
};

inline void BayesLMg::change_X(const arma::mat& Xin){
  X = Xin;
  p = X.n_cols;
  
  if(sampling_mcmc){
    XtX = X.t() * X;
    Mi = 1.0/g * XtX + arma::speye(p,p)*.01;
    inv_var_post = Mi + XtX;
    Sigma = arma::inv_sympd(inv_var_post);
    mu = Sigma * X.t() * ycenter;
    mutSimu = arma::conv_to<double>::from(mu.t()*inv_var_post*mu);
    if(fix_sigma){
      alpha = 0.0;
      beta = 0.0;
      sigmasq = 1.0;
      sample_beta();
    } else {
      alpha = 2.1; // parametrization: a = mean^2 / variance + 2
      beta = alpha-1;  //                  b = (mean^3 + mean*variance) / variance
      alpha_n = alpha + n/2.0;
      beta_n = beta + 0.5*(-mutSimu + yty);
      
      sample_sigmasq();
      sample_beta();
    }
  }
  //yPxy = arma::conv_to<double>::from(y.t() * X * arma::inv_sympd(XtX) * X.t() * y);
  yPxy = arma::conv_to<double>::from(ycenter.t() * bmfuncs::hat(X) * ycenter);
  //clog << yPxy << endl;
  marglik = get_marglik(fix_sigma);
}

inline void BayesLMg::sample_beta(){
  b = (bmrandom::rndpp_mvnormal(1, mu, Sigma*sigmasq)).row(0).t();
  xb = X*b;
  icept = arma::mean(y-xb);
  linear_predictor = icept + xb;
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
  yPxy = arma::conv_to<double>::from(ycenter.t() * bmfuncs::hat(X) * ycenter);
  marglik = get_marglik(fix_sigma);
};

inline void BayesSelect::change_X(const arma::mat& X){
  p = X.n_cols;
  
  //yPxy = arma::conv_to<double>::from(y.t() * X * arma::inv_sympd(XtX) * X.t() * y);
  yPxy = arma::conv_to<double>::from(ycenter.t() * bmfuncs::hat(X) * ycenter);
  //clog << yPxy << endl;
  marglik = get_marglik(fix_sigma);
}


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
  double marglik;
  
  arma::vec icept_stored;
  arma::mat gamma_stored;
  arma::mat beta_stored;
  arma::vec sigmasq_stored;
  
  VarSelMCMC(const arma::vec&, const arma::mat&, const arma::vec&, double, double, bool, int);
};


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
  marglik = model.marglik;
  
  //clog << "test  2" << endl;
  for(int m=0; m<mcmc; m++){
    sampling_order = arma::regspace(0, p-1); // bmrandom::rndpp_shuffle(p_indices);
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

}

#endif