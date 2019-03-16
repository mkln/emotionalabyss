#ifndef RCPP_bmdataman
#define RCPP_bmdataman

#include <RcppArmadillo.h>
#include <random>
#include <ctime>
#include <math.h>
#include <cstdlib>
#include <omp.h>

#define NUM_THREADS 4

using namespace std;

// #pragma omp parallel for num_threads(NUM_THREADS)
// functions not specific to modular models

namespace bmdataman {

inline arma::vec bmms_setdiff(const arma::vec& x, const arma::vec& y) {
  std::vector<int> a = arma::conv_to< std::vector<int> >::from(arma::sort(x));
  std::vector<int> b = arma::conv_to< std::vector<int> >::from(arma::sort(y));
  std::vector<int> out;
  
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::inserter(out, out.end()));
  
  return arma::conv_to<arma::vec>::from(out);
}

inline arma::uvec bmms_usetdiff(const arma::uvec& x, const arma::uvec& y) {
  std::vector<int> a = arma::conv_to< std::vector<int> >::from(arma::sort(x));
  std::vector<int> b = arma::conv_to< std::vector<int> >::from(arma::sort(y));
  std::vector<int> out;
  
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::inserter(out, out.end()));
  
  return arma::conv_to<arma::uvec>::from(out);
}



inline arma::mat X2Dgrid_alt(const arma::vec& x1, const arma::vec& x2){
  int n = x1.n_elem*x2.n_elem, 
    i=0, j=0;
  arma::uvec ixes = arma::regspace<arma::uvec>(0, n-1);
  arma::mat rr = arma::zeros(n, 2);
  arma::umat subscripts = arma::ind2sub(arma::size(x1.n_elem, x2.n_elem), ixes);
  for(unsigned int r=0; r<subscripts.n_cols; r++){
    i = subscripts(0, r);
    j = subscripts(1, r);
    rr(i*x2.n_elem+j, 0) = x1(i);
    rr(i*x2.n_elem+j, 1) = x2(j);
  }
  return(rr);
}

inline arma::mat X2Dgrid(const arma::vec& x1, const arma::vec& x2){
  arma::mat rr = arma::zeros(x1.n_elem*x2.n_elem, 2);
  for(unsigned int i=0; i<x1.n_elem; i++){
    for(unsigned int j=0; j<x2.n_elem; j++){
      rr(i*x2.n_elem+j, 0) = x1(i);
      rr(i*x2.n_elem+j, 1) = x2(j);
    }
  }
  return(rr);
}

inline arma::mat exclude(const arma::mat& test, const arma::vec& excl){
  arma::vec keepers = arma::ones(test.n_cols);
  for(unsigned int e=0; e<excl.n_elem; e++){
    keepers(excl(e)) = 0;
  }
  //cout << "exclude cols " << endl;
  return test.cols(arma::find(keepers));
}


inline arma::vec nonzeromean(const arma::mat& mat_mcmc){
  arma::vec result = arma::zeros(mat_mcmc.n_rows);
  for(unsigned int j=0; j<mat_mcmc.n_rows; j++){
    arma::vec thisrow = mat_mcmc.row(j).t();
    arma::vec nnzero = thisrow(arma::find(thisrow));
    result(j) = nnzero.n_elem > 0 ? arma::mean(nnzero) : 0.0;
  }
  return result;
}

inline arma::vec col_eq_check(const arma::mat& A){
  arma::vec is_same_as = arma::ones(A.n_cols) * -1;
  
  for(unsigned int i1=0; (i1<A.n_cols); i1++){
    for(unsigned int i2=A.n_cols-1; i2 > i1; i2--){
      if(approx_equal(A.col(i1), A.col(i2), "absdiff", 1e-10)){
        if(is_same_as(i2) == -1) { 
          is_same_as(i2) = i1;
        }
      } 
    }
  }
  return is_same_as;
}

inline arma::vec col_sums(const arma::mat& matty){
  return arma::sum(matty, 0).t();
}

inline arma::mat hat_alt(const arma::mat& X){
  arma::mat U;
  arma::vec s;
  arma::mat V;
  int c = X.n_cols > X.n_rows ? X.n_rows : X.n_cols;
  arma::svd(U,s,V,X);
  return U.cols(0,c-1) * U.cols(0,c-1).t();
}

inline arma::mat hat(const arma::mat& X){
  if(X.n_cols > X.n_rows){
    return hat_alt(X);
  } else {
    arma::mat iv;
    bool gotit = arma::inv_sympd(iv, X.t() * X);
    if(gotit){
      return X * iv * X.t();
    } else { 
      return hat_alt(X);
    }
  }
}

inline arma::mat cube_mean(const arma::cube& X, int dim){
  return arma::mean(X, dim);
}

inline arma::mat cube_sum(const arma::cube& X, int dim){
  return arma::sum(X, dim);
}

inline arma::mat cube_prod(const arma::cube& x, int dim){
  int p1 = x.n_rows;
  int p2 = x.n_cols;
  int p3 = x.n_slices;
  if(dim == 0){
    // return p2xp3 matrix
    arma::mat result(p2, p3);
    arma::vec vecpro(p1);
    for(unsigned int j=0; j<p2; j++){
      for(unsigned int z=0; z<p3; z++){
        vecpro = x.subcube( 0, j, z, 
                            p1-1, j, z );
        result(j, z) = arma::prod(vecpro);
      }
    }
    return result;
  }
  if(dim == 1){
    // return p1xp3 matrix
    arma::mat result(p1, p3);
    arma::rowvec vecpro(p2);
    for(unsigned int i=0; i<p1; i++){
      for(unsigned int z=0; z<p3; z++){
        vecpro = x.subcube( i, 0, z, 
                            i, p2-1, z);
        result(i, z) = arma::prod(vecpro); //
      }
    }
    return result;
  }
  if(dim == 2){
    // return p1xp2 matrix
    arma::mat result(p1, p2);
    arma::vec vecpro(p3);
    for(unsigned int i=0; i<p1; i++){
      for(unsigned int j=0; j<p2; j++){
        vecpro = x.subcube( i, j, 0, 
                            i, j, p3-1 );
        result(i,j) = arma::prod(vecpro);
      }
    }
    return result;
  }
  return arma::zeros(0,0);
}

// Vector index to matrix subscripts
// 
// Get matrix subscripts from corresponding vector indices (both start from 0).
// This is a utility function using Armadillo's ind2sub function.
// @param index a vector of indices
// @param m a matrix (only its size is important)
inline arma::mat index_to_subscript(const arma::uvec& index, const arma::mat& m){
  return arma::conv_to<arma::mat>::from(arma::ind2sub(arma::size(m), index));
}

// gini coefficient
inline double gini(arma::vec& x, int p){
  arma::vec fixedx = arma::zeros(2 + x.n_elem);
  fixedx(0) = -1;
  //clog << x.t() << endl;
  fixedx.subvec(1, x.n_elem) = x;
  //clog << "see" << endl;
  fixedx(fixedx.n_elem-1) = p-1;
  
  arma::vec xg = arma::sort(arma::diff(fixedx));
  int n = xg.n_elem;
  double g1top = 0;
  for(int i=0; i<xg.n_elem; i++){
    g1top += (n - i) * xg(i);
  }
  //clog << n << " " << g1top << " " << arma::accu(xg) << endl;
  double gix = 1.0/n * (n + 1 - 2* g1top/(arma::accu(xg)));
  return gix;
}

inline arma::vec find_avail(const arma::vec& tied, int n){
  arma::vec ls = arma::linspace(0, n-1, n);
  return bmdataman::bmms_setdiff(ls, tied);
}

inline arma::uvec find_first_unique(const arma::vec& x){
  arma::uvec ixes(x.n_elem);
  ixes.fill(0);
  int uqsize = 0;
  arma::vec sortx = arma::sort(x);
  ixes(0) = 0;
  double prev=sortx(0);
  for(unsigned int i=1; i<x.n_elem; i++){
    if(sortx(i) > prev){ // changed value
      uqsize ++;
      ixes(uqsize) = i;
      prev = sortx(i);
    }
  }
  return ixes.subvec(0, uqsize);
}

inline arma::vec find_ties(const arma::vec& x){
  arma::vec ones = arma::ones(x.n_elem);
  ones.elem(find_first_unique(x)).fill(0);
  arma::vec partresult = arma::conv_to<arma::vec>::from(arma::find(ones)) - 1;
  arma::vec minusone1 = -1*arma::ones(1);
  return bmdataman::bmms_setdiff(partresult, minusone1);
}

inline arma::vec pnorm01_vec(const arma::vec& x, int lower=1, int logged=0){
  /*arma::vec p(x.n_elem);
   for(unsigned int i = 0; i<x.n_elem; i++){
   p(i) = R::pnorm(x(i), 0, 1, lower, logged);
   }
   return(p);*/
  Rcpp::NumericVector xn = Rcpp::wrap(x);
  return Rcpp::pnorm(xn, 0.0, 1.0, lower, logged);
}

inline arma::vec qnorm01_vec(const arma::vec& x, int lower=1, int logged=0){
  /*arma::vec q(x.n_elem);
   for(unsigned int i = 0; i<x.n_elem; i++){
   q(i) = R::qnorm(x(i), 0, 1, lower, logged);
   }*/
  Rcpp::NumericVector xn = Rcpp::wrap(x);
  return Rcpp::qnorm(xn, 0.0, 1.0);
}

inline arma::vec log1p_vec(const arma::vec& x){
  return log(1 + x);
}

inline arma::uvec usetdiff(const arma::uvec& x, const arma::uvec& y) {
  std::vector<int> a = arma::conv_to< std::vector<int> >::from(arma::sort(x));
  std::vector<int> b = arma::conv_to< std::vector<int> >::from(arma::sort(y));
  std::vector<int> out;
  
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::inserter(out, out.end()));
  return arma::conv_to<arma::uvec>::from(out);
}


}

#endif