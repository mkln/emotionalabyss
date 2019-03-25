#ifndef RCPP_bmfuncs
#define RCPP_bmfuncs

#include <RcppArmadillo.h>
#include <random>
#include <ctime>
#include <math.h>
#include <cstdlib>

using namespace std;

// functions 

namespace bmfuncs {
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
  return bmms_setdiff(ls, tied);
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
  return bmms_setdiff(partresult, minusone1);
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


inline arma::mat div_by_colsum(const arma::mat& J){
  arma::vec jcs = col_sums(J);
  arma::mat result = arma::zeros(J.n_rows, J.n_cols);
  for(unsigned int i=0; i<result.n_cols; i++){
    result.col(i) = J.col(i) / jcs(i);
  }
  return result;
}



inline arma::mat single_split(const arma::mat& Jcoarse, int where, int p){
  int which_row = where;
  int which_col = arma::conv_to<int>::from(arma::find(Jcoarse.row(which_row), 1, "first"));
  
  arma::mat slice_upleft = Jcoarse.submat(0, which_col, which_row, which_col);
  arma::mat slice_downright = Jcoarse.submat(which_row+1, which_col, p-1, which_col);
  arma::mat upright = arma::zeros(which_row+1, 1);
  arma::mat downleft = arma::zeros(p-which_row-1, 1);
  arma::mat up = arma::join_rows(slice_upleft, upright);
  arma::mat down = arma::join_rows(downleft, slice_downright);
  arma::mat splitted = arma::join_cols(up, down); 
  return splitted;
}

inline arma::mat single_split_new(const arma::mat& Jcoarse, int where, int p){
  arma::mat splitted = arma::zeros(Jcoarse.n_rows, Jcoarse.n_cols+1);
  
  int which_row = where;
  int which_col = arma::conv_to<int>::from(arma::find(Jcoarse.row(which_row), 1, "first"));
  
  //arma::mat slice_upleft = Jcoarse.submat(0, which_col, which_row, which_col);
  //arma::mat slice_downright = Jcoarse.submat(which_row+1, which_col, p-1, which_col);
  //arma::mat upright = arma::zeros(which_row+1, 1);
  //arma::mat downleft = arma::zeros(p-which_row-1, 1);
  //arma::mat up = arma::join_rows(slice_upleft, upright);
  //arma::mat down = arma::join_rows(downleft, slice_downright);
  
  splitted.submat(0, which_col, which_row, which_col) = Jcoarse.submat(0, which_col, which_row, which_col);
  splitted.submat(which_row+1, which_col+1, p-1, which_col+1) = Jcoarse.submat(which_row+1, which_col, p-1, which_col);
  //arma::mat splitted = arma::join_cols(up, down); 
  return splitted;
}

inline arma::mat multi_split(const arma::vec& pones, const arma::vec& splits, int p){
  arma::vec jumps = arma::zeros(p);
  arma::mat splitted = arma::zeros(p, splits.n_elem+1);
  for(unsigned int i=0; i<splits.n_elem; i++){
    jumps(splits(i)+1) = 1;
  }
  jumps = arma::cumsum(jumps);
  for(unsigned int i=0; i<jumps.n_elem; i++){ // splitted col = splitted(i) 
    splitted(i, jumps(i)) = 1;
  }
  return splitted;
}


inline arma::mat exclude(const arma::mat& test, const arma::vec& excl){
  arma::vec keepers = arma::ones(test.n_cols);
  for(unsigned int e=0; e<excl.n_elem; e++){
    keepers(excl(e)) = 0;
  }
  //cout << "exclude cols " << endl;
  return test.cols(arma::find(keepers));
}


inline arma::mat multi_split_old(const arma::mat& Jcoarse, 
                             const arma::vec& where, int p){
  //int p = arma::accu(Jcoarse);
  //int c = Jcoarse.n_cols;
  
  arma::vec unique_where = arma::unique(where);
  arma::vec excluding(1);
  arma::vec orig_cols_splitted = arma::zeros(Jcoarse.n_cols);
  arma::mat slice_upleft, slice_downright, upright, downleft, up, down, splitted;
  
  arma::mat splitting_mat = Jcoarse;
  arma::mat temp(Jcoarse.n_rows, 0);
  //cout << Jcoarse << endl;
  for(unsigned int w=0; w<unique_where.n_elem; w++){
    unsigned int which_row = unique_where(w);
    //cout << "multi_split: inside loop. splitting row " << which_row << " of " << endl;
    //cout << Jcoarse << endl; 
    int which_col = arma::conv_to<int>::from(arma::find(Jcoarse.row(which_row), 1, "first"));
    
    //cout << "splitting at row " << which_row << endl;
    //cout << "corresponding to " << which_col << " in the original matrix" << endl;
    //cout << "current status of splitted cols " << orig_cols_splitted.t() << endl;
    
    if(orig_cols_splitted(which_col) == 1){
      // we had already split this column before
      // this means we are splitting a column from the matrix of splits
      //cout << "we have already split this column before, hence taking this matrix " << endl << temp << endl;
      which_col = arma::conv_to<unsigned int>::from(arma::find(temp.row(which_row), 1, "first"));
      //cout << "hence the column we need to look at is actually " << which_col << endl;
      if((which_row>=temp.n_rows-1) || (temp(which_row+1, which_col) != 0)){
        splitted = single_split(temp, which_row, p);
        //cout << "and we obtain the following matrix " << endl << splitted << endl;
        excluding(0) = which_col;
        temp = exclude(temp, excluding);
        temp = arma::join_horiz(temp, splitted);
        //cout << "status up to now: " << endl << temp << endl;
      } //else {
      // cout << "ineffective split " << endl;
      //}
    } else {
      // first time we split this column on the original matrix
      //cout << "did not split this column before" << endl;
      orig_cols_splitted(which_col) = 1;
      //cout << which_row << " : " << which_col << endl; //" [ " << priorprob.t() <<  " ] " << endl;
      if((which_row>=Jcoarse.n_rows-1) || (Jcoarse(which_row+1, which_col) != 0)){
        //cout << "starting from the original " << endl << Jcoarse << endl;
        splitted = single_split(Jcoarse, which_row, p);
        //cout << "we obtain the following matrix " << endl << splitted << endl;
        temp = arma::join_horiz(temp, splitted);
        //cout << "status up to now: " << endl << temp << endl;
      } //else {
      //cout << "ineffective split " << endl;
      //}
    }
    //cout << "===================================" << endl;
  }
  //cout << test1 << endl << test2 << endl << test << endl;
  return temp;
}



inline arma::mat multi_split_ones(const arma::vec& where, int p){
  arma::vec unique_where = arma::unique(where);
  int wsize = unique_where.n_elem;
  arma::mat result = arma::zeros(p, wsize+1);
  arma::vec wheresort = arma::sort(unique_where);
  arma::vec whereadd = arma::zeros(wsize+2);
  whereadd(0) = -1;
  whereadd.subvec(1,wsize) = wheresort;
  whereadd(wsize+1) = p-1;
  arma::vec sizes = arma::diff(whereadd);
  //clog << sizes << endl;
  int rowcount = 0;
  for(int i=0; i<unique_where.n_elem+1; i++){
    //clog << "from:" << rowcount << " to:" << rowcount+sizes(i) << " column:" << i << endl;
    result.submat(rowcount, i, rowcount+sizes(i)-1, i).fill(1);
    rowcount += sizes(i);
  }
  return result;
}

inline arma::mat multi_split_ones_v2(const arma::vec& where, int p){
  arma::vec unique_where = arma::unique(where);
  int wsize = unique_where.n_elem;
  arma::mat result = arma::zeros(2, wsize+1);
  arma::vec wheresort = arma::sort(unique_where);
  arma::vec whereadd = arma::zeros(wsize+2);
  whereadd(0) = -1;
  whereadd.subvec(1,wsize) = wheresort;
  whereadd(wsize+1) = p-1;
  arma::vec sizes = arma::diff(whereadd);
  //clog << sizes << endl;
  int rowcount = 0;
  for(int i=0; i<wsize+1; i++){
    //clog << "from:" << rowcount << " to:" << rowcount+sizes(i) << " column:" << i << endl;
    result.submat(0,i,0,i) = rowcount;
    result.submat(1,i,1,i) = rowcount+sizes(i)-1;
    rowcount += sizes(i);
  }
  return result;
}

inline arma::mat multi_split_nonnested(const arma::mat& prevmat, arma::vec newsplits, int p){
  arma::vec jumps = arma::zeros(p);
  for(unsigned int i=0; i<newsplits.n_elem; i++){
    jumps(newsplits(i)+1) = 1;
  }
  jumps = 1+arma::cumsum(jumps);
  int totcol = prevmat.n_cols;
  
  arma::vec colchecker = arma::zeros(p);
  arma::vec cols_to_be_changed = arma::zeros(0);
  arma::vec num_changes_bycol = arma::zeros(0);
  
  for(unsigned int j=0; j<prevmat.n_cols; j++){
    // if a jump happens when this column has 1, then the colchecker will have >2 unique val 
    // one must be 0, the other can be 1 or more, but only if there's another one there has been a change
    colchecker = prevmat.col(j) % jumps;
    arma::vec uniques = arma::unique(colchecker);
    bool changed = uniques.n_elem > 2;
    if(changed) { 
      cols_to_be_changed = arma::join_vert(cols_to_be_changed, arma::ones(1)*j);
      num_changes_bycol = arma::join_vert(num_changes_bycol, arma::ones(1)*(uniques.n_elem-2));
      totcol = totcol + uniques.n_elem - 2;
    }
  }
  //clog << cols_to_be_changed.t() << endl;
  //clog << num_changes_bycol.t() << endl;
  //clog << totcol << endl;
  arma::vec cols_notto_be_changed = arma::zeros(0);
  arma::mat returnmat = arma::zeros(prevmat.n_rows, totcol);
  int changing=0;
  int ccol = 0;
  for(unsigned int j=0; j<prevmat.n_cols; j++){
    //clog << "j = " << j << endl;
    if(j == cols_to_be_changed(changing)){
      //clog << "column needs to be changed, with " << num_changes_bycol(changing) << " changes" << endl;
      int target = j;
      arma::vec targetvec = prevmat.col(target);
      for(int c=0; c<num_changes_bycol(changing); c++){
        //clog << "  change c=" << c;
        int addhere = prevmat.n_cols + ccol;
        //clog << " will affect column " << addhere << endl;
        jumps = jumps-1;
        jumps.elem(arma::find(jumps==-1)).fill(0);
        returnmat.col(addhere) = targetvec % jumps;
        returnmat.col(target) = targetvec - returnmat.col(addhere);
        ccol ++;
        target = addhere;
        targetvec = returnmat.col(target);
      }
      if(changing < cols_to_be_changed.n_elem-1) {
        changing++; 
      }
    } else {
      //clog << "column does not need to be changed" << endl;
      cols_notto_be_changed = arma::join_vert(cols_notto_be_changed, arma::ones(1)*j);
      returnmat.col(j) = prevmat.col(j);
    }
  }
  
  //clog << cols_notto_be_changed.t() << endl;
  returnmat = exclude(returnmat, cols_notto_be_changed);
  
  // fix
  for(unsigned int j=0; j<returnmat.n_cols; j++){
    arma::vec replacement = returnmat.col(j);
    replacement.elem(arma::find(replacement > 1)).fill(1);
    replacement.elem(arma::find(replacement < 0)).fill(0);
    returnmat.col(j) = replacement;
  }
  
  return returnmat;
}


inline arma::vec split_fix(const arma::field<arma::vec>& in_splits, int stage){
  arma::vec splits_if_any = in_splits(stage)(arma::find(in_splits(stage)!=-1));
  return splits_if_any;
  //cout << "OUT SPLITS " << in_splits << endl;
}

inline arma::field<arma::vec> stage_fix(const arma::field<arma::vec>& in_splits){
  //cout << "IN SPLITS " << in_splits << endl;
  //cout << "theoretical number of stages : " << in_splits.n_elem << endl;
  int n_stages = 0;
  for(unsigned int i=0; i<in_splits.n_elem; i++){
    if(in_splits(i).n_elem > 0){
      bool actual_stage = false;
      for(unsigned int j=0; j<in_splits(i).n_elem; j++){
        if(in_splits(i)(j)>-1){
          actual_stage = true;
        }
      }
      n_stages = actual_stage ? (n_stages+1) : n_stages;
    }
  }
  //cout << "actual number of stages ; " << n_stages << endl;
  int loc = 0;
  arma::field<arma::vec> split_seq(n_stages);
  for(unsigned int i=0; i<in_splits.n_elem; i++){
    if(in_splits(i).n_elem > 0){
      arma::vec splits_if_any = in_splits(i)(arma::find(in_splits(i)!=-1));
      if(splits_if_any.n_elem > 0){
        split_seq(loc) = splits_if_any;
        loc++;
      }
    }
  }
  return split_seq;
  //cout << "OUT SPLITS " << in_splits << endl;
}

inline arma::vec stretch_vec(const arma::mat& locations, const arma::vec& base, int dim){
  arma::vec bigvec = arma::zeros(dim);
  for(int i=0; i<base.n_elem; i++){
    bigvec.subvec(locations(0,i), locations(1,i)).fill(base(i));
  }
  return bigvec;
}


inline arma::mat reshaper(const arma::field<arma::mat>& J_field, int s){
  arma::mat stretcher = (J_field(s).t() * J_field(s-1));
  arma::mat normalizer = col_sums(J_field(s));
  //stretcher.transform( [](double val) { return (val>0 ? 1 : 0); } );
  for(unsigned int j=0; j<stretcher.n_rows; j++){
    stretcher.row(j) = stretcher.row(j)/normalizer(j);
  }
  return(stretcher);
}

inline arma::field<arma::vec> splits_truncate(const arma::field<arma::vec>& splits, int k){
  int k_effective = splits.n_elem > k ? k : splits.n_elem;
  if(k_effective<1){
    k_effective=1;
  }
  arma::field<arma::vec> splits_new(k_effective);
  for(int i=0; i<k_effective; i++){
    splits_new(i) = splits(i);
  }
  return splits_new;
}

inline arma::field<arma::mat> splits_augmentation(const arma::field<arma::mat>& splits){
  arma::field<arma::mat> splits_augment(splits.n_elem);
  // append previous splits to currents
  splits_augment(0) = splits(0);
  for(unsigned int i=1; i<splits.n_elem; i++){
    splits_augment(i) = arma::join_vert(splits_augment(i-1), splits(i));
  }
  return splits_augment;
}

inline arma::field<arma::mat> merge_splits(const arma::field<arma::mat>& old_splits, 
                                           const arma::field<arma::mat>& new_splits){
  arma::field<arma::mat> splits(old_splits.n_elem);
  for(unsigned int s = 0; s<new_splits.n_elem; s++){
    splits(s) = arma::join_vert(old_splits(s), new_splits(s));
  }
  return splits;
}

inline arma::field<arma::mat> Ares(const arma::mat& L1, const arma::mat& L2, const arma::mat& L){
  // returns A = [A1 A2]' matrix such that [L1 L2] A = L
  arma::field<arma::mat> result(2);
  arma::mat A = arma::pinv(arma::join_horiz(L1, L2)) * L;
  result(0) = A.rows(0, L1.n_cols-1);
  result(1) = A.rows(L1.n_cols, A.n_rows-1);
  return result;
} 

inline arma::vec update_scale(const arma::mat& Di, 
                              const arma::mat& D1,
                              const arma::mat& D2,
                              const arma::field<arma::mat>& A,
                              const arma::mat& X1,
                              const arma::mat& X2,
                              const arma::vec& theta1,
                              const arma::vec& theta2){
 // run model at some scale, get residuals and run another model at another scale
 // whats the implied regression coefficient at the new scale?
 return Di * ( (A(0).t()*D1.t() + A(1).t()*X2.t()*X1) * theta1 + A(1).t()*D2*theta2 );
}


inline double ilogit(const double& x, const double& r){
  return 1.0/(1 + exp(-.5/r * x));
}

inline double tline(const double& x, const double& m){
  double y = m*x + .5;
  if(y < 0){
    return 0;
  }
  if(y > 1){
    return 1;
  }
  return y;
}


inline arma::vec Jcol_ilogitsmooth(const arma::vec& J, double r){
  if(r == 0){
    return J;
  }
  double p = J.n_elem;
  r = r*p / 100.0;
  arma::vec ix_ones = arma::conv_to<arma::vec>::from(arma::find(J==1));
  arma::vec result = arma::zeros(J.n_elem);
  double meanix_min;
  double meanix_max;
  
  meanix_min = ix_ones.min();
  meanix_max = ix_ones.max();
  
  if(meanix_max-meanix_min < 2){
    meanix_min -= .5;
    meanix_max += .5;
  }
  if( r > 0 ){
    for(unsigned int i=0; i<result.n_elem; i++){
      double where = (i+0.0);
      result(i) = 1 + ilogit(where - meanix_min, r) - ilogit(where - meanix_max, r);
    }
  } else {
    for(unsigned int i=0; i<result.n_elem; i++){
      double where = (i+0.0);
      result(i) = min( tline(where - meanix_min, -.1/r), 1 - tline(where - meanix_max, -.1/r));
    }
  }
  return (result-result.min())/(result.max()-result.min());
}



inline arma::vec Jcol_pnormsmooth(const arma::vec& J, double r){
  if(r == 0){
    return J;
  }
  double p = J.n_elem;
  r = r*p / 100.0;
  arma::vec ix_ones = arma::conv_to<arma::vec>::from(arma::find(J==1));
  arma::vec result = arma::zeros(J.n_elem);
  
  double meanix_min = ix_ones.min();
  double meanix_max = ix_ones.max();
  if(meanix_max-meanix_min < 2){
    meanix_min -= .5;
    meanix_max += .5;
  }
  for(unsigned int i=0; i<result.n_elem; i++){
    double where = (i+0.0);
    result(i) = R::pnorm(where, meanix_min+.0, r, 1, 0);
    result(i) = result(i) + 1-R::pnorm(where, meanix_max+.0, r, 1, 0);
  }
  return (result-result.min())/(result.max()-result.min());
}

inline arma::mat J_smooth(const arma::mat& J, double radius, bool nested){
  if(radius==0){
    return J;
  }
  arma::mat result = arma::zeros(J.n_rows, J.n_cols);
  for(unsigned int j=0; j<J.n_cols; j++){
    result.col(j) = Jcol_ilogitsmooth(J.col(j), radius);
  }
  if(nested){
    for(unsigned int i=0; i<J.n_rows; i++){
      result.row(i) = result.row(i) / arma::accu(result.row(i));
    }
  }
  return result;
}

inline arma::mat wavelettize(const arma::mat& J){
  arma::vec Jcs = col_sums(J)/2;
  arma::mat Jw = J;
  for(unsigned int j=0; j<J.n_cols; j++){
    int cc=0;
    int i=0;
    while( (cc < Jcs(j)) & (i < J.n_rows)){
      //for(unsigned int i=0; i<J.n_rows; i++){
      if(J(i,j)==1){
        Jw(i,j) = -1;
        cc++;
      } 
      i++;
    }
  }
  return J;
}

inline double centerloc(const arma::vec& Jcol){
  arma::vec locs = arma::conv_to<arma::vec>::from(arma::find(Jcol==1));
  return arma::mean(locs);
}

inline arma::vec splits_to_centers(const arma::vec& bigsplits, int p){  
  arma::vec splits_sorted = arma::sort(bigsplits);
  arma::vec fullsplits = arma::unique(arma::join_vert(arma::join_vert(arma::zeros(1), splits_sorted+1), p*arma::ones(1)));
  arma::vec lengths = arma::diff(fullsplits);
  arma::vec centers = arma::zeros(splits_sorted.n_elem + 1);
  for(int i=0; i<centers.n_elem; i++){
    if(lengths(i)==1){
      centers(i) = fullsplits(i);
    } else {
      centers(i) = fullsplits(i) + lengths(i)/2.0 - .5;
    }
  }
  return centers;
}


inline arma::mat Jcentercover(const arma::mat& J, double radius){
  radius += .5; // 
  arma::mat Jnew = J;
  for(unsigned int j=0; j<J.n_cols; j++){
    double cloc = centerloc(J.col(j));
    for(unsigned int i=0; i<J.n_rows; i++){
      if( abs(i - cloc) > radius ){
        Jnew(i, j) = 0;
      }
    }
  }
  return Jnew;
}


}

#endif