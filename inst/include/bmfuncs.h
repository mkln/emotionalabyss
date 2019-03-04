#ifndef RCPP_bmfuncs
#define RCPP_bmfuncs

#include <RcppArmadillo.h>
#include <random>
#include <ctime>
#include <math.h>
#include <cstdlib>
#include "bmdataman.h"

using namespace std;

// functions more specific to modular models

namespace bmfuncs {

inline double split_struct_ratio(arma::vec prop_split, arma::vec orig_split, int p, double param){
  //return exp(-param*(gini(prop_split,p) + gini(orig_split,p)));
  return 1.0; 
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
        temp = bmdataman::exclude(temp, excluding);
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
  arma::mat normalizer = bmdataman::col_sums(J_field(s));
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

}

#endif