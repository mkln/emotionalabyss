#ifndef RCPP_bm2d
#define RCPP_bm2d

#include <RcppArmadillo.h>
#include <random>
#include <ctime>
#include <math.h>
#include <cstdlib>
#include "bmfuncs.h" 

using namespace std;

// functions for 2d manipulation

namespace bm2d {
// from matrix Sx2 where each row is a 2D split
// and p1, p2 is the grid dimension
// return a masking matrix size p1xp2 with 1 at split locations
inline arma::mat splitsub_to_splitmask(const arma::field<arma::mat>& splits, int p1, int p2){
  // given dimensions p1xp2 and the splits
  // returns matrix of zeros + l in split locations
  int lev = splits.n_elem;
  arma::mat mask = arma::zeros(p1, p2);
  for(unsigned int l=0; l<lev; l++){
    for(unsigned int i=0; i<splits(l).n_rows; i++){
      mask(splits(l)(i,0), splits(l)(i,1)) = l+1;
    }
  }
  // lower right corner cannot be used if other than voronoi
  //mask(p1-1, p2-1) = -1;
  return mask;
}

// obtain Sx2 split matrix from a split mask
inline arma::field<arma::mat> splitmask_to_splitsub(const arma::mat& splitmask){
  int lev = splitmask.max();
  arma::field<arma::mat> subs(lev);
  for(unsigned int l=0; l<lev; l++){
    arma::uvec split_locs_in_mask = arma::find(splitmask==l+1);
    subs(l) = arma::trans(arma::conv_to<arma::mat>::from( arma::ind2sub(arma::size(splitmask), split_locs_in_mask)));
  }
  return subs;
}


/*
 *              FUNCTIONS FOR GROUPING MASKS
 *                   AND COARSENING
 */

/*
 // from a starting grouping mask, split the relevant region 
 // using the new split onesplit. for BLOCKS, not voronoi
 inline arma::mat mask_onesplit(const arma::mat& startmat, 
 const arma::vec& onesplit, int seq){
 int p1 = startmat.n_rows;
 int p2 = startmat.n_cols;
 //int maxval = startmat.max();
 arma::mat mask = startmat;
#pragma omp parallel
 {
 for(unsigned int i=0; i<p1; i++){
 for(unsigned int j=0; j<p2; j++){
 //if(startmat(i,j) == startmat(onesplit(0), onesplit(1))){
 if(i <= onesplit(0)){
 if(j <= onesplit(1)){
 mask(i,j)=startmat(i,j)+1*pow(10, seq);
 } else {
 mask(i,j)=startmat(i,j)+2*pow(10, seq);
 }
 } else {
 if(j <= onesplit(1)){
 mask(i,j)=startmat(i,j)+3*pow(10, seq);
 } else {
 mask(i,j)=startmat(i,j)+4*pow(10, seq);
 }
 }
 //}
 } 
 }
 }
 return(mask-mask.min());
 }
 
 // make a grouping mask from split matrix
 // a grouping max assigns each element of the grid to a numbered group
 inline arma::mat splitsub_to_groupmask_blocks(const arma::mat& splits, int p1, int p2){
 // splits is a nsplit x 2 matrix
 arma::mat splitted = arma::zeros(p1, p2);
 for(unsigned int i=0; i<splits.n_rows; i++){
 if((splits(i,0)==p1-1) & (splits(i,1)==p2-1)){
 cout << splits.t() << endl;
 throw std::invalid_argument("edge splits not allowed: nothing to split");
 } else {
 if((splits(i,0)>p1-1) || (splits(i,1)>p2-1) || (splits(i,0)<0) || (splits(i,1)<0)){
 throw std::invalid_argument("one split outside splitting region");
 } else {
 splitted = mask_onesplit(splitted, splits.row(i).t(), i);
 }
 }
 }
 return(splitted);
 }
 
 */

inline arma::mat row_intersection(const arma::mat& mat1, const arma::mat& mat2){
  arma::mat inter = -1*arma::zeros(mat1.n_rows<mat2.n_rows? mat1.n_rows : mat2.n_rows, 2);
  int c=0;
  //#pragma omp parallel
  //{
  for(unsigned int i=0; i<mat1.n_rows; i++){
    for(unsigned int j=0; j<mat2.n_rows; j++){
      if(arma::approx_equal(mat1.row(i), mat2.row(j), "absdiff", 0.002)){
        inter.row(c) = mat1.row(i);
        c++;
      }
    }
  }
  //}
  if(c>0){
    return inter.rows(0,c-1);
  } else {
    return arma::zeros(0,2);
  }
}

inline arma::mat row_difference(const arma::mat& mat1, const arma::mat& mat2){
  arma::mat diff = -1*arma::zeros(mat1.n_rows, 2);
  int c=0;
  for(unsigned int i=0; i<mat1.n_rows; i++){
    bool foundit = false;
    for(unsigned int j=0; j<mat2.n_rows; j++){
      if(arma::approx_equal(mat1.row(i), mat2.row(j), "absdiff", 0.002)){
        foundit = true;
      }
    }
    if(foundit == false){
      diff.row(c) = mat1.row(i);
      c++;
    }
  }

  if(c>0){
    return diff.rows(0,c-1);
  } else {
    return arma::zeros(0,2);
  }
}

//with H. voronoi tessellation OLD
inline arma::mat splitsub_to_groupmask2(arma::field<arma::mat> splits, int p1, int p2){
  // splits is a nsplit x 2 matrix
  arma::vec distances = arma::ones(splits(0).n_rows);
  arma::mat splitted = arma::zeros(p1, p2);
  // level 0
  for(unsigned int i=0; i<p1; i++){
    for(unsigned int j=0; j<p2; j++){
      for(unsigned int s=0; s<splits(0).n_rows; s++){
        distances(s) = pow(0.0+i-splits(0)(s,0), 2) + pow(0.0+j-splits(0)(s,1), 2);
      }
      //clog << distances << endl;
      splitted(i, j) = distances.index_min();
    } 
  }
  
  splits = bmfuncs::splits_augmentation(splits);

  // other levels
  int lev = splits.n_elem;
  for(unsigned int l=1; l<splits.n_elem; l++){
    // subset the search on the points in the groupmask that
    // belong to the same group in the previous level
    int splits_at_prev_lev = splits(l-1).n_rows;
    // loop over possible values of previous levels
    for(unsigned int s=0; s<splits_at_prev_lev; s++){
      // for each split at this level,
      // subset matrix elements with same value of split
      arma::uvec locs = arma::find(splitted == splitted( splits(l-1)(s,0), splits(l-1)(s,1) ));
      // relevant splits are only those that are also in the same area
      arma::mat all_locs_subs = arma::conv_to<arma::mat>::from(arma::ind2sub(arma::size(splitted), locs));
      arma::mat relevant = row_intersection(all_locs_subs.t(), splits(l));
      // distance of subset points from all relevant splits at this level
      arma::mat distances = arma::ones(relevant.n_rows);
      if(distances.n_elem > 0){
        
        for(unsigned int i=0; i<locs.n_elem; i++){ // like for i, for j, but vectorized
          arma::uvec inde = arma::ind2sub(arma::size(splitted), locs(i));
          for(unsigned int r=0; r<relevant.n_rows; r++){
            distances(r) = pow(0.0+inde(0) - relevant(r,0), 2) + pow(0.0+ inde(1) - relevant(r,1), 2);
          }
          splitted(inde(0), inde(1)) += (1+distances.index_min()) * pow(55, l);
        }
      }
    }
  }
  return(splitted);
}
  

//with H. voronoi tessellation, new
inline arma::mat splitsub_to_groupmask(arma::field<arma::mat> splits, int p1, int p2){
  // splits is a nsplit x 2 matrix
  arma::vec distances = arma::ones(splits(0).n_rows);
  arma::mat splitted = arma::zeros(p1, p2);
  // level 0
  for(unsigned int i=0; i<p1; i++){
    for(unsigned int j=0; j<p2; j++){
      for(unsigned int s=0; s<splits(0).n_rows; s++){
        distances(s) = pow(0.0+i-splits(0)(s,0), 2) + pow(0.0+j-splits(0)(s,1), 2);
      }
      //clog << distances << endl;
      splitted(i, j) = distances.index_min();
    } 
  }
  
  // other levels
  int lev = splits.n_elem;
  for(unsigned int l=1; l<splits.n_elem; l++){
    // subset the search on the points in the groupmask that
    // belong to the same group in the previous level
    int splits_at_prev_lev = splits(l-1).n_rows;
    // loop over possible values of previous levels
    for(unsigned int s=0; s<splits_at_prev_lev; s++){
      // for each split at this level,
      // subset matrix elements with same value of split
      arma::uvec locs = arma::find(splitted == splitted( splits(l-1)(s,0), splits(l-1)(s,1) ));
      // relevant splits are only those that are also in the same area
      arma::mat all_locs_subs = arma::conv_to<arma::mat>::from(arma::ind2sub(arma::size(splitted), locs));
      arma::mat relevant = row_intersection(all_locs_subs.t(), splits(l));
      // distance of subset points from all relevant splits at this level
      arma::mat distances = arma::ones(relevant.n_rows);
      if(distances.n_elem > 0){
        for(unsigned int i=0; i<locs.n_elem; i++){ // like for i, for j, but vectorized
          arma::uvec inde = arma::ind2sub(arma::size(splitted), locs(i));
          for(unsigned int r=0; r<relevant.n_rows; r++){
            distances(r) = pow(0.0+inde(0) - relevant(r,0), 2) + pow(0.0+ inde(1) - relevant(r,1), 2);
          }
          splitted(inde(0), inde(1)) += (1+distances.index_min()) * pow(55, l);
        }
      }
    }
  }
  return(splitted);
}

//with Bubbles-Voronoi tessellation
inline arma::mat splitsub_to_groupmask_bubbles(arma::field<arma::mat> splits, int p1, int p2, 
                                               double radius_in, double dec=1, bool circle=true){
  double radius = radius_in;
  arma::mat splitted = arma::zeros(p1, p2);
  for(unsigned int l=0; l<splits.n_elem; l++){
    if(splits(l).n_rows != 0){
      radius = dec*(splits.n_elem - l)*radius_in;
      for(unsigned int s=0; s<splits(l).n_rows; s++){
        arma::vec distances = arma::ones(splits(l).n_rows);
        int iloc = splits(l)(s,0);
        int jloc = splits(l)(s,1);
        int mini = max(0, (int)round(iloc-radius));
        int maxi = min(p2, (int)round(iloc+radius));
        int minj = max(0, (int)round(jloc-radius));
        int maxj = min(p1, (int)round(jloc+radius));
        //clog << mini << " " << maxi << " " << minj << " " << maxj << endl;
        //arma::mat newsplitted = splitted;
        for(unsigned int i=mini; i<maxi; i++){
          for(unsigned int j=minj; j<maxj; j++){
            for(unsigned int r=0; r<splits(l).n_rows; r++){
              distances(r) = pow(0.0+i-splits(l)(r,0), 2) + pow(0.0+j-splits(l)(r,1), 2);
            }
            int minfound = distances.index_min();
            if((s == minfound)  
                 //& (splitted(i,j)==splitted(iloc,jloc))
                 ){
              if(circle){
                if(pow(pow( abs(0.0 + i - iloc), 2) + pow( abs(0.0 +j - jloc), 2 ), .5) < radius){
                   splitted(i, j) += (1+s) * pow(55, l);
                }
              } else {
                splitted(i, j) += (1+s) * pow(55, l);
              }
            } 
          }
        }
        //splitted = newsplitted;
      }
    }
  }
  return(splitted);
}

// extract region numbers (labels) from a grouping mask
inline arma::vec mat_unique(const arma::mat& A){
  arma::uvec uvals = arma::find_unique(A);
  return(A.elem(uvals));
}

// returns a matrix where all unselected regions are set to 0
inline arma::mat mask_oneval(const arma::mat& A, const arma::mat& mask, int val){
  arma::uvec uvals = arma::find(mask != val);
  arma::mat retmat = A;
  retmat.elem(uvals).fill(0.0);
  return(retmat);
}

// using a grouping mask, sum values in a matrix corresponding to 
// the same group (2d coarsening operation)
inline double mask_oneval_sum(const arma::mat& A, const arma::mat& mask, int val){
  arma::uvec uvals = arma::find(mask == val);
  return(arma::accu(A.elem(uvals)));
}

// transform a regressor matrix to a vector using grouping mask as coarsening
inline arma::vec mat_to_vec_by_region(const arma::mat& A, const arma::mat& mask, 
                                      const arma::vec& unique_regions){
  //arma::vec unique_regions = mat_unique(mask);
  int n_unique_regions = unique_regions.n_elem;
  arma::vec vectorized_mat = arma::zeros(n_unique_regions);
  for(unsigned int r=0; r<n_unique_regions; r++){
    vectorized_mat(r) = mask_oneval_sum(A, mask, unique_regions(r));
  }
  return(vectorized_mat);
}

// using a grouping mask, transforma a cube to a matrix
inline arma::mat cube_to_mat_by_region(const arma::cube& C, const arma::mat& mask, 
                                       const arma::vec& unique_regions){
  // cube is assumed dimension (p1, p2, n)
  int n_unique_regions = unique_regions.n_elem;
  arma::mat matricized_cube = arma::zeros(C.n_slices, n_unique_regions);
  for(unsigned int r=0; r<n_unique_regions; r++){
    // every row in the cube is a matrix observation
    for(unsigned int i=0; i<C.n_slices; i++){
      // we transform every matrix into a vector by region
      matricized_cube(i,r) = mask_oneval_sum(C.slice(i), mask, unique_regions(r));
    }
  }
  return(matricized_cube);
}

// given a vectorized beta vector, a vector of labels, and a grouping mask
// return a matrix of size size(mask) filling it with beta using regions
inline arma::mat unmask_vector(const arma::vec& beta, const arma::vec& regions, const arma::mat& mask){
  // takes a vector and fills a matrix of dim size(mask) using regions
  arma::mat unmasked_vec = arma::zeros(arma::size(mask));
  for(unsigned int r=0; r<regions.n_elem; r++){
    unmasked_vec.elem(arma::find(mask == regions(r))).fill(beta(r));
  }
  return(unmasked_vec);
}

/*
 *              FUNCTIONS TO MOVE SPLIT LOCATIONS
 *                   TO APPROPRIATE NEW LOCS
 */

// focuses around onesplit, with radius
inline arma::mat splitmask_focus(const arma::mat& mask_of_splits, arma::vec onesplit, int radius_int=1){
  int x1 = onesplit(0) - radius_int;
  int x2 = onesplit(0) + radius_int;
  int y1 = onesplit(1) - radius_int;
  int y2 = onesplit(1) + radius_int;
  int xlim = mask_of_splits.n_rows-1;
  int ylim = mask_of_splits.n_cols-1;
  x1 = max(0, x1);
  x2 = min(x2, xlim);
  y1 = max(0, y1);
  y2 = min(y2, ylim);
  
  // focusing
  arma::mat around_split = mask_of_splits.submat(x1, y1, x2, y2);
  return around_split;
}

// given focus of a split matrix, counts the zeros (ie number of available locations to move to)
inline int number_availables(const arma::mat& splitmask_focus){
  int c1 = arma::accu(1-splitmask_focus.elem(arma::find(splitmask_focus == 0)));
  return c1;
}


}

#endif