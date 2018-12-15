// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// setdiff
arma::vec setdiff(const arma::vec& x, const arma::vec& y);
RcppExport SEXP _emotionalabyss_setdiff(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(setdiff(x, y));
    return rcpp_result_gen;
END_RCPP
}
// X2Dgrid
arma::mat X2Dgrid(const arma::vec& x1, const arma::vec& x2);
RcppExport SEXP _emotionalabyss_X2Dgrid(SEXP x1SEXP, SEXP x2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type x1(x1SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type x2(x2SEXP);
    rcpp_result_gen = Rcpp::wrap(X2Dgrid(x1, x2));
    return rcpp_result_gen;
END_RCPP
}
// X2Dgrid_alt
arma::mat X2Dgrid_alt(const arma::vec& x1, const arma::vec& x2);
RcppExport SEXP _emotionalabyss_X2Dgrid_alt(SEXP x1SEXP, SEXP x2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type x1(x1SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type x2(x2SEXP);
    rcpp_result_gen = Rcpp::wrap(X2Dgrid_alt(x1, x2));
    return rcpp_result_gen;
END_RCPP
}
// exclude
arma::mat exclude(const arma::mat& test, const arma::vec& excl);
RcppExport SEXP _emotionalabyss_exclude(SEXP testSEXP, SEXP exclSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type test(testSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type excl(exclSEXP);
    rcpp_result_gen = Rcpp::wrap(exclude(test, excl));
    return rcpp_result_gen;
END_RCPP
}
// nonzeromean
arma::vec nonzeromean(const arma::mat& mat_mcmc);
RcppExport SEXP _emotionalabyss_nonzeromean(SEXP mat_mcmcSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type mat_mcmc(mat_mcmcSEXP);
    rcpp_result_gen = Rcpp::wrap(nonzeromean(mat_mcmc));
    return rcpp_result_gen;
END_RCPP
}
// col_eq_check
arma::vec col_eq_check(const arma::mat& A);
RcppExport SEXP _emotionalabyss_col_eq_check(SEXP ASEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    rcpp_result_gen = Rcpp::wrap(col_eq_check(A));
    return rcpp_result_gen;
END_RCPP
}
// col_sums
arma::vec col_sums(const arma::mat& matty);
RcppExport SEXP _emotionalabyss_col_sums(SEXP mattySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type matty(mattySEXP);
    rcpp_result_gen = Rcpp::wrap(col_sums(matty));
    return rcpp_result_gen;
END_RCPP
}
// hat_alt
arma::mat hat_alt(const arma::mat& X);
RcppExport SEXP _emotionalabyss_hat_alt(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(hat_alt(X));
    return rcpp_result_gen;
END_RCPP
}
// hat
arma::mat hat(const arma::mat& X);
RcppExport SEXP _emotionalabyss_hat(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(hat(X));
    return rcpp_result_gen;
END_RCPP
}
// cube_mean
arma::mat cube_mean(const arma::cube& X, int dim);
RcppExport SEXP _emotionalabyss_cube_mean(SEXP XSEXP, SEXP dimSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::cube& >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type dim(dimSEXP);
    rcpp_result_gen = Rcpp::wrap(cube_mean(X, dim));
    return rcpp_result_gen;
END_RCPP
}
// cube_sum
arma::mat cube_sum(const arma::cube& X, int dim);
RcppExport SEXP _emotionalabyss_cube_sum(SEXP XSEXP, SEXP dimSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::cube& >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type dim(dimSEXP);
    rcpp_result_gen = Rcpp::wrap(cube_sum(X, dim));
    return rcpp_result_gen;
END_RCPP
}
// cube_prod
arma::mat cube_prod(const arma::cube& x, int dim);
RcppExport SEXP _emotionalabyss_cube_prod(SEXP xSEXP, SEXP dimSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::cube& >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type dim(dimSEXP);
    rcpp_result_gen = Rcpp::wrap(cube_prod(x, dim));
    return rcpp_result_gen;
END_RCPP
}
// index_to_subscript
arma::mat index_to_subscript(const arma::uvec& index, const arma::mat& m);
RcppExport SEXP _emotionalabyss_index_to_subscript(SEXP indexSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type index(indexSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(index_to_subscript(index, m));
    return rcpp_result_gen;
END_RCPP
}
// gini
double gini(arma::vec& x, int p);
RcppExport SEXP _emotionalabyss_gini(SEXP xSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(gini(x, p));
    return rcpp_result_gen;
END_RCPP
}
// find_avail
arma::vec find_avail(const arma::vec& tied, int n);
RcppExport SEXP _emotionalabyss_find_avail(SEXP tiedSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type tied(tiedSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(find_avail(tied, n));
    return rcpp_result_gen;
END_RCPP
}
// find_first_unique
arma::uvec find_first_unique(const arma::vec& x);
RcppExport SEXP _emotionalabyss_find_first_unique(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(find_first_unique(x));
    return rcpp_result_gen;
END_RCPP
}
// find_ties
arma::vec find_ties(const arma::vec& x);
RcppExport SEXP _emotionalabyss_find_ties(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(find_ties(x));
    return rcpp_result_gen;
END_RCPP
}
// rndppll_mvnormal
arma::mat rndppll_mvnormal(int n, const arma::vec& mean, const arma::mat& sigma);
RcppExport SEXP _emotionalabyss_rndppll_mvnormal(SEXP nSEXP, SEXP meanSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(rndppll_mvnormal(n, mean, sigma));
    return rcpp_result_gen;
END_RCPP
}
// rndpp_mvnormal
arma::mat rndpp_mvnormal(int n, const arma::vec& mean, const arma::mat& sigma);
RcppExport SEXP _emotionalabyss_rndpp_mvnormal(SEXP nSEXP, SEXP meanSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(rndpp_mvnormal(n, mean, sigma));
    return rcpp_result_gen;
END_RCPP
}
// rndpp_mvnormal2
arma::mat rndpp_mvnormal2(int n, const arma::vec& mu, const arma::mat& sigma);
RcppExport SEXP _emotionalabyss_rndpp_mvnormal2(SEXP nSEXP, SEXP muSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type mu(muSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(rndpp_mvnormal2(n, mu, sigma));
    return rcpp_result_gen;
END_RCPP
}
// rndpp_mvnormalnew
arma::mat rndpp_mvnormalnew(int n, const arma::vec& mean, const arma::mat& sigma);
RcppExport SEXP _emotionalabyss_rndpp_mvnormalnew(SEXP nSEXP, SEXP meanSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(rndpp_mvnormalnew(n, mean, sigma));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_emotionalabyss_setdiff", (DL_FUNC) &_emotionalabyss_setdiff, 2},
    {"_emotionalabyss_X2Dgrid", (DL_FUNC) &_emotionalabyss_X2Dgrid, 2},
    {"_emotionalabyss_X2Dgrid_alt", (DL_FUNC) &_emotionalabyss_X2Dgrid_alt, 2},
    {"_emotionalabyss_exclude", (DL_FUNC) &_emotionalabyss_exclude, 2},
    {"_emotionalabyss_nonzeromean", (DL_FUNC) &_emotionalabyss_nonzeromean, 1},
    {"_emotionalabyss_col_eq_check", (DL_FUNC) &_emotionalabyss_col_eq_check, 1},
    {"_emotionalabyss_col_sums", (DL_FUNC) &_emotionalabyss_col_sums, 1},
    {"_emotionalabyss_hat_alt", (DL_FUNC) &_emotionalabyss_hat_alt, 1},
    {"_emotionalabyss_hat", (DL_FUNC) &_emotionalabyss_hat, 1},
    {"_emotionalabyss_cube_mean", (DL_FUNC) &_emotionalabyss_cube_mean, 2},
    {"_emotionalabyss_cube_sum", (DL_FUNC) &_emotionalabyss_cube_sum, 2},
    {"_emotionalabyss_cube_prod", (DL_FUNC) &_emotionalabyss_cube_prod, 2},
    {"_emotionalabyss_index_to_subscript", (DL_FUNC) &_emotionalabyss_index_to_subscript, 2},
    {"_emotionalabyss_gini", (DL_FUNC) &_emotionalabyss_gini, 2},
    {"_emotionalabyss_find_avail", (DL_FUNC) &_emotionalabyss_find_avail, 2},
    {"_emotionalabyss_find_first_unique", (DL_FUNC) &_emotionalabyss_find_first_unique, 1},
    {"_emotionalabyss_find_ties", (DL_FUNC) &_emotionalabyss_find_ties, 1},
    {"_emotionalabyss_rndppll_mvnormal", (DL_FUNC) &_emotionalabyss_rndppll_mvnormal, 3},
    {"_emotionalabyss_rndpp_mvnormal", (DL_FUNC) &_emotionalabyss_rndpp_mvnormal, 3},
    {"_emotionalabyss_rndpp_mvnormal2", (DL_FUNC) &_emotionalabyss_rndpp_mvnormal2, 3},
    {"_emotionalabyss_rndpp_mvnormalnew", (DL_FUNC) &_emotionalabyss_rndpp_mvnormalnew, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_emotionalabyss(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
