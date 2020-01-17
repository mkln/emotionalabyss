

clean_rebuild <- function(package_name=NULL){
  
  if(is.null(package_name)){
    package_name <- tail(strsplit(getwd(), "/")[[1]],1)
  }
  
  .First <- function(){
    library_cmd <- sprintf("library(%s)", package_name)
    cat(library_cmd, "\n")
    library(package_name, character.only=T)
  }
  
  list2env(list(".First"=.First), envir=.GlobalEnv)
  
  cat("save.image\n")
  save.image(".RData")
  
  cat("Rcpp::compileAttributes()\n")
  Rcpp::compileAttributes()
  
  build_cmd <- sprintf("R CMD INSTALL --preclean --no-multiarch --with-keep.source ../%s", package_name)
  cat(build_cmd, "\n")
  system(build_cmd)
  
  cat("startup::restart()\n")
  startup::restart()
  
}