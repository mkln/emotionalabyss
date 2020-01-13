

clean_rebuild <- function(package_name=NULL){
  
  if(is.null(package_name)){
    package_name <- tail(strsplit(getwd(), "/")[[1]],1)
  }
  
  temp_file <- sprintf("/tmp/%s_build_temp.RData", package_name)
  cat("save.image --> ", temp_file, "\n")
  save.image(temp_file)
  
  cat("Rcpp::compileAttributes()\n")
  Rcpp::compileAttributes()
  
  build_cmd <- sprintf("R CMD INSTALL --preclean --no-multiarch --with-keep.source ../%s", package_name)
  cat(build_cmd, "\n")
  system(build_cmd)
  
  cat("startup::restart()\n")
  startup::restart()
  
  cat("load", temp_file, "\n")
  load(temp_file)
  
  rm_cmd <- sprintf("rm %s", temp_file)
  cat(rm_cmd, "\n")
  system(rm_cmd)
  
  library_cmd <- sprintf("library(%s)", package_name)
  cat(library_cmd, "\n")
  library(package_name, character.only=T)
}