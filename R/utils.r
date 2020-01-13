

clean_rebuild <- function(package_name){
  
  temp_file <- sprintf("/tmp/%s_build_temp.RData", package_name)
  cat("save.image --> ", temp_file, "\n")
  save.image(temp_file)
  
  cat("Rcpp::compileAttributes()")
  Rcpp::compileAttributes()
  
  build_cmd <- sprintf("R CMD INSTALL --preclean --no-multiarch --with-keep.source ../%s", package_name)
  cat(build_cmd, "\n")
  system(build_cmd)
  
  cat("startup::restart()\n")
  startup::restart()
  
  cat("load", temp_file)
  load(temp_file)
  
  rm_cmd <- sprintf("rm %s", temp_file)
  cat(rm_cmd)
  system(rm_cmd)
  
}