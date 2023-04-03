# A function to compute the absolute weighted correlation between exposure
# and confounders.
# params:
#   w: a vector of exposure values
#   vw: a vector of weights
#   c_num: a data.frame of numerical confounders
#   c_cat: a data.frame of categorical confounders

absolute_weighted_corr_df <- function(w,
                                      vw,
                                      c_num,
                                      c_cat){
  
  
 
  # detect numeric columns
  col_n <- colnames(c_num)

  # detect factorial columns
  col_f <- colnames(c_cat)

  c_cat[] <- lapply(c_cat, as.factor)

  absolute_corr_n <- absolute_corr_f <- NULL

  if (length(col_n) > 0) {
    absolute_corr_n<- sapply(col_n,function(i){
      abs(wCorr::weightedCorr(x = w,
                              y = c_num[,i],
                              weights = vw,
                              method = c("spearman")))})
    absolute_corr_n <- unlist(absolute_corr_n)
    names(absolute_corr_n) <- col_n
  }

  if (length(col_f) > 0) {
    internal_fun<- function(i){
      abs(wCorr::weightedCorr(x = w,
                              y = c_cat[, i],
                              weights = vw,
                              method = c("Polyserial")))}

    absolute_corr_f <- c()
    for (item in col_f){
      if (length(unique(c_cat[[item]])) == 1 ){
        absolute_corr_f <- c(absolute_corr_f, NA)
      } else {
        absolute_corr_f <- c(absolute_corr_f, internal_fun(item))
      }
    }
    names(absolute_corr_f) <- col_f
  }

  absolute_corr <- c(absolute_corr_n, absolute_corr_f)

  if (sum(is.na(absolute_corr)) > 0){
    warning(paste("The following features generated missing values: ",
                  names(absolute_corr)[is.na(absolute_corr)],
                  "\nIn computing mean covariate balance, they will be ignored."))
  }

  df <- data.frame(name = character(), value = numeric())

  for (i in names(absolute_corr)){
    df <- rbind(df, data.frame(name=i, value=absolute_corr[[i]]))
  }

  return(df)
}