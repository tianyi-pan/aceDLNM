#' Title formula for Effect of exposure process
#'
#' @param x exposure variable
#' @param t time variable
#'
#' @return a formula for model fitting in \code{aceDLNM}. 
#' @export
sX <- function(t, x){
  xvar <- as.character(substitute(x))
  tvar <- as.character(substitute(t))
  return(list(x = xvar, t = tvar))
}

# function from mam https://github.com/awstringer1/mam/blob/master/R/helper.R
newsparsemat <- function(n,m) {
  methods::new('dgCMatrix',Dim=as.integer(c(n,m)),p=rep(0L,m+1),i=integer(0),x=numeric(0))
}
# ##
# getsX <- function(sXobject, dat) {
#   x <- dat[,sXobject$x]
#   t <- dat[,sXobject$t]
#   return(data.frame(x = x, t = t))
# }
