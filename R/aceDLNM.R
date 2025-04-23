#' Title Fit a Adaptive Cumulative Exposure Distributed Lag Non-Linear Model
#' (ACE-DLNM)
#'
#' @param formula the formula for the effect of exposure at time t regarding the
#'   exposure history. Use sX() for the distributed lag term. For example y ~
#'   sX(t, x).
#' @param smooth the formula for the other smooth functions. The smooths are
#'   specified by mgcv::s. For example ~ s(z). Note also that the Gaussian
#'   random effect models are accommodated, by setting s(bs = "re").
#' @param unpen.smooth the formula for the other unpenalized smooth functions.
#' @param fe.varying the formula for time-varying covariates in linear
#'   components. For example ~ z
#' @param fe.cont the formula for time-invariant covariates in linear
#'   components. For example ~ z
#' @param dat the data frame containing the variables.
#' @param maxL the maximum lag, default 14.
#' @param kw the dimension for B-spline for weight function, default 20.
#' @param kE the dimension for B-spline for ACE response-exposure-function,
#'   default 20.
#' @param interpolate whether or not to interpolate the exposures (using cubic
#'   B-spline), default \code{TRUE}.
#' @param kx.per500 the number of knots to fit exposures using B-spline. The
#'   argument works only if \code{interpolate = FALSE}.
#' @param E.min the lower bound of ACE. The lower bound is set as the minimum of
#'   the exposure if E.min is missing, since this the range we are typically
#'   interested in.
#' @param pc the point constraint for ACE response-exposure-function, i.e. f(x0)
#'   = 0. Use sum-to-zero constraint if \code{pc = NULL}. Default \code{pc =
#'   NULL}.
#' @param par.start the starting values for BFGS in the outer optimization. If
#'   the argument is missing, the starting values are the fitted values from
#'   \code{mgcv::bam}.
#' @param par.fix the values of smoothing/dispersion parameters which is fixed
#'   in the outer optimization. For example, set \code{par.fix = c(10,NA,NA)}
#'   for fixing log(theta)=10 if the prior knowledge believes there is no
#'   over-dispersion.
#' @param upper.bound the upper bound of parameters in BFGS.
#' @param CI the confidence level, default 0.95.
#' @param CI.R the number of sampling in the sampling method for CI, default
#'   1000.
#' @param CI.seed the random seed in the sampling method for CI, default 123.
#' @param delta.method whether or not to use the delta method for CI,default
#'   \code{FALSE}.
#' @param eta whether or not to report the CI for the linear predictor eta =
#'   log(mu), default \code{FALSE}.
#' @param hessian whether or not to return the numerically differentiated
#'   Hessian matrix from \code{optim}, default \code{FALSE}.
#' @param GD whether or not to use the gradient descent if BFGS fails (max gr >
#'   GD.grtol), default \code{TRUE}.
#' @param GD.grtol the tolerance of maximum gradient, default 1. The maximum
#'   numbers of steps for gradient descent is 2. But the BFGS converges in most
#'   cases.
#' @param verbose whether or not to print the progress and diagnostic
#'   information, such as the gradients and function values in inner and middle
#'   optimizations.
#'
#' @return Object of class \code{aceDLNM_fit}. S3 function \code{summary},
#'   \code{residuals}, and \code{predict}.
#' @importFrom mgcv s
#' @importFrom data.table as.data.table
#' @importFrom data.table setorder
#' @export
aceDLNM <- function(formula,
                    smooth = NULL,
                    unpen.smooth = NULL,
                    fe.varying = NULL,
                    fe.cont = NULL,
                    offset = NULL,
                    dat = NULL,
                    maxL = 14, kw = 20, kE = 20,
                    interpolate = TRUE,
                    kx.per500 = 300,
                    E.min,
                    E.max,
                    pc = NULL,
                    par.start, par.fix, upper.bound,
                    CI = 0.95,
                    CI.R = 1000, CI.seed = 123,
                    delta.method = FALSE,
                    eta = FALSE,
                    hessian = FALSE,
                    GD = TRUE,
                    GD.grtol = 1,
                    check.BFGS = FALSE,
                    verbose = TRUE) {

  if(length(formula) != 3) stop("Incorrect formula. Please set a formula for example y~sX(t,x).")
  sXobject <- eval(formula[[3]])

  ## change character columns to factor. support random effects defined by mgcv::s(bs = "re")
  chr_col <- which(sapply(dat, class) == "character")
  if(length(chr_col) >= 1) {
    for(col. in chr_col) dat[,col.] <- factor(dat[,col.])
  }



  sXdat <- dat
  ## change colnames in dataframe as x, t and y
  colnames(sXdat)[which(colnames(sXdat) == sXobject$x)] <- "x"
  colnames(sXdat)[which(colnames(sXdat) == sXobject$t)] <- "t"
  colnames(sXdat)[which(colnames(sXdat) == as.character(formula[[2]]))] <- "y"

  ## byvar
  byvar <- sXobject$by


  maxLreal <- maxL+1

  shift <- min(min(sXdat$x), 0)
  sXdat$x <- sXdat$x - shift # make exposure process non-negative
  formula.list <- list(formula = formula,
                       fe.cont = fe.cont,
                       fe.varying = fe.varying,
                       smooth = smooth)

  ### CONSTRUCTIONS #######
  ## time non-varying for group
  if(length(unique(sXdat$t)) < nrow(sXdat)) {

    if(is.null(fe.cont) & is.null(byvar)) stop("The exposure process data are duplicated at some time point. Please provide fe.cout.")

    if(!is.null(fe.cont)){
      group_name.fe.cont <- fe.cont[[2]]
      if(length(group_name.fe.cont) >= 2) {
        group_name.fe.cont <- as.character(group_name.fe.cont[2:length(group_name.fe.cont)])
      } else {
        group_name.fe.cont <- as.character(group_name.fe.cont)
      }
    } else {
      group_name.fe.cont <- NULL
    }
    if(!is.null(byvar)) {
      group_name.byvar <- byvar
    } else {
      group_name.byvar <- NULL
    }

    group_name <- c(group_name.fe.cont, group_name.byvar)

    ## split data
    sXdatlist <- split(as.data.table(sXdat), by = group_name) # use split in data.table to preserve order

  } else {
    sXdatlist <- list(sXdat)
  }

  ## set t starting at 1 and sort
  min_t <- min(sXdat$t)
  sXdatlist <- lapply(sXdatlist, function(sXdati) {
    sXdati$t <- sXdati$t - min_t + 1
    setorder(sXdati, t)
    return(data.frame(sXdati))
  })


  ## smooths from mgcv
  SSw <- mgcv::smoothCon(s(l, bs = "bs", k = kw), absorb.cons = FALSE, data = data.frame(l = seq(0, maxLreal, length.out = max(maxLreal, 500))))[[1]]
  knots_w <- SSw$knots

  SSwCon <- mgcv::smoothCon(s(l, bs = "bs", k = kw), absorb.cons = TRUE, data = data.frame(l = seq(0, maxLreal, length.out = max(maxLreal, 500))))[[1]]
  QRw <- qr(t(SSw$X) %*% as.vector(rep(1,nrow(SSw$X))))
  Qw <- qr.Q(QRw, complete = TRUE)
  Zw <- Qw[,2:ncol(Qw)]
  # max(abs(SSw$X %*% Zw - SSwCon$X))
  # max(abs(t(Zw) %*% SSw$S[[1]] %*% Zw - SSwCon$S[[1]]))
  SwCon <- SSwCon$S[[1]]


  ## absorb SwCon
  EEw <- eigen(SwCon)
  pw <- ncol(EEw$vectors)
  rw <- sum(EEw$values > .Machine$double.eps)
  mw <- pw - rw
  URw <- EEw$vectors[,1:rw]
  UFw <- EEw$vectors[,rw + (1:mw)]
  if (!is.matrix(UFw)) UFf <- cbind(UFw) # ensure UFw in a matrix
  Dpw <- as.matrix(Matrix::Diagonal(rw,1 / sqrt(Reduce(c,lapply(EEw$values,function(x) x[x>.Machine$double.eps])))))

  Uwpen <- as.matrix(cbind(URw %*% Dpw, UFw))
  SwI <- diag(1, nrow = pw, ncol = pw) # identity matrix
  SwI[rw + (1:mw), rw + (1:mw)] <- 0 # new penalty matrix
  Zwnew <- Zw %*% Uwpen
  # f(E) = Bf_matrix * Ufpen * alpha_f_repa
  # DRf <- as.matrix(Matrix::Diagonal(rf,sqrt(Reduce(c,lapply(EEf$values,function(x) x[x>.Machine$double.eps])))))
  # Ufpen %*% rbind(t(URf %*% DRf), t(UFf)) = identity matrix
  # alpha_f_repa <- rbind(t(URf %*% DRf), t(UFf)) %*% alpha_f
  # alpha_f <- Ufpen %*% alpha_f_repa
  ## in cpp, we obtain alpha_f_repa. In R, we tranfer it to alpha_f


  ### 0. distributed lag term
  ## preparations for distributed lag terms
  if(verbose) cat("****** Start prepraration ****** \n")
  DLprepare <- lapply(sXdatlist, function(sXdati){
    x <- sXdati$x
    y <- sXdati$y
    t <- sXdati$t
    Nti <- length(y) - maxL
    if((kx.per500 > 300) || (interpolate == TRUE)) {
      # if(verbose) cat("Interpolate the exposure process. \n")
      kx <- Nti+maxL + 4 + 2 # number of knots for X(t)
      # kx <- Nti+maxL + 4 # number of knots for X(t)
      interpolate <- TRUE
    } else{
      kx <- kx.per500 * ifelse(Nti < 500, 1, round(Nti/500)) # number of knots for X(t)
    }

    ### 0.1 Model exposure process
    if(verbose) start <- Sys.time()
    if(!interpolate) {
      # if(verbose) cat("Start modelling exposure process ... ", "\n")
      SSx <- mgcv::smoothCon(s(t, bs = "bs", k = kx),
                             absorb.cons = FALSE,
                             data = data.frame(t = t))[[1]] ## reparameterize it later
      knots_x <- SSx$knots
      # X <- SSx$X

      ## sum-to-zero reparameterization for SSx
      QRx <- qr(t(SSx$X) %*% as.vector(rep(1,nrow(SSx$X))))
      Qx <- qr.Q(QRx, complete = TRUE)
      Zx <- Qx[,2:ncol(Qx)]
      ## Check whether the design matrices are identical
      # X_repa <- SSx$X %*% Zx
      # max(unname(model.matrix(xt.fit))[,-1] - SSx$X %*% Zx) # SAME

      xt.fit <- mgcv::gam(x~s(t, bs = "bs", k = kx), data = data.frame(x = x, t = t))
      ## coefficients for X(t)
      alpha_x <- xt.fit$coefficients
      if(verbose) cat("modelling exposure process takes: ",
                      round(difftime(Sys.time(),start,units = 'secs'), 5),
                      "seconds. \n", "Consider setting interpolate = TRUE, if it (mgcv::gam function) takes too long. \n")
    } else {
      # knots_x <- c(rep(t[1]-1-0.2,4), t[1]-0.5-0.001, t, t[length(t)]+0.5+0.001, rep(t[length(t)]+1+0.2,4))
      knots_x <- c(t[1]-1-0.3, rep(t[1]-1-0.2,3), t[1]-0.5-0.001, t, t[length(t)]+0.5+0.001, rep(t[length(t)]+1+0.2,3), t[length(t)]+1+0.3)
      # knots_x <- c(rep(t[1]-1.5,3),t[1]-0.99, t, t[length(t)]+0.99, rep(t[length(t)]+1.5,3))
      X <- splines::splineDesign(knots = knots_x, x = knots_x, outer.ok = TRUE)
      ## interpolate = TRUE
      ## set points for boundary and auxiliary boundary
      Xsparse <- as(X, "dgCMatrix")
      alpha_x <- Interpolate(Xsparse, c(rep(0,4), x[1], x, x[length(x)], rep(0,4)))
      # alpha_x <- Interpolate(Xsparse, c(rep(0,4),x,rep(0,4)))
      xt.fit <- "interpolate"
      # if(verbose) cat("modelling exposure process takes: ",
      #                 round(difftime(Sys.time(),start,units = 'secs'), 5),
      #                 "seconds. \n")
    }


    ### 0.2 Integration
    removed.t <- t[1:maxL]
    t <- t[-(1:maxL)] # delete the first maxL days
    y <- y[-(1:maxL)] # delete the first maxL days
    x <- x[-(1:maxL)] # delete the first maxL days

    ### integration
    # if(verbose){
    #   cat("Start integration ... ", "\n")
    #   start <- Sys.time()
    # }
    if(!interpolate) {
      integral <- Integral(knots_x, knots_w, kx, kw, maxLreal, Zx, Zwnew, t+0.5, alpha_x, FALSE)
    } else {
      integral <- Integral_interpolate(knots_x, knots_w, kx, kw, maxLreal, Zwnew, t+0.5, alpha_x, FALSE)
    }
    # if(verbose) cat("integration takes: ",
    #                 round(difftime(Sys.time(),start,units = 'secs'), 5), "seconds. \n")

    ## linear predictor s(t) = f(\int w(l) X(t-l) dl) = f (B_inner(t) %*% alpha_w)
    ## where B_inner(t) = alpha_x %*% D, dim(D) = c(kx, kw), D_{p,q} = \int b_{xp}(t-l)b_{wq}(l) dl.
    ## b_{xp} is the p-th basis function for X(t)
    ## Stack B_inner(t) for t = (1:Nt)+40, we have B_inner. dim(B_inner) = c(Nt, kw)
    ## See Section X.X in Paper ...
    B_inner <- integral$AlphaxD

    ## For constraint 1: \int w(l)^2 dl = 1
    ## equivalent to w^T %*% Dw %*% w = 1
    Dw <- integral$Dw

    E.maxi <- ceiling(max(integral$Xt2))
    E.mini <- floor(min(x))

    return(list(B_inner = B_inner,
                Dw = Dw,
                E.max = E.maxi,
                E.min = E.mini,
                removed.t = removed.t
                ))
  })


  B_inner <- do.call("rbind", lapply(DLprepare, "[[", "B_inner"))

  Dw <- DLprepare[[1]]$Dw
  if(missingArg(E.max)) E.max <- do.call("max", lapply(DLprepare, "[[", "E.max"))
  if(missingArg(E.min)) E.min <- do.call("min", lapply(DLprepare, "[[", "E.min"))
  removed.t <- lapply(DLprepare, "[[", "removed.t")




  ## remove the starting time points
  sXdatlist <- mapply(function(sXdati, removed.ti) return(sXdati[-which(sXdati$t %in% removed.ti),]),
                      sXdatlist, removed.t, SIMPLIFY = FALSE)
  if(verbose) cat("****** Finish preparation ****** \n")

  sXdat <- do.call("rbind", sXdatlist)

  rm(DLprepare) # release memory
  rm(sXdatlist) # release memory

  ## 1. time-varying fixed effects
  if(!is.null(fe.varying)) {
    Xlin <- stats::model.matrix(fe.varying,data=sXdat)[,-1,drop=F]
  } else {
    Xlin <- matrix(1, nrow = nrow(sXdat))
  }
  if(!is.null(unpen.smooth)) {
    unpen.smooth <- lapply(lapply(attr(terms(unpen.smooth),"term.labels"), function(text) parse(text = text)), eval)
    unpen.SS <- lapply(lapply(unpen.smooth,mgcv::smoothCon,data=sXdat,absorb.cons = TRUE),'[[',1) ## construct unpenalized smooth terms
    unpen.Xlist <- lapply(unpen.SS,'[[','X')
    unpen.X <- Reduce(cbind,unpen.Xlist) ## design matrix
    Xlin <- cbind(Xlin, unpen.X)
  }
  ### code following https://github.com/awstringer1/mam/blob/master/R/mam.R
  ## start following ##
  ## 2.2 smooth term
  numsmooth <- 0 # initialize
  if(!is.null(smooth)){
    smooth <- lapply(lapply(attr(terms(smooth),"term.labels"), function(text) parse(text = text)), eval)

    SS <- lapply(lapply(smooth,mgcv::smoothCon,data=sXdat,absorb.cons = TRUE),'[[',1) ## construct smooth terms
    numsmooth <- length(smooth) # Number of smooth terms

    EE <- lapply(lapply(lapply(SS,'[[','S'),'[[',1),eigen) ## eigen decomposition for penalty matrix

    p <- sapply(lapply(EE,'[[','vectors'),ncol) ## dimension of penalty matrix
    r <- sapply(lapply(EE,'[[','values'),function(x) sum(x>.Machine$double.eps)) ## rank of penalty matrix
    m <- p-r ## dim of null space (minus intercept)
    URlist <- mapply(function(x,y) x[ ,1:y],lapply(EE,'[[','vectors'),r,SIMPLIFY = FALSE)
    UFlist <- mapply(
      function(x,y,z) {
        if (y<z) return(x[ ,(1+y):z])
        newsparsemat(z,z)
      },lapply(EE,'[[','vectors'),r,p,SIMPLIFY = FALSE)
    URlist <- lapply(URlist,cbind) # Ensure they stay matrices
    UFlist <- lapply(UFlist,cbind) # Ensure they stay matrices

    UR <- Matrix::bdiag(URlist)
    UF <- Matrix::bdiag(UFlist)
    # if m=1 UF gets coerced to numeric
    if (!is.matrix(UF)) UF <- cbind(UF)

    Dpi <- Matrix::Diagonal(sum(r),1 / sqrt(Reduce(c,lapply(lapply(EE,'[[','values'),function(x) x[x>.Machine$double.eps]))))

    Xlist <- lapply(SS,'[[','X')
    X <- Reduce(cbind,Xlist) ## design matrix

    Xrand <- as.matrix(X %*% UR %*% Dpi) ## reparametrized
    Xfix <- as.matrix(X %*% UF)
    dups <- !base::duplicated(t(Xfix)) & apply(Xfix,2,function(x) !all(x==0)) # Remove the duplicated intercepts
    if (length(dups) > 1) Xfix <- Xfix[ ,which(dups)]

    model.choice = "with.smooth"

    rm(Xlist)
    rm(X)

  } else{
    model.choice = "without.smooth"
  }


  # 2.3 add the intercept
  if(exists("Xfix")) {
    Xfix <- cbind(Xlin,Xfix) ## linear effect + unpenalized columns
  } else {
    Xfix <- Xlin
  }
  if (any(apply(Xfix,2,function(x) all(x==1)))) Xfix <- Xfix[,-(apply(Xfix,2,function(x) all(x==1)))]

  if (!is.null(fe.cont)){
    Xgroup <- stats::model.matrix(fe.cont,data=sXdat)[,-1,drop=FALSE]
    if (any(apply(Xgroup,2,function(x) all(x==1)))) Xgroup <- Xgroup[,-(apply(Xgroup,2,function(x) all(x==1)))] # remove the intercept
    Xfix <- cbind(1, Xgroup, Xfix) # add the intercept to the first column
  } else {
    Xfix <- cbind(1,Xfix) # add the intercept to the first column
  }


  ### END following https://github.com/awstringer1/mam/blob/master/R/mam.R #######

  ## 2.4 offset
  if(!is.null(offset)) {
    Xoffset.original <- as.vector(stats::model.matrix(offset,data=sXdat)[,-1,drop=F])
    Xoffset.original.min <- min(Xoffset.original)
    Xoffset <- Xoffset.original - Xoffset.original.min
  } else {
    Xoffset.original <- as.vector(rep(0, nrow(sXdat)))
    Xoffset.original.min <- 0
    Xoffset <- as.vector(rep(0, nrow(sXdat)))
  }




  ### Model Fitting

  N <- nrow(sXdat)
  x <- sXdat$x
  y <- sXdat$y
  t <- sXdat$t

  if(missingArg(upper.bound)) upper.bound <- c(10, 15, 12)
  lower.bound <- c(-2,-8,-8)


  SSf <- mgcv::smoothCon(s(E, bs = "bs", k = kE), absorb.cons = FALSE,
                         data = data.frame(E = seq(E.min, E.max, length.out = N)))[[1]]
  if(is.null(pc)) {
    ## sum-to-zero constraint reparameterization for SSf
    SSfCon <- mgcv::smoothCon(s(E, bs = "bs", k = kE), absorb.cons = TRUE,
                         data = data.frame(E = seq(E.min, E.max, length.out = N)))[[1]]

    V <- t(SSf$X) %*% as.vector(rep(1,nrow(SSf$X)))

  } else {
    ## point constraint reparameterization for SSf
    # SSfCon <- mgcv::smoothCon(s(E, bs = "bs", k = kE, pc = pc), absorb.cons = TRUE,
    #                      data = data.frame(E = seq(E.min, E.max, length.out = N)))[[1]]
    ## NOTE: (s(E, bs = "bs", k = kE, pc = pc) will give a different reparametrization.
    # max(abs(SSf$X %*% Zf - SSfCon$X)) is not zero!
    # We build the design matrix and penalty matrix by hand

    V <- matrix(Bsplinevec2(pc, SSf$knots, 4), ncol = 1)

  }
  # V_tmp <- V
  # V_tmp[1] <- V_tmp[1] - sqrt(sum(V^2))
  # H <- diag(1, kE) - 2 * (V_tmp %*% t(V_tmp)) / sum(V_tmp^2)
  # Zf <- H[,-1] # same results
  QRf <- qr(V)
  Qf <- qr.Q(QRf, complete = TRUE)
  Zf <- Qf[,2:ncol(Qf)]
  ## Check whether the design matrices are identical
  # max(abs(SSf$X %*% Zf - SSfCon$X))
  # max(abs(t(Zf) %*% SSf$S[[1]] %*% Zf - SSfCon$S[[1]]))


  if(is.null(pc)) {
    SfCon <- SSfCon$S[[1]]
  } else {
    ## NOTE: (s(E, bs = "bs", k = kE, pc = pc) will give a different reparametrization.
    # max(abs(SSf$X %*% Zf - SSfCon$X)) is not zero!
    # We build the design matrix and penalty matrix by hand
    SfCon <- t(Zf) %*% SSf$S[[1]] %*% Zf
  }



  ## absorb SfCon
  EEf <- eigen(SfCon)
  # EEf$values
  # EEf$vectors
  pf <- ncol(EEf$vectors)
  rf <- sum(EEf$values > .Machine$double.eps)
  mf <- pf - rf
  URf <- EEf$vectors[,1:rf]
  UFf <- EEf$vectors[,rf + (1:mf)]
  if (!is.matrix(UFf)) UFf <- cbind(UFf) # ensure UFf in a matrix
  Dpf <- as.matrix(Matrix::Diagonal(rf,1 / sqrt(Reduce(c,lapply(EEf$values,function(x) x[x>.Machine$double.eps])))))

  Ufpen <- as.matrix(cbind(URf %*% Dpf, UFf))
  SfI <- diag(1, nrow = pf, ncol = pf) # identity matrix
  SfI[rf + (1:mf), rf + (1:mf)] <- 0 # new penalty matrix

  # f(E) = Bf_matrix * Ufpen * alpha_f_repa
  # DRf <- as.matrix(Matrix::Diagonal(rf,sqrt(Reduce(c,lapply(EEf$values,function(x) x[x>.Machine$double.eps])))))
  # Ufpen %*% rbind(t(URf %*% DRf), t(UFf)) = identity matrix
  # alpha_f_repa <- rbind(t(URf %*% DRf), t(UFf)) %*% alpha_f
  # alpha_f <- Ufpen %*% alpha_f_repa
  ## in cpp, we obtain alpha_f_repa. In R, we tranfer it to alpha_f
  Zfnew <- Zf %*% Ufpen



  ## set starting values
  phi.init.default  <- rep(0,kw-1)

  alpha_f.init.default <- rep(0,kE-1)

  betaF.init.default <- rep(0, ncol(Xfix))

  ## A list to store the optimization results at given smoothing log_theta.given, log_smoothing_f.given, log_smoothing_w.given.
  if(model.choice == "with.smooth") {
    betaR.init.default <- rep(0,ncol(Xrand))
    LAMLenv <- list(par = NULL,
                    fn = NULL,
                    gr = NULL,
                    mod = list(phi.mod = phi.init.default, alpha_f.mod = alpha_f.init.default,
                              betaF.mod = betaF.init.default,
                              betaR.mod = betaR.init.default))
  } else {
    LAMLenv <- list(par = NULL,
                fn = NULL,
                gr = NULL,
                mod = list(phi.mod = phi.init.default, alpha_f.mod = alpha_f.init.default,
                            betaF.mod = betaF.init.default))
  }

  ## upper bound
  if(model.choice == "with.smooth") {
    upper.bound <- c(upper.bound, rep(15, numsmooth))
    lower.bound <- c(lower.bound, rep(-8, numsmooth))
  }


  ## profile likelihood
  getLAML <-  function(par) {
    if(verbose) cat("parameters:", par, "\n")
    log_theta.given <- par[1]
    log_smoothing_f.given <- par[2]
    log_smoothing_w.given <- par[3]
    if(model.choice == "with.smooth") logsmoothing.given <- par[-(1:3)]
    ## Inner opt
    if(is.null(LAMLenv$par) | !all(par == LAMLenv$par)) {
      ## Inner opt
      phi.init <- LAMLenv$mod$phi.mod
      if(any(is.nan(phi.init))) phi.init <- phi.init.default
      alpha_f.init <- LAMLenv$mod$alpha_f.mod
      if(any(is.nan(alpha_f.init))) alpha_f.init <- alpha_f.init.default
      betaF.init <- LAMLenv$mod$betaF.mod
      if(any(is.nan(betaF.init))) betaF.init <- betaF.init.default
      if(model.choice == "with.smooth"){
        betaR.init <- LAMLenv$mod$betaR.mod
        if(any(is.nan(betaF.init))) betaR.init <- betaR.init.default
        LAML.results <- aceDLNMopt(y, B_inner, SSf$knots, SwI, SfI, Dw,
                                      Xrand, Xfix, Zfnew, Xoffset, r,
                                      alpha_f.init,
                                      phi.init,
                                      log_theta.given, log_smoothing_f.given, log_smoothing_w.given,
                                      betaR.init, betaF.init, logsmoothing.given,
                                      verbose)
        LAMLenv$par <<- par
        LAMLenv$mod$phi.mod <<- LAML.results$phi.mod
        LAMLenv$mod$alpha_f.mod <<- LAML.results$alpha_f.mod
        LAMLenv$mod$betaF.mod <<- LAML.results$betaF.mod
        LAMLenv$mod$betaR.mod <<- LAML.results$betaR.mod
        LAMLenv$fn <<- LAML.results$LAML.fn
        LAMLenv$gr <<- LAML.results$LAML.gradient
      } else {
        LAML.results <- aceDLNMopt_nosmooth(y, B_inner, SSf$knots, SwI, SfI, Dw,
                                               Xfix, Zfnew, Xoffset,
                                               alpha_f.init,
                                               phi.init,
                                               log_theta.given, log_smoothing_f.given, log_smoothing_w.given,
                                               betaF.init,
                                               verbose)
        LAMLenv$par <<- par
        LAMLenv$mod$phi.mod <<- LAML.results$phi.mod
        LAMLenv$mod$alpha_f.mod <<- LAML.results$alpha_f.mod
        LAMLenv$mod$betaF.mod <<- LAML.results$betaF.mod
        LAMLenv$fn <<- LAML.results$LAML.fn
        LAMLenv$gr <<- LAML.results$LAML.gradient
      }

    }
  }



  LAML.fn <- function(par){
    getLAML(par)
    if(verbose) cat("LAML: ", LAMLenv$fn, "\n\n")
    LAMLenv$fn
  }


  ## gradient of LAML
  LAML.gr <- function(par){
    getLAML(par)
    if(verbose) cat("Gradient: ", LAMLenv$gr, "\n\n")
    LAMLenv$gr
  }



  if(missingArg(par.fix)){
    par.fix <- rep(NA, 3)
    if(model.choice == "with.smooth") par.fix <- rep(NA, 3+numsmooth)
  }
  par.fix.id <- !sapply(par.fix, is.na)


  ### model fitting
  if(verbose) {
    cat("Start model fitting... \n")
  }
  start <- Sys.time()


  if(missingArg(par.start)) {
    par.start <- c(6, 6, 6)
    if(model.choice == "with.smooth") par.start <- c(par.start, rep(6, numsmooth))
    tryCatch(
      expr = {
        ## fit a GAM using bam()
        y.mean <- mean(sXdat$y, na.rm = TRUE)

        formula.mgcv <- paste0("y~s(x, bs='bs', k = ", kE, ")")

        formula.list.mgcv <- Filter(Negate(is.null), formula.list[names(formula.list) %in% c("smooth","fe.varying")])
        if(length(formula.list.mgcv) > 0) {
          formula.other.mgcv <- paste(formula.list.mgcv, collapse = "+")
          formula.other.mgcv <- gsub("~", "", formula.other.mgcv)
          formula.other.mgcv <- gsub("\"", "'", formula.other.mgcv)
          formula.mgcv <- paste0(formula.mgcv, "+", formula.other.mgcv)
        }
        formula.mgcv <- as.formula(formula.mgcv)
        utils::capture.output(
          mod.mgcv <- mgcv::bam(formula.mgcv, data = sXdat, family = nb(), discrete = TRUE)
        )
        par.start[1] <- max(family(mod.mgcv)$getTheta(), log(2*y.mean)) # choose a larger theta if theta from bam() is smaller than 2*mean(y). Var = mu + mu^2/theta. i.e. init. var = 1.5*mu
        par.start[2] <- log(mod.mgcv$sp)[1]
        par.start[3] <- 6
        if(length(par.start) > 3) par.start[4:length(par.start)] <- log(mod.mgcv$sp)[-1]
        if(par.start[1] > 7) par.start[1] <- 7 # do not set a huge log(theta)
        if(par.start[2] > 11) par.start[2] <- 11 # do not set a huge log(smoothing_f)

        for (i in 1:length(par.start)) {
          if((par.start[i] > (upper.bound[i] - 1)) | (par.start[i] < lower.bound[i])) par.start[i] <- 7
        }
        if(verbose) cat("Use results from mgcv::bam with same-day exposure as the initial guess for BFGS. \n ")
        rm(mod.mgcv)
      },
      # warning = function(w) {
      #   if(verbose) {
      #     cat("some warnings from bam() for setting the initial guess. Its influence on the model fitting is ignorable, but you can set par.start by hand. \n")
      #     warning(w)
      #   }
      # },
      error = function(e) {
        cat("use default starting values for BFGS ... \n")
      }
    )

  } else {
    for (i in 1:length(par.start)) {
      if((par.start[i] > (upper.bound[i] - 1)) | (par.start[i] < lower.bound[i])) par.start[i] <- 7
    }
  }
  success <- FALSE
  itrtry <- 0
  while((!success) & (itrtry < 4)) {
    itrtry <- itrtry + 1
    tryCatch(
      expr = {
        opt.LAML <- optim(par.start[!par.fix.id],
                        fn = function(par.){
                                            par.fn <- par.fix
                                            par.fn[!par.fix.id] <- par.
                                            return(LAML.fn(par.fn))
                                            }, # objective function
                        gr = function(par.){
                                            par.fn <- par.fix
                                            par.fn[!par.fix.id] <- par.
                                            return(LAML.gr(par.fn)[!par.fix.id])
                                            }, # gradient function
                        method = "L-BFGS-B",
                        lower = lower.bound[!par.fix.id],
                        upper = upper.bound[!par.fix.id],
                        control = list(trace = verbose),
                        hessian = hessian
                        )

        if(all(opt.LAML$par == par.start[!par.fix.id])){
          ## get stuck in the starting values
          par.start[!par.fix.id] <- par.start[!par.fix.id]/2
          opt.LAML <- optim(par.start[!par.fix.id],
                            fn = function(par.){
                                                par.fn <- par.fix
                                                par.fn[!par.fix.id] <- par.
                                                return(LAML.fn(par.fn))
                                                }, # objective function
                            gr = function(par.){
                                                par.fn <- par.fix
                                                par.fn[!par.fix.id] <- par.
                                                return(LAML.gr(par.fn)[!par.fix.id])
                                                }, # gradient function
                            method = "L-BFGS-B",
                            lower = lower.bound[!par.fix.id],
                            upper = upper.bound[!par.fix.id],
                            control = list(trace = verbose),
                            hessian = hessian)
        }
        success <- TRUE
      },
      error = function(e) {
        cat("resetting starting values... \n")
      }
    )
    if((!success)) {
      ## reset starting values for log_theta and log_smoothing_w
      if(!par.fix.id[1]) par.start[1] <- c(0,0,3,3)[itrtry]
      if(!par.fix.id[3]) par.start[3] <- c(8,2,10,2)[itrtry]
      if(par.fix.id[1] & par.fix.id[3]) stop("Error. Please try other par.start or par.fix")
      LAMLenv <- list(par = NULL,
                      fn = NULL,
                      gr = NULL,
                      mod = list(phi.mod = phi.init.default, alpha_f.mod = alpha_f.init.default,
                                  betaF.mod = betaF.init.default))
    }
  }
  if(GD) {
    # do gradient descent step for gradient of f and w
    GDitermax <- 2
    GDiter <- 0
    while((max(abs(LAMLenv$gr[2:3])) > GD.grtol) & (GDiter < GDitermax)) {
      GDiter <- GDiter + 1
      if(verbose) cat("Gradient descent step. Attempt ", GDiter, "/", GDitermax, " \n")
      par.restart <- LAMLenv$par
      GDstep <- 1*LAMLenv$gr[2:3]
      par.restart[2:3] <- par.restart[2:3] - GDstep # update f and w
      for (i in 2:3) {
        # satisfy box constraint
        if((par.restart[i] >= upper.bound[i]-1)) par.restart[i] = upper.bound[i]-1
        # reset par if the step size is too large
        if((par.restart[i] <= max(3,lower.bound[i]+1))) par.restart[i] = max(3,lower.bound[i]+1)
      }
      opt.LAML <- optim(par.restart[!par.fix.id],
                        fn = function(par.){
                          par.fn <- par.fix
                          par.fn[!par.fix.id] <- par.
                          return(LAML.fn(par.fn))
                        }, # objective function
                        gr = function(par.){
                          par.fn <- par.fix
                          par.fn[!par.fix.id] <- par.
                          return(LAML.gr(par.fn)[!par.fix.id])
                        }, # gradient function
                        method = "L-BFGS-B",
                        lower = lower.bound[!par.fix.id],
                        upper = upper.bound[!par.fix.id],
                        control = list(trace = verbose),
                        hessian = hessian)
    }
  }
  if(max(abs(LAMLenv$gr[2:3])) > GD.grtol) if(verbose) cat("the gradient of LAML is large.\n")
  if((!success) & itrtry >= 4) stop("Error. Please try other par.start")

  ## reset starting value once if log-lambda reaches the boundary.
  # retrybound <- FALSE
  # if((!par.fix.id[2]) & ((opt.LAML$par[2] >= upper.bound[2]-0.1) & LAML.gr(opt.LAML$par)[2] < -1e-2)) {
  #   cat("log-lambda-f reaches the boundary. Resetting starting values.\n")
  #   par.start <- opt.LAML$par
  #   par.start[2] <- 8
  #   retrybound <- TRUE
  # }
  # if((!par.fix.id[3]) & ((opt.LAML$par[3] >= upper.bound[3]-0.1) & LAML.gr(opt.LAML$par)[3] < -1e-2)) {
  #   cat("log-lambda-w reaches the boundary. Resetting starting values.\n")
  #   par.start <- opt.LAML$par
  #   par.start[3] <- 8
  #   retrybound <- TRUE
  # }
  # if(retrybound) {
  #   opt.LAML <- optim(par.start[!par.fix.id],
  #                         fn = function(par.){
  #                                             par.fn <- par.fix
  #                                             par.fn[!par.fix.id] <- par.
  #                                             return(LAML.fn(par.fn))
  #                                             }, # objective function
  #                         gr = function(par.){
  #                                             par.fn <- par.fix
  #                                             par.fn[!par.fix.id] <- par.
  #                                             return(LAML.gr(par.fn)[!par.fix.id])
  #                                             }, # gradient function
  #                         method = "L-BFGS-B",
  #                         lower = lower.bound[!par.fix.id],
  #                         upper = upper.bound[!par.fix.id],
  #                         control = list(trace = verbose),
  #                         hessian = FALSE
  #                         )
  # }

  opttime <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  if(verbose){
    cat("Finished model fitting. It took ", round(opttime,5), " seconds.\n", sep = "")
  }


  ### obtain estimate ########
  ## point estimate
  log_theta.opt <- LAMLenv$par[1]
  log_smoothing_f.opt <- LAMLenv$par[2]
  log_smoothing_w.opt <- LAMLenv$par[3]
  alpha_f.opt <- LAMLenv$mod$alpha_f.mod
  phi.opt <- LAMLenv$mod$phi.mod
  phi_long <- as.vector(c(1, phi.opt))
  alpha_w.opt <- phi_long / as.numeric(sqrt(t(phi_long) %*% Dw %*% phi_long))
  betaF.opt <- LAMLenv$mod$betaF.mod

  out = list(opt = opt.LAML,
             env = LAMLenv,
             CI.sample = NULL,
             point = list(log_theta = log_theta.opt,
                          log_smoothing_f = log_smoothing_f.opt,
                          log_smoothing_w = log_smoothing_w.opt,
                          alpha_f = alpha_f.opt,
                          alpha_w = alpha_w.opt,
                          phi = phi.opt,
                          betaF = betaF.opt),
             smooth = list(wl = SSwCon, fE = NULL))
  if(is.null(pc)) {
    out$smooth$fE <- SSfCon
  } else {
    ## NOTE: (s(E, bs = "bs", k = kE, pc = pc) will give a different reparametrization.
    # max(abs(SSf$X %*% Zf - SSfCon$X)) is not zero!
    # We build the design matrix and penalty matrix by hand
    out$smooth$fE <- SSf
  }
  if(model.choice == "with.smooth") {
    log_smoothing.opt <- LAMLenv$par[-(1:3)]
    betaR.opt <- LAMLenv$mod$betaR.mod
    out$point$betaR = betaR.opt
    out$point$log_smoothing = log_smoothing.opt
  }

  ## CI
  if(verbose) {
    cat("Start sampling for CI ... \n")
    start <- Sys.time()
  }
  if(model.choice == "with.smooth") {
    sampled <- aceDLNMCI(y, B_inner, SSf$knots, SwI, SfI, Dw,
                            Xrand, Xfix, Zfnew, Xoffset, r,
                            LAMLenv$mod$alpha_f.mod,
                            phi.opt,
                            log_theta.opt, log_smoothing_f.opt, log_smoothing_w.opt,
                            betaR.opt, betaF.opt, log_smoothing.opt,
                            CI.R, CI.seed,
                            eta, delta.method, verbose)
  } else {
    sampled <- aceDLNMCI_nosmooth(y, B_inner, SSf$knots, SwI, SfI, Dw,
                                      Xfix, Zfnew, Xoffset,
                                     LAMLenv$mod$alpha_f.mod,
                                      phi.opt,
                                      log_theta.opt, log_smoothing_f.opt, log_smoothing_w.opt,
                                      betaF.opt,
                                      CI.R, CI.seed,
                                      eta, delta.method, verbose)
  }

  E <- B_inner %*% out$point$alpha_w
  Bf <- sapply(E, function(Ei) Bsplinevec2Con(Ei, SSf$knots, 4, Zfnew))
  eta1 <- as.vector(t(Bf) %*% out$point$alpha_f)
  eta2 <- as.vector(Xfix %*% out$point$betaF)
  if(exists("Xrand")) eta2 <- eta2 + as.vector(Xrand %*% out$point$betaR)
  out$eta.est <- eta1 + eta2 + Xoffset

  eta_E.est <- eta1 + mean(eta2)
  eta_other.est <- eta2 - mean(eta2)

  out$eta_E = data.frame(est = eta_E.est)
  out$eta_other = data.frame(est = eta_other.est)

  if(eta) {
    eta_E.CI <- apply(sampled$eta_E_sample_mat, 2, function(row.) quantile(row., c((1-CI)/2, 1-(1-CI)/2)))
    eta_other.CI <- apply(sampled$eta_other_sample_mat, 2, function(row.) quantile(row., c((1-CI)/2, 1-(1-CI)/2)))
    out$eta_E = data.frame(est = eta_E.est,
                           ll = eta_E.CI[1,],
                           ul = eta_E.CI[2,])
    out$eta_other = data.frame(est = eta_other.est,
                               ll = eta_other.CI[1,],
                               ul = eta_other.CI[2,])
  }
  if(verbose){
    runningtime <- as.numeric(difftime(Sys.time(), start, units = "secs"))
    cat("Finished sampling for CI. It took ", round(runningtime,5), " seconds.\n", sep = "")
  }


  out$CI.sample <- sampled
  out$CI <- CI
  ## return formula
  out$formula <- formula.list

  ## return data
  out$data <- list(maxL = maxL,
                   B_inner = B_inner,
                   Dw = Dw,
                   E.max = E.max,
                   E.min = E.min,
                   kE = kE,
                   kw = kw,
                   y = y,
                   x = x,
                   t = t,
                   pc = pc # point constraint
                   )
  out$data$Xfix = Xfix
  out$data$knots_f = SSf$knots
  out$data$knots_w = knots_w
  out$data$Zf = Zf
  out$data$Ufpen = Ufpen
  out$data$Zw = Zw
  out$data$Uwpen = Uwpen
  out$data$kx.per500 = kx.per500
  out$data$shift = shift
  out$data$offset = list(Xoffset.original = Xoffset.original,
                         Xoffset.original.min = Xoffset.original.min,
                         Xoffset = Xoffset)
  if(model.choice == "with.smooth") {
    out$data$Xrand = Xrand
    out$data$UR = UR
    out$data$UF = UF
    out$data$Dpi = Dpi
    out$data$r = r
    out$data$m = m
    out$smooth$others = SS
  }
  # if(!is.null(unpen.smooth)) {
  #   out$smooth$unpen.smooth = unpen.SS
  # }
  out$inputdata <- dat
  out$modeldata <- sXdat
  out$opttime <- opttime
  out$penalty <- "second"
  out$interpolate <- interpolate

  ## check convergence
  out$eigval_Hessian_inner <- eigen(sampled$Hessian_inner)$values
  ## check convergence of BFGS
  if(check.BFGS) {
    if(is.null(opt.LAML$hessian)){
      if(verbose) cat("start obtain Hessian matrix. \n")
      par.start[!par.fix.id] <- opt.LAML$par
      H.LAML <- optimHess(par.start[!par.fix.id],
                        fn = function(par.){
                                            par.fn <- par.fix
                                            par.fn[!par.fix.id] <- par.
                                            return(LAML.fn(par.fn))
                                            }, # objective function
                        gr = function(par.){
                                            par.fn <- par.fix
                                            par.fn[!par.fix.id] <- par.
                                            return(LAML.gr(par.fn)[!par.fix.id])
                                            }, # gradient function
                        control = list(trace = verbose)
                        )
      out$hessian <- H.LAML

      e.H <- eigen(H.LAML)

      out$eigen.hessian <- e.H

      evals <- e.H$values

      if(abs(prod(evals)) < 1e-3) evals <- evals + sqrt(sum((out$env$gr)^2))

      suggest.step <- e.H$vectors %*% diag(1/abs(evals)) %*% t(e.H$vectors) %*% out$env$gr

      out$suggest.step <- suggest.step
      if(verbose) cat("finish obtain Hessian matrix. \n")
      if( sqrt(sum((out$env$gr)^2)) > 0.2) cat("BFGS might not converge. You could try other par.start and rerun the model.")
    }

    if(min(out$eigval_Hessian_inner) < 0.01) {
      warning("The optimization algorithm might not converge. Try rerunning the model with par.start = ", c(out$opt$par - out$suggest.step), "\n")
    }

  } else {
    if(min(out$eigval_Hessian_inner) < 0.01) {
      warning("The optimization algorithm might not converge. Try rerunning the model with par.start = ", c(out$opt$par), "\n")
    }
  }

  structure(out, class = "aceDLNM_fit") # S3 class
}
