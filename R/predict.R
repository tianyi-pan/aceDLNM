#' Title Obtain predictions from the fitted model by aceDLNM
#'
#' @param object object of class \code{aceDLNM_fit}.
#' @param dat the data frame containing the variables.
#' @param CI.fit whether or not to report the CI, default \code{TRUE}. 
#' @param ...
#'
#' @return
#' @export
predict.aceDLNM_fit <- function(object, dat, CI.fit = TRUE, ...) {

  if(missingArg(dat)) {
    if (!is.null(object$eta)) {
      return(object$eta)
    } else {
      dat <- object$inputdata
    }
  }

  smooth = object$formula$smooth
  formula = object$formula$formula
  fe.cont = object$formula$fe.cont
  fe.varying = object$formula$fe.varying
  kw = object$data$kw
  kE = object$data$kE
  CI = object$CI
  maxL = object$data$maxL
  maxLreal = maxL+1

  dat[, as.character(formula[[2]])] <- 0 # create an empty y

  sXobject <- eval(formula[[3]])

  sXdat <- dat

  sXdat$x <- sXdat$x - object$data$shift # make exposure process non-negative

  ## change colnames in dataframe as x, t and y
  colnames(sXdat)[which(colnames(sXdat) == sXobject$x)] <- "x"
  colnames(sXdat)[which(colnames(sXdat) == sXobject$t)] <- "t"
  colnames(sXdat)[which(colnames(sXdat) == as.character(formula[[2]]))] <- "y"


  ### CONSTRUCTIONS #######
  ## time non-varying for group
  if (length(unique(sXdat$t)) < nrow(sXdat)) {
    if (is.null(fe.cont))
      stop("The exposure process data are duplicated at some time point. Please provide fe.cout.")
    # group <- stats::model.matrix(fe.cont,data=sXdat)[,-1,drop=FALSE]
    ## remove the intercept
    # if (any(apply(group,2,function(x) all(x==1)))) group <- group[,-(apply(group,2,function(x) all(x==1)))]
    group_name <- fe.cont[[2]]
    if (length(group_name) >= 2) {
      group_name <- as.character(group_name[2:length(group_name)])
    } else {
      group_name <- as.character(group_name)
    }

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

  SSwCon <- object$smooth$wl
  knots_w <- SSwCon$knots
  Zw <- object$data$Zw
  Zf <- object$data$Zf

  knots_f <- object$data$knots_f
  interpolate <- object$interpolate
  kx.per500 <- object$data$kx.per500

  ### 0. distributed lag term
  ## preparations for distributed lag terms
  DLprepare <- lapply(sXdatlist, function(sXdati) {
    x <- sXdati$x
    t <- sXdati$t
    Nti <- length(x) - maxL
    if ((kx.per500 > 300) || (interpolate == TRUE)) {
      kx <- Nti + maxL + 4 + 2 # number of knots for X(t)
      interpolate <- TRUE
    } else{
      kx <- kx.per500 * ifelse(Nti < 500, 1, round(Nti / 500)) # number of knots for X(t)
    }

    ### 0.1 Model exposure process
    if (!interpolate) {
      SSx <- mgcv::smoothCon(s(t, bs = "bs", k = kx),
                             absorb.cons = FALSE,
                             data = data.frame(t = t))[[1]] ## reparameterize it later
      knots_x <- SSx$knots
      # X <- SSx$X

      ## sum-to-zero reparameterization for SSx
      QRx <- qr(t(SSx$X) %*% as.vector(rep(1, nrow(SSx$X))))
      Qx <- qr.Q(QRx, complete = TRUE)
      Zx <- Qx[, 2:ncol(Qx)]
      ## Check whether the design matrices are identical
      # X_repa <- SSx$X %*% Zx
      # max(unname(model.matrix(xt.fit))[,-1] - SSx$X %*% Zx) # SAME

      xt.fit <- mgcv::gam(x ~ s(t, bs = "bs", k = kx), data = data.frame(x = x, t = t))
      ## coefficients for X(t)
      alpha_x <- xt.fit$coefficients
    } else {
      knots_x <- c(rep(t[1] - 1 - 0.2, 3),
                   t[1] - 1 - 0.2,
                   t[1] - 1 - 0.001,
                   t,
                   t[length(t)] + 0.5 + 0.001,
                   t[length(t)] + 0.5 + 0.2,
                   rep(t[length(t)] + 0.5 + 0.2, 3))
      X <- splines::splineDesign(knots = knots_x,
                                 x = knots_x,
                                 outer.ok = TRUE)
      ## interpolate = TRUE
      ## set points for boundary and auxiliary boundary
      Xsparse <- as(X, "dgCMatrix")
      alpha_x <- Interpolate(Xsparse, c(rep(0, 5), x, rep(0, 5)))
      xt.fit <- "interpolate"
    }


    ### 0.2 Integration
    removed.t <- t[1:maxL]
    t <- t[-(1:maxL)] # delete the first maxL days
    x <- x[-(1:maxL)] # delete the first maxL days

    ### integration
    if (!interpolate) {
      integral <- Integral(knots_x, knots_w, kx, kw, maxLreal, Zx, Zw, t+0.5, alpha_x, TRUE)
    } else {
      integral <- Integral_interpolate(knots_x, knots_w, kx, kw, maxLreal, Zw, t+0.5, alpha_x, TRUE)
    }

    B_inner <- integral$AlphaxD

    return(list(B_inner = B_inner, removed.t = removed.t))
  })


  B_inner <- do.call("rbind", lapply(DLprepare, "[[", "B_inner"))
  removed.t <- lapply(DLprepare, "[[", "removed.t")
  ## remove the starting time points
  sXdatlist <- mapply(function(sXdati, removed.ti)
    return(sXdati[-which(sXdati$t %in% removed.ti), ]), sXdatlist, removed.t, SIMPLIFY = FALSE)
  sXdat <- do.call("rbind", sXdatlist)

  shift_t <-  max(object$modeldata$t) - max(object$inputdata[,sXobject$t])
  sXdat$t <- sXdat$t + min_t - 1 - shift_t


  ## 1. time-varying fixed effects
  if (!is.null(fe.varying)) {
    Xlin <- stats::model.matrix(fe.varying, data = sXdat)[, -1, drop = F]
  } else {
    Xlin <- matrix(1, nrow = nrow(sXdat))
  }

  ### code following https://github.com/awstringer1/mam/blob/master/R/mam.R
  ## start following ##
  ## 2.2 smooth term
  numsmooth <- 0 # initialize
  if (!is.null(smooth)) {
    smooth <- lapply(lapply(attr(terms(smooth), "term.labels"), function(text)
      parse(text = text)), eval)

    SS <- object$smooth$others

    XXpred <- Reduce(cbind,lapply(SS,mgcv::PredictMat,data = sXdat)) ## smooth terms for preddat

    numsmooth <- length(smooth) # Number of smooth terms

    EE <- lapply(lapply(lapply(SS, '[[', 'S'), '[[', 1), eigen) ## eigen decomposition for penalty matrix

    p <- sapply(lapply(EE, '[[', 'vectors'), ncol) ## dimension of penalty matrix
    r <- sapply(lapply(EE, '[[', 'values'), function(x)
      sum(x > .Machine$double.eps)) ## rank of penalty matrix
    m <- p - r ## dim of null space (minus intercept)
    URlist <- mapply(function(x, y)
      x[, 1:y], lapply(EE, '[[', 'vectors'), r, SIMPLIFY = FALSE)
    UFlist <- mapply(function(x, y, z) {
      if (y < z)
        return(x[, (1 + y):z])
      newsparsemat(z, z)
    }, lapply(EE, '[[', 'vectors'), r, p, SIMPLIFY = FALSE)
    URlist <- lapply(URlist, cbind) # Ensure they stay matrices
    UFlist <- lapply(UFlist, cbind) # Ensure they stay matrices

    UR <- Matrix::bdiag(URlist)
    UF <- Matrix::bdiag(UFlist)
    # if m=1 UF gets coerced to numeric
    if (!is.matrix(UF))
      UF <- cbind(UF)

    Dpi <- Matrix::Diagonal(sum(r), 1 / sqrt(Reduce(c, lapply(lapply(EE, '[[', 'values'), function(x)
      x[x > .Machine$double.eps]))))

    Xrand <- as.matrix(XXpred %*% UR %*% Dpi)
    Xfix <- as.matrix(XXpred %*% UF)

    dups <- !base::duplicated(t(Xfix)) &
      apply(Xfix, 2, function(x)
        ! all(x == 0)) # Remove the duplicated intercepts
    if (length(dups) > 1)
      Xfix <- Xfix[, which(dups)]

    model.choice = "with.smooth"
  } else{
    model.choice = "without.smooth"
  }


  # 2.3 add the intercept
  if (exists("Xfix")) {
    Xfix <- cbind(Xlin, Xfix) ## linear effect + unpenalized columns
  } else {
    Xfix <- Xlin ## linear effect + unpenalized columns
  }
  if (any(apply(Xfix, 2, function(x)
    all(x == 1))))
    Xfix <- Xfix[, -(apply(Xfix, 2, function(x)
      all(x == 1)))]

  if (!is.null(fe.cont)) {
    Xgroup <- stats::model.matrix(fe.cont, data = sXdat)[, -1, drop = FALSE]
    if (any(apply(Xgroup, 2, function(x)
      all(x == 1))))
      Xgroup <- Xgroup[, -(apply(Xgroup, 2, function(x)
        all(x == 1)))] # remove the intercept
    Xfix <- cbind(1, Xgroup, Xfix) # add the intercept to the first column
  } else {
    Xfix <- cbind(1, Xfix) # add the intercept to the first column
  }


  E <- B_inner %*% object$point$alpha_w
  Bf <- sapply(E, function(Ei)
    Bsplinevec2Con(Ei, knots_f, 4, Zf))
  eta1 <- as.vector(t(Bf) %*% object$point$alpha_f)
  eta2 <- as.vector(Xfix %*% object$point$betaF)
  if (exists("Xrand"))
    eta2 <- eta2 + as.vector(Xrand %*% object$point$betaR)
  eta.est <- eta1 + eta2


  if (CI.fit) {
    R.CI <- nrow(object$CI.sample[[1]])
    eta.sample <- sapply(1:R.CI, function(i) {
      E <- B_inner %*% object$CI.sample$alpha_w_sample[i, ]
      Bf <- sapply(E, function(Ei)
        Bsplinevec2Con(Ei, knots_f, 4, Zf))
      eta1 <- as.vector(t(Bf) %*% object$CI.sample$alpha_f_sample[i, ])
      eta2 <- as.vector(Xfix %*% object$CI.sample$betaF_sample[i, ])
      if (exists("Xrand"))
        eta2 <- eta2 + as.vector(Xrand %*% object$CI.sample$betaR_sample[i, ])
      return(eta1 + eta2)
    })

    eta.CI <- apply(eta.sample, 1, function(row.)
      quantile(row., c((1 - CI) / 2, 1 - (1 - CI) / 2)))
    eta.ll <- eta.CI[1, ]
    eta.ul <- eta.CI[2, ]
    return(data.frame(est = eta.est, ll = eta.ll, ul = eta.ul))
  }

  return(data.frame(est = eta.est))
}
