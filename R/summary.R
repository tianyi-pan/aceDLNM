#' Title summary() Method for Objects of Class 'aceDLNM_fit'
#'
#' @param object object of class \code{aceDLNM_fit}.
#' @param E.eval a vector of ACE where the ACE response-exposure-function is
#'   evaluated. The default is a sequence from 0.0025 quantile to 0.9975
#'   quantile of estimated E, if the argument is missing.
#' @param others.eval a data frame containing variables where the other smooth
#'   functions are evaluated.
#' @param plot whether or not to show the plots, default \code{FALSE}.
#' @param true.function the list containing the true functions.
#' @param E0 if \code{E0 = NULL} (default), report f(E) for ACEREF, else report
#'   f(E) - f(E0).
#' @param ...
#'
#' @import dplyr
#' @return a list containing (1) the estimated functions evaluated at
#'   \code{E.eval} and \code{others.eval} or the default values, (2) AIC, and
#'   (3) plots.
#' @export
summary.aceDLNM_fit <- function(object, E.eval, l.eval, others.eval = NULL,
                                   plot = FALSE, true.function,
                                   contrast = FALSE,
                                   contrast.lower = 0, contrast.upper = 1,
                                   E0 = NULL, ...){

  ## point estimate
  pc <- object$data$pc
  alpha_w <- object$point$alpha_w
  alpha_f <- object$point$alpha_f
  kw <- object$data$kw
  kE <- object$data$kE

  Ufpen <- object$data$Ufpen
  Uwpen <- object$data$Uwpen

  if(missingArg(l.eval)) l.eval <- seq(0, object$data$maxL+1, length.out = max(object$data$maxL+1, 500))
  wl.fit <- function(lnew) mgcv::PredictMat(object$smooth$wl, data = data.frame(l = lnew)) %*% Uwpen %*% alpha_w[2:kw] + alpha_w[1]

  # wl.fit <- function(lnew) Bsplinevec2Con(lnew, object$data$knots_w, 4, object$data$Zw %*% Uwpen) %*% alpha_w[2:kw] + alpha_w[1]
  # wl.fit <- Vectorize(wl.fit)

  wl.mode <- c(wl.fit(l.eval))

  E.mode <- c(object$data$B_inner %*% alpha_w)
  E.mode.quantile <- quantile(E.mode, c(0.0025, 0.9975))
  if(missingArg(E.eval)) E.eval <- seq(max(E.mode.quantile[1],0), E.mode.quantile[2], length.out = 500)
  if(is.null(pc)) {
    fE.fit <- function(Enew) mgcv::PredictMat(object$smooth$fE, data = data.frame(E = Enew)) %*% Ufpen %*% alpha_f
  } else {
    fE.fit <- function(Enew) mgcv::PredictMat(object$smooth$fE, data = data.frame(E = Enew)) %*% object$data$Zf %*% Ufpen %*% alpha_f
  }
  # fE.fit <- function(Enew) Bsplinevec2Con(Enew, object$data$knots_f, 4, object$data$Zf) %*% Ufpen %*% alpha_f
  # fE.fit <- Vectorize(fE.fit)

  fE.mode <- c(fE.fit(E.eval))

  if(!is.null(E0)) {
    fE0.mode <- fE.mode - c(fE.fit(E0))
  }
  if(is.null(pc)) {
    fE.mode.mean <- mean(fE.mode)
  } else {
    fE.mode.mean <- 0
  }
  fE.mode <- fE.mode - fE.mode.mean

  out <- list(l.eval, wl.mode, E.mode, fE.mode)


  ## CI
  R.CI <- nrow(object$CI.sample[[1]])
  wl.results <- lapply(1:R.CI, function(i){
    alpha_w_sample <- object$CI.sample$alpha_w_sample[i,]

    wl.fit <- function(lnew) mgcv::PredictMat(object$smooth$wl, data = data.frame(l = lnew)) %*% Uwpen %*% alpha_w_sample[2:kw] + alpha_w_sample[1]
    # wl.fit <- function(lnew) Bsplinevec2Con(lnew, object$data$knots_w, 4, object$data$Zw %*% Uwpen) %*% alpha_w_sample[2:kw] + alpha_w_sample[1]
    # wl.fit <- Vectorize(wl.fit)
    wl.est <- c(wl.fit(l.eval))

    return(cbind(l.eval, wl.est, rep(i, length(wl.est))))
  })
  wl.results <- do.call(rbind.data.frame, wl.results)
  colnames(wl.results) <- c("l", "est", "iter")

  fE.results <- lapply(1:R.CI, function(i){
    alpha_f_sample <- object$CI.sample$alpha_f_sample[i,]
    if(is.null(pc)) {
      fE.fit <- function(Enew) mgcv::PredictMat(object$smooth$fE, data = data.frame(E = Enew)) %*% Ufpen %*% alpha_f_sample
    } else {
      fE.fit <- function(Enew) mgcv::PredictMat(object$smooth$fE, data = data.frame(E = Enew)) %*% object$data$Zf %*% Ufpen %*% alpha_f_sample
    }
    # fE.fit <- function(Enew) Bsplinevec2Con(Enew, object$data$knots_f, 4, object$data$Zf) %*% alpha_f_sample
    # fE.fit <- Vectorize(fE.fit)

    fE.est <- c(fE.fit(E.eval))
    # fE.est  <- fE.est - fE.mode.mean
    if(is.null(pc)) {
      fE.est <- fE.est - mean(fE.est)
    }
    return(cbind(E.eval, fE.est, rep(i, length(fE.est))))
  })
  fE.results <- do.call(rbind.data.frame, fE.results)
  colnames(fE.results) <- c("E", "est", "iter")

  if(!is.null(E0)) {
    fE0.results <- lapply(1:R.CI, function(i){
      alpha_f_sample <- object$CI.sample$alpha_f_sample[i,]
      if(is.null(pc)) {
        fE.fit <- function(Enew) mgcv::PredictMat(object$smooth$fE, data = data.frame(E = Enew)) %*% Ufpen %*% alpha_f_sample
      } else {
        fE.fit <- function(Enew) mgcv::PredictMat(object$smooth$fE, data = data.frame(E = Enew)) %*% object$data$Zf %*% Ufpen %*% alpha_f_sample
      }
      fE.est <- c(fE.fit(E.eval))
      fE0.est <- fE.est - c(fE.fit(E0))
      return(cbind(E.eval, fE0.est, rep(i, length(fE0.est))))
    })
    fE0.results <- do.call(rbind.data.frame, fE0.results)
    colnames(fE0.results) <- c("E", "est", "iter")
  }

  ## contrast
  if(contrast) {
    if(contrast.upper > object$data$maxL+1) contrast.upper <- object$data$maxL+1
    if(contrast.lower < 0) contrast.lower <- 0
    contrast.results <- lapply(1:R.CI, function(i){
      alpha_w_sample <- object$CI.sample$alpha_w_sample[i,]
      alpha_f_sample <- object$CI.sample$alpha_f_sample[i,]

      #  wl.fit <- function(lnew) mgcv::PredictMat(object$smooth$wl, data = data.frame(l = lnew)) %*% alpha_w_sample[2:kw] + alpha_w_sample[1]
      wl.fit <- function(lnew) Bsplinevec2Con(lnew, object$data$knots_w, 4, object$data$Zw %*% Uwpen) %*% alpha_w_sample[2:kw] + alpha_w_sample[1]
      wl.fit <- Vectorize(wl.fit)

      if(is.null(pc)) {
        fE.fit <- function(Enew) mgcv::PredictMat(object$smooth$fE, data = data.frame(E = Enew)) %*% Ufpen %*% alpha_f_sample
      } else {
        fE.fit <- function(Enew) mgcv::PredictMat(object$smooth$fE, data = data.frame(E = Enew)) %*% object$data$Zf %*% Ufpen %*% alpha_f_sample
      }
      # fE.fit <- function(Enew) Bsplinevec2Con(Enew, object$data$knots_f, 4, object$data$Zf) %*% Ufpen %*% alpha_f_sample
      # fE.fit <- Vectorize(fE.fit)

      # TODO: rewrite the integration in cpp
      E.increase <- integrate(wl.fit, lower = contrast.lower, upper = contrast.upper)$value

      contrast <- fE.fit(E.eval + E.increase) - fE.fit(E.eval)

      return(cbind(E.eval, contrast, rep(i, length(E.eval))))
    })
    contrast.results <- do.call(rbind.data.frame, contrast.results)
    colnames(contrast.results) <- c("E", "est", "iter")
  }

  ## get quantile
  wl.df <- group_by(wl.results, l) %>%
    summarize(ll = quantile(est, 0.025), ul = quantile(est, 0.975)) %>%
    mutate(mode = wl.mode)
  fE.df <- group_by(fE.results, E) %>%
    summarize(ll = quantile(est, 0.025), ul = quantile(est, 0.975)) %>%
    mutate(mode = fE.mode)
  if(!is.null(E0)) {
    fE0.df <- group_by(fE0.results, E) %>%
      summarize(ll = quantile(est, 0.025), ul = quantile(est, 0.975)) %>%
      mutate(mode = fE0.mode)
  }
  if(contrast) {
    contrast.df <- group_by(contrast.results, E) %>%
    summarize(ll = quantile(est, 0.025), ul = quantile(est, 0.975), mean = mean(est))
  }

  out <- list(samples = list(wl = wl.results,
                             fE = fE.results),
              est = list(wl = wl.df,
                         fE = fE.df))
  if(contrast) {
    out$samples$contrast = contrast.results
    out$est$contrast = contrast.df
  }
  if(!is.null(E0)) {
    out$samples$fE0 = fE0.results
    out$est$fE0 = fE0.df
  }


  if(!missingArg(true.function)) {
    ## wl
    out$est$wl$true <- true.function$wl(l.eval)

    ## fE
    if(is.null(pc)) {
      out$est$fE$true <- true.function$fE(E.eval)
      out$est$fE$true <- out$est$fE$true - mean(out$est$fE$true)
    } else {
      out$est$fE$true <- true.function$fE(E.eval) - true.function$fE(pc)

    }
  }


  ### Other smooth terms
  if(!is.null(object$smooth$others)) {
    # others.eval
    SS <- object$smooth$others

    smooth.name <- unlist(lapply(SS, "[[", "term")) # var name of smooth terms
    if(is.null(others.eval)) {
      others.eval <- data.frame(lapply(object$modeldata[,smooth.name, drop=FALSE],
                                       function(col){
                                         # support random effects
                                         if(class(col) == "factor") return(factor(rep(col[1], 500), levels = levels(col)))
                                         else return(seq(min(col), max(col), length.out = 500))
                                       }))
    }
    Xpred <- lapply(SS, function(a){
      preddat <- data.frame(others.eval[,a$term])
      colnames(preddat) <- a$term
      mgcv::PredictMat(a, data = preddat)
    })
    Xpred <- Reduce(cbind, Xpred)
    Xrandpred <- as.matrix(Xpred %*% object$data$UR %*% object$data$Dpi) ## reparametrized
    Xfixpred <- as.matrix(Xpred %*% object$data$UF)
    Xfixpred <- Xfixpred[,apply(Xfixpred, 2, function(col.) sum(col. != 0) > 0), drop = FALSE]


    betaF_R <- object$point$betaF[-seq_len(ncol(object$data$Xfix) - sum(object$data$m))] # unpenalized terms
    splitR_index <- c() # index for betaR
    for (i in seq_along(object$data$r)) {
      splitR_index <- c(splitR_index, rep(i, object$data$r[i]))
    }
    splitF_index <- c() # index for betaF
    for (i in seq_along(object$data$m)) {
      splitF_index <- c(splitF_index, rep(i, object$data$m[i]))
    }


    Xrandpred_split <- lapply(split(seq_len(ncol(Xrandpred)), splitR_index),
                              function(j) Xrandpred[,j])
    betaR_split <- split(object$point$betaR, splitR_index)

    Xfixpred_split <- lapply(split(seq_len(ncol(Xfixpred)), splitF_index),
                             function(j) Xfixpred[,j])
    betaF_split <- split(betaF_R, splitF_index)


    ## obtain fitted value
    fittedR <- mapply(function(a,b) as.matrix(a) %*% as.vector(b),
                      Xrandpred_split, betaR_split, SIMPLIFY = FALSE)
    fittedF <- mapply(function(a,b) as.matrix(a) %*% as.vector(b),
                      Xfixpred_split, betaF_split, SIMPLIFY = FALSE)


    if(length(fittedF) == 0) {
      ## if there is no unpenalized component in smooth
      fitted <- fittedR
    } else {
      insertname <- setdiff(names(fittedR), names(fittedF)) # fill the fittedF to match the length
      if(length(insertname) != 0) {
        for (jj in insertname) fittedF[[jj]] <- matrix(0, nrow = nrow(fittedR[[1]]), ncol = 1)
      }
      ## order by name
      fittedF <- fittedF[order(names(fittedF))]

      fitted <- mapply(function(a,b) a+b, fittedR, fittedF, SIMPLIFY = FALSE)
    }

    fitted <- do.call("rbind", fitted)

    smooth.mode <- data.frame(x = as.matrix(as.vector(sapply(others.eval, as.numeric)), ncol = 1),
                           var = rep(colnames(others.eval), each = nrow(others.eval)),
                           mode = fitted)

    fitted.CI <- lapply(1:R.CI, function(i){
      betaF_sample <- object$CI.sample$betaF_sample[i,]
      betaR_sample <- object$CI.sample$betaR_sample[i,]
      betaR_sample_split <- split(betaR_sample, splitR_index)
      betaF_sample_split <- split(betaF_sample[-seq_len(ncol(object$data$Xfix) - sum(object$data$m))],
                                  splitF_index)

      fittedR_sample <- mapply(function(a,b) as.matrix(a) %*% as.vector(b),
                        Xrandpred_split, betaR_sample_split, SIMPLIFY = FALSE)
      fittedF_sample <- mapply(function(a,b) as.matrix(a) %*% as.vector(b),
                        Xfixpred_split, betaF_sample_split, SIMPLIFY = FALSE)
      if(length(fittedF_sample) == 0) {
        ## if there is no unpenalized component in smooth
        fitted_sample <- fittedR_sample
      } else {
        insertname <- setdiff(names(fittedR_sample), names(fittedF_sample)) # fill the fittedF to match the length
        if(length(insertname) != 0) {
          for (jj in insertname) fittedF_sample[[jj]] <- matrix(0, nrow = nrow(fittedR_sample[[1]]), ncol = 1)
        }
        ## order by name
        fittedF_sample <- fittedF_sample[order(names(fittedF_sample))]

        fitted_sample <- mapply(function(a,b) a+b, fittedR_sample, fittedF_sample, SIMPLIFY = FALSE)
      }
      fitted_sample <- do.call("rbind", fitted_sample)

      return(data.frame(x = as.matrix(as.vector(sapply(others.eval, as.numeric)), ncol = 1),
                        var = rep(smooth.name, each = nrow(others.eval)),
                        est = fitted_sample,
                        iter = rep(i, length(fitted_sample))))
    })
    fitted.CI <- do.call(rbind.data.frame, fitted.CI)

    ## get quantile
    smooth.df <- group_by(fitted.CI, x, var) %>%
      summarize(ll = quantile(est, 0.025), ul = quantile(est, 0.975))

    smooth.df <- left_join(smooth.mode, smooth.df, by = c("x","var"))

    if(!missingArg(true.function)) {
      if(!is.null(true.function$smooth)){
        true.function$smooth <- Vectorize(true.function$smooth)
        smooth.df$true <- true.function$smooth(smooth.df$x, smooth.df$var)

        # if(center) {
        #   modemean <- group_by(smooth.df, var) %>% summarize(modemean = mean(mode)) %>% select(modemean) %>% as.numeric()
        #   truemean <- group_by(smooth.df, var) %>% summarize(truemean = mean(true)) %>% select(modemean) %>% as.numeric()
        #
        #   smooth.df$true <- smooth.df$true - truemean
        #   smooth.df$mode <- smooth.df$mode - modemean
        #   smooth.df$ul <- smooth.df$ul - modemean
        #   smooth.df$ll <- smooth.df$ll - modemean
        # }
      }
    }

    # remove re terms
    re_id <- which(unlist(lapply(lapply(SS, attr, "class"), "[[", 1)) == "random.effect")

    if(length(re_id) >= 1) smooth.df <- dplyr::filter(smooth.df, !var %in% unique(smooth.df$var)[re_id])

    out$est$smooth = smooth.df

  }

  if(plot) {
    oask <- devAskNewPage(TRUE)
    on.exit(devAskNewPage(oask))

    with(out$est$fE, plot(E, mode, type = "l", ylim = c(min(ll), max(ul)), ylab = "est", xlab = "weighted exposure"))
    with(out$est$fE, lines(E, ll, lty = "dashed"))
    with(out$est$fE, lines(E, ul, lty = "dashed"))
    if(!missingArg(true.function)) with(out$est$fE, lines(E, true, col = "blue"))

    with(out$est$wl, plot(l, mode, type = "l", ylim = c(min(ll), max(ul)), ylab = "est", xlab = "weight"))
    with(out$est$wl, lines(l, ll, lty = "dashed"))
    with(out$est$wl, lines(l, ul, lty = "dashed"))
    if(!missingArg(true.function)) with(out$est$wl, lines(l, true, col = "blue"))

    if(!is.null(object$smooth$others)) {
      for(var. in unique(smooth.df$var)){
        smooth.df. <- filter(smooth.df, var == var.)
        with(smooth.df., plot(x, mode, type = "l", ylim = c(min(ll), max(ul)), ylab = "est", xlab = var.))
        with(smooth.df., lines(x, ll, lty = "dashed"))
        with(smooth.df., lines(x, ul, lty = "dashed"))
        if(!missingArg(true.function)){
          if(!is.null(true.function$smooth)) with(smooth.df., lines(x, true, col = "blue"))
        }
      }
    }
    if(!is.null(E0)) {
      with(out$est$fE0, plot(E, mode, type = "l", ylim = c(min(ll), max(ul)), ylab = paste0("f(E) - f(", E0, ")"), xlab = "weighted exposure"))
      with(out$est$fE0, lines(E, ll, lty = "dashed"))
      with(out$est$fE0, lines(E, ul, lty = "dashed"))
    }
    if(contrast) {
      with(out$est$contrast, plot(E, mean, type = "l", ylim = c(min(0,ll), max(0,ul)), ylab = "contrast", xlab = "weighted exposure"))
      with(out$est$contrast, lines(E, ll, lty = "dashed"))
      with(out$est$contrast, lines(E, ul, lty = "dashed"))
      abline(h = 0)
    }
    devAskNewPage(oask)
  }

  # AIC
  if(is.null(pc)) {
    SfCon <- object$smooth$fE$S[[1]]
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

    SwCon <- object$smooth$wl$S[[1]]
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

    if(!is.null(object$formula$smooth)) {
      AIC <- ConditionalAIC(object$data$y, object$data$B_inner, object$smooth$fE$knots, SwI, SfI, object$data$Dw,
                            object$data$Xrand, object$data$Xfix, object$data$Zf %*% object$data$Ufpen, object$data$offset$Xoffset, object$data$r,
                            object$point$alpha_f,
                            object$point$phi,
                            object$point$log_theta,
                            object$point$log_smoothing_f,
                            object$point$log_smoothing_w,
                            object$point$betaR, object$point$betaF,
                            object$point$log_smoothing)
    } else {
      AIC <- ConditionalAIC_nosmooth(object$data$y, object$data$B_inner, object$smooth$fE$knots, SwI, SfI, object$data$Dw,
                                     object$data$Xfix, object$data$Zf %*% object$data$Ufpen, object$data$offset$Xoffset,
                                     object$point$alpha_f,
                                     object$point$phi,
                                     object$point$log_theta,
                                     object$point$log_smoothing_f,
                                     object$point$log_smoothing_w,
                                     object$point$betaF)
    }

    out$AIC = AIC
  }


  invisible(out)
}
