#' Title Randomized Quantile Residuals
#'
#' @param object object of class \code{aceDLNM_fit}.
#' @param seed random seed, default 123.
#' @param plot whether or not to show the plots, default \code{FALSE}.
#' @param nsample the number of sampling from the datasets if number of
#'   observations is larger than nsample, default \code{1e4}.
#' @param curver whether or not to draw a curve for residual plot using
#'   \code{ggplot2::stat_smooth(method = "gam")}.
#' @param ...
#'
#' @import dplyr
#' @import ggplot2
#' @return a list containing the randomized quantile residuals and the plots.
#' @references Dunn, Peter K., and Gordon K. Smyth. \dQuote{Randomized quantile
#'   residuals}. Journal of Computational and graphical statistics 5.3 (1996):
#'   236-244.
#' @export
residuals.aceDLNM_fit <- function(object, seed = 123, plot = FALSE,
                                     nsample = 1e4, curve = FALSE, ...){
  set.seed(seed)

  eta <- object$eta$est
  y <- object$modeldata$y
  x <- object$modeldata$x
  t <- object$modeldata$t

  mu <- exp(eta)
  theta <- exp(object$point$log_theta) # theta
  if(length(y) > nsample) {
    sample.id <- sample(1:length(y), nsample)
  } else {
    sample.id <- 1:length(y)
  }


  rqr <- mapply(function(yi, mui){
    yilim <- max(yi-1,0)
    ai <- pnbinom(yilim, size = theta, mu = mui)
    bi <- pnbinom(yi, size = theta, mu = mui)
    ui <- runif(1, min = ai, max = bi)
    ri <- qnorm(ui)
    return(ri)
  },
  y[sample.id], mu[sample.id])

  out <- list(res = data.frame(rqr = rqr,
                               y = y[sample.id],
                               mu = mu[sample.id],
                               x = x[sample.id],
                               t = t[sample.id]),
              p.qq = NULL,
              p.res = NULL
              )

  p.qq <- ggplot(data.frame(xx = rqr), aes(sample = xx)) + stat_qq() + stat_qq_line() + xlab("Theoretical quantiles") + ylab("Sample quantiles") + theme_bw()
  out$p.qq <- p.qq
  p.res <- ggplot(data.frame(t =t[sample.id], xx = rqr), aes(x = t, y = xx)) +
      geom_point() + geom_hline(yintercept = 0) +
      scale_x_continuous(breaks = seq(min(out$res$t), max(out$res$t), by = round((max(out$res$t) - min(out$res$t))/50))) +
      xlab("Time") + ylab("Randomized Quantile Residuals") + theme_bw() +
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
  if(curve) p.res <- p.res + stat_smooth(method = "gam")
  out$p.res <- p.res

  if(plot) {
    oask <- devAskNewPage(TRUE)
    on.exit(devAskNewPage(oask))
    dev.hold()
    plot(p.qq)
    dev.flush()

    dev.hold()
    plot(p.res)
    dev.flush()
    devAskNewPage(oask)
    invisible(out)
  } else {
    return(out)
  }

}
