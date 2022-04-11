LQR_Bayesboot <- function(P_tra, pred_tra, pred_for, h, N_boot) {
  # This function provides a forecast y(t), given data observed until the time interval t-2
  
  # Output:
  # The output of the function is a [1x1] point prediction obtained from the [9 x N_boot] matrix,
  # containing N_boot bootstrap samples of the 9 predictive quantiles at coverages 0.1,0.2,...,0.9
  
  # Inputs:
  # 1) Y_tra and pred_tra are always the same preprocessed training data.
  # 2) pred_for is a data frame containing predictors 
  # 3) h is the number of the target 10-min interval of the day. E.g., the interval 10:00-10:10 
  #    is the 61th interval of the day.
  # 4) N_boot is the desired number of bootstrap samples. 
  
  ## ----------------------------- SET SEED FOR REPRODUCIBILITY -------------------------------
  set.seed(h)
  
  ## ----------------------------- DATA READING AND PREPROCESSING -----------------------------
  # Store data in a data frame 
  DATA_tra <- data.frame("P"=P_tra, "Plag"=pred_tra[,1], "Qlag"=pred_tra[,2],
                                    "avgP"=pred_tra[,3], "avgQ"=pred_tra[,4])
  DATA_for <- data.frame("P"=NaN,   "Plag"=pred_for[,1], "Qlag"=pred_for[,2],
                                    "avgP"=pred_for[,3], "avgQ"=pred_for[,4])

  ## ----------------------------- MAKING PREDICTIONS -----------------------------
  Q01 <- seq(0.1,0.9,0.1)
  P_prev <- array(rep(NaN, length(Q01)*N_boot), c(length(Q01),N_boot))
  P_prev2 <- array(rep(NaN, 1*3), c(1,3))

  # Train and forecast data
  train_h <- seq(from = h, to = length(DATA_tra[,1]), by = 144)
  DATA_train_h <- DATA_tra[train_h,]
  DATA_forec_h <- DATA_for

  # Forecasting model
  for (q in c(1,5,9))
  {
    qr_prev <- function(d)     # Definition of the forecasting function (quantile regression model)
    { mdl_fit <- rq( P ~ Plag + Qlag + avgP + avgQ +
                         Plag*Qlag + avgP*avgQ,
                     tau = Q01[q], data = d)
      predict.rq(mdl_fit, DATA_forec_h)
    }

    bootstrap_res <- as.matrix( bayesboot(DATA_train_h, qr_prev, R = N_boot) ) # Apply Bayesian bootstrap
    for (n in 1:N_boot)
      {P_prev[q,n] <- bootstrap_res[n,1]}
  }

  P_prev[is.nan(P_prev)] <- 0
#  P_prev[P_prev < 0] <- 0
  for (kk in (c(1,2,3)))
  { P_prev2[kk] <- mean(P_prev[(1+(kk-1)*4),]) }
  P_prev2 <- P_prev2
}




