# Path of installed packages
user_name_str <- Sys.info()[[7]]
current_path <- getwd()
R_libraries_path <- paste("C:/Users/",user_name_str,"/Documents/R/win-library/4.0", sep="")
Data_path <- paste(current_path, "/data/", sep="")
data_file <- "data_tra_q_rt.RDATA"

.libPaths(c(R_libraries_path))
sink("nul")

# Load required librarues
library(bayesboot)
library(zoo, warn.conflicts = FALSE)
suppressPackageStartupMessages(library(quantreg))


# Actual function
LQR_Bayesboot <- function(predQ_for,h,N_boot) {
  # This function provides a forecast y(t), given data observed until the time interval t-2

  # Output:
  # The output of the function is a [1x1] point prediction obtained from the [9 x N_boot] matrix,
  # containing N_boot bootstrap samples of the 9 predictive quantiles at coverages 0.1,0.2,...,0.9

  # Inputs:
  # 1) predQ_for is a data frame containing predictors Q(t-144),Q(t-2),Q(t-3),Q(t-4),Q(t-5),Q(t-6)
  # 2) h is the number of the target 10-min interval of the day. E.g., the interval 10:00-10:10
  #    is the 61th interval of the day.
  # 3) N_boot is the desired number of bootstrap samples.

  load(paste(Data_path, data_file,sep=""))
  ## ----------------------------- SET SEED FOR REPRODUCIBILITY -------------------------------
  set.seed(h)

  ## ----------------------------- DATA READING AND PREPROCESSING -----------------------------
  # Normalize the inputs
  base_Q <- list("xmin" = min(Q_tra), "xmax" = max(Q_tra))
  Q_tra <- (Q_tra-base_Q$xmin)/(base_Q$xmax-base_Q$xmin)
  predQ_tra <- (predQ_tra-base_Q$xmin)/(base_Q$xmax-base_Q$xmin)
  predQ_for <- (predQ_for-base_Q$xmin)/(base_Q$xmax-base_Q$xmin)

  # Store data in a data frame
  DATAQ_tra <- data.frame("Q"=Q_tra,   "Qlag144"=predQ_tra[,1],"Qlag2"=predQ_tra[,2],
                                         "Qlag3"=predQ_tra[,3],"Qlag4"=predQ_tra[,4],
                                         "Qlag5"=predQ_tra[,5],"Qlag6"=predQ_tra[,6])
  DATAQ_for <- data.frame("Q_for"=NaN, "Qlag144"=predQ_for[,1],"Qlag2"=predQ_for[,2],
                                         "Qlag3"=predQ_for[,3],"Qlag4"=predQ_for[,4],
                                         "Qlag5"=predQ_for[,5],"Qlag6"=predQ_for[,6])

  ## ----------------------------- MAKING PREDICTIONS -----------------------------
  Q01 <- seq(0.1,0.9,0.1)
  Q_prev <- array(rep(NaN, length(Q01)*N_boot), c(length(Q01),N_boot))

  # Train and forecast data
  train_h <- seq(from = h, to = length(DATAQ_tra[,1]), by = 144)
  DATA_train_h <- DATAQ_tra[train_h,]
  DATA_forec_h <- DATAQ_for

  # Forecasting model
  for (q in 5)
  {
    qr_prev <- function(d)     # Definition of the forecasting function (quantile regression model)
    { mdl_fit <- rq( Q ~ Qlag2 + Qlag3 + Qlag4 + Qlag5 + Qlag6 +
                         Qlag2*Qlag144 + Qlag2*Qlag6,
                     tau=Q01[q], data = d)
      predict.rq(mdl_fit, DATA_forec_h)
    }

    bootstrap_res <- as.matrix( bayesboot(DATA_train_h, qr_prev, R = N_boot) ) # Apply Bayesian bootstrap
    for (n in 1:N_boot)
      {Q_prev[q,n] <- bootstrap_res[n,1]}
  }

  Q_prev[is.nan(Q_prev)] <- 0
  Q_prev[Q_prev < 0] <- 0
  Q_prev <- mean(Q_prev[5,])   # This line will change after the optimization of the sample pick
  Q_prev <- base_Q$xmin + Q_prev * (base_Q$xmax - base_Q$xmin)
}

