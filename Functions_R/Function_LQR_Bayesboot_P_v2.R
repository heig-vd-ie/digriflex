##### Path of installed packages
user_name_str <- Sys.info()[[7]]
R_libraries_path <- paste("C:/Users/",user_name_str,"/Documents/R/win-library/4.0", sep="")
Data_path <- paste("C:/Users/",user_name_str,"/Desktop/DiGriFlex_Code/Data/", sep="")
# if(Sys.info()[[7]] == "mohammad.rayati"){
#   R_libraries_path <- "C:/Users/mohammad.rayati/Documents/R/win-library/4.0"
#   Data_path <- "C:/Users/mohammad.rayati/Desktop/DiGriFlex_Code/Data/"
# }
# if(Sys.info()[[7]] == "labo-reine-iese"){
#   R_libraries_path <- "C:/Users/labo-reine-iese/Documents/R/win-library/4.0"
#   Data_path <- "C:/Users/labo-reine-iese/Desktop/DiGriFlex_Code/Data/"
# }
.libPaths(c(R_libraries_path))
sink("nul")

##### Load required librarues
library(bayesboot)
library(zoo, warn.conflicts = FALSE)
suppressPackageStartupMessages(library(quantreg))


##### Actual function
LQR_Bayesboot <- function(predP_for,h,N_boot) {
  # This function provides a forecast y(t), given data observed until the time interval t-2
  
  # Output:
  # The output of the function is a [1x1] point prediction obtained from the [9 x N_boot] matrix,
  # containing N_boot bootstrap samples of the 9 predictive quantiles at coverages 0.1,0.2,...,0.9
  
  # Inputs:
  # 1) predP_for is a data frame containing predictors P(t-144),P(t-2),P(t-3),P(t-4),P(t-5),P(t-6)
  # 2) h is the number of the target 10-min interval of the day. E.g., the interval 10:00-10:10
  #    is the 61th interval of the day.
  # 3) N_boot is the desired number of bootstrap samples.

  load(paste(Data_path, "DataP_tra.RData",sep=""))
  ## ----------------------------- SET SEED FOR REPRODUCIBILITY -------------------------------
  set.seed(h)
  
  ## ----------------------------- DATA READING AND PREPROCESSING -----------------------------
  # Normalize the inputs 
  base_P <- list("xmin" = min(P_tra), "xmax" = max(P_tra))
  P_tra <- (P_tra-base_P$xmin)/(base_P$xmax-base_P$xmin)
  predP_tra <- (predP_tra-base_P$xmin)/(base_P$xmax-base_P$xmin)
  predP_for <- (predP_for-base_P$xmin)/(base_P$xmax-base_P$xmin)
  
  # Store data in a data frame 
  DATAP_tra <- data.frame("P"=P_tra,   "Plag144"=predP_tra[,1],"Plag2"=predP_tra[,2],
                                         "Plag3"=predP_tra[,3],"Plag4"=predP_tra[,4],
                                         "Plag5"=predP_tra[,5],"Plag6"=predP_tra[,6])
  DATAP_for <- data.frame("P_for"=NaN, "Plag144"=predP_for[,1],"Plag2"=predP_for[,2],
                                         "Plag3"=predP_for[,3],"Plag4"=predP_for[,4],
                                         "Plag5"=predP_for[,5],"Plag6"=predP_for[,6])

  ## ----------------------------- MAKING PREDICTIONS -----------------------------
  Q01 <- seq(0.1,0.9,0.1)
  P_prev <- array(rep(NaN, length(Q01)*N_boot), c(length(Q01),N_boot))
  
  # Train and forecast data
  train_h <- seq(from = h, to = length(DATAP_tra[,1]), by = 144) 
  DATA_train_h <- DATAP_tra[train_h,]
  DATA_forec_h <- DATAP_for
    
  # Forecasting model
  for (q in seq(1,length(Q01),1))
  {
    qr_prev <- function(d)     # Definition of the forecasting function (quantile regression model)
    { mdl_fit <- rq( P ~ Plag2 + Plag3 + Plag4 + Plag5 + Plag6 + Plag144 +
                         Plag2*Plag144 + Plag2*Plag6,
                     tau=Q01[q], data = d)
      predict.rq(mdl_fit, DATA_forec_h) 
    } 
      
    bootstrap_res <- as.matrix( bayesboot(DATA_train_h, qr_prev, R = N_boot) ) # Apply Bayesian bootstrap
    for (n in 1:N_boot)
      {P_prev[q,n] <- bootstrap_res[n,1]}
  }

  P_prev[is.nan(P_prev)] <- 0
  P_prev[P_prev < 0] <- 0
  P_prev <- mean(P_prev[5,])   # This line will change after the optimization of the sample pick
  P_prev <- P_prev
}




