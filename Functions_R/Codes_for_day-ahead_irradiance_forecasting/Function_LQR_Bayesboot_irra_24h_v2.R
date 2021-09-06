##### Path of installed packages
user_name_str <- Sys.info()[[7]]
R_libraries_path <- paste("C:/Users/",user_name_str,"/Documents/R/win-library/4.0", sep="")
Data_path <- paste("C:/Users/",user_name_str,"/Desktop/DiGriFlex_Code/Functions_R/Codes_for_day-ahead_irradiance_forecasting/", sep="")
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
LQR_Bayesboot <- function(pred_for,h,N_boot) {
  # This function provides a forecast y(t) with t belonging to day D, 
  # given data observed until 18:00 of day D-1
  
  # Output:
  # The output of the function is a [1x1] point prediction obtained from the [9 x N_boot] matrix,
  # containing N_boot bootstrap samples of the 9 predictive quantiles at coverages 0.1,0.2,...,0.9
  
  # Inputs:
  # 1) irra_tra and pred_tra are always the same preprocessed training data.
  # 2) pred_for is a data frame containing predictors for the target forecast horizon
  # 3) h is the number of the target 10-min interval of the day. E.g., the interval 10:00-10:10 
  #    is the 61th interval of the day; for h<=33 and h>=121 (before 5:30 and after 20:00) 
  #    the output is directly 0.
  # 4) N_boot is the desired number of bootstrap samples. 

  load(paste0(Data_path,"DATA_tra.RData"))
  ## ----------------------------- SET SEED FOR REPRODUCIBILITY -------------------------------
  set.seed(h)

  pred_for <- data.frame(pred_for[c(1)],pred_for[c(2)],pred_for[c(3)],pred_for[c(4)],pred_for[c(5)],
                         pred_for[c(6)],pred_for[c(7)],pred_for[c(8)],pred_for[c(9)],pred_for[c(10)])
  ## ----------------------------- DATA READING AND PREPROCESSING -----------------------------
  # Normalize the inputs 
  base_irra <- list("xmin" = min(irra_tra), "xmax" = max(irra_tra))
  irra_tra <- (irra_tra-base_irra$xmin)/(base_irra$xmax-base_irra$xmin)
  pred_tra[,c(1,6)] <- (pred_tra[,c(1,6)]-base_irra$xmin)/(base_irra$xmax-base_irra$xmin)
  pred_for[,c(1,6)] <- (pred_for[,c(1,6)]-base_irra$xmin)/(base_irra$xmax-base_irra$xmin)

  base_pres <- list("xmin" = min(pred_tra[,2]), "xmax" = max(pred_tra[,2]))
  pred_tra[,c(2,7)] <- (pred_tra[,c(2,7)]-base_pres$xmin)/(base_pres$xmax-base_pres$xmin)
  pred_for[,c(2,7)] <- (pred_for[,c(2,7)]-base_pres$xmin)/(base_pres$xmax-base_pres$xmin)

  base_relh <- list("xmin" = min(pred_tra[,3]), "xmax" = max(pred_tra[,3]))
  pred_tra[,c(3,8)] <- (pred_tra[,c(3,8)]-base_relh$xmin)/(base_relh$xmax-base_relh$xmin)
  pred_for[,c(3,8)] <- (pred_for[,c(3,8)]-base_relh$xmin)/(base_relh$xmax-base_relh$xmin)
  
  base_temp <- list("xmin" = min(pred_tra[,4]), "xmax" = max(pred_tra[,4]))
  pred_tra[,c(4,9)] <- (pred_tra[,c(4,9)]-base_temp$xmin)/(base_temp$xmax-base_temp$xmin)
  pred_for[,c(4,9)] <- (pred_for[,c(4,9)]-base_temp$xmin)/(base_temp$xmax-base_temp$xmin)
  
  base_wind <- list("xmin" = min(pred_tra[,5]), "xmax" = max(pred_tra[,5]))
  pred_tra[,c(5,10)] <- (pred_tra[,c(5,10)]-base_wind$xmin)/(base_wind$xmax-base_wind$xmin)
  pred_for[,c(5,10)] <- (pred_for[,c(5,10)]-base_wind$xmin)/(base_wind$xmax-base_wind$xmin)
  
  # Store data in a data frame ---------------------------------------------------
  DATA_tra <- data.frame("irra"=irra_tra, "irralag"=pred_tra[,1], "preslag"=pred_tra[,2],
                                          "relhlag"=pred_tra[,3], "templag"=pred_tra[,4],
                                          "windlag"=pred_tra[,5], "avgirra"=pred_tra[,6],
                                          "avgpres"=pred_tra[,7], "avgrelh"=pred_tra[,8],
                                          "avgtemp"=pred_tra[,9], "avgwind"=pred_tra[,10])
  DATA_for <- data.frame("irra"=NaN,      "irralag"=pred_for[,1], "preslag"=pred_for[,2],
                                          "relhlag"=pred_for[,3], "templag"=pred_for[,4],
                                          "windlag"=pred_for[,5], "avgirra"=pred_for[,6],
                                          "avgpres"=pred_for[,7], "avgrelh"=pred_for[,8],
                                          "avgtemp"=pred_for[,9], "avgwind"=pred_for[,10])

  ## ----------------------------- MAKING PREDICTIONS -----------------------------
  Q01 <- seq(0.1,0.9,0.1)
  irra_prev <- array(rep(NaN, length(Q01)*N_boot), c(length(Q01),N_boot))
  irra_prev2 <- array(rep(NaN, 1*3), c(1,3))
  
  if (h<=33)      # For h<=33 and h>=121 (before 5:30 and after 20:00) the output is directly 0
    {irra_prev <- array(rep(0, length(Q01)*N_boot), c(length(Q01),N_boot))}
  if (h>=121)
    {irra_prev <- array(rep(0, length(Q01)*N_boot), c(length(Q01),N_boot))}
  if (h>=34 & h <=120) {
    
    # Train and forecast data
    train_h <- seq(from = h, to = length(DATA_tra[,1]), by = 144) 
    DATA_train_h <- DATA_tra[train_h,]
    DATA_forec_h <- DATA_for
    
    # Forecasting model
    # for (q in seq(1,length(Q005),1))
    for (q in c(1,5,9))
    {
      qr_prev <- function(d)     # Definition of the forecasting function (quantile regression model)
      { mdl_fit <- rq( irra ~ irralag + preslag + relhlag + templag + windlag +
                              avgirra + avgpres + avgrelh + avgtemp + avgwind +
                              irralag*avgirra,
                       tau=Q01[q], data = d)
      predict.rq(mdl_fit, DATA_forec_h) 
      } 
      
      bootstrap_res <- as.matrix( bayesboot(DATA_train_h, qr_prev, R = N_boot) ) # Apply Bayesian bootstrap
      for (n in 1:N_boot)
        {irra_prev[q,n] <- bootstrap_res[n,1]}
    }
  }
  
  irra_prev[is.nan(irra_prev)] <- 0
  irra_prev[irra_prev < 0] <- 0
  for (kk in (c(1,2,3)))
  { irra_prev2[kk] <- mean(irra_prev[(1+(kk-1)*4),]) }
  irra_prev2 <- base_irra$xmin + irra_prev2*(base_irra$xmax-base_irra$xmin)
}




