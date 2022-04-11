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
LQR_Bayesboot <- function(pred_for,h,N_boot) {
  # This function provides a forecast y(t), given data observed until the time interval t-2
  
  # Output:
  # The output of the function is a [1x1] point prediction obtained from the [9 x N_boot] matrix,
  # containing N_boot bootstrap samples of the 9 predictive quantiles at coverages 0.1,0.2,...,0.9
  
  # Inputs:
  # 1) pred_for is a data frame containing predictors irra(t-144),irra(t-2),irra(t-3),irra(t-4),irra(t-5),irra(t-6)
  # 2) h is the number of the target 10-min interval of the day. E.g., the interval 10:00-10:10
  #    is the 61th interval of the day; for h<=33 and h>=121 (before 5:30 and after 20:00) 
  #    the output is directly 0.
  # 3) N_boot is the desired number of bootstrap samples.

  load(paste(Data_path,"Data_tra_Irr.RData",sep=""))
  ## ----------------------------- SET SEED FOR REPRODUCIBILITY -------------------------------
  set.seed(h)
  
  ## ----------------------------- DATA READING AND PREPROCESSING -----------------------------
  # Normalize the inputs 
  base_irra <- list("xmin" = min(irra_tra), "xmax" = max(irra_tra))
  irra_tra <- (irra_tra-base_irra$xmin)/(base_irra$xmax-base_irra$xmin)
  pred_tra <- (pred_tra-base_irra$xmin)/(base_irra$xmax-base_irra$xmin)
  pred_for <- (pred_for-base_irra$xmin)/(base_irra$xmax-base_irra$xmin)
  
  # Store data in a data frame 
  DATA_tra <- data.frame("irra"=irra_tra, "irralag144"=pred_tra[,1],"irralag2"=pred_tra[,2],
                                            "irralag3"=pred_tra[,3],"irralag4"=pred_tra[,4],
                                            "irralag5"=pred_tra[,5],"irralag6"=pred_tra[,6])
  DATA_for <- data.frame("irra_for"=NaN,  "irralag144"=pred_for[,1],"irralag2"=pred_for[,2],
                                            "irralag3"=pred_for[,3],"irralag4"=pred_for[,4],
                                            "irralag5"=pred_for[,5],"irralag6"=pred_for[,6])

  ## ----------------------------- MAKING PREDICTIONS -----------------------------
  Q01 <- seq(0.1,0.9,0.1)
  irra_prev <- array(rep(NaN, length(Q01)*N_boot), c(length(Q01),N_boot))
  
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
    for (q in 5)
    {
      qr_prev <- function(d)     # Definition of the forecasting function (quantile regression model)
      { mdl_fit <- rq( irra ~ irralag144 + irralag2 + irralag3 + irralag4 + irralag5 + irralag6 + 
                              irralag144*irralag2 + irralag2*irralag6,
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
  irra_prev <- mean(irra_prev[5,])   # This line will change after the optimization of the sample pick
  irra_prev <- base_irra$xmin + irra_prev * (base_irra$xmax - base_irra$xmin)
}




