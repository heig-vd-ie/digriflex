library(bayesboot)
library(zoo)
library(quantreg)

rm(list=ls())

source('Function_LQR_Bayesboot.R', echo = F)

load("C:\Users\mohammad.rayati\Desktop\20210315\Functions\R_functions/DATA_tra.RData") 
                    # This file must contain all the available training data with 10-min  
                    # resolution; it won't change during real-time operation. The first entry 
                    # is measured in the 00:00-00:10 interval of the first available day.

N_boot <- 300       # This is the desired number of bootstrap samples.


Results <- array(NaN, 144)    # In real-time usage, this line is deleted.

# h <- 61           # h is the number of the target 10-min interval of the day. E.g., the 
                    # interval 10:00-10:10 is the 61th interval of the day. Note that 
                    # for h<=33 and h>=121 (before 5:30 and after 20:00) the output is directly 0

for (h in seq(1,144,1)) {   # In real-time usage, this "for" loop shall be deleted and the line above shall be uncommented

  load("C:\Users\mohammad.rayati\Desktop\20210315\Functions\R_functions/DATA_for.RData") 
                    # This file is called here only for testing the performance. It contains
                    # the predictors used in the model at each iteration: 
                    # - the last available observations, i.e., P(t-2), irra(t-2);
                    # - the observations of the day before, i.e., P(t-144), irra(t-144).
                    # In real-time usage, these four variables should be passed from Python.
  pred_for <- pred_for[h,]    # In real-time usage, this line shall be deleted.
  # pred_for <- data.frame(Plag144_for,Plag2_for,irralag144_for,irralag2_for) # In real-time usage, this line shall be uncommented.

  Results[h] <- LQR_Bayesboot(P_tra,pred_tra,pred_for,h,N_boot)    # This line calls the function.

}    # In real time usage, this line shall be commented.