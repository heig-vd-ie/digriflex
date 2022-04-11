library(bayesboot)
library(zoo)
library(quantreg)

rm(list=ls())

if (paste(Sys.getenv("COMPUTERNAME")) == "DESKTOP-1KHCJFC")
{str1 = "D:/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/Test_2021_09_07 load_24h/"}
if (paste(Sys.getenv("COMPUTERNAME")) == "DESKTOP-JPI6MNC")
{str1 = "C:/Users/Pasquale/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/Test_2021_09_07 load_24h/"}
if (paste(Sys.getenv("COMPUTERNAME")) == "LAPTOP-QPQ208J3")
{str1 = "C:/Users/defal/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/Test_2021_09_07 load_24h/"}

source(paste0(str1,"Function_LQR_Bayesboot_Q_24h_v4.R"), echo = F)

load(paste0(str1,"DATA_tra_v4.RData"))
load(paste0(str1,"DATA_for_v4.RData"))
N_boot <- 30       # This is the desired number of bootstrap samples.

# Normalize the inputs 
base_P <- list("xmin" = min(P_tra), "xmax" = max(P_tra))
P_tra <- (P_tra-base_P$xmin)/(base_P$xmax-base_P$xmin)
pred_tra[,c(1,3,5,7)] <- (pred_tra[,c(1,3,5,7)]-base_P$xmin)/(base_P$xmax-base_P$xmin)
pred_for[,c(1,3,5,7)] <- (pred_for[,c(1,3,5,7)]-base_P$xmin)/(base_P$xmax-base_P$xmin)

base_Q <- list("xmin" = min(Q_tra), "xmax" = max(Q_tra))
Q_tra <- (Q_tra-base_Q$xmin)/(base_Q$xmax-base_Q$xmin)
pred_tra[,c(2,4,6,8)] <- (pred_tra[,c(2,4,6,8)]-base_Q$xmin)/(base_Q$xmax-base_Q$xmin)
pred_for[,c(2,4,6,8)] <- (pred_for[,c(2,4,6,8)]-base_Q$xmin)/(base_Q$xmax-base_Q$xmin)

Results_Q <- array(rep(NaN, 144*3), c(144,3))    # In real-time usage, this line is deleted.

# h <- 61           # h is the number of the target 10-min interval of the day. E.g., the 
                    # interval 10:00-10:10 is the 61th interval of the day
for (h in seq(1,144,1)) 
{
  print(paste0(h, "/ 144"))
  Results_Q[h,c(1,2,3)] <- LQR_Bayesboot(Q_tra, pred_tra, pred_for[h,], h, N_boot)
}

Results_abs_Q <- Results_Q*(base_Q$xmax-base_Q$xmin) + base_Q$xmin

DATA_comparison <- data.frame(Q_for, Results_abs_Q)

save("Results_Q","base_P","base_Q","Results_abs_Q","Q_for", file = paste0(str1,"Results_pask_Q_v4.RData"))
write.csv("Results_abs_Q", paste0(str1,"Results_pask_Q_v4.csv")) 
