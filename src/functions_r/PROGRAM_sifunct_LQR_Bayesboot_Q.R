library(bayesboot)
library(zoo)
library(quantreg)

rm(list=ls())

if (paste(Sys.getenv("COMPUTERNAME")) == "DESKTOP-1KHCJFC")
{str1 = "D:/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/Test_2021_07_15 load/"}
if (paste(Sys.getenv("COMPUTERNAME")) == "DESKTOP-JPI6MNC")
{str1 = "C:/Users/Pasquale/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/Test_2021_07_15 load/"}
if (paste(Sys.getenv("COMPUTERNAME")) == "LAPTOP-QPQ208J3")
{str1 = "C:/Users/defal/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/Test_2021_07_15 load/"}

source(paste0(str1,"Function_LQR_Bayesboot_Q.R"), echo = F)

load(paste0(str1,"DATAQ_tra.RData"))
load(paste0(str1,"DATAQ_for.RData"))
N_boot <- 200       # This is the desired number of bootstrap samples.

Results_Q <- array(NaN, 144)    # In real-time usage, this line is deleted.

# h <- 61           # h is the number of the target 10-min interval of the day. E.g., the 
                    # interval 10:00-10:10 is the 61th interval of the day

for (h in seq(1,144,1)) 
{
  print(paste0(h, "/ 144"))
  Results_Q[h] <- LQR_Bayesboot(Q_tra, predQ_tra, predQ_for[h,], h, N_boot)
}

base_Q <- list("xmin" = min(Q_tra), "xmax" = max(Q_tra))
Results_abs_Q <- Results_Q*(base_Q$xmax-base_Q$xmin) + base_Q$xmin

DATA_comparison <- data.frame(Q_for,Results_abs_Q)

save("Results_Q","base_Q","Results_abs_Q","Q_for", file = paste0(str1,"Results_pask_Q.RData"))
write.csv("Results_abs_Q", paste0(str1,"Results_pask_Q.csv")) 
