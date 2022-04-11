library(bayesboot)
library(zoo)
library(quantreg)

rm(list=ls())

if (paste(Sys.getenv("COMPUTERNAME")) == "DESKTOP-1KHCJFC")
{str1 = "D:/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/Test_2021_08_31 irragg_24h/"}
if (paste(Sys.getenv("COMPUTERNAME")) == "DESKTOP-JPI6MNC")
{str1 = "C:/Users/Pasquale/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/Test_2021_08_31 irragg_24h/"}
if (paste(Sys.getenv("COMPUTERNAME")) == "LAPTOP-QPQ208J3")
{str1 = "C:/Users/defal/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/Test_2021_08_31 irragg_24h/"}

source(paste0(str1,"Function_LQR_Bayesboot_irra_24h.R"), echo = F)

load(paste0(str1,"DATA_tra.RData")) 
N_boot <- 100       # This is the desired number of bootstrap samples.

Results <- array(NaN, 144)    # In real-time usage, this line is deleted.

# h <- 61           # h is the number of the target 10-min interval of the day. E.g., the 
                    # interval 10:00-10:10 is the 61th interval of the day. Note that 
                    # for h<=33 and h>=121 (before 5:30 and after 20:00) the output is directly 0

load(paste0(str1,"DATA_for.RData")) 

for (h in seq(1,144,1)) 
{ print(h)
  Results[h] <- LQR_Bayesboot(irra_tra,pred_tra,pred_for[h,],h,N_boot) }

base_irra <- list("xmin" = min(irra_tra), "xmax" = max(irra_tra))
Results_abs <- Results*(base_irra$xmax-base_irra$xmin) + base_irra$xmin

DATA_comparison <- data.frame(irra_for,Results_abs)

save("Results","base_irra","Results_abs","irra_for", file = paste0(str1,"Results_pask.RData"))
write.csv("Results_abs", paste0(str1,"Results_pask.csv")) 
