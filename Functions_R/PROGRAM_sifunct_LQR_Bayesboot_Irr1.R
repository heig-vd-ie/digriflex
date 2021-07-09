library(bayesboot)
library(zoo)
library(quantreg)

rm(list=ls())

source(paste0(str1,"Function_LQR_Bayesboot_irra.R"), echo = F)

load(paste0(str1,"DATA_tra.RData")) 
N_boot <- 300       # This is the desired number of bootstrap samples.

Results <- array(NaN, 54)    # In real-time usage, this line is deleted.

# h <- 61           # h is the number of the target 10-min interval of the day. E.g., the 
                    # interval 10:00-10:10 is the 61th interval of the day. Note that 
                    # for h<=33 and h>=121 (before 5:30 and after 20:00) the output is directly 0

load(paste0(str1,"pred_tot.RData")) 
irra_new <- DATA$I[22507:22704]

irralag144_for <- irra_new[1:54]
irralag2_for <- irra_new[(144-2+1):(144-2+54)]
irralag3_for <- irra_new[(144-3+1):(144-3+54)]
irralag4_for <- irra_new[(144-4+1):(144-4+54)]
irralag5_for <- irra_new[(144-5+1):(144-5+54)]
irralag6_for <- irra_new[(144-6+1):(144-6+54)]
pred_for <- data.frame(irralag144_for,irralag2_for,irralag3_for,irralag4_for,irralag5_for,irralag6_for) 

irra_for <- irra_new[(144+1):(144+54)]

for (h in seq(42+1,42+54,1)) 
{
  Results[(h-42)] <- LQR_Bayesboot(irra_tra,pred_tra,pred_for[(h-42),],h,N_boot)
}

base_irra <- list("xmin" = min(irra_tra), "xmax" = max(irra_tra))
Results_abs <- Results*(base_irra$xmax-base_irra$xmin) + base_irra$xmin

DATA_comparison <- data.frame(irra_for,Results_abs)

save("Results","base_irra","Results_abs","irra_for", file = paste0(str1,"Results_pask.RData"))
write.csv("Results_abs", paste0(str1,"Results_pask.csv")) 
