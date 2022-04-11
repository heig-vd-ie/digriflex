library(R.matlab)

rm(list=ls())

if (paste(Sys.getenv("COMPUTERNAME")) == "DESKTOP-1KHCJFC")
{str1 = "D:/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/"}
if (paste(Sys.getenv("COMPUTERNAME")) == "DESKTOP-JPI6MNC")
{str1 = "C:/Users/Pasquale/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/"}
if (paste(Sys.getenv("COMPUTERNAME")) == "LAPTOP-QPQ208J3")
{str1 = "C:/Users/defal/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/"}

load(paste0(str1,"DATA.RData")) 

irra[irra < 0] <- 0             # negative irradiance is set at 0

L <- length(irra)

irra_tra <- irra[(288+1):(L-144)]
irra_for <- irra[((L-144)+1):L]

# Creating lagged time series - TRAINING ------------------------------------------
irralag144_tra <- array(rep(0,L-288-144))
preslag144_tra <- array(rep(0,L-288-144))
relhlag144_tra <- array(rep(0,L-288-144))
templag144_tra <- array(rep(0,L-288-144))
windlag144_tra <- array(rep(0,L-288-144))
for (i in (288+1):(L-144)) 
{ irralag144_tra[i-288] <- irra[i-144]
  preslag144_tra[i-288] <- pres[i-144]
  relhlag144_tra[i-288] <- relh[i-144]
  templag144_tra[i-288] <- temp[i-144]
  windlag144_tra[i-288] <- wind[i-144] }

irralag288_tra <- array(rep(0,L-288-144))
preslag288_tra <- array(rep(0,L-288-144))
relhlag288_tra <- array(rep(0,L-288-144))
templag288_tra <- array(rep(0,L-288-144))
windlag288_tra <- array(rep(0,L-288-144))
for (i in (288+1):(L-144)) 
{ irralag288_tra[i-288] <- irra[i-288]
  preslag288_tra[i-288] <- pres[i-288]
  relhlag288_tra[i-288] <- relh[i-288]
  templag288_tra[i-288] <- temp[i-288]
  windlag288_tra[i-288] <- wind[i-288] }

# Selecting one or two days before, depending on the hour of the day --------------
irralag_tra <- array(rep(0,L-288-144))
preslag_tra <- array(rep(0,L-288-144))
relhlag_tra <- array(rep(0,L-288-144))
templag_tra <- array(rep(0,L-288-144))
windlag_tra <- array(rep(0,L-288-144))
for (i in (288+1):(L-144))
{ int <- i %% 144
  if (int >= 1 && int <= 108)
  { irralag_tra[i-288] <- irralag144_tra[i-288]
    preslag_tra[i-288] <- preslag144_tra[i-288] 
    relhlag_tra[i-288] <- relhlag144_tra[i-288] 
    templag_tra[i-288] <- templag144_tra[i-288] 
    windlag_tra[i-288] <- windlag144_tra[i-288] }
  else
  { irralag_tra[i-288] <- irralag288_tra[i-288]
    preslag_tra[i-288] <- preslag288_tra[i-288] 
    relhlag_tra[i-288] <- relhlag288_tra[i-288] 
    templag_tra[i-288] <- templag288_tra[i-288] 
    windlag_tra[i-288] <- windlag288_tra[i-288] }
}

# Average weather condition during the last observable day ------------------------
avgirra_tra <- array(rep(0,L-288-144))
avgpres_tra <- array(rep(0,L-288-144))
avgrelh_tra <- array(rep(0,L-288-144))
avgtemp_tra <- array(rep(0,L-288-144))
avgwind_tra <- array(rep(0,L-288-144))
for (i in (288+1):(L-144)) 
{ D <- ((i-1) %/% 144)+1
  avgirra_tra[i-288] <- mean(irra[(144*(D-2)+1):(144*(D-2)+108)])
  avgpres_tra[i-288] <- mean(pres[(144*(D-2)+1):(144*(D-2)+108)])
  avgrelh_tra[i-288] <- mean(relh[(144*(D-2)+1):(144*(D-2)+108)])
  avgtemp_tra[i-288] <- mean(temp[(144*(D-2)+1):(144*(D-2)+108)])
  avgwind_tra[i-288] <- mean(wind[(144*(D-2)+1):(144*(D-2)+108)]) }

# Pool the predictors
pred_tra <- data.frame(irralag_tra,preslag_tra,relhlag_tra,templag_tra,windlag_tra,
                       avgirra_tra,avgpres_tra,avgrelh_tra,avgtemp_tra,avgwind_tra)


## SAVE DATA -------------------------------------------------------------------------------------
save(pred_tra,irra_tra, file = paste0(str1,"Test_2021_08_31 irragg_24h/DATA_tra.RData"))

