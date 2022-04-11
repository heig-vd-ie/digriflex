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
irralag144_for <- array(rep(0,144))
preslag144_for <- array(rep(0,144))
relhlag144_for <- array(rep(0,144))
templag144_for <- array(rep(0,144))
windlag144_for <- array(rep(0,144))
for (i in (L-144+1):(L)) 
{ irralag144_for[i-(L-144)] <- irra[i-144]
  preslag144_for[i-(L-144)] <- pres[i-144]
  relhlag144_for[i-(L-144)] <- relh[i-144]
  templag144_for[i-(L-144)] <- temp[i-144]
  windlag144_for[i-(L-144)] <- wind[i-144] }

irralag288_for <- array(rep(0,144))
preslag288_for <- array(rep(0,144))
relhlag288_for <- array(rep(0,144))
templag288_for <- array(rep(0,144))
windlag288_for <- array(rep(0,144))
for (i in (L-144+1):(L)) 
{ irralag288_for[i-(L-144)] <- irra[i-288]
  preslag288_for[i-(L-144)] <- pres[i-288]
  relhlag288_for[i-(L-144)] <- relh[i-288]
  templag288_for[i-(L-144)] <- temp[i-288]
  windlag288_for[i-(L-144)] <- wind[i-288] }

# Selecting one or two days before, depending on the hour of the day --------------
irralag_for <- array(rep(0,144))
preslag_for <- array(rep(0,144))
relhlag_for <- array(rep(0,144))
templag_for <- array(rep(0,144))
windlag_for <- array(rep(0,144))
for (i in (L-144+1):(L))
{ int <- i %% 144
  if (int >= 1 && int <= 108)
  { irralag_for[i-(L-144)] <- irralag144_for[i-(L-144)]
    preslag_for[i-(L-144)] <- preslag144_for[i-(L-144)] 
    relhlag_for[i-(L-144)] <- relhlag144_for[i-(L-144)] 
    templag_for[i-(L-144)] <- templag144_for[i-(L-144)] 
    windlag_for[i-(L-144)] <- windlag144_for[i-(L-144)] }
  else
  { irralag_for[i-(L-144)] <- irralag288_for[i-(L-144)]
    preslag_for[i-(L-144)] <- preslag288_for[i-(L-144)] 
    relhlag_for[i-(L-144)] <- relhlag288_for[i-(L-144)] 
    templag_for[i-(L-144)] <- templag288_for[i-(L-144)] 
    windlag_for[i-(L-144)] <- windlag288_for[i-(L-144)] }
}

# Average weather condition during the last observable day ------------------------
avgirra_for <- array(rep(0,144))
avgpres_for <- array(rep(0,144))
avgrelh_for <- array(rep(0,144))
avgtemp_for <- array(rep(0,144))
avgwind_for <- array(rep(0,144))
for (i in (L-144+1):(L)) 
{ D <- ((i-1) %/% 144)+1
  avgirra_for[i-(L-144)] <- mean(irra[(144*(D-2)+1):(144*(D-2)+108)])
  avgpres_for[i-(L-144)] <- mean(pres[(144*(D-2)+1):(144*(D-2)+108)])
  avgrelh_for[i-(L-144)] <- mean(relh[(144*(D-2)+1):(144*(D-2)+108)])
  avgtemp_for[i-(L-144)] <- mean(temp[(144*(D-2)+1):(144*(D-2)+108)])
  avgwind_for[i-(L-144)] <- mean(wind[(144*(D-2)+1):(144*(D-2)+108)]) }

# Pool the predictors
pred_for <- data.frame(irralag_for,preslag_for,relhlag_for,templag_for,windlag_for,
                       avgirra_for,avgpres_for,avgrelh_for,avgtemp_for,avgwind_for)


## SAVE DATA -------------------------------------------------------------------------------------
save(pred_for,irra_for, file = paste0(str1,"Test_2021_08_31 irragg_24h/DATA_for.RData"))

