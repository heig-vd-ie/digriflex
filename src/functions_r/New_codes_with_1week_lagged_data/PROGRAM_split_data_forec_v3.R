library(R.matlab)

rm(list=ls())

if (paste(Sys.getenv("COMPUTERNAME")) == "DESKTOP-1KHCJFC")
{str1 = "D:/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/"}
if (paste(Sys.getenv("COMPUTERNAME")) == "DESKTOP-JPI6MNC")
{str1 = "C:/Users/Pasquale/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/"}
if (paste(Sys.getenv("COMPUTERNAME")) == "LAPTOP-QPQ208J3")
{str1 = "C:/Users/defal/OneDrive - Università degli Studi di Napoli Parthenope/PROGETTI DI RICERCA/CARPITA 2018/2021_04 R function for Mohammad/"}

# load(paste0(str1,"DATA_PQ_FINAL.RData")) 
# 
# NEW_dates1 <- seq(as.POSIXct("2016-02-01 00:10:00", tz = "UTC"), as.POSIXct("2016-06-10 00:00:00", tz = "UTC"), "10 min")
# NEW_dates2 <- seq(as.POSIXct("2016-06-17 00:10:00", tz = "UTC"), as.POSIXct("2017-01-19 00:00:00", tz = "UTC"), "10 min")
# NEW_dates3 <- seq(as.POSIXct("2017-01-26 00:10:00", tz = "UTC"), as.POSIXct("2017-06-21 00:00:00", tz = "UTC"), "10 min")
# NEW_dates4 <- seq(as.POSIXct("2017-07-05 00:10:00", tz = "UTC"), as.POSIXct("2018-03-24 00:00:00", tz = "UTC"), "10 min")
# NEW_dates5 <- seq(as.POSIXct("2018-04-28 00:10:00", tz = "UTC"), as.POSIXct("2018-12-07 00:00:00", tz = "UTC"), "10 min")
# NEW_dates6 <- seq(as.POSIXct("2018-12-21 00:10:00", tz = "UTC"), as.POSIXct("2019-07-12 00:00:00", tz = "UTC"), "10 min")
# NEW_dates7 <- seq(as.POSIXct("2019-07-26 00:10:00", tz = "UTC"), as.POSIXct("2020-03-06 00:00:00", tz = "UTC"), "10 min")
# NEW_dates8 <- seq(as.POSIXct("2020-07-10 00:10:00", tz = "UTC"), as.POSIXct("2021-04-01 00:00:00", tz = "UTC"), "10 min")
# NEW_dates <- c(NEW_dates1, NEW_dates2, NEW_dates3, NEW_dates4, NEW_dates5, NEW_dates6, NEW_dates7, NEW_dates8)
# rm(NEW_dates1, NEW_dates2, NEW_dates3, NEW_dates4, NEW_dates5, NEW_dates6, NEW_dates7, NEW_dates8)
# 
# DATA <- data.frame(NEW_dates, array(0, length(NEW_dates)), array(0, length(NEW_dates)))
# colnames(DATA) <- c("Date and time","P","Q")
# 
# set.seed(230586)
# 
# k <- 0
# for (i in 1:length(NEW_dates))
# {
#   print(paste(i, "/", length(NEW_dates)))
#   
#   if (i %% 6 == 1 || i %% 6 == 3 || i %% 6 == 4 || i %% 6 == 0)
#   {
#     k <- k+1
#     DATA[i,2:3] <- NEW_DATA[k,2:3]
#   }
#   if (i %% 6 == 2 || i %% 6 == 5)
#   {
#     delta <- abs(NEW_DATA[k,2]-NEW_DATA[(k+1),2])/2
#     DATA[i,2] <- mean(c(NEW_DATA[k,2],NEW_DATA[(k+1),2])) + runif(1, -delta*0.25, +delta*0.25)
#     delta <- abs(NEW_DATA[k,3]-NEW_DATA[(k+1),3])/2
#     DATA[i,3] <- mean(c(NEW_DATA[k,3],NEW_DATA[(k+1),3])) + runif(1, -delta*0.25, +delta*0.25)
#   }
# }
# save("DATA", file = paste0(str1,"DATA_PQ_FINAL_10min.RData"))
load(paste0(str1,"DATA_PQ_FINAL_10min.RData"))

P <- DATA$P
Q <- DATA$Q

P[P < 0] <- 0             # negative active power is set at 0

L <- length(P)

P_tra <- P[(1008+1):(L-144)]
P_for <- P[((L-144)+1):L]

Q_tra <- Q[(1008+1):(L-144)]
Q_for <- Q[((L-144)+1):L]

# Creating lagged time series - TRAINING ------------------------------------------
Plag144_for <- array(rep(0,144))
Qlag144_for <- array(rep(0,144))
for (i in (L-144+1):(L)) 
{ Plag144_for[i-(L-144)] <- P[i-144]
  Qlag144_for[i-(L-144)] <- Q[i-144] }

Plag288_for <- array(rep(0,144))
Qlag288_for <- array(rep(0,144))
for (i in (L-144+1):(L)) 
{ Plag288_for[i-(L-144)] <- P[i-288]
  Qlag288_for[i-(L-144)] <- Q[i-288] }

Plag1008_for <- array(rep(0,144))
Qlag1008_for <- array(rep(0,144))
for (i in (L-144+1):(L)) 
{ Plag1008_for[i-(L-144)] <- P[i-1008]
  Qlag1008_for[i-(L-144)] <- Q[i-1008] }

# Selecting one or two days before, depending on the hour of the day --------------
Plag_for <- array(rep(0,144))
Qlag_for <- array(rep(0,144))
for (i in (L-144+1):(L))
{ int <- i %% 144
  if (int >= 1 && int <= 108)
  { Plag_for[i-(L-144)] <- Plag144_for[i-(L-144)]
    Qlag_for[i-(L-144)] <- Qlag144_for[i-(L-144)] }
else
  { Plag_for[i-(L-144)] <- Plag288_for[i-(L-144)]
    Qlag_for[i-(L-144)] <- Qlag288_for[i-(L-144)] }
}

# Average weather condition during the last observable day ------------------------
avgP_for <- array(rep(0,144))
avgQ_for <- array(rep(0,144))
for (i in (L-144+1):(L)) 
{ D <- ((i-1) %/% 144)+1
  avgP_for[i-(L-144)] <- mean(P[(144*(D-2)+1):(144*(D-2)+108)])
  avgQ_for[i-(L-144)] <- mean(Q[(144*(D-2)+1):(144*(D-2)+108)]) }

# Pool the predictors
pred_for <- data.frame(Plag_for, Qlag_for, Plag1008_for, Qlag1008_for, avgP_for, avgQ_for)

## SAVE DATA TRAINING -----------------------------------------------------------------------------
save(pred_for, P_for, Q_for, file = paste0(str1,"Test_2021_09_07 load_24h/DATA_for_v3.RData"))




