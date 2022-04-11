##### Path of installed packages
user_name_str <- Sys.info()[[7]]
R_libraries_path <- paste("C:/Users/",user_name_str,"/Documents/R/win-library/4.0", sep="")
Data_path <- paste("C:/Users/",user_name_str,"/Desktop/DiGriFlex_Code/Data/", sep="")
.libPaths(c(R_libraries_path))
sink("nul")

##### Load required librarues
library(bayesboot)
library(zoo, warn.conflicts = FALSE)
suppressPackageStartupMessages(library(quantreg))


##### Actual function
DayAhead_Bayesboot <- function(pred_for) {
  output <- data.frame(col1 = t(data.frame(pred_for[145:288])), col2 = 0 * t(data.frame(pred_for[145:288])),
                       col3 = 0 * t(data.frame(pred_for[145:288])))
  output <-  output
}
