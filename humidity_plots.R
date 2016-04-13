library(ggplot2)
library(lubridate)
library(readr)
library(scales)

# The competition data is stored in the ../input directory
train <- read_csv("./data/train.csv")

# Write some basic stats to the log
cat("Number of training rows ", nrow(train), "\n")
head(train)
train$hour  <- hour(ymd_hms(train$datetime))
train$times <- as.POSIXct(strftime(ymd_hms(train$datetime), format="%H:%M:%S"), format="%H:%M:%S")
train$jitterTimes <- train$times+minutes(round(runif(nrow(train),min=0,max=59)))
train$day <- wday(ymd_hms(train$datetime), label=TRUE)
train$month <- month(ymd_hms(train$datetime))
train$year <- year(ymd_hms(train$datetime))
train[train$year == 2012,]$month <- train[train$year == 2012,]$month + 12
#summary(train)
train$reg_sc <- NA
reg_sc_wd0 <- tapply(train[train$workingday ==0,]$casual, as.factor(train[train$workingday ==0,]$month), FUN = function(x) x/max(x))
reg_sc_wd0[[1]]
#reg_sc_wd0 <- lapply(reg_sc_wd0, cbind)
reg_sc_wd0f <- c()
for (i in 1:24) reg_sc_wd0f <- c(reg_sc_wd0f, reg_sc_wd0[[i]])
#summary(reg_sc_wd0f)
tapply(reg_sc_wd0f, as.factor(train[train$workingday ==0,]$month), summary)
train$reg_sc <- NA
train[train$workingday ==0,]$reg_sc <- reg_sc_wd0f

p <- ggplot(train[train$workingday==0,], aes(x=jitterTimes, color=humidity, y=reg_sc)) +
  geom_point(aes(colour = humidity)) +
  theme_light(base_size=20) +
  xlab("Hour of the Day") +
  scale_x_datetime(breaks = date_breaks("4 hours"), labels=date_format("%I:%M %p")) + 
  ylab("Casual: Bike Rentals (working day):\n Scaled per Month (rentals_month/max(rentals_month)") +
  scale_colour_gradientn("Hum ", colours=c("#5e4fa2", "#3288bd", "#66c2a5", "#abdda4", "#e6f598", "#fee08b", "#fdae61", "#f46d43", "#d53e4f", "#9e0142")) +
  ggtitle("On weekends, any deducible effect of humidity?\n") +
  theme(axis.text=element_text(size=10), axis.title.y=element_text(size=12), axis.title.x=element_text(size=12), plot.title=element_text(size=18))
p
ggsave("humidity__casual_rentals_scaledpermonth.png", p)
