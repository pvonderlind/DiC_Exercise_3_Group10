library(readr)
library(ggplot2)
library(reshape2)
library(scales)

setwd("C:\\Users\\danie\\tuWien\\8_semester\\DiC\\DiC_Exercise_3_Group10\\base-docker-app\\obj_detection_container-vanilla")
res_loc <- read_csv("results_local.csv")
res_loc$request_time <- res_loc$request_time*1000
res_loc$prediction_time <- res_loc$prediction_time*1000

data_mod <- melt(res_loc, id.vars='...1', measure.vars=c('request_time', 'prediction_time'))



mean(res_loc$prediction_time)
mean(res_loc$request_time)



res_rem <- read_csv("results_remote.csv")
res_rem$request_time <- res_rem$request_time*1000
res_rem$prediction_time <- res_rem$prediction_time*1000

data_mod2 <- melt(res_rem, id.vars='...1', measure.vars=c('request_time', 'prediction_time'))


ggplot(data_mod) + geom_boxplot(aes(x=value, y=variable, color=variable)) +  scale_x_log10(limits=c(43, 1816)) + xlab("time in ms") + ylab("") + theme(legend.position = "none") +scale_y_discrete(labels=c("request","prediction"))
ggplot(data_mod2) + geom_boxplot(aes(x=value, y=variable, color=variable)) +  scale_x_log10(limits=c(43, 1816)) + xlab("time in ms") + ylab("") + theme(legend.position = "none") +scale_y_discrete(labels=c("request","prediction"))



mean(res_rem$prediction_time)
mean(res_rem$request_time)

