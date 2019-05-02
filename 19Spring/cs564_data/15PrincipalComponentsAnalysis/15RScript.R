library(rela)
library(psych)
library(ca)
library(ggplot2)
library(factoextra)

#page 3
edat17 <- read.csv('index2017_data.csv', header=TRUE)
names(edat17)

#page 4
edat <- edat17[,-c(1:2)] #exclude 1:2 columns
str(edat)
for(i in c(1:4,7:10))
  edat[,i] <- as.numeric(edat[,i])

#page 14
library(psych)
pca <- principal(edat, nfactors=2, rotate='none')
biplot.psych(pca, col=c("black","red"), cex=c(0.5,1), arrow.len=0.08,
             main=NULL, labels=edat17[,1])

#page 15
dat <- read.csv("USA2016PresidentElection.csv",header=T)
head(dat,4)
xd <- dat[,-c(1,4,5)] #exclude state names
library(psych)
pcusa <- principal(xd, nfactors=2, rotate="none")
pcusa$loadings





