library: psych, DescTools,  coefplot

#page 5
HP <- read.csv('HappyIndex.csv', header=TRUE)
str(HP)
HD <- data.frame(Rank=HP[,1],Country=HP[,2],LifeExpectancy=HP[,4],Wellbeing=HP[,5],
                 Footprint=HP[,7],InequalityOutcome=HP[,8],HPI=HP[,11])


#page 7
xa = c("LifeExpectancy","Wellbeing","Footprint",
       "InequalityLifeOutcome","HPI")
par(mfrow=c(2,3))
for(i in 3:7) boxplot(HD[,i],notch=T,col='azure',xlab=xa[i-2])
par(mfrow=c(1,1))


#page 9
attach(HD); par(mfrow=c(3,2))
hist(LifeExpectancy,breaks=40,
     freq=FALSE,col='cyan')
lines(density(LifeExpectancy),col=2)
hist(Wellbeing,breaks=40,
     freq=FALSE,col='cyan')
lines(density(Wellbeing),col=2)
hist(Footprint,breaks=40,
     freq=FALSE,col='cyan')
lines(density(Footprint),col=2)
hist(InequalityOutcome,breaks=40,
     freq=FALSE,col='cyan')
lines(density(InequalityOutcome),col=2)
hist(HPI,breaks=40,
     freq=FALSE,col='cyan')
lines(density(HPI),col=2)
par(mfrow=c(1,1)); detach(HD)


#page 13
library(googleVis)
GC <- gvisGeoChart(HD, locationvar='Country', colorvar='LifeExpectancy', 
                   options=list(width=800,height=500, 
                                backgroundColor='lightblue',
                                colorAxis="{values:[48.9,57.6,66.3,75.0,83.6],
                                colors:[ \'magenta',\'orange',\'bisque',\'aquamarine',\'green']}"))
GT <- gvisTable(HRank[,c(1:3)],options=list(width=300,height=500))
plot(gvisMerge(GC,GT,horizontal=TRUE))


#page 16
fitL <- lm(HPI ~ LifeExpectancy, data=HD)
title = paste('HPI = ',round(fitL$coefficients[1],3),
              '+', round(fitL$coefficients[2],3), 'x LifeExpectancy')
plot(HD[,3],HD[,7], pch=21, bg='cyan', xlab='LifeExpectancy Score',
     ylab='Happy Planet Index (2016)', main=title)
abline(fitL, col=2)


#page 19
library(dplyr)
Mean_by_Cluster <- HDC %>% 
  select(HPI, LifeExpectancy, Wellbeing, Cluster) %>%
  group_by(Cluster) %>% 
  summarise(mean_HPI=mean(HPI), mean_LifeExpectancy=mean(LifeExpectancy),
            mean_Wellbeing=mean(Wellbeing)) 
Mean_by_Cluster


#page 20
cols <- c("purple","green","magenta")
par(mfrow=c(3,1),mar=c(2,4,1,1))
boxplot(HPI~Cluster, boxwex=0.75,xlab="Cluster", 
        ylab="Happy Planet Index",col=cols, data=HDC)
boxplot(LifeExpectancy~Cluster, boxwex=0.75, xlab="Cluster", 
        ylab="Life Expectancy",col=cols, data=HDC)
boxplot(Wellbeing~Cluster, boxwex=0.75,xlab="Cluster", 
        ylab="Wellbeing", col=cols, data=HDC)
par(mfrow=c(1,1))

