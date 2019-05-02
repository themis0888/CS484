
#library: HSAUR3, Rmisc, ggplot2, psych

#page 4
boxplot(count~spray, data=InsectSprays,
        xlab='Type of Insect Spray',
        ylab='Number of Dead Insects', col=2:7)
abline(h=mean(InsectSprays$count),col='gray')

#page 7
df1 = 5; df2 = 66
a = 0.05
Fc = qf(1-a, df1, df2)
Fc

#page 8
print(model.tables(aov.out,"means"),digits=3)  
plot.design(InsectSprays)

#page 9
tkh <- TukeyHSD(aov.out, conf.level=0.95)
tkh
plot(tkh, las=1)

#page 12
library(HSAUR3)
head(schooldays,2)

#page 13
library(Rmisc)
sum = summarySE(schooldays,measurevar="absent", 
                groupvars=c("race","school"))

library(ggplot2)
pd = position_dodge(0.3)
ggplot(sum, aes(x=school,y=absent,color=race)) + 
  geom_errorbar(aes(ymin=absent-se,ymax=absent+se), 
                width=0.2,size=0.7,position=pd) +
  geom_point(shape=16, size=3, position=pd) +
  scale_color_manual(values=c("red","blue")) +  
  theme(legend.position=c(0.13,0.85))  

#page 15
print(model.tables(aov2,"means"),digits=4)
plot.design(absent ~ race+school, data=schooldays)

#page 16
with(schooldays, interaction.plot(x.factor=school, 
                                  trace.factor=race, response=absent, col=2:3))

#page 21
cdt <- read.csv("ComparingColleges.csv",header=T)
attach(cdt); dim(cdt)
head(cdt)
table(cdt$School_Type)

#page 22
Y <- cbind(SAT,Acceptance,StudentP,PhDP,GradP,Top10P)
table(School_Type)
fit <- manova(Y ~ School_Type)
summary(fit,test="Pillai")

#page 24
tapply(StudentP,School_Type,mean)
tapply(PhDP,School_Type,mean)
tapply(Top10P,School_Type,mean)
par(mai=c(0.8,0.8,0.2,0.2),mfrow=c(1,3))
plot.design(StudentP~School_Type)
plot.design(PhDP ~ School_Type)
plot.design(Top10P ~ School_Type)

#page 25
library(psych)
db <- describeBy(cdt[,3:8],School_Type)
db$"Lib Arts"[,c(1:5,8:10,13)]
db$"Univ"[,c(1:5,8:10,13)]







