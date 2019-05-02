#library: gplots, gmodels, sjPlot, grid, vcd


#page 4
O = c(169,58,56,18,253,45,38,90)
tc = c("Frequency")
tr = c("Chinese","Indian","Korean","Maori","NZ European",
       "Other European","Pacific","Other")

#page 9
dt1 <- array(c(11,23,22, 33,14,13, 7,9,14), dim=c(3,3),
             dimnames=list("Residence City"=c("Boston","Montreal","Montpellier"),
                           "Favorite Baseball Team"=c("Blue Jays","Red Socks","Yankees")))
(dt1 <- as.table(dt1))

#page 12
library(gmodels)
CrossTable(dt1,prop.c=FALSE,prop.chisq=FALSE,prop.t=FALSE,
           expected=TRUE,format="SPSS")


#page 13
library(gplots)
balloonplot(t(dt1), label=TRUE, show.margins=FALSE,
            main="Balloon Plot for Residence City by Baseball Team")


#page 14
curve(dchisq(x,df=4),0,20,200,xlab= expression(chi^2),
      ylab="Probability Density")
qc <- qchisq(1-0.05,df=4)
abline(v=qc,col=4) #critical chi-square
abline(v=ct1$statistic,col=2) #statistic chi-square
legend("topright",c("Critical","Statistic"),lty=1,
       col=c(4,2))


#page 15
dt2 <- read.csv("SurveyData.csv",header=T)
head(dt2[,1:6],4)

#page 16
University <- factor(dt2$univ,levels=1:2,labels=c("Y","K"))
#c4 : I accept quickly a new fashion. (negative 1-5 positive)
FashionAcceptance <- factor(dt2$c4)
tb2 <- table(University,FashionAcceptance)
tb2
addmargins(tb2)

#page 16
library(gplots)
balloonplot(t(tb2), label=TRUE, show.margins=FALSE,
            main="Balloon Plot for Two Universities by FashionAcceptance")

#page 17
library(sjPlot)
set_theme(geom.label.size=4,axis.textsize=1.1,
          legend.pos="bottom")
sjp.xtab(University,FashionAcceptance,type="bar",y.offset=0.01,
         margin="row",coord.flip=TRUE,wrap.labels=7,
         geom.colors="Set2",show.summary=TRUE)


#page 19
bs <- read.csv("BoyScout.csv",header=TRUE)

#page 22
library(grid); library(vcd)
mosaic(bs3,shade=TRUE,legend=TRUE)
