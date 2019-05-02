# HW 6
library('gmodels')
library('grid')
library('vcd')
library('tidyverse')

# Prob 1
dt1 <- array(c(14, 4, 7, 10, 15, 9, 3, 11, 5), dim=c(3,3),
              dimnames=list('Preference'=c('Music', 'News-talk', 'Sports'),
                            'Age'=c('Young Adult', 'Middle Age', 'Older Adult')))

dt <- as.table(dt1)
CrossTable(dt1, prop.c=F, prop.chisq=F, prop.t=F,
           expected=T, format='SPSS')

(ct1 <- chisq.test(dt1))

curve(dchisq(x,df=4),0,20,200,xlab= expression(chi^2),
      ylab="Probability Density")
qc <- qchisq(1-0.05,df=4)
abline(v=qc,col=4) #critical chi-square
abline(v=ct1$statistic,col=2) #statistic chi-square
legend("topright",c("Critical","Statistic"),lty=1,
       col=c(4,2))

mosaic(dt1, shade=T, legend=T)

# Prob 2 
# 1)
bs <- data.frame(Arthritis)
head(Arthritis)
bs %>% select('Treatment', 'Sex', 'Improved')

bs3 <- xtabs(~Treatment+Sex+Improved, data=bs)

# 2)
mosaic(bs3, shade=T, legend=T)

# 3)
mantelhaen.test(bs3)

# 4)
doubledecker(bs3)

# 5)
bs <- bs[which(bs['Sex']=='Female'),]
bs %>% select('Treatment', 'Improved')
bs3 <- xtabs(~Treatment+Improved, data=bs)
mosaic(bs3, gp=shading_max, legend=T)
