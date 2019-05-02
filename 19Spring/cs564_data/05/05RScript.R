##### 05 Probability, Distributions, and Hypothesis Test #####

#library:  


#page 3
xp <- sample(c("sucess","fail"), 100, replace=T, prob=c(0.8,0.2))
table(xp)

#page 4
x1=-4; x2=4
x <- seq(x1, x2, 0.1); y <- dnorm(x)
plot(x, y, type='l', xlim=c(x1,x2), ylim=c(0,0.4), 
     xlab="x", ylab="Probability Density")

#page 5
x <-  c(72,49,81,52,31, 38,81,58,58,73,
        43,56,45,54,40, 81,60,52,52,38,
        79,83,63,58,59, 71,89,73,77,60,
        65,60,69,88,75, 59,52,75,70,93,
        90,62,91,61,53, 83,32,49,39,57,
        39,28,67,74,61, 42,39,76,68,65,
        58,49,72,29,70, 56,48,60,36,79,
        72,65,40,49,37, 63,72,58,62,46)
hist(x,freq = F)
lines(density(x))

qqnorm(x)
qqline(x)

#page 8
m <- mean(x); s<- sd(x)
s1 <- pnorm(m+s, m, s)
s0 <- pnorm(m,m,s)
s10 <- round(s1 - s0, 4)*100; s10

#page 19
plot(function(x) dt(x,df=10), -4,4, ylim=c(0,0.4))

#page 20
a=0.05; (tc=qt(1-a/2,df=15)) 
tc
#t distribution
dt(x,df=15)
curve(dt(x,df=15),-5,5,200,
      col=4,ylab="Probability",
      xlab="t-value")
abline(v=tc,col='red')

#page 21
x = c(446,450,458,452,456,462,449,460,467,455)
cat("Confidence intervals:",ci," for x\n")

#page 31
data(energy, package="ISwR")
str(energy)
head(energy)

#page 32
var.test(expend~stature, data=energy)$p.value 
two <- t.test(expend~stature, data=energy,
              alternative="two.sided", var.equal=TRUE)
two

#page 33
load("glib.RData")
x2=5; t_curve(x2,-tc)
text(ts,0.15,paste("t=",ts))
abline(v=ts)

#page 34
boxplot(expend~stature, data=energy, xlab="Woman group",
        ylab="24 hour energy expenditure in MJ", 
        col=c('cyan','magenta'))

#page 35
dt <- read.csv("PrePost.csv", header=T)


