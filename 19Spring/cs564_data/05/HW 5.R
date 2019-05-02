# HW5
library('robustbase')

# Prob 1
# 1)
x <- mtcars[,'mpg']
axis_x <- seq(5,40,0.1)
y <- dnorm(axis_x, mean=mean(x), sd = sqrt(var(x)))
shapiro.test(x)
plot(axis_x, y, type='l')
hist(x, freq=F)
lines(density(x), col='86')
lines(axis_x,y, col='500')

qqnorm(x)
qqline(x)

# 2)
x1 <- mtcars[which(mtcars[,'am']==0),][,'mpg']
x2 <- mtcars[which(mtcars[,'am']==1),][,'mpg']
t.test(x1, x2, paired=F)
t.test(x1)
t.test(x2)



# Prob 2 
x = NOxEmissions[,'LNOx']
m = mean(x) # 4.378
n = length(x) # 8088
s = sd(x) # 0.937
a = 0.05; tc = qt(p=1-a/2, df=n-1) # 1.96
e = tc * s / sqrt(n) # 0.0204 ...?? -> 0.016
# Q. The number of observation??
# e = tc * s # 1.837 -> 0.946
ci = m + c(-e,e)
cat("Confidence intervals:", ci, " for x\n")

hist(x, freq=F) 
lines(density(x))
t.test(x)

length(which(x<ci[1])) / n
length(which(x>ci[2])) / n

# ratio of the instances inside of the intervals 
length(which((x<ci[2] & x>ci[1]))) / n










