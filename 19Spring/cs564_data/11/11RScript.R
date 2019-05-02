#library: TH.data, rpart, grid, mvtnorm, partykit, rattle

data(GlaucomaM, package='TH.data')
dim(GlaucomaM)

table(GlaucomaM$Class)

library(rpart)
cfit <- rpart(Class ~ ., data=GlaucomaM, method='class')
cfit

win.graph(8,7.5)
library(grid); library(partykit)
plot(as.party(cfit), tp_args=list(id=FALSE))


library(quint)
form1 <- I(cesdt1-cesdt3) ~ cond | nationality+marital+
  wcht1+age+trext+comorbid+disopt1+uncomt1+negsoct1
control1 <- quint.control(maxl=5, B=2)
quint1 <- quint(form1, data=subset(bcrp, cond<3), control=control1)

quint1$si
plot(quint1)


form2 <- I(physt3-physt1) ~ cond | cesdt1+negsoct1+uncomt1+
  disopt1+comorbid+age+wcht1+nationality+marital+trext
quint2 <- quint(form2, data=subset(bcrp, cond<3))
plot(quint2)