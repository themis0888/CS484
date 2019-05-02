#####  Regression Analysis #####
# library: ggvis, nycflights13, dplyr, coefplot, psych, COUNT, 
install.packages(c('ggvis', 'nycflights13', 'dplyr', 'coefplot', 'psych', 'COUNT'))

str(longley)
ur <- lm(Employed ~ GNP, data=longley)
summary(ur)

with(longley, plot(GNP, Employed, pch=21, bg='cyan'))
lines(longley$GNP, ur$fitted.values, col='red')

predict(ur, list(GNP=300))
predict(ur, list(GNP=c(300, 500)))

#page 8
library(ggvis)
longley %>% ggvis(~GNP, ~Employed) %>%
  layer_points() %>%
  layer_model_predictions(model="lm",se=TRUE,stroke:="blue")

#page 9
library(nycflights13)
flights_df <- as.data.frame(flights)
head(flights_df,2)

#page 12
library(psych)
cordat <- base_dat %>% 
  select(dep_delay,arr_delay,distance,air_time)
pairs.panels(cordat)

#page 16
Hours <- c(0.50,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,2.50,
           2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50)
Pass	<- c(0,0,0,0,0,0,1,0,1,0, 1,0,1,0,1,1,1,1,1,1)
passhour <- data.frame(Hours, Pass)
head(passhour,3); tail(passhour,3)

#page 18
b = coef(out1)
x = 1:5; P = 1.0/(1+exp(-b[1]-b[2]*x))
cat("Probabilities of passing exam:\n",
    round(P,3),"for",x,"hours study")

#page 19
H = -b[1] / b[2]
cat("Boundary hour to pass exam:",H,"\n")

plot(Pass~Hours, pch=20,col="blue",
     main='Fitted Logistic Regression Line with Observed Data')
lines(Hours, out1$fitted, type="l", col="red")
abline(h=0.5,v=H,col="gray")


#page 21
data(badhealth, package="COUNT")
head(badhealth,2)
sapply(badhealth, function(x) length(unique(x)))
table(badhealth$badh)

#page 22
library(psych)
pairs.panels(badhealth)

