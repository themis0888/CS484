library(googleVis)
library(dplyr)
library(gapminder)
library(ggplot2); 
library(plotly)
library(devtools)
library(Rcpp)
library(githubinstall)
install_github('ramnathv/rCharts', force= TRUE)
library(rCharts)
require(reshape2)


bubble <- gvisBubbleChart(Fruits, idvar='Fruit', 
                          xvar='Sales', yvar='Expenses',
                          colorvar = 'Year', sizevar = 'Profit')

plot(bubble)

library(dplyr)
dat <- Fruits %>% group_by(Fruit) %>%
  summarize(Sale = mean(Sales), Expense=mean(Expenses))
dat

bar <- gvisBarChart(dat, xvar='Fruit',
        yvar=c('Sale', 'Expense'))
plot(bar)

ck <- gvisCandlestickChart(OpenClose, 
        xvar = 'Weekday', low='Low', open='Open',
        high='High', options=list(legend='none'))
plot(ck)

aq <- na.omit(airquality) %>% select(Ozone, Temp, Month) %>%
  group_by(Month) %>%
  summarize(OzoneMean=mean(Ozone), TempMean=mean(Temp))
aq <- data.frame(aq); head(aq)

Line <- gvisLineChart(aq, xvar='Month', yvar=c('OzoneMean','TempMean'),
                      options=list(gvis.editor='Edit me!'))
plot(Line)

library(ggvis)
mtcars %>% 
  ggvis(~wt, ~mpg, fill:='red', stroke:='black',
        size:=input_slider(10,100,label='point size'),
        opacity:= input_slider(0,1,label='opacity')) %>%
  layer_points()

iris %>% group_by(Species) %>%
  summarize(SubTotal=sum(Sepal.Length))

iris %>% ggvis(~Species, ~Sepal.Length, 
               fill:=input_select(c('red', 'green', 'blue'),
                                                           label='Fill Color')) %>% layer_bars()
iris %>% ggvis(~Sepal.Length, ~Sepal.Width, fill=~Species, 
               size:=input_slider(10,50, label='point size'), 
               opacity:=input_slider(0.1, 1, label='opacity')) %>%
    layer_points()

library(gapminder)                                  
library(ggplot2); library(plotly)
gap <- gapminder %>% 
  filter(year==1977) %>%
  ggplot(aes(x=gdpPercap, y=lifeExp,
             size = pop, color=continent)) +
  geopm_point() + scale_
           
