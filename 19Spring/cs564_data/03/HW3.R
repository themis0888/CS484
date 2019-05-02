# HW3 


library(xml2)
library(rvest)
library(ggplot2)

# Prob1 
web <- read_html("https://en.wikipedia.org/wiki/World_Happiness_Report")
tbl <- html_nodes(web, "table")

tbl5 <- tbl %>% .[5] %>%
  html_table(fill=TRUE)
  
table5 = as.data.frame(tbl5)

ggplot(table5, aes(x=table5[,3],y =table5[,4]), xtitle=xlab("Score") + ylab("GDP per Capita")) + stat_smooth(level = 0.95) + geom_point()

# Prob2
library(gtrendsR)
LS <- gtrends(keyword=c("LG","SAMSUNG"), geo=c("KR", "KR","US", "US"), time="today 12-m")
plot(LS)

