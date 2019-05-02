# HW 9
library(maps)
library(mapdata)
library(googleVis)

# Problem 1
map('worldHires', c('UK', 'Ireland', 'Isle of Man', 'Isle of Wight', 'Wales'),
    fill=TRUE, col="green", xlim=c(-11,3), ylim=c(49,60.9))

map.cities(country = "UK", capitals = 1, col= 'blue')


# Problem 2
data("Population")
head(Population[, 1:3], 3)

WP = data.frame(Country=Population$Country,
                 Population.in.millions=round(Population$Population/1e6,0),
                 Rank = as.numeric(Population$Rank))

G1 <- gvisGeoChart(WP, "Country", "Population.in.millions", "Rank", 
                   options=list(dataMode="regions", width=800, height=600))

GT <- gvisTable(WP, options=list(width=300,height=340))
G2 <- gvisMerge(G1,GT,horizontal=TRUE)

plot(G2)

