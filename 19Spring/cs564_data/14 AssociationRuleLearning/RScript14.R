library(grid)
library(vcd)
library(arules)
library(arulesViz)

#page 14
tab['bottled beer','red/blush wine']
tab['red/blush wine','red/blush wine']
48 / 189  #0.2539683
tab['white wine','white wine']
tab['bottled beer','white wine']
22 / 187  #0.1176471

#page 15
rules <- apriori(Groceries, parameter=list(supp=0.0015,conf=0.3),
                 appearance=list(default="lhs",rhs='bottled beer'))

#page 16
library(arules); library(arulesViz)
plot(rules, method="graph", measure='confidence', shading='lift',
     control=list(type="items"))

#page 17
plot(rules, method="grouped", control=list(type="items"))

#page 19
url <- "http://www.rdatamining.com/data/titanic.raw.rdata"
download.file(url, destfile="titanic.raw.RData", mode="wb")
load("titanic.raw.RData")

#page 20
library(arules)
titanic_rules <- apriori(titanic.raw, 
    parameter=list(minlen=2,supp=0.005,conf=0.8),
    appearance=list(rhs=c("Survived=No","Survived=Yes"),
    default="lhs"), control=list(verbose=FALSE))
quality(titanic_rules) <- round(quality(titanic_rules), digits=5)


