#library: gclus, dplyr, NbClust, cluster, factoextra 

library(gclus)
library(dplyr)
data(wine)
scaled_wine <- scale(wine) %>% as.data.frame()
scaled_wine2 <- scaled_wine[-1]
head(scaled_wine2,2)

library(NbClust)
?NbClust

NbClust(scaled_wine2, method='complete', index='hartigan')$Best.nc
NbClust(scaled_wine2, method='complete', index='kl')$Best.nc

NbClust(scaled_wine2, method='complete')


library(cluster)
?clusGap
Gap <- clusGap(scaled_wine2, FUNcluster=pam, K.max=15)

print(Gap, method = 'firstmax')

library(factoextra)
fviz_gap_stat(Gap)

data(iris)
head(iris)
iris.scaled <- scale(iris[, -5])

library(NbClust)
nb <- NbClust(iris.scaled, distance = 'euclidean', min.nc = 2,
              max.nc = 10, method = 'complete', index = 'all')
n <- nb$Best.nc[1]
kc <- kmeans(iris.scaled, centers=n, nstart=4)
kc

clusplot(iris.scaled, kc$cluster, color=T, shade=T, labels=2)

sobj <- silhouette(kc$cluster, dist(iris.scaled))
summary(sobj)
plot(sobj, col=2:4)

pamx <- pam(iris.scaled, 3)
summary(pamx)
plot(pamx)


head(USArrests)
NbClust(USArrests, method='complete', index='hartigan')$Best.nc
ds <- dist(USArrests, method='euclidean')
hcst <- hclust(ds, method='complete')
plot(hcst, labels=rownames(USArrests), cex=0.8)
rect.hclust(hcst, 3)



