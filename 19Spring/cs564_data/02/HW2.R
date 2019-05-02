# HW2 

library(MASS)
library(ggplot2)
library(datasets)

#Prob 1

data(Animals)
brain <- Animals[,1]
body <- Animals[,2]

plot(log(body), log(brain), xlab="Body (Log)", ylab ="Brain (Log)")
text(log(body), log(brain), row.names(Animals), cex=2/3)

ggplot(Animals, aes(x=log(body), y=log(brain), label=row.names(Animals)),
	xtitle=xlab("Body (Log)") + ylab("Brain (Log)")) + 
	geom_point() + geom_text()

#Prob 2 

caffe <- matrix(c(652, 1537,598,242,36,46,38,21,218,327,106,67),
	nrow=3,byrow=T)
rownames(caffe) <- c("Married","Prev.married","Single")
colnames(caffe) <- c("0","1-150","151-300",">300")

total_caffe <- margin.table(caffe,1)
total_caffe
barplot(total_caffe)

#Prob 3

ggplot(data=iris, aes(x=Sepal.Width)) + 
geom_histogram(binwidth=0.3, color="black", aes(fill=Species)) + 
xlab("Sepal Width") + ylab("Frequency") + 
ggtitle("Histogram of Sepal Width")


#Prob 4

data(airquality)

aq_trim <- airquality[which(airquality$Month == 7 |
	airquality$Month == 8 |
	airquality$Month == 9), ]
aq_trim$Month <- factor(aq_trim$Month,
	labels = c("July", "August", "September"))


ggplot(aq_trim, aes(x = Day, y = Ozone, fill = Month, size = Wind)) +
ggtitle("Air Quality in New York by Day") +
labs(x = "Day of the month", y = "Ozone (ppb)") +
scale_x_continuous(breaks = seq(1, 31, 5)) + geom_point(shape = 21)


