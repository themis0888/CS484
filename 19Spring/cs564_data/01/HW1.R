x=c(1,2,3,4,5,2,4,3,5,1,2,3,4,5,1,2)
y=c("Red","Green","Blue","Magenta")
y[x]

A_pre <- c(1,2,3,0,1,4,5,2,4)
B_pre <- c(2,3,0,-1,2,5,3,9,2)
A <- matrix(A_pre, nrow = 3, ncol = 3, byrow = TRUE)
B <- matrix(B_pre, nrow = 3, ncol = 3, byrow = TRUE)

C = A * B 

sum(state.x77[,"Income"] < 4000)

which(state.x77[,"Income"] == max(state.x77[,"Income"]))
