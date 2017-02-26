#########################################################
# Construct an Adaboost algorithm using decision stumps as weak learner to classify USPS data
#########################################################

setwd("D:\\LiYuan\\ME_CU_MS\\Fall15\\ML Stat-Enroll\\Homework\\HW3")

uspsdata <- read.table("uspsdata.txt", sep="")
uspscl <- read.table("uspscl.txt", sep="")

indicator <- function(condition) ifelse(condition, 1, 0)

#########################################################
# train function
# implement decision stumps
#########################################################


train <- function(X, w, y){
  dat <- cbind(X, y, w)
  Qf <- NULL
  Qb <- NULL
  Q_grid <- NULL
  m_list <- NULL
  # decision stump classifier
  for (i in 1:ncol(X)){
    xi <- dat[order(dat[,i]),][,c(i,(ncol(X)+1),(ncol(X)+2))]

    for (j in 1:(nrow(y))){
      Qf[j] <- (crossprod(xi[,3]*indicator(xi[,2]!=-1), indicator(xi[,1]<=xi[j,1])) + 
                  crossprod(xi[,3]*indicator(xi[,2]!=1), indicator(xi[,1]>xi[j,1]))) / sum(w)
      Qb[j] <- (crossprod(xi[,3]*indicator(xi[,2]!=1), indicator(xi[,1]<=xi[j,1])) + 
                  crossprod(xi[,3]*indicator(xi[,2]!=-1), indicator(xi[,1]>xi[j,1]))) / sum(w)
    }
    
    if (length(which(Qb==min(Qb))) <= length(which(Qf==min(Qf)))) {
      t <- xi[which.min(Qf), 1]
      m <- 1
      Q_grid <- rbind(Q_grid, Qf)
    }else if (length(which(Qb==min(Qb))) > length(which(Qf==min(Qf)))) {
      t <- xi[which.min(Qb), 1]
      m <- -1
      Q_grid <- rbind(Q_grid, Qb)
    }
    m_list <- c(m_list,m)

  }
  
  j_split <- which.min(apply(Q_grid,1,min))
  x_optimal <- dat[order(X[,j_split]),][,c(j_split,(ncol(X)+1)),(ncol(X)+2)]
  t_split <- x_optimal[which.min(Q_grid[j_split,]),1]
  m <- m_list[j_split]
  pars <- c(j_split, t_split, m)
  return(pars)
}

#########################################################
# classify function
#########################################################

classify <- function(X, pars){
  j_opt <- pars[1]
  t_opt <- pars[2]
  m_l <- pars[3]
  y <- rep(0, nrow(X))
  ind <- which(X[,j_opt] > t_opt)
  y[ind] <- m_l
  y[-ind] <- -m_l
  return(y)
}

#########################################################
# implementation of train and classfiy for decision stumps
# for adaBoost training function
#########################################################
  
adaBoost <- function(X, y, B) {
  w <- rep(1/nrow(y), nrow(y))
  alpha <- NULL
  allPars <- NULL
  final_classifier <- rep(0, nrow(y))
  for(b in 1:B){  
    pars <- train(X, w, y)
    label <- classify(X, pars)
    err <- crossprod(w, indicator(y!=label)) / sum(w)
    alpha[b] <- log((1-err)/err)
    w <- w*exp(as.numeric(alpha[b])*indicator(y!=label))
    allPars <- rbind(allPars, pars)
  }
  return(list(alpha=alpha, allPars=allPars))

}

#########################################################
# agg_class function
#########################################################

agg_class <- function(X, alpha, allPars){
  
  B <- length(alpha)
  cl <- NULL
  if (B==1){
    cl <- cbind(cl, classify(X, t(allPars)))
  }else {
    for (b in 1:B){
      cl <- cbind(cl, classify(X, allPars[b,]))
    }
  }
  return(sign(cl%*%alpha))
}

#########################################################
# evaluation of adaboost algorithm on the USPS data using
# 5-fold cross validation with 40 weak learners
#########################################################

X <- uspsdata
y <- as.vector(uspscl)
w <- rep(1/nrow(y), nrow(y))

ada <- adaBoost(X, y, B)
c_hat <- agg_class(X, ada$alpha, ada$allPars)

K <- 5
B <- 40

err_train_mat <- matrix(0, B, K)
err_test_mat <- matrix(0, B, K)
for (k in 1:K){
  ind_test <- c(((k-1)*round(nrow(X)/K)+1):(round(nrow(X)/K)*k))
  train_ada <- adaBoost(X[-ind_test,], as.matrix(y[-ind_test,]), B)
  for (b in 1:B){
    c_hat_train <- agg_class(X[-ind_test,], train_ada$alpha[1:b], train_ada$allPars[1:b,])
    c_hat_test <- agg_class(X[ind_test,], train_ada$alpha[1:b], train_ada$allPars[1:b,])
    err_train_mat[b,k] <- sum(c_hat_train!=y[-ind_test,]) / nrow(X)
    err_test_mat[b,k] <- sum(c_hat_test!=y[ind_test,]) / nrow(X)
  }

}

err_train <- NULL
err_test <- NULL
for (b in 1:B){
  err_train <- c(err_train, mean(err_train_mat[b,]))
  err_test <- c(err_test, mean(err_test_mat[b,]))
}

# plot of training error and teating error as function of b
library(ggplot2)
ggplot(err_train, aes(x=1:B, y=err_train)) + geom_line() + geom_point() +
  ggtitle("Training Error of AdaBoost") +
  xlab("Number of Weak Learners") + ylab("Misclassification Rate") +
  ylim(0, 0.15)
ggplot(err_test, aes(x=1:B, y=err_test)) + geom_line() + geom_point() + 
  ggtitle("Testing Error of AdaBoost") +
  xlab("Number of Weak Learners") + ylab("Misclassification Rate") +
  ylim(0,0.15)


