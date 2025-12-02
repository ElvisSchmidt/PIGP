library(rstan)
library(MASS)
library(mvtnorm)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

#--------------------Compute important quantities---------------
pre_process <- function(I,K1,K2,l1x,l1t) {
  M <- nrow(I)
  K <- matrix(0,2*M,2*M)
  KL <- matrix(0,2*M,2*M)
  LK <- matrix(0,2*M,2*M)
  LKL <- matrix(0,2*M,2*M)
  
  for (i in 1:M) {
    for (j in 1:M) {
      dx <- I[i,1] - I[j,1]
      dt <- I[i,2] - I[j,2]
      
      K[i,j] <- K1[i,j]
      K[i+M,j+M] <-K2[i,j]
      LK[i,j] <- (-1/ (l1x^2) +  dx^2/(l1x^4)) * K1[i,j] #Upper left block
      LK[i+M,j] <- (- dt/(l1t^2) )* K1[i,j]
      KL[i,j] <- LK[i,j]
      KL[i,j+M] <- (- LK[i+M,j]) #Lower left block
      
      LKL[i,j] <- (3/l1x^4 - 6*dx^2/l1x^6 + dx^4/l1x^8 ) * K1[i,j]
      LKL[i+M,j] <- (-1/l1x^2 + dx^2/l1x^4) * (-dt/l1t^2) * K1[i,j]
      LKL[i,j+M] <- (-1/l1x^2 + dx^2/l1x^4) * ( dt/l1t^2) * K1[i,j]
      LKL[i+M,j+M] <- (1/l1t^2 - dt^2/l1t^4) *K1[i,j]
    }
  }
  
  
  
  m <- t(solve(t(K), t(LK)))
  K_deriv <- LKL - m %*% KL
  K_deriv <- (K_deriv + t(K_deriv))/2 #+1e-12*diag(nrow(K_deriv))
  evals <- eigen(K_deriv)$values
  evallsK <- eigen(K)$values
  list(K=K,KL=KL,LK=LK,LKL=LKL,m=m,K_deriv=K_deriv)  
}
#-----------------------Data Preparing-------------------
#Load Data
data <- read.csv("SimulatedDataKPP.csv", header = TRUE)
#data <- read.csv("CellData24-48.csv", header = TRUE)
# Extract columns
y <- data[, 3]
x <- data[, 1]
t <- data[, 2]
noise <- rnorm(length(y), mean = 0, sd = 0.01) 
y <- y +noise# normalize y
# Normalize x and t to [0, 1]
x_norm <- (x - min(x)) / (max(x) - min(x))
t_norm <- (t - min(t)) / (max(t) - min(t))
X_norm <- cbind(x_norm, t_norm)
#X_norm <- cbind(x,t)
print(range(X_norm[,1]))  # should be 0 → 1
print(range(X_norm[,2]))
print( length(X_norm) )
#total nr of datapoints
N = length(y)
#----------------Data Training-------------------------

l1x <- 0.16
l1t <- 1.63
outputscale <- 0.45
sigma_init <- 0.01
cat("----Hyperparameter tuning results C1:-------")
cat(paste("\nl1x:             ", l1x, "\nl1t:             ", l1t, "\noutputscale:    ", outputscale, "\ninitial sigma: ", sigma_init,"\n"))
K1 <-matrix(0,N,N)
LK_prel <- matrix(0,N,N)
for (i in (1:N)){
  for (j in (1:N)){
    dx <- X_norm[i,1] - X_norm[j,1]
    dt <- X_norm[i,2] - X_norm[j,2]
    K1[i,j]= outputscale^2*exp(-0.5*(dx^2/l1x^2 +dt^2/l1t^2) )
    
    if (i==j) { K1[i,j] <- K1[i,j]+1e-6}
    LK_prel[i,j] <- (-1/ (l1x^2) +  dx^2/(l1x^4)) * K1[i,j]  
  }
}
#-------------------------OBTAIN DATA FOR C2---------------------------------
C1_hat <- as.vector(K1 %*% solve(K1+sigma_init^2*diag(N),y))
C1_hat <- pmax(C1_hat,0) #Force negative values to be zero

C2_synthetic <- as.vector(LK_prel %*% solve(K1, C1_hat))
outputscale2 <- 20
l2x <- 0.113
l2t <- 0.456
sigma_init2 <- 1e-6
cat("----Hyperparameter tuning results C2:-------")
cat(paste("\nl2x:             ", l2x, "\nl2t:             ", l2t, "\noutputscale2:    ", outputscale2, "\ninitial sigma: ", sigma_init2,"\n"))
K2 <-matrix(0,N,N)
for (i in (1:N)){
  for (j in (1:N)){
    dx <- X_norm[i,1] - X_norm[j,1]
    dt <- X_norm[i,2] - X_norm[j,2]
    K2[i,j] = outputscale2^2*exp(-0.5*(dx^2/l2x^2 +dt^2/l2t^2) )
    if (i==j) { K2[i,j] <- K2[i,j]+1e-6}
  }
}
C2_hat <- as.vector(K2 %*% solve(K2+sigma_init2^2*diag(N),C2_synthetic))
pars <- pre_process(X_norm,K1,K2,l1x,l1t)
K <- pars$K
m <- pars$m
K_deriv <- pars$K_deriv
LK <- pars$LK
KL <- pars$KL
LKL <- pars$LKL

#-------------------------INITIALIZE MAP ESTIMATE----------------------------
C = as.vector(c(C1_hat,C2_hat))
#cat("C:", C)
#cat("MAX m*C: ",max(m %*% C),"\n")
#cat("MAX K_DERIV: ",max(K_deriv),"\n")
#cat("MIN K_DERIV: ",min(abs(K_deriv)),"\n")

#print(max(K1))
#print(max(m))
#cat("Diag LKL:",(LKL[1,1]), LKL[N+1,N+1],"\n")
#cat("Diag mC:", (m%*%KL)[1,1], (m%*%KL)[N+1,N+1],"\n")
#cat("Diag K_deriv:",diag(K_deriv),"\n")
#cat("Diag K:",diag(K),"\n")
#cat("Diag LKL:",diag(LKL),"\n")
#cat("Diag mKL", diag(m %*% KL),"\n")
#cat("MAX m*C: ",m %*% C,"\n")
#print(m %*% as.vector(cbind(y,C2_synthetic)))
initializeMAP <- stan_model(file="Initialize_KPP.stan", model_name = "Initialize MAP")
init_fit <- optimizing(initializeMAP, data=list(N=N,K=K,m=m,K_deriv=K_deriv,C1=C1_hat,C2=C2_hat) , iter=3000,
                       init = list(logcap=log(1),logD = log(500*24/1800^2), loglambda=log(24*0.05)),
                       as_vector=FALSE,verbose=TRUE,
                       tol_rel_grad = 0,   # gradient tolerance
                       tol_param = 1e-10)   # parameter tolerance)

#print(init_fit$par[c("D_scale","lambda_scale","cap")])
posterior <- stan_model(file = "FisherKPP_Posterior.stan",
                        model_name = "KPPposterior")
fit <- optimizing(posterior, data=list(X=X_norm, N=N, K=K, K_deriv=K_deriv, y=y,ysynth=C2_synthetic, m=m, mu=0, beta=1 ),
                  init= list(logD=log(1000*24/1900^2),logcap=log(1.8),loglambda=log(24*0.03),C1=C1_hat,C2=C2_hat),
                  iter=30000, verbose=TRUE,as_vector=FALSE,algorithm="LBFGS")
print(fit$par[c("D_scale","lambda_scale","cap","sigma2","dsigma2")])
print(fit$par[c("p1","p2","p3","p4")])
n_chains=1
MAP_list <- replicate(n_chains, fitMAP$par, simplify = FALSE)
samples <-sampling(posterior,data=list(X=X_norm, N=N, K=K, K_deriv=K_deriv, y=y,ysynth=C2_synthetic, m=m, mu=0, beta=1 ),
                   init= MAP_list, chains=n_chains,
                   iter=3000, verbose=FALSE, control = list(max_treedepth = 10))
print(samples,pars=c("D_scale","lambda_scale","cap","sigma2"),digits=5)
post <- rstan::extract(samples)
df <- as.data.frame(post)
write.csv(df, "posterior_draws_samples.csv", row.names = FALSE)
#C1_opt <- fit$par$C1
#C2_opt <- fit$par$C2
#print(C1_opt-y)
#print(C2_opt-C2_synthetic)
print(C2_synthetic)
RHS <- c(C2_hat, 100*24/1900^2*C2_synthetic + 0.06*24*y*(1-y/1.77))
RHS_col <- matrix(RHS,ncol=1)
err <- RHS_col - m%*%C
#print(length(RHS))
#print(length(m%*%C))
temp <- cbind(RHS_col, m%*%C, err,diag(K_deriv),err*solve(K_deriv,err))
#options(max.print = 1e7)
#print(temp)
#print( t(err)%*%solve(K_deriv,err) )
#print(min(C1_hat))
