library(rstan)
library(future.apply)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

#----------------------Functions-------------------------------
pre_process <- function(I,y, l1, l2,outputscale, nugget=1e-8) {
  M <- nrow(I)
  
  K  <- matrix(0, M, M)
  LK <- matrix(0, M, M)
  KL <- matrix(0, M, M)
  C  <- matrix(0, M, M)
  
  for (i in 1:M) {
    for (j in 1:M) {
      dx <- I[i,1] - I[j,1]
      dt <- I[i,2] - I[j,2]
      
      K[i,j] <- outputscale^2*exp(-0.5 * ( (dx^2) / (l1^2) + (dt^2) / (l2^2) ))
      if (i == j) K[i,j] <- K[i,j] + nugget
      
      LK[i,j] <- (-dx / (l1^2) - dt / (l2^2)) * K[i,j]
      KL[i,j] <- ( dx / (l1^2) + dt / (l2^2)) * K[i,j]
      
      C[i,j] <- ( (1 / (l1^2) - dx^2 / (l1^4)) +
                    (1 / (l2^2) - dt^2 / (l2^4)) -
                    2 * dx * dt / (l1^2 * l2^2) ) * K[i,j]
    }
  }
  
  C=(C+t(C))/2
  K = (K+t(K))/2
  # compute m = KL %*% K^{-1}
  # safer to use Cholesky if K is positive definite
  m <- t(solve(t(K), t(LK)))
  K_deriv <- C-m %*% KL
  ev = eigen(K)
  evals <- ev$values
  evecs <- ev$vectors 
  #Transformations from u to u_KL and vice versa if one wants to experiment with KL expansions
  u_to_uKL <- evecs %*% diag(sqrt(evals))/sqrt(outputscale)
  uKL_to_u <- diag(1/sqrt(evals)) %*% t(evecs)*sqrt(outputscale)
  #Estimate of GP mean mu
  N <- length(y)
  ones <- rep(1, N)
  a <- t(evecs) %*% ones
  b <- t(evecs) %*% y 
  a_div_e <- a / evals     
  mu <- as.numeric( crossprod(a_div_e, b) / crossprod(a_div_e, a) ) #mean estimate, used in PIGP paper but not for us
  list(K = K, LK = LK, KL = KL, C = C, m = m, K_deriv= K_deriv, u_to_uKL = u_to_uKL, uKL_to_u = uKL_to_u, C = C,evals=evals, evecs=evecs)
}

#----------------------------------DATA-----------------------------------------
set.seed(123)
# Load data
data <- read.csv("GitData0-1.csv", header = TRUE)

# Extract columns
x <- data[,1]
t <- data[,2]
y <- data[,3]
X <- cbind(x, t)
# Sample sizes
size <- 60  #for data points
size2 <- 60 #data points + possible additional points to inform the GP
points <- length(x) 
# Random indices without replacement
indices <- sample(1:points, size = size2, replace = FALSE)
# Gaussian noise
noise <- rnorm(size, mean = 0, sd = 0.01) 
# Select X and y with noise
X_dat <- X[1:size, ]
y_dat <- y[1:size] + noise
I <- X[1:size2,]
#-------------------------------------------------------------------------------
#-------------------------TRAIN DATA AND PREPROCESS ----------------------------
header_path <- normalizePath('pmatern.cpp', winslash = '/') #if one wants to add external cpp code, not used here
kernel_train <- stan_model  (
  file = "Train_Kernel2.stan",
  model_name = "Train Kernel",
  allow_undefined = TRUE,
  includes = paste0('\n#include \"', header_path, '\"\n')
)

#Find optimal hyperparameters. Initialize optimizition for a grid, take parameters with best posterior value
# ------------- grid -------------
ell_x_vals <- seq(0.3, 2, length.out = 1)
ell_t_vals <- seq(0.02, 2, length.out = 1)
outputscale_vals <- seq(0.5, 3, length.out = 1)

init_grid <- expand.grid(ell_x=ell_x_vals,
                         ell_t=ell_t_vals,
                         os=outputscale_vals)

n_inits <- nrow(init_grid)

# ------------- make initializations  -------------
make_init_fun <- function(i){
  function(){
    list(
      lengthscale = c(init_grid$ell_x[i], init_grid$ell_t[i]),
      outputscale = init_grid$os[i]
    )
  }
}

# ------------- run the optimization-------------

run_one <- function(i, kernel_train, size, size2, X_dat, y_dat, I){
  fit_hp <- optimizing(
    kernel_train,
    data = list(N=size, M=size2, X=X_dat, y=y_dat, I=I),
    init = make_init_fun(i),
    algorithm="LBFGS",
    as_vector=FALSE,
    iter=60000,
    verbose=FALSE
  )
  list(idx=i, value=fit$value, fit=fit)
}

results <-list()
values <- numeric(n_inits)

for(i in seq_len(nrow(init_grid))){
  fit_hp <- optimizing(
    kernel_train,
    data = list(N=size, M=size2, X=X_dat, y=y_dat, I=I),
    init = make_init_fun(i),
    algorithm="LBFGS",
    as_vector=FALSE,
    iter=30000,
    verbose=TRUE
  )
  
  results[[i]] <- fit_hp
  values[i] <- fit_hp$value
}

# pick best run
best_idx <- which.max(values)
fit_hp <- results[[best_idx]]
#GP hyperparameters
l <- fit_hp$par$lengthscale
l1 <- l[1]
l2 <- l[2]
outputscale <- fit_hp$par$outputscale
sigma_init <- fit_hp$par$sigma
cat("----Hyperparameter tuning results:-------")
cat(paste("\nl1:             ", l1, "\nl2:             ", l2, "\noutputscale:    ", outputscale, "\ninitials sigma: ", sigma_init,"\n"))
cat("Value Max: ",max(values)," Value Min: ",min(values),"\n")
pars <- pre_process(I,y,l1,l2,outputscale)
m <- pars$m
K <- pars$K
K_deriv <- pars$K_deriv
u_to_uKL <- pars$u_to_uKL
uKL_to_u <- pars$uKL_to_u
mu <- pars$mu #not used
KL <- pars$KL
LK <- pars$LK
C <- pars$C
evals <- pars$evals
evecs <-pars$evecs
#-------------------------------------------------------------------------------
#----------------------------GET INITIALIZATION---------------------------------
header_path <- normalizePath('pmatern.cpp', winslash = '/')
initialize_MAP <- stan_model  (
  file = "Initialize_GP.stan",
  model_name = "Initialize",
  allow_undefined = TRUE,
  includes = paste0('\n#include \"', header_path, '\"\n')
)
fit_init <- optimizing(initialize_MAP, data=list(N=size, M=size2, tau=X_dat, y=y_dat, I=I, outputscale=outputscale, noise = sigma_init^2,m=m, K=K, K_deriv=K_deriv),
                       init=list("theta0"=0,"theta1"=0,"theta2"=0),
                       as_vector=FALSE)
cat("---------Initial values:---------\n")
theta0 <- fit_init$par["theta0"]
theta1 <- fit_init$par["theta1"]
theta2 <- fit_init$par["theta2"]
mu_est <- as.numeric(fit_init$par["mu_est"])
cat(paste("theta0: ",theta0, "\ntheta1:",theta1,"\ntheta2:",theta2,"\nmu_est:",mu_est))
u_KL <- fit_init$par$u_KL
u_est <- fit_init$par$u_est
#-------------------------------------------------------------------------------
#------------------------------RUN HMC-----------------------------------------
header_path <- normalizePath('pmatern.cpp', winslash = '/') #not used
HMC_model <- stan_model  (
  file = "PDE_GP_post.stan",
  model_name = "RUN_HMC",
  allow_undefined = TRUE,
  includes = paste0('\n#include \"', header_path, '\"\n')
)
  init_list <- list(
    "theta0" = theta0,
    "theta1" = theta1,
    "theta2" = theta2,
    "u" = u_est,
    "sigma" = sigma_init)
n_chains=3
beta = 1
fitMAP <- optimizing(HMC_model, data=list(N=size, M=size2, tau=X_dat, y=y_dat, I=I, m=m, K=K, K_deriv=K_deriv,beta=beta),
                init=init_list,iter=200000,verbose=FALSE,as_vector=FALSE)
MAP_list <- replicate(n_chains, fitMAP$par, simplify = FALSE)

cat("\n ------ MAP estimate ---------------\n")
print(fitMAP$par[c("theta0","theta1","theta2","sigma")])

fit <- sampling(HMC_model, data=list(N=size, M=size2, tau=X_dat, y=y_dat, I=I, m=m, K=K, K_deriv=K_deriv),
                  init=MAP_list,
                       iter=1000,chains=n_chains,refresh=500, control = list(max_treedepth = 13))
#cat("\n------Predictions:-------\n")
#print(fit,pars=c("theta0","theta1","theta2","sigma"),digits=5)
cat("MAX K_DERIV: ",max(K_deriv),"\n")
cat("MIN K_DERIV: ",min(abs(K_deriv)),"\n")
cat("MAX K_DERIV: ",diag(K_deriv),"\n")
cat("MAX m*y: ",max(m %*% y),"\n")
cat("MAX K",max(K),"\n")
cat("Diag C", diag(C))
cat("Diag mKL", diag( m %*% KL))
cat("Eval K", min(eigen(K)$values), min(abs(eigen(K_deriv)$values)))
# SAVE DATA
RHS <- y_dat*(1 -1.8*x)
temp <- cbind(m%*%y, RHS, m%*%y-RHS,diag(K_deriv))
print(temp)
err <- RHS-m%*%y
print(t(err)%*%solve(diag(diag(K_deriv)),err))
#post <- rstan::extract(fit)
#df <- as.data.frame(post)
#write.csv(df, "posterior_draws_001.csv", row.names = FALSE)


