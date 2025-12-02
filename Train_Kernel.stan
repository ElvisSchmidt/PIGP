data{
  int N;
  vector[N] y;
  matrix[N,2] X;
  
}
transformed data{
  
}
parameters{
  real log_lambda;
  real log_l1;
  real log_l2;
}
transformed parameters{
  
  real lambda = exp(log_lambda);
  real l1 =exp(log_l1);
  real l2 = exp(log_l2);
  //Kernel
matrix[N, N] K;
for (i in 1:N) {
  for (j in 1:N) {
    real dx = (X[i,1] - X[j,1]) / l1;
    real dt = (X[i,2] - X[j,2]) / l2;
    K[i,j] = exp(-0.5 * (square(dx) + square(dt)));
  }
}
K += 1e-8 * diag_matrix(rep_vector(1.0, N)); //stability
  // R = K + lambda * I
vector[N] ones = rep_vector(1.0,N);
matrix[N,N] R = K + lambda*diag_matrix(ones);
vector[N] evals = eigenvalues_sym(R);
matrix[N,N] evecs = eigenvectors_sym(R);
vector[N] a = evecs' *ones;
vector[N] b =  evecs'*y;
vector[N] a_div_e = a./evals;
real mean_est = dot_product(a_div_e,b)/dot_product(a_div_e,a);
vector[N] resid = y - mean_est;
vector[N] err = evecs'*resid;
vector[N] err_div_e = err ./ evals;
real outputscale = dot_product(err_div_e, err) / N;
}
model{
  #priors
  //target += normal_lpdf(log_l1 | 0, 1);
  //target += normal_lpdf(log_l2 | 0, 1);
  //target += normal_lpdf(log_lambda | -7, 2);
  target += -0.5*log(outputscale) - 0.5*mean(log(evals));
}

generated quantities {
  real out = (y-mean_est)' * inverse_spd(R)*(y-mean_est);
  real sigma2 = out*exp(log_lambda);
  real sigma = sqrt(sigma2);
}
