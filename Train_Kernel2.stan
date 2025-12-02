
functions{
  matrix L_cov_exp_quad_ARD(vector[] x, real outputscale, vector l, real delta) {
    int N = size(x);
    matrix[N,N] K;
    real sq_outputscale =square(outputscale);
    for (i in 1:(N-1)) {
      
      K[i,i] = sq_outputscale+ delta;
      for (j in (i+1):N) {
        K[i,j]= sq_outputscale * exp(-0.5*dot_self( (x[i]-x[j])./l^2 ));
        K[j,i] = K[i,j];
      }
    }
  
  K[N,N] = sq_outputscale + delta;
  return cholesky_decompose(K);
  }
}
data {
  int<lower=1> N; //number of data
  int<lower=1> M; //number of GP points
  vector[2] X[N]; //space-time points
  vector[2] I[M]; //GP points
  vector[N] y; //data
}
transformed data {
  real delta = 1e-9; //Ensure positive definiteness
  real mu = 0;
  vector[N] mu_vec = rep_vector(mu,N);
}
parameters {
  vector<lower=0,upper=10>[2] lengthscale;
  real<lower=0,upper=10> outputscale;
  real<lower=0> sigma;
  vector[N] eta;
}
model {
  vector[N] f;
  {  matrix[N, N] L_K = L_cov_exp_quad_ARD(I,outputscale,lengthscale,delta);
  f = L_K * eta+ mu_vec;
  }
  lengthscale ~inv_gamma(5,5);
  outputscale ~ normal(0,1);
  sigma~ std_normal();
  y ~ normal(f, sigma);
  eta ~ std_normal();
}

