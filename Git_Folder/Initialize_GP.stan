data {
  int<lower=0> M;
  int<lower=0> N;
  matrix[N,2] tau;
  matrix[M,2] I;
  vector[N] y;
  matrix[M,M] K;
  matrix[M,M] m;
  matrix[M,M] K_deriv;
  matrix[M,M] u_to_uKL;
  real outputscale;
  real noise;
}

transformed data {
  matrix[N,N] K_tau;
  matrix[M,N] K_Itau;
  vector[M] u;
  for (i in 1:N) {
    for (j in 1:N) {
      K_tau[i,j] =K[i,j];
      if (i == j)
        K_tau[i,j] += noise; 
    }
  }

  for (i in 1:M) {
    for (j in 1:N) {
      K_Itau[i,j] = K[i,j];
    }
  }
  //---------initial estimate for u------------
  vector[N] ones = rep_vector(1.0, N); // column of ones
  vector[N] Kinv_y = mdivide_left_spd(K_tau, y);
  vector[N] Kinv_ones = mdivide_left_spd(K_tau, ones);
  real mu =  dot_product(ones, Kinv_y) / dot_product(ones, Kinv_ones);
  vector[M] muM = rep_vector(mu,M);
  vector[N] muN = rep_vector(mu,N);
  u = muM + K_Itau * mdivide_left_spd(K_tau,y-muN);
  vector[M] LU_GP = m*(u-muM);
}

parameters {
  real theta0;
  real theta1;
  real theta2;
}
transformed parameters {
  vector[M] partial_sol;
  for (n in 1:M) {
    partial_sol[n] = u[n]*(theta0+theta1*I[n,1]+theta2*I[n,1]^2);
    } 
}
model {
  partial_sol ~ multi_normal(LU_GP,K_deriv);
}
generated quantities{
  //KL-expand u
 real mu_est = mu;
 vector[M] u_est = u;
 vector[M] u_KL = u_to_uKL*(u-mu);
}
