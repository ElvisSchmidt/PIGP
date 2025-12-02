data {
  int<lower=0> M;
  int<lower=0> N;
  matrix[N,2] tau;
  matrix[M,2] I;
  vector[N] y; 
  matrix[M,M] K;
  matrix[M,M] K_deriv;
  matrix[M,M] m;
  real beta;
}
transformed data {
  vector[M] dmu=rep_vector(0,M);
  vector[M] mu_vec = rep_vector(0,M);
}
parameters {
  real theta0;
  real theta1;
  real theta2;
  vector[M] u;
  real<lower=0> sigma;
}
transformed parameters {
  vector[M] Lu;
  for (n in 1:M) {
    Lu[n] = u[n]*(theta0+theta1*I[n,1]+theta2*I[n,1]^2);
    } 
  vector[M] err = Lu - m*(u-mu_vec);

}
model {
  //target += multi_normal_lpdf( u | mu_vec , K)/beta;
  target += -0.5/beta*to_row_vector(u-mu_vec)*(mdivide_left_spd(K,u-mu_vec));

  target += (-0.5/sigma^2*dot_self(u-y) - 0.5*N*log(sigma^2));
  //target += multi_normal_lpdf(u | y , sigma^2*diag_matrix(rep_vector(1,N)));
  
  //target += multi_normal_lpdf(Lu | dmu+m*(u-mu_vec) , K_deriv)/beta;
  target += -0.5/beta* to_row_vector(err) * ( mdivide_left_spd(K_deriv,err) );
}
