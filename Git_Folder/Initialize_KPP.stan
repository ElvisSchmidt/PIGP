data{
  int<lower=0> N;
  matrix[2*N,2*N] K;
  vector[N] C1;
  vector[N] C2;
  matrix[2*N,2*N] m;
  matrix[2*N,2*N] K_deriv;
  
}
transformed data{
  vector[2*N] C;
  C[1:N] = C1;
  C[N+1:2*N] = C2;
}
parameters{
  real logD;
  real loglambda;
  real logcap;
}
transformed parameters{
  real D = exp(logD);
  real lambda = exp(loglambda);
  real cap = exp(logcap);
  vector[2*N] RHS;
  vector[N] F = lambda* C1 .*(1 - C1 /1.77);
  RHS[1:N] = C2;
  RHS[N+1:2*N] = D * C2 + F;
  vector[2*N] err = RHS-m*C;
  real D_scale = 1850.0^2/24*D;
  real lambda_scale = lambda/24;
}
model{
    target += -0.5* to_row_vector(err) * ( mdivide_left_spd(K_deriv,err));
}
