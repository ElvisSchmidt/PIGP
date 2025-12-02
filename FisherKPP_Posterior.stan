data {
  int<lower=0> N;
  matrix[N,2] X;
  vector[N] y;
  vector[N] ysynth;
  matrix[2*N,2*N] K;
  matrix[2*N,2*N] K_deriv;
  matrix[2*N,2*N] m;
  real mu;
  real beta;
}
transformed data{
  vector[2*N] mu_vec = rep_vector(mu,2*N);
  vector[2*N] dat;
  dat[1:N] = y;
  dat[N+1:2*N] = ysynth;
  //real sigma2 = 0.009^2; //variance
  #real dsigma2 = 1e-8;
  //real cap = 1.77;
  //real lambda= 24.0*0.06;

}
parameters{
  vector[N] C1;
  vector[N] C2;
  real logD; //diffusitivity
  real<upper=log(3.0)> logcap;
  real loglambda;
  //real<lower=0> lambda; //proliferation rate
  //real<lower=0> cap; //Carrying capacity
  real<upper=log(0.1)> logsigma2; //variance
  real dlogsigma2;

}
transformed parameters{
  real D = exp(logD);
  real cap=exp(logcap);
  real lambda=exp(loglambda);
  real sigma2 = exp(logsigma2);
  real dsigma2=exp(dlogsigma2);
  vector[2*N] C;
  C[1:N] = C1;
  C[N+1:2*N] = C2;
  vector[2*N] RHS;
  for (n in 1:N) {
  RHS[n] = C2[n];
  RHS[N+n] = D*C2[n] + lambda*C1[n]*(1-C1[n]/cap);
  }
  vector[2*N] err = RHS- m*(C-mu_vec);
  real D_scale = 1850.0^2/24.0*D;
  real lambda_scale = lambda/24.0;
}

model {
  //PRIOR
    target += -0.5*to_row_vector(C-mu_vec)*(mdivide_left_spd(K,C-mu_vec));
  //LIKELIHOOD
    target += (-0.5/sigma2*dot_self(C1-y) - 0.5*N*log(sigma2));
    target += (-0.5/dsigma2*dot_self(C2-ysynth)-0.5*N*log(dsigma2));
  //PDE INFORMED PRIOR
    target += -0.5* to_row_vector(err) * ( mdivide_left_spd(K_deriv,err));
    
}

generated quantities{
  real p1 = -0.5/beta*to_row_vector(C-mu_vec)*(mdivide_left_spd(K,C-mu_vec));
  real p2 = (-0.5/sigma2*dot_self(C1-y) - 0.5*N*log(sigma2));
  real p3 = (-0.5/dsigma2*dot_self(C2-ysynth) - 0.5*N*log(dsigma2));
  real p4 = -0.5/beta* to_row_vector(err) * ( mdivide_left_spd(K_deriv,err) );
}
