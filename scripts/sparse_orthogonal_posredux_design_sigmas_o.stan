// Sparse Bayesian Independent Component Analysis, V is orthogonal on the modulons
// M (or S, beta in our case :) ) is sparse 
// In previous iterations tau was dependendant on a positive-ordered constrained hyperprior. 
// Since this did not identified the modulons, we have removed the hyperprior and made tau itself positive-ordered.
//
// For identification, the caller must provide p0 as a vector, which is downsampled from the TF-operon distribution
//
// This model assigns a different covariate $z$ per gene but the modulon is still operon-based.
// To account for the background expression, we introduce an extra vector.
data {
  int<lower=1> N_gene; // Number of rows
  int<lower=1> N_cond; // Number of columns
  int<lower=1> N_ope;  // Number of operons
  int<lower=1> N_run;  // Number of runs (total replicates)
  matrix[N_gene, N_run] Y; // Data matrix
  int<lower=1> N_mode; // Number of principal components
  real<lower=0> sigma; // Standard deviation of measurement model
  // Sparse parameters
  int<lower=1> p0;  // estimated number of non-zero U parameters
  real<lower=1> slab_df;
  real<lower=0> slab_scale;
  real<lower=1> nu_global;
  real<lower=1> nu_local;
  real<lower=0> beta_v;
  // Design matrices
  array[N_run] int<lower=1,upper=N_cond> exp_design;  // run (index) -> condition
  array[N_gene] int<lower=1,upper=N_ope> ope_design;  // gene (index) -> operon

  int<lower=0,upper=1> likelihood;
}

transformed data {
  // N_ope
  // vector<lower=1>[N_mode] scale_global = p0 / ((N_mode - p0) .* sqrt(N_ope));
  // N_cond one
  // vector[N_mode] scale_global = p0 ./ ((N_mode - p0) .* sqrt(N_cond));
  // new one (right?)
  real scale_global = p0 ./ ((N_ope - p0) .* sqrt(N_cond));
  // for (i in 1:N_mode) {
  //    p0_int[i] = to_int(p0[i]);
  // }
}

parameters {
  vector<lower=0>[N_ope*N_mode] aux_tau;    // aux to t-student
  vector<lower=0>[N_ope*N_mode] aux_alpha;  // aux to t-student 
  matrix<lower=0>[N_mode, N_cond] X_V;      // latent activity matrix
  // sparse parameters
  vector<lower=0>[N_mode] caux;
  vector<lower=0>[N_mode] tau_sp;
  matrix[N_ope, N_mode] z;
  vector[N_gene] background;
}

transformed parameters {
  // REPARAM: U ~ t-student ( nu_local, 0, 1 ), saved to compute kappa
  matrix[N_gene, N_mode] beta;      // sparse modulon matrix
  matrix[N_mode, N_cond] V_t;       // orthogonal activity matrix
  matrix[N_ope, N_mode] kappa;      // shrinkage factor
  vector[N_mode] meff;              // effective parameters per modulonk
  {   
    // Constructing V as orthogonal component of the polar decomposition of X_V
    vector[N_cond] eval_trans_V; // Transformation of eigenvalues for polar decomp.
    tuple(matrix[N_cond, N_cond], vector[N_cond]) evec_eval = eigendecompose_sym(crossprod(X_V)); // eigendecomposition of z_V'*z_V (stable & eff)
    for (i in 1:N_cond) {
      // add an epsilon to remove numerical errors
      eval_trans_V[i] = 1.0 / sqrt(evec_eval.2[i] + 1e-10);
    }
    V_t = (X_V * evec_eval.1 * diag_matrix(eval_trans_V) * evec_eval.1');

    // sparsity (reg. horseshoe)
    matrix[N_ope, N_mode] U = to_matrix(aux_alpha ./ sqrt(aux_tau), N_ope, N_mode);
    vector[N_mode] c = slab_scale * sqrt(caux);
    matrix[N_ope, N_mode] U_m;
    for (m in 1:N_mode){
      U_m[,m] = sqrt(c[m]^2 * square(U[,m]) ./ (c[m]^2 + tau_sp[m]^2 * square(U[,m]))) * tau_sp[m];
    }
    beta = (U_m .* z)[ope_design,];

    // operon-modulon belonging
    for (m in 1:N_mode){
      kappa[,m] = 1 / (1 + N_cond * sqrt(sigma) * tau_sp[m]^2 * U[,m]^2 * 1^2);
    }
  }
  for (m in 1:N_mode){
    meff[m] = sum(1 - kappa[,m]);
  }
}

model { 
  // activity matrix
  to_vector(X_V) ~ normal(0, beta_v);

  // Orthogonal prior specification
  // first, the params for the latent t-student M matrix
  real half_nu = 0.5 * nu_local;
  aux_tau ~ gamma(half_nu, half_nu);
  aux_alpha ~ std_normal();

  // Sparse prior specification
  caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df); 
  tau_sp ~ student_t(nu_global, scale_global*sigma, scale_global*sigma);
  to_vector(z) ~ std_normal();
  background ~ student_t(2, 0, 0.5);
  // Likelihood
  if (likelihood == 1) {
    // beta <- Modulon (modulon, gene)
    // V <- Activity matrix (modulon, condition)
    matrix[N_gene, N_cond] y_hat = (beta * V_t);
    for (cond in 1:N_cond) {
      y_hat[, cond] = y_hat[, cond] + background;
    }
    to_vector(Y) ~ normal(to_vector(y_hat[, exp_design]), sigma);
  }
}

generated quantities {
  matrix[N_cond,N_mode] V = V_t';
}
