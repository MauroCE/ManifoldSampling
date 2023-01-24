library(gk)

# Set Parameter and Observed Data
ystar = c(2.20517002,  3.06752075,  4.38657593,  3.17309127,  4.77882485,
          19.1077929 ,  2.26343794,  5.06907625,  2.17385853,  3.4596839 ,
          2.64154572,  6.720638  ,  2.49741528,  3.80894688,  2.3511728 ,
          1.86440246,  3.62924583,  9.07515891,  3.81038952,  2.44666584,
          3.33609323,  4.46608741,  6.08458601,  2.39127591,  4.25704094,
          3.47303588,  2.96840863,  3.01332173,  2.56712164,  2.59191003,
          6.75826702,  3.3215288 ,  2.65689358,  1.58910621,  2.49148956,
          2.64493561,  2.34425868,  2.32575127,  4.65269195,  2.79470526,
          2.15631608,  2.32382798,  2.76001451,  5.74571668,  2.25107554,
          8.1856495 ,  2.65395516,  2.15025159,  3.79460562,  7.31370972)
theta0 = c(3, 1, 2, 0.5)
N = 100000
Sigma0 = 0.1*diag(4)

# Define a uniform log density
uniform_log_density = function(theta){
  log(as.numeric(all((theta >= 0) & (theta <= 10))))
}

# First run
out = mcmc(ystar, N=N, get_log_prior=uniform_log_density, theta0=theta0, Sigma0=Sigma0)


# Compute correct covariance matrix
Sigmahat = cov(out)

# Second run
N_SECOND_RUN=200000
out2 = mcmc(ystar, N=N_SECOND_RUN, get_log_prior=uniform_log_density, theta0=theta0, Sigma0=Sigmahat)

write.table(out2, file="gk_mcmc_samples2.txt")
