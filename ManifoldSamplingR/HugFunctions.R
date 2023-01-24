library(pracma)
library(ggplot2)
library(tidyr)
library(Ecdat)

HugTangentialMultivariate <- function(x0, T, B, N, alpha, logpi, jac, method='qr'){
  # q samples v, lq evaluates log-density for v distribution
  stopifnot(method == 'qr' || method == 'linear')
  # Assign correct projection function
  project <- if(method == 'qr') qr_project else linear_project
  # Store samples and acceptances
  samples = x0
  acceptances = rep(0, N)
  # Since we will always use MVN to sample v, we automatically create functions here
  sample_v <- function() rnorm(length(x0))
  logpdf_v <- function(v) sum(dnorm(v, log=TRUE))
  for (i in 1:N){
    # Sample spherical velocity
    v0s = sample_v()
    # Squeeze
    v0 = v0s - alpha * project(v0s, jac(x0))
    # Housekeeping
    x = x0
    v = v0
    logu = log(runif(1)) # Uniform for Metropolis-Hastings step
    delta = T / B        # Step size
    for (j in 1:B){
      x = x + delta * v / 2          # Move
      v = v - 2 * project(v, jac(x)) # Bounce
      x = x + delta * v / 2          # Move
    }
    # Unsqueeze
    v = v + (alpha/ (1 - alpha)) * project(v, jac(x))
    # Metropolis-Hastings accept-reject step
    if (logu <= logpi(x) + logpdf_v(v) - logpi(x0) - logpdf_v(v0s)){
      samples = vstack(samples, c(x))
      acceptances[i] = 1
      x0 = x
    } else {
      samples = vstack(samples, c(x0))
      acceptances[i] = 0
    }
  }
  output = list(samples=samples[2:nrow(samples), ], acceptprob=acceptances/N)
  return(output)
}

# Finds an initial point near the manifold
find_point_on_manifold <- function(m, ystar, maxiter=100){
  i = 0
  while (i <= maxiter){
    i = i + 1
    # Sample parameter and latent variables
    theta = runif(4, min=0, max=10)
    z     = rnorm(m)
    xi    = c(theta, z)
    # Solve
    func <- function(xi) c(f(xi) - ystar, rep(0, 4))
    opt_result = lsqnonlin(func, xi)
    if (opt_result$errno == 0 || opt_result$errno == 1){
      cat("Error Number ", opt_result$errno)
      return(opt_result$x)
    } else {
      cat("Iteration ", i, " failed.")
    }
  }
}

# Constraint function
f <- function(xi){
  z2gk(xi[5:length(xi)], xi[1], xi[2], xi[3], xi[4])
}

# Constraint Jacobian
Jf_transpose <- function(xi){
  # Grab parameters (a not needed)
  b = xi[2]
  g = xi[3]
  k = xi[4]
  z = xi[5:length(xi)]
  m = length(z)
  # Compute rows one at a time
  row1 = rep(1, m)
  row2 = (1 + 0.8 * (1 - exp(-g * z)) / (1 + exp(-g * z))) * ((1 + z**2)**k) * z
  row3 = 8 * b * (z**2) * ((1 + z**2)**k) * exp(g*z) / (5 * (1 + exp(g*z))**2)
  row4 = b*z*((1+z**2)**k)*(1 + 9*exp(g*z))*log(1 + z**2) / (5*(1 + exp(g*z)))
  row5 = diag(b*((1+z**2)**(k-1))*(((18*k + 9)*(z**2) + 9)*exp(2*g*z) + (8*g*z**3 + (20*k + 10)*z**2 + 8*g*z + 10)*exp(g*z) + (2*k + 1)*z**2 + 1) / (5*(1 + exp(g*z))**2))
  vstack(row1, row2, row3, row4, row5)
}

Jf <- function(xi){
  t(Jf_transpose(xi))
}

# Log Prior for ABC
logprior <- function(xi){
  theta = xi[1:4]
  z     = xi[5:length(xi)]
  sum(dunif(theta, max=10, log=TRUE)) + sum(dnorm(z, log=TRUE))
}

# Epanechnikov Kernel
log_epanechnikov <- function(xi, ystar, eps){
  u = norm(f(xi) - ystar, type="2")
  log((3 * (1 - (u**2 / eps**2)) / (4 * eps)) * as.integer(u <= eps))
}

# Log ABC Posterior
logpost <- function(xi, ystar, eps){
  logprior(xi) + log_epanechnikov(xi, ystar, eps)
}


# Stolen from gk package
z2gk = function(z, A, B, g, k, c=0.8){
  ##Essentially this function calculates
  ##x = A + B * (1+c*tanh(g*z/2)) * z*(1+z^2)^k
  ##But treats some edge cases carefully

  ##Recycle inputs to same length as output
  n = max(length(z), length(A), length(B), length(g), length(k), length(c))
  zeros = rep(0, n)
  z = z + zeros
  A = A + zeros
  B = B + zeros
  g = g + zeros
  k = k + zeros
  c = c + zeros

  ##Standard calculatations
  z_squared = z^2
  term1 = (1+c*tanh(g*z/2))
  term2 = z*(1+z_squared)^k

  ##Correct edge cases
  ##(likely to be rare so no need for particularly efficient code)
  term1[g==0] = 1 ##Avoids possibility of 0*Inf
  zbig = which(is.infinite(z_squared))
  term2[zbig] = sign(z[zbig]) * abs(z[zbig])^(1 + 2*k[zbig]) ##Correct when abs(z) large or infinite

  ##Return output
  return(A + B*term1*term2)
}

# Project using QR decomposition
qr_project <- function(v, J){
  Q = qr.Q(qr(t(J)))
  Q %*% (t(Q) %*% v)
}

# Project solving the linear system directly
linear_project <- function(v, J){
  t(J) %*% solve(J %*% t(J), J %*% v)
}

# Utility function to stack vectors together and un-naming the rows
vstack <- function(...){
  unname(rbind(...))
}

#### CODE TO RUN
# Settings and data generation
theta0 = c(3.0, 2.0, 1.0, 0.5)
m      = 50
eps    = 2.0
ystar  = rgk(m, theta0[1], theta0[2], theta0[3], theta0[4])
logpi  = function(xi) logpost(xi, ystar, eps)
T      = 0.2
B      = 5
N      = 10000
alpha  = 0.0
# Find point on manifold
xi0    = find_point_on_manifold(m, ystar)
# Run HUG
hug_out = HugTangentialMultivariate(xi0, T, B, N, alpha, logpi, Jf)
hug_samples = hug_out$samples
hug_ap      = hug_out$acceptprob
# Transform samples into dataframe
samples_df = as.data.frame(hug_samples[, 1:4])
# Plot histogram of HUG samples
theta0_df = data.frame(key = c("V1", "V2", "V3", "V4"), value=theta0)
ggplot(gather(samples_df), aes(value)) +
  geom_histogram(aes(y = ..density..), bins=20, fill='white', color=1) +
  geom_density(color='blue', lwd=0.8) +
  facet_wrap(~key, scales='free') +
  geom_vline(data=theta0_df, aes(xintercept=value), linetype="dotted", color='red', size=0.8)
# Use Prangle's abc function
fx_rate = Ecdat::Garch$cd
nx = length(fx_rate)
log_return = log(fx_rate[2:nx] / fx_rate[1:(nx-1)])




