functions {
  // Eigens definierte Funktionen
  real my_func(real x) {
    return x^2;
  }
}

data {
  // INPUT Daten
  int<lower=1> N; // Int >= 1
  vector[N] x; // Vektor Länge N
  matrix[N,3] X; // Nx3 Matrix
  array[N] int<lower=0, upper=1> y; // Binary array
  real<lower=0> positive; //Positive  (reelle) Zahl
}

transformed data {
  // Einmalige Berechnungen (Ableitungen) aus den Daten
  vector[N] x_centered = x - mean(x); // Nach dem Mittelwert zentriertes x
  real x_sd = sd(x); // Standartabweichung von x
}

parameters {
  // Welche Parameter sollen geschätzt werden?
  real mu;
  real<lower=0> sigma;
  simplex[3] theta; // Wahrscheinlichkeitsverteilung mit 3 Werten, die zu 1 summieren
  ordered[4] cutpoints; // c1 < c2 < c3 < c4 (Jeder folgende Eintrag muss strikt größer als der vorherige Eintrag sein)
  vector[N] z; // Vektor mit N parametern

}

transformed parameters {
  // Ableitungen aus Parameters
  vector[N] y_hat = mu + sigma * z;
}

model {
  // Priors

  mu ~ normal(0,10);
  sigma ~ exponential(1);
  z ~ normal(0,1);

  // Likelihood
  y ~ bernoulli_logit(y_hat);

  target += normal_lpdf(mu | 0, 10);
}

generated quantities {
   // Post processing
   vector[N] y_pred;
   vector[N] log_lik;

   for(i in 1:N) {
    y_pred[i] = normal_rng(y_hat[i], sigma);
    log_lik[i] = normal_lpdf(y[i] | y_hat[i], sigma);
   }
}
