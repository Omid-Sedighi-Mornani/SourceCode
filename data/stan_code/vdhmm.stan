functions{
  //Calculates the transition probability matrix given the duration d in state s_from and time difference between observations t
  //The diagonal element is simply log(inv_logit(intercept+d*lambda+gamma*t))
  real calc_tpm(int s_from, int s_to, vector intercept, real d, real lambda,real t,real gamma, real senti, real epsilon ,vector[] l_tpm){
    real diag;
    diag =  - log1p_exp(-(intercept[s_from] + lambda * d + gamma * t + senti * epsilon));
    if(s_from == s_to){
      return diag;
    }else{
      return l_tpm[s_from,s_to] + log1m_exp(diag);
    }
  }
  vector get_duration(int C, row_vector Q, int[] Rating, real[] Days, real[] Sentiment, int T , 
  int S , vector l_pi , vector[] l_tpm , vector[] emission, vector intercept, 
  real lambda, real gamma, real epsilon, 
  real omega_0, vector omega_tilde, vector omega_state){
    vector[S] forw[T];
    vector[S] duration[T];
    vector[S] temp_tpm[S];
    vector[T] c;
    vector[S] weighted_days;
    vector[S] logit_duration;
    for(s in 1:S){
      forw[1,s] = l_pi[s] + emission[s,Rating[1]];
      duration[1,s] = 0.0;
      weighted_days[s] = 0.0;
    }
    c[1] = -log_sum_exp(forw[1]);
    forw[1] += c[1];
    for(t in 2:T){
      for(s in 1:S){
        vector[S] acc; 
        for(s_from in 1:S){
          temp_tpm[s_from,s] = calc_tpm(s_from, s, 
                                        intercept,duration[t-1,s_from], 
                                        lambda, 
                                        log(1+Days[t]-Days[t-1]), 
                                        gamma,
                                        Sentiment[t-1], 
                                        epsilon , 
                                        l_tpm);
          acc[s_from] = forw[t-1,s_from] + temp_tpm[s_from,s];
        }
        forw[t,s] = log_sum_exp(acc) + emission[s,Rating[t]];
      }
      c[t] = -log_sum_exp(forw[t]);
      forw[t] += c[t];
      for(s in 1:S){
        real dur_temp = duration[t-1,s];
        real day_temp = weighted_days[s];
        duration[t,s] = log1p_exp(dur_temp + c[t] + 
                                  emission[s,Rating[t]] + temp_tpm[s,s] + forw[t-1,s] - forw[t,s]);
        weighted_days[s] = log_sum_exp(log(Days[t]-Days[t-1]), day_temp + c[t] + 
                                            emission[s,Rating[t]] + 
                                            temp_tpm[s,s] + forw[t-1,s] - forw[t,s]);
      }
      
    }
    
    logit_duration = duration[T]; 
    
    return duration[T];
  }
  row_vector compute_ll(int[] Rating, real[] Days, real[] Sentiment, int T , int S , vector l_pi , vector[] l_tpm , vector[] emission, vector intercept, real lambda, real gamma, real epsilon){
    vector[S] forw[T];
    vector[S] duration[T];
    vector[S] temp_tpm[S];
    vector[T] c;
    row_vector[S+1] results;
    for(s in 1:S){
      forw[1,s] = l_pi[s] + emission[s,Rating[1]];
      duration[1,s] = 0.0;
    }
    c[1] = -log_sum_exp(forw[1]);
    forw[1] += c[1];
    for(t in 2:T){
      for(s in 1:S){
        vector[S] acc;
        for(s_from in 1:S){
          temp_tpm[s_from,s] = calc_tpm(s_from,s,intercept,
                                        duration[t-1,s_from], 
                                        lambda,
                                        log(1+Days[t]-Days[t-1]),
                                        gamma,
                                        Sentiment[t-1], 
                                        epsilon, 
                                        l_tpm);
          acc[s_from] = forw[t-1,s_from] + temp_tpm[s_from,s];
        }
        forw[t,s] = log_sum_exp(acc) + emission[s,Rating[t]];
      }
      c[t] = -log_sum_exp(forw[t]);
      forw[t] += c[t];
      for(s in 1:S){
        real dur_temp = duration[t-1,s];
        duration[t,s] = log1p_exp(dur_temp + c[t] + 
                                  emission[s,Rating[t]] + temp_tpm[s,s] + forw[t-1,s] - forw[t,s]);
      }
      
    }
    
    results[S+1] = -sum(c);
    results[1:S] = duration[T]';
    
    return results;
  }
}
data {
  int<lower=1> S; // number of states
  int<lower=0> N_total; // total number of restaurants considered
  int<lower=0> N_train; // number of restaurants in the training set N_test == N_total-N_train
  int<lower=1> N_obs; // Total number of reviews
  int<lower=0> nCovs; // Number of covariates
  int<lower=1> Time[N_total];//Number of reviews for each restaurant sum(Time) == N_obs
  int<lower=0,upper=1> Closed[N_total]; //Indicator for each restaurant if closed == 1
  real<lower=0> Days[N_obs]; // number of days since first review of restaurant
  int<lower=1,upper=5> Ratings[N_obs]; //All ratings
  real Sentiment[N_obs];
  matrix[N_train,nCovs] Q; // thinned Q matrix of Covariates for training
  matrix[nCovs,nCovs] R;
  matrix[N_total-N_train,nCovs] X_test; //Covariates for test set
}
transformed data {
  //prior parameter vectors for states and pain levels
  int pos_obs[N_total];
  vector[S-1] prior_tpm[S];
  {
    for(s in 1:(S)){
      prior_tpm[s] = rep_vector(1,S-1);
    }
    for(m in 1:N_total){
      pos_obs[m] = (m == 1 ? 1 : pos_obs[m-1]+Time[m-1]);
    }
  }
}
parameters {
  //State sequence
  simplex[S] pi; //initial state distribution for each patien
  simplex[S-1] tpm[S] ;//Transition probability matrix for each patient, time independent
  vector[S] intercept;
  real lambda;
  real gamma;
  real epsilon;
  //State-Obs link
  vector[S-1] state_emission_raw;
  simplex[5] probs;
  vector[S] R2;
  //Closing
  vector[S-1] omega_state_raw;
  real<lower=0> omega_state_mid;
  vector[nCovs] omega_tilde;
  real omega_0;
  
}
transformed parameters{
  vector[N_train] log_lik;
  vector[5] emission[S];
  row_vector[S] duration_mean;
  row_vector[S] duration_sd;
  matrix[N_train,S] Duration;
  vector[S] l_pi = log(pi);
  vector[S] l_tpm[S];
  vector[S] omega_state;
  {
    vector[S] state_emission;
    int i = 1;
    for(s in 1:S){
      if(s == 2){
        omega_state[s] = omega_state_mid;
      }else{
        omega_state[s] = omega_state_raw[i];
        i = i + 1;
      }
    }
    for(s in 1:S){
      real scale = inv_sqrt(1 - inv_logit(R2[s]));
      vector[4] cuts = scale * inv_Phi(cumulative_sum(probs[1:4]));
      state_emission[s] = s == 1 ? 0.0 : state_emission[s-1] + exp(state_emission_raw[s-1]);
      l_tpm[s,s] = negative_infinity();
      l_tpm[s,1:(s-1)] = log(tpm[s,1:(s-1)]);
      l_tpm[s,(s+1):S] = log(tpm[s,s:(S-1)]);
      for(r in 1:5){
        emission[s,r] = ordered_probit_lpmf(r|state_emission[s],cuts);
      }
    }
    for(m in 1:N_train){
      int Rating_vec[Time[m]] = Ratings[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      real Days_vec[Time[m]] = Days[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      real Sentiment_vec[Time[m]] = Sentiment[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      row_vector[S+1] results;
      results =  compute_ll(Rating_vec, Days_vec, Sentiment_vec, Time[m], 
                               S, l_pi, l_tpm, emission, intercept, lambda, gamma, epsilon);
      log_lik[m] = results[S+1];
      Duration[m] = results[1:S];
    }
    for(s in 1:S){
      vector[N_train] temp = Duration[,s];
      duration_mean[s] = mean(temp);
      duration_sd[s] = sd(temp);
      Duration[,s] = (temp - mean(temp))/sd(temp);
    }
  }
}
model {
  //State sequence-----
  pi ~ dirichlet(rep_vector(1,S));
  intercept ~ normal(0,5);
  lambda ~ std_normal();
  gamma ~ std_normal();
  epsilon ~ std_normal();
  for(s in 1:S){
    tpm[s] ~ dirichlet(prior_tpm[s]);
  }
  //State-Obs link-----
  probs ~ dirichlet(rep_vector(0.1,5));
  state_emission_raw ~ normal(0,5);
  R2 ~ normal(0,5);
  //Closing-----
  omega_0 ~ normal(0,5);
  omega_tilde ~ std_normal();
  omega_state_raw ~ student_t(7,0,1);
  omega_state_mid ~ std_normal();
  //log prob increasing
  Closed[1:N_train] ~ bernoulli_logit(omega_0 + Q * omega_tilde + Duration * omega_state);
  target += sum(log_lik);
}
generated quantities{
  vector[N_total] close_prob;
  vector[N_total - N_train] log_lik_test;
  vector[nCovs] omega_cov = R\omega_tilde;
  {
    for(m in 1:N_total){
      int Rating_vec[Time[m]] = Ratings[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      real Days_vec[Time[m]] = Days[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      real Sentiment_vec[Time[m]] = Sentiment[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      row_vector[S+1] results;
      row_vector[S] scaled_duration;
      results =  compute_ll(Rating_vec, Days_vec, Sentiment_vec ,Time[m], 
                               S, l_pi, l_tpm, emission,
                               intercept, lambda, gamma, epsilon);
      scaled_duration = (results[1:S] - duration_mean)./duration_sd;                         
      if(m <= N_train){
        close_prob[m] = inv_logit(omega_0 + Q[m]*omega_tilde + scaled_duration*omega_state);
      }else{
        close_prob[m] = inv_logit(omega_0 + X_test[m-N_train]*omega_cov + scaled_duration*omega_state);
        log_lik_test[m-N_train] = results[S+1];
      }                         
      
    }
  }
}

