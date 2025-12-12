functions{
  //Calculates the transition probability matrix given the duration d in state s_from and time difference between observations t
  //The diagonal element is simply log(inv_logit(intercept+d*lambda+gamma*t))
  real calc_tpm(int s_from, int s_to, array[] real intercept, real d, real lambda,real t,real gamma, real senti, real epsilon , array[,] real l_tpm){
    real diag;
    diag =  - log1p_exp(-(intercept[s_from] + lambda * d + gamma * t + senti * epsilon));
    if(s_from == s_to){
      return diag;
    }else{
      return l_tpm[s_from,s_to] + log1m_exp(diag);
    }
  }
  array[] int get_states_viterbi(array[] int Rating, array[] real Days, array[] real Sentiment, int T, int S, array[] real l_pi,
        array[,] real l_tpm, array[,] real emission, array[] real intercept, real lambda, real gamma, real epsilon){
    array[T] int hidden_seq;
    {
      real log_p_max;
      array[T, S] int back_track;
      array[T, S] real best_logp;
      array[T] vector[S] forw;
      array[T] vector[S] duration;
      array[S] vector[S] temp_tpm;
      vector[T] c;
      for(s in 1:S){
        best_logp[1,s] = l_pi[s] + emission[s,Rating[1]];
        forw[1,s] = l_pi[s] + emission[s,Rating[1]];
        duration[1,s] = 0.0;
      }
      c[1] = -log_sum_exp(forw[1]);
      forw[1] = forw[1] + c[1];
      for(t in 2:T){
        for(s in 1:S){
          vector[S] acc;
          best_logp[t,s] = negative_infinity();
          for(s_from in 1:S){
            real logp;
            temp_tpm[s_from,s] = calc_tpm(s_from,
                                          s,
                                          intercept,
                                          duration[t-1,s_from],
                                          lambda,
                                          log(1+Days[t]-Days[t-1]),
                                          gamma,
                                          Sentiment[t-1],
                                          epsilon,
                                          l_tpm);
            logp = best_logp[t-1,s_from] + temp_tpm[s_from,s] + emission[s,Rating[t]];
            acc[s_from] = forw[t-1,s_from] + temp_tpm[s_from,s];
            if(logp > best_logp[t,s]){
              back_track[t,s] = s_from;
              best_logp[t,s] = logp;
            }
          }
          forw[t,s] = log_sum_exp(acc) + emission[s,Rating[t]];
        }
        c[t] = -log_sum_exp(forw[t]);
        forw[t] = forw[t] + c[t];
        for(s in 1:S){
          real dur_temp = duration[t-1,s];
          duration[t,s] = log1p_exp(dur_temp + c[t] +
                                  emission[s,Rating[t]] + temp_tpm[s,s] + forw[t-1,s] - forw[t,s]);
        }
      }
      log_p_max = max(best_logp[T]);
      for(s in 1:S){
        if(best_logp[T,s] == log_p_max){
          hidden_seq[T] = s;
        }
      }
      for(t in 1:(T-1)){
        hidden_seq[T-t] = back_track[T-t+1,hidden_seq[T-t+1]];
      }
    }
      return hidden_seq;
  }
  array[] int get_states_live(array[] int Rating, array[] real Days, array[] real Sentiment, int T, int S, array[] real l_pi,
        array[,] real l_tpm, array[,] real emission, array[] real intercept, real lambda, real gamma, real epsilon){
    array[T] int hidden_seq;
    {
      real log_p_max;
      array[T, S] int back_track;
      array[T, S] real best_logp;
      array[T] vector[S] forw;
      array[T] vector[S] duration;
      array[S] vector[S] temp_tpm;
      vector[T] c;
      real logp;
      for(s in 1:S){
        best_logp[1,s] = l_pi[s] + emission[s,Rating[1]];
        forw[1,s] = l_pi[s] + emission[s,Rating[1]];
        duration[1,s] = 0.0;
      }
      c[1] = -log_sum_exp(forw[1]);
      forw[1] = forw[1] + c[1];
      logp = negative_infinity();
      for(s in 1:S){
        if(forw[1,s] > logp){
          logp = forw[1,s];
          hidden_seq[1] = s;
        }
      }
      for(t in 2:T){
        for(s in 1:S){
          vector[S] acc;
          for(s_from in 1:S){
            temp_tpm[s_from,s] = calc_tpm(s_from,s,intercept,duration[t-1,s_from],lambda,log(1+Days[t]-Days[t-1]),gamma,Sentiment[t-1],epsilon,l_tpm);
            logp = best_logp[t-1,s_from] + temp_tpm[s_from,s] + emission[s,Rating[t]];
            acc[s_from] = forw[t-1,s_from] + temp_tpm[s_from,s];
          }
          forw[t,s] = log_sum_exp(acc) + emission[s,Rating[t]];
        }
        c[t] = -log_sum_exp(forw[t]);
        forw[t] = forw[t] + c[t];
        logp =negative_infinity();
        for(s in 1:S){
          real dur_temp = duration[t-1,s];
          duration[t,s] = log1p_exp(dur_temp + c[t] +
                                  emission[s,Rating[t]] + temp_tpm[s,s] + forw[t-1,s] - forw[t,s]);
          if(forw[t,s] > logp){
            logp = forw[t,s];
            hidden_seq[t] = s;
          }
        }
      }
    }
      return hidden_seq;
  }
  array[] vector get_states_live_dist(array[] int Rating, array[] real Days, array[] real Sentiment, int T, int S, array[] real l_pi,
        array[,] real l_tpm, array[,] real emission, array[] real intercept, real lambda, real gamma, real epsilon){
    array[T] vector[S] hidden_seq;
    {
      real log_p_max;
      array[T, S] int back_track;
      array[T, S] real best_logp;
      array[T] vector[S] forw;
      array[T] vector[S] duration;
      array[S] vector[S] temp_tpm;
      vector[T] c;
      real logp;
      for(s in 1:S){
        best_logp[1,s] = l_pi[s] + emission[s,Rating[1]];
        forw[1,s] = l_pi[s] + emission[s,Rating[1]];
        duration[1,s] = 0.0;
      }
      c[1] = -log_sum_exp(forw[1]);
      forw[1] = forw[1] + c[1];
      hidden_seq[1] = forw[1];
      for(t in 2:T){
        for(s in 1:S){
          vector[S] acc;
          for(s_from in 1:S){
            temp_tpm[s_from,s] = calc_tpm(s_from,s,intercept,duration[t-1,s_from],lambda,log(1+Days[t]-Days[t-1]),gamma, Sentiment[t-1], epsilon,l_tpm);
            logp = best_logp[t-1,s_from] + temp_tpm[s_from,s] + emission[s,Rating[t]];
            acc[s_from] = forw[t-1,s_from] + temp_tpm[s_from,s];
          }
          forw[t,s] = log_sum_exp(acc) + emission[s,Rating[t]];
        }
        c[t] = -log_sum_exp(forw[t]);
        forw[t] = forw[t] + c[t];
        hidden_seq[t] = forw[t];
        for(s in 1:S){
          real dur_temp = duration[t-1,s];
          duration[t,s] = log1p_exp(dur_temp + c[t] +
                                  emission[s,Rating[t]] + temp_tpm[s,s] + forw[t-1,s] - forw[t,s]);
        }
      }
    }
      return hidden_seq;
  }
  array[] int get_states(array[] int Rating, array[] real Days, array[] real Sentiment, int T, int S, array[] real l_pi,
        array[,] real l_tpm, array[,] real emission, array[] real intercept, real lambda, real gamma, real epsilon,int method){
          array[T] int hidden_seq;
          if(method == 1){
            hidden_seq = get_states_live(Rating, Days,Sentiment, T, S, l_pi, l_tpm, emission, intercept, lambda, gamma,epsilon);
          } else{
            hidden_seq = get_states_viterbi(Rating, Days,Sentiment, T, S, l_pi, l_tpm, emission, intercept, lambda, gamma, epsilon);
          }
          return hidden_seq;
    }
}
data {
  int<lower=1> S; // number of states
  int<lower=0> N_total; // total number of restaurants considered
  int<lower=0> N_train; // number of restaurants in the training set N_test == N_total-N_train
  int<lower=0> N_samples;
  int<lower=1> N_obs; // Total number of reviews
  array[N_total] int<lower=1> Time;//Number of reviews for each restaurant sum(Time) == N_obs
  array[N_obs] real<lower=0> Days; // number of days since first review of restaurant
  array[N_obs] int<lower=1,upper=5> Ratings; //All ratings
  array[N_obs] real Sentiment; //All ratings
  //Parameters
  //State_seq
  array[S] real l_pi;
  array[S, S] real l_tpm;
  real lambda;
  real gamma;
  real epsilon;
  array[N_train, S] real intercept;
  array[S] real intercept_mean;
  //Emission
  array[S, 5] real emission;
  int<lower=1,upper=3> method;
}
transformed data {
  //prior parameter vectors for states and pain levels
  array[N_total] int pos_obs;
  array[S] vector[S-1] prior_tpm;
  {
    for(s in 1:(S)){
      prior_tpm[s] = rep_vector(1,S-1);
    }
    for(m in 1:N_total){
      pos_obs[m] = (m == 1 ? 1 : pos_obs[m-1]+Time[m-1]);
    }
  }
}

model {

}
generated quantities{
  array[N_obs] int latent_state;
  array[N_obs] vector[S] latent_probs;
  if(method < 3){
  {
    for(m in 1:N_total){
      array[Time[m]] int Rating_vec = Ratings[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      array[Time[m]] real Days_vec = Days[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      array[Time[m]] real Senti_vec = Sentiment[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      array[Time[m]] int state_seq;
      if(m<=N_train){
        state_seq =  get_states(Rating_vec, Days_vec, Senti_vec, Time[m],
                                 S, l_pi, l_tpm, emission,
                                 intercept[m,], lambda, gamma,epsilon,method);
      }else{
        state_seq =  get_states(Rating_vec, Days_vec, Senti_vec, Time[m],
                                 S, l_pi, l_tpm, emission,
                                 intercept_mean, lambda, gamma,epsilon,method);
      }
      latent_state[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))] = state_seq;
  }
  }
  }else{

  {
    for(m in 1:N_total){
      array[Time[m]] int Rating_vec = Ratings[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      array[Time[m]] real Days_vec = Days[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      array[Time[m]] real Senti_vec = Sentiment[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      array[Time[m]] vector[S] state_seq;
      if(m<=N_train){
        state_seq =  get_states_live_dist(Rating_vec, Days_vec, Senti_vec, Time[m],
                                 S, l_pi, l_tpm, emission,
                                 intercept[m,], lambda, gamma,epsilon);
      }else{
        state_seq =  get_states_live_dist(Rating_vec, Days_vec, Senti_vec, Time[m],
                                 S, l_pi, l_tpm, emission,
                                 intercept_mean, lambda, gamma,epsilon);
      }
      latent_probs[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))] = state_seq;
  }
  }
  }
}

