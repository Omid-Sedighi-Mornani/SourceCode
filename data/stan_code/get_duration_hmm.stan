functions{
  //Calculates the transition probability matrix given the duration d in state s_from and time difference between observations t
  //The diagonal element is simply log(inv_logit(intercept+d*lambda+gamma*t))
  real calc_tpm(int s_from, int s_to, real[] intercept, real[,] l_tpm){
    real diag;
    diag =  - log1p_exp(-(intercept[s_from]));
    if(s_from == s_to){
      return diag;
    }else{
      return l_tpm[s_from,s_to] + log1m_exp(diag);
    }
  }
  row_vector get_duration_day(int[] Rating, real[] Days, real[] Sentiment, int T , int S , real[] l_pi , real[,] l_tpm , real[,] emission, real[] intercept){
    vector[S] forw[T];
    vector[S] duration[T];
    vector[S] temp_tpm[S];
    vector[T] c;
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
          temp_tpm[s_from,s] = calc_tpm(s_from, 
                                        s, 
                                        intercept,
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
    return duration[T]';
  }
  row_vector get_duration(int[] Rating, real[] Days, real[] Sentiment, int T , int S , real[] l_pi , real[,] l_tpm , real[,] emission, real[] intercept){
    vector[S] forw[T];
    vector[S] duration[T];
    row_vector[S] weighted_days;
    vector[S] temp_tpm[S];
    vector[T] c;
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
          temp_tpm[s_from,s] = calc_tpm(s_from,s, 
                                        intercept,
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
        weighted_days[s] = log_sum_exp(log(1+Days[t]-Days[t-1]), 
                                        day_temp + c[t] + emission[s,Rating[t]] + temp_tpm[s,s] + forw[t-1,s] - forw[t,s]);
        duration[t,s] = log1p_exp(dur_temp + c[t] + 
                                  emission[s,Rating[t]] + temp_tpm[s,s] + forw[t-1,s] - forw[t,s]);
      }
      
    }
    return exp(weighted_days);
  }
  row_vector[] get_duration_all(int[] Rating, real[] Days, real[] Sentiment, int T , int S , real[] l_pi , real[,] l_tpm , real[,] emission, real[] intercept){
    vector[S] forw[T];
    row_vector[S] duration[T];
    vector[S] temp_tpm[S];
    vector[T] c;
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
          temp_tpm[s_from,s] = calc_tpm(s_from,s, 
                                        intercept,
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
    return duration;
  }
}
data {
  int<lower=1> S; // number of states
  int<lower=0> N_total; // total number of restaurants considered
  int<lower=0> N_train; // number of restaurants in the training set N_test == N_total-N_train
  int<lower=0> N_samples;
  int<lower=1> N_obs; // Total number of reviews
  int<lower=1> Time[N_total];//Number of reviews for each restaurant sum(Time) == N_obs
  real<lower=0> Days[N_obs]; // number of days since first review of restaurant
  int<lower=1,upper=5> Ratings[N_obs]; //All ratings
  real Sentiment[N_obs];
  //Parameters
  //State_seq
  real l_pi[S];
  real l_tpm[S,S];
  real intercept[S];
  //Emission
  real emission[S,5];
  int<lower=1,upper=3> method; 
}
transformed data {
  //prior parameter vectors for states and pain levels
  int pos_obs[N_total];
  {
    for(m in 1:N_total){
      pos_obs[m] = (m == 1 ? 1 : pos_obs[m-1]+Time[m-1]);
    }
  }
}

model {
}
generated quantities{
  matrix[N_total,S] duration;
  row_vector[S] full_duration[N_obs];
  {
    if(method == 1){
    for(m in 1:N_total){
      int Rating_vec[Time[m]] = Ratings[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      real Days_vec[Time[m]] = Days[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      real Senti_vec[Time[m]] = Sentiment[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      duration[m] =  get_duration(Rating_vec, Days_vec, Senti_vec, Time[m], 
                                 S, l_pi, l_tpm, emission,
                                 intercept);
    }
    }else if(method == 2){
      for(m in 1:N_total){
      int Rating_vec[Time[m]] = Ratings[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      real Days_vec[Time[m]] = Days[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      real Senti_vec[Time[m]] = Sentiment[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      duration[m] =  get_duration_day(Rating_vec, Days_vec, Senti_vec, Time[m], 
                                 S, l_pi, l_tpm, emission,
                                 intercept);
    }
      }
      else{
      for(m in 1:N_total){
      int Rating_vec[Time[m]] = Ratings[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      real Days_vec[Time[m]] = Days[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      real Senti_vec[Time[m]] = Sentiment[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))];
      row_vector[S] temp[Time[m]];
      temp =  get_duration_all(Rating_vec, Days_vec, Senti_vec, Time[m], 
                                 S, l_pi, l_tpm, emission,
                                 intercept);
      full_duration[(1+sum(Time[1:(m-1)])):(sum(Time[1:m]))] = temp;                           
      }
    }
  }
}
