library(dplyr)
library(readr)

y_true <- read_csv('y_true.csv')
NN3_1_1 <- read_csv('y_NN3_1_1st.csv')
NN3_1_2 <- read_csv('y_NN3_1_2nd.csv')
NN3_1_3 <- read_csv('y_NN3_1_3rd.csv')
NN3_1_4 <- read_csv('y_NN3_1_4th.csv')
NN3_1_5 <- read_csv('y_NN3_1_5th.csv')
NN3_1_6 <- read_csv('y_NN3_1_6th.csv')
NN3_1_7 <- read_csv('y_NN3_1_7th.csv')
NN3_1_8 <- read_csv('y_NN3_1_8th.csv')

NN3_2_1 <- read_csv('y_NN3_2_1st.csv')
NN3_2_2 <- read_csv('y_NN3_2_2nd.csv')
NN3_2_3 <- read_csv('y_NN3_2_3rd.csv')
NN3_2_4 <- read_csv('y_NN3_2_4th.csv')
NN3_2_5 <- read_csv('y_NN3_2_5th.csv')
NN3_2_6 <- read_csv('y_NN3_2_6th.csv')
NN3_2_7 <- read_csv('y_NN3_2_7th.csv')
NN3_2_8 <- read_csv('y_NN3_2_8th.csv')

NN3_3_1 <- read_csv('y_NN3_3_1st.csv')
NN3_3_2 <- read_csv('y_NN3_3_2nd.csv')
NN3_3_3 <- read_csv('y_NN3_3_3rd.csv')
NN3_3_4 <- read_csv('y_NN3_3_4th.csv')
NN3_3_5 <- read_csv('y_NN3_3_5th.csv')
NN3_3_6 <- read_csv('y_NN3_3_6th.csv')
NN3_3_7 <- read_csv('y_NN3_3_7th.csv')
NN3_3_8 <- read_csv('y_NN3_3_8th.csv')

NN3_4_1 <- read_csv('y_NN3_4_1st.csv')
NN3_4_2 <- read_csv('y_NN3_4_2nd.csv')
NN3_4_3 <- read_csv('y_NN3_4_3rd.csv')
NN3_4_4 <- read_csv('y_NN3_4_4th.csv')
NN3_4_5 <- read_csv('y_NN3_4_5th.csv')
NN3_4_6 <- read_csv('y_NN3_4_6th.csv')
NN3_4_7 <- read_csv('y_NN3_4_7th.csv')
NN3_4_8 <- read_csv('y_NN3_4_8th.csv')

NN3_1 <- bind_rows(NN3_1_1,NN3_1_2,NN3_1_3,NN3_1_4,
                    NN3_1_5,NN3_1_6,NN3_1_7,NN3_1_8)

NN3_2 <- bind_rows(NN3_2_1,NN3_2_2,NN3_2_3,NN3_2_4,
                    NN3_2_5,NN3_2_6,NN3_2_7,NN3_2_8)

NN3_3 <- bind_rows(NN3_3_1,NN3_3_2,NN3_3_3,NN3_3_4,
                   NN3_3_5,NN3_3_6,NN3_3_7,NN3_3_8)

NN3_4 <- bind_rows(NN3_4_1,NN3_4_2,NN3_4_3,NN3_4_4,
                   NN3_4_5,NN3_4_6,NN3_4_7,NN3_4_8)

size <- read_csv('size.csv')
y_true1 <- y_true %>% bind_cols(., NN3_1,NN3_2,NN3_3)
y_true2 <- y_true1 %>% mutate( MEAN = 1/3 * (NN3_1 + NN3_2 + NN3_3  ))

tidy1 <- left_join(y_true2, size, by = c("permno" = 'permno', 'DATE' = 'DATE') )

tidy2 <- tidy1 %>% select(-c(X1,NN3...4,NN3...5,NN3...6) )

## 데이터 변환 끝 

tidy3<- tidy2 %>% 
    group_by(DATE) %>% mutate( Rank = ntile(MEAN,10))
tidy4 <- tidy3 %>% ungroup() %>%
    rename(y_real = excess_return)
# 1이 낮은거 ~ 10이 제일 높은 거 

Date <-  tidy3 %>% ungroup() %>% select(DATE) %>% distinct()
port_tb <- tibble( Date = Date$DATE ) # Where to save portfolio value

P1 = seq(0, length = length(Date$DATE))
P2 = seq(0, length = length(Date$DATE))
P3 = seq(0, length = length(Date$DATE))
P4 = seq(0, length = length(Date$DATE))
P5 = seq(0, length = length(Date$DATE))
P6 = seq(0, length = length(Date$DATE))
P7 = seq(0, length = length(Date$DATE))
P8 = seq(0, length = length(Date$DATE))
P9 = seq(0, length = length(Date$DATE))
P10 = seq(0, length = length(Date$DATE))


for ( i in 1:length(Date$DATE))     {
    P1[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 1) %>% 
        mutate( weight = mvel1/sum(mvel1) ) %>%
        mutate( r_part = weight * y_real) %>% 
        select(r_part) %>% sum();
    
    P2[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 2) %>% 
        mutate( weight = mvel1/sum(mvel1) ) %>%
        mutate( r_part = weight * y_real) %>% 
        select(r_part) %>% sum();
    
    P3[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 3) %>% 
        mutate( weight = mvel1/sum(mvel1) ) %>%
        mutate( r_part = weight * y_real) %>% 
        select(r_part) %>% sum();
    
    P4[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 4) %>% 
        mutate( weight = mvel1/sum(mvel1) ) %>%
        mutate( r_part = weight * y_real) %>% 
        select(r_part) %>% sum();
    
    P5[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 5) %>% 
        mutate( weight = mvel1/sum(mvel1) ) %>%
        mutate( r_part = weight * y_real) %>% 
        select(r_part) %>% sum();
    
    P6[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 6) %>% 
        mutate( weight = mvel1/sum(mvel1) ) %>%
        mutate( r_part = weight * y_real) %>% 
        select(r_part) %>% sum();
    
    P7[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 7) %>% 
        mutate( weight = mvel1/sum(mvel1) ) %>%
        mutate( r_part = weight * y_real) %>% 
        select(r_part) %>% sum();
    
    P8[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 8) %>% 
        mutate( weight = mvel1/sum(mvel1) ) %>%
        mutate( r_part = weight * y_real) %>% 
        select(r_part) %>% sum();
    
    P9[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 9) %>% 
        mutate( weight = mvel1/sum(mvel1) ) %>%
        mutate( r_part = weight * y_real) %>% 
        select(r_part) %>% sum() ;
    
    P10[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 10) %>% 
        mutate( weight = mvel1/sum(mvel1) ) %>%
        mutate( r_part = weight * y_real) %>% 
        select(r_part) %>% sum() 
}

port_tb <- port_tb %>% 
    mutate(P1 = P1, P2 = P2 , P3 = P3, P4 = P4,
           P5 = P5, P6= P6, P7 = P7, P8 = P8,
           P9 = P9, P10 = P10)
port_tb1 <- port_tb %>% 
    mutate( log_ret_P1 = log(1 + P1),
            log_ret_P2 = log(1 + P2),
            log_ret_P3 = log(1 + P3),
            log_ret_P4 = log(1 + P4),
            log_ret_P5 = log(1 + P5),
            log_ret_P6 = log(1 + P6),
            log_ret_P7 = log(1 + P7),
            log_ret_P8 = log(1 + P8),
            log_ret_P9 = log(1 + P9),
            log_ret_P10 = log(1 + P10),
    ) %>% 
    mutate( CS1 = cumsum(log_ret_P1), 
            CS2 = cumsum(log_ret_P2),
            CS3 = cumsum(log_ret_P3), 
            CS4 = cumsum(log_ret_P4),
            CS5 = cumsum(log_ret_P5) , 
            CS6 = cumsum(log_ret_P6),
            CS7 = cumsum(log_ret_P7) , 
            CS8 = cumsum(log_ret_P8),
            CS9 = cumsum(log_ret_P9), 
            CS10 = cumsum(log_ret_P10),
            RET1 = exp(CS1) -1,
            RET2 = exp(CS2) -1,
            RET3 = exp(CS3) -1,
            RET4 = exp(CS4) -1,
            RET5 = exp(CS5) -1,
            RET6 = exp(CS6) -1,
            RET7 = exp(CS7) -1,
            RET8 = exp(CS8) -1,
            RET9 = exp(CS9) -1,
            RET10 = exp(CS10) -1 )
#####################
#####################


E1 = seq(0, length = length(Date$DATE))
E2 = seq(0, length = length(Date$DATE))
E3 = seq(0, length = length(Date$DATE))
E4 = seq(0, length = length(Date$DATE))
E5 = seq(0, length = length(Date$DATE))
E6 = seq(0, length = length(Date$DATE))
E7 = seq(0, length = length(Date$DATE))
E8 = seq(0, length = length(Date$DATE))
E9 = seq(0, length = length(Date$DATE))
E10 = seq(0, length = length(Date$DATE))

for ( i in 1:length(Date$DATE))     {
    E1[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 1) %>% select(y_real) %>% 
        summarise(., r1 = mean(y_real)) %>% as.numeric()
    
    E2[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 2) %>% select(y_real) %>% 
        summarise(., r1 = mean(y_real)) %>% as.numeric()
    
    E3[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 3) %>% select(y_real) %>% 
        summarise(., r1 = mean(y_real)) %>% as.numeric()
    
    E4[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 4) %>% select(y_real) %>% 
        summarise(., r1 = mean(y_real)) %>% as.numeric()
    
    E5[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 5) %>% select(y_real) %>% 
        summarise(., r1 = mean(y_real)) %>% as.numeric()
    
    E6[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 6) %>% select(y_real) %>% 
        summarise(., r1 = mean(y_real)) %>% as.numeric()
    
    E7[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 7) %>% select(y_real) %>% 
        summarise(., r1 = mean(y_real)) %>% as.numeric()
    
    E8[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 8) %>% select(y_real) %>% 
        summarise(., r1 = mean(y_real)) %>% as.numeric()
    
    E9[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 9) %>% select(y_real) %>% 
        summarise(., r1 = mean(y_real)) %>% as.numeric()
    
    E10[i] = tidy4 %>% filter(., DATE == Date$DATE[i]) %>% 
        filter( Rank == 10) %>% select(y_real) %>% 
        summarise(., r1 = mean(y_real)) %>% as.numeric()
}


port_tbE <- port_tb %>% 
    mutate(P1 = E1, P2 = E2 , P3 = E3, P4 = E4,
           P5 = E5, P6= E6, P7 = E7, P8 = E8,
           P9 = E9, P10 = E10)

port_tb1E <- port_tbE %>% 
    mutate( log_ret_P1 = log(1 + P1),
            log_ret_P2 = log(1 + P2),
            log_ret_P3 = log(1 + P3),
            log_ret_P4 = log(1 + P4),
            log_ret_P5 = log(1 + P5),
            log_ret_P6 = log(1 + P6),
            log_ret_P7 = log(1 + P7),
            log_ret_P8 = log(1 + P8),
            log_ret_P9 = log(1 + P9),
            log_ret_P10 = log(1 + P10),
    ) %>% 
    mutate( CS1 = cumsum(log_ret_P1), 
            CS2 = cumsum(log_ret_P2),
            CS3 = cumsum(log_ret_P3), 
            CS4 = cumsum(log_ret_P4),
            CS5 = cumsum(log_ret_P5) , 
            CS6 = cumsum(log_ret_P6),
            CS7 = cumsum(log_ret_P7) , 
            CS8 = cumsum(log_ret_P8),
            CS9 = cumsum(log_ret_P9), 
            CS10 = cumsum(log_ret_P10),
            RET1 = exp(CS1) -1,
            RET2 = exp(CS2) -1,
            RET3 = exp(CS3) -1,
            RET4 = exp(CS4) -1,
            RET5 = exp(CS5) -1,
            RET6 = exp(CS6) -1,
            RET7 = exp(CS7) -1,
            RET8 = exp(CS8) -1,
            RET9 = exp(CS9) -1,
            RET10 = exp(CS10) -1 )


write.csv(port_tb1,'NN3_final.csv')
write.csv(port_tb1E,'NN3_final_equal.csv')
