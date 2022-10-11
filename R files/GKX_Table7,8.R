# Gu Kelly Xiu 
### Table 8 : Fama French 5 analysis
# Author : Seongdeok Ko
# Import Packages
library(broom)
library(dplyr)
library(readr)
library(readxl)
library(sandwich)
library(lmtest)
FF6 <- read_xlsx("FF6.xlsx")
#####################################################
### PLS
pls <- read.csv("pls.csv")

r_10m1 <- pls$P10 * 100 - pls$P1 * 100

pls_mean <- mean(r_10m1)
pls_IR <- mean(r_10m1)/sd(r_10m1)

reg_pls <- lm( r_10m1 ~ FF6$RMRF + FF6$SMB + FF6$HML + 
                   FF6$RMW + FF6$CMA + FF6$UMD)

coeftest(reg_pls, vcov = NeweyWest(reg_pls, lag = 6) ) %>% tidy
#####################################################
p1 = pls$P1 ; p2 = pls$P2 ; p3 = pls$P3 ; p4 = pls$P4;
p5 = pls$P5 ; p6 = pls$P6 ; p7 = pls$P7 ; p8 = pls$P8;
p9 = pls$P9 ; p10 = pls$P10

##################### Mean Return 
print(c(mean(p1)*100,mean(p2)*100,mean(p3)*100,
        mean(p4)*100,mean(p5)*100,mean(p6)*100,
        mean(p7)*100,mean(p8)*100,mean(p9)*100,
        mean(p10)*100))
##################### Standard Deviation
print(c(sd(p1)*100,sd(p2)*100,
        sd(p3)*100,sd(p4)*100,
        sd(p5)*100,sd(p6)*100,
        sd(p7)*100,sd(p8)*100,
        sd(p9)*100,sd(p10)*100  )  )
#################### Sharpe Ratio
print(c(mean(p1)/sd(p1) * sqrt(12),
        mean(p2)/sd(p2) * sqrt(12),
        mean(p3)/sd(p3) * sqrt(12),
        mean(p4)/sd(p4) * sqrt(12),
        mean(p5)/sd(p5) * sqrt(12),
        mean(p6)/sd(p6) * sqrt(12),
        mean(p7)/sd(p7) * sqrt(12),
        mean(p8)/sd(p8) * sqrt(12),
        mean(p9)/sd(p9) * sqrt(12),
        mean(p10)/sd(p10) * sqrt(12) ))

#####################################################
### PCR
### 
pcr <- read.csv("pcr.csv")
r_10m1 <- pcr$P10 * 100 - pcr$P1 * 100
pcr_IR<- mean(r_10m1)/sd(r_10m1)
pcr_mean <- mean(r_10m1)
reg_pcr <- lm( r_10m1 ~ FF6$RMRF + FF6$SMB + FF6$HML +
                   FF6$RMW + FF6$CMA + FF6$UMD)

coeftest(reg_pcr, vcov = NeweyWest(reg_pcr, lag = 6) ) %>% tidy()
#####################################################
p1 = pls$P1 ; p2 = pls$P2 ; p3 = pls$P3 ; p4 = pls$P4;
p5 = pls$P5 ; p6 = pls$P6 ; p7 = pls$P7 ; p8 = pls$P8;
p9 = pls$P9 ; p10 = pls$P10

##################### Mean Return 
print(c(mean(p1)*100,mean(p2)*100,mean(p3)*100,
        mean(p4)*100,mean(p5)*100,mean(p6)*100,
        mean(p7)*100,mean(p8)*100,mean(p9)*100,
        mean(p10)*100))
##################### Standard Deviation
print(c(sd(p1)*100,sd(p2)*100,
        sd(p3)*100,sd(p4)*100,
        sd(p5)*100,sd(p6)*100,
        sd(p7)*100,sd(p8)*100,
        sd(p9)*100,sd(p10)*100  )  )
#################### Sharpe Ratio
print(c(mean(p1)/sd(p1) * sqrt(12),
        mean(p2)/sd(p2) * sqrt(12),
        mean(p3)/sd(p3) * sqrt(12),
        mean(p4)/sd(p4) * sqrt(12),
        mean(p5)/sd(p5) * sqrt(12),
        mean(p6)/sd(p6) * sqrt(12),
        mean(p7)/sd(p7) * sqrt(12),
        mean(p8)/sd(p8) * sqrt(12),
        mean(p9)/sd(p9) * sqrt(12),
        mean(p10)/sd(p10) * sqrt(12) ))

###########################
###########################
### Elastic Net 
enet <- read.csv("enet.csv")

r_10m1 <- enet$P10 * 100 - enet$P1 * 100

enet_IR<- mean(r_10m1)/sd(r_10m1)
enet_mean <-  mean(r_10m1)
reg_enet <- lm( r_10m1 ~ FF6$RMRF + FF6$SMB + 
                    FF6$HML + FF6$RMW + FF6$CMA + FF6$UMD)
reg_enet %>% coeftest(.,  vcov = NeweyWest(reg_enet, lag = 6) ) %>% tidy()

###################################
###################################

p1 = enet$P1 ; p2 = enet$P2
p3 = enet$P3 ; p4 = enet$P4
p5 = enet$P5 ; p6 = enet$P6
p7 = enet$P7 ; p8 = enet$P8
p9 = enet$P9 ; p10 = enet$P10

print(c(mean(p1)*100,mean(p2)*100,mean(p3)*100,
        mean(p4)*100,mean(p5)*100,mean(p6)*100,
        mean(p7)*100,mean(p8)*100,mean(p9)*100,
        mean(p10)*100))

print(c(sd(p1)*100,sd(p2)*100,
        sd(p3)*100,sd(p4)*100,
        sd(p5)*100,sd(p6)*100,
        sd(p7)*100,sd(p8)*100,
        sd(p9)*100,sd(p10)*100  )  )

print(c(mean(p1)/sd(p1) * sqrt(12),
        mean(p2)/sd(p2) * sqrt(12),
        mean(p3)/sd(p3) * sqrt(12),
        mean(p4)/sd(p4) * sqrt(12),
        mean(p5)/sd(p5) * sqrt(12),
        mean(p6)/sd(p6) * sqrt(12),
        mean(p7)/sd(p7) * sqrt(12),
        mean(p8)/sd(p8) * sqrt(12),
        mean(p9)/sd(p9) * sqrt(12),
        mean(p10)/sd(p10) * sqrt(12) ))


################### GLM

glm<- read.csv("glm.csv")


r_10m1 <- glm$P10 * 100 - glm$P1 * 100

glm_IR <- mean(r_10m1)/sd(r_10m1)
glm_mean <- mean(r_10m1)

reg_glm <- lm( r_10m1 ~ FF6$RMRF + FF6$SMB + FF6$HML + FF6$RMW + FF6$CMA + FF6$UMD)
reg_glm %>% coeftest(., vcov = NeweyWest(., lag = 6)) %>% tidy()


#############################

p1 = glm$P1 ; p2 = glm$P2
p3 = glm$P3 ; p4 = glm$P4
p5 = glm$P5 ; p6 = glm$P6
p7 = glm$P7 ; p8 = glm$P8
p9 = glm$P9 ; p10 = glm$P10

################################
print(c(mean(p1)*100,mean(p2)*100,mean(p3)*100,
        mean(p4)*100,mean(p5)*100,mean(p6)*100,
        mean(p7)*100,mean(p8)*100,mean(p9)*100,
        mean(p10)*100))

print(c(sd(p1)*100,sd(p2)*100,
        sd(p3)*100,sd(p4)*100,
        sd(p5)*100,sd(p6)*100,
        sd(p7)*100,sd(p8)*100,
        sd(p9)*100,sd(p10)*100  )  )

print(c(mean(p1)/sd(p1) * sqrt(12),
        mean(p2)/sd(p2) * sqrt(12),
        mean(p3)/sd(p3) * sqrt(12),
        mean(p4)/sd(p4) * sqrt(12),
        mean(p5)/sd(p5) * sqrt(12),
        mean(p6)/sd(p6) * sqrt(12),
        mean(p7)/sd(p7) * sqrt(12),
        mean(p8)/sd(p8) * sqrt(12),
        mean(p9)/sd(p9) * sqrt(12),
        mean(p10)/sd(p10) * sqrt(12) ))

### NN3 #########################

NN3 <- read.csv("NN3_final.csv")

r_10m1 <- NN3$P10 * 100 - NN3$P1 * 100

NN3_IR <- mean(r_10m1)/sd(r_10m1)
NN3_mean <- mean(r_10m1)

reg_NN3 <- lm( r_10m1 ~ FF6$RMRF + FF6$SMB + FF6$HML + FF6$RMW + FF6$CMA + FF6$UMD)

reg_NN3 %>% coeftest(., vcov = NeweyWest(., lag = 6)) %>% tidy()

p1 = NN3$P1 ; p2 = NN3$P2
p3 = NN3$P3 ; p4 = NN3$P4
p5 = NN3$P5 ; p6 = NN3$P6
p7 = NN3$P7 ; p8 = NN3$P8
p9 = NN3$P9 ; p10 = NN3$P10


print(c(mean(p1)*100,mean(p2)*100,mean(p3)*100,
        mean(p4)*100,mean(p5)*100,mean(p6)*100,
        mean(p7)*100,mean(p8)*100,mean(p9)*100,
        mean(p10)*100))

print(c(sd(p1)*100,sd(p2)*100,
        sd(p3)*100,sd(p4)*100,
        sd(p5)*100,sd(p6)*100,
        sd(p7)*100,sd(p8)*100,
        sd(p9)*100,sd(p10)*100  )  )

print(c(mean(p1)/sd(p1) * sqrt(12),
        mean(p2)/sd(p2) * sqrt(12),
        mean(p3)/sd(p3) * sqrt(12),
        mean(p4)/sd(p4) * sqrt(12),
        mean(p5)/sd(p5) * sqrt(12),
        mean(p6)/sd(p6) * sqrt(12),
        mean(p7)/sd(p7) * sqrt(12),
        mean(p8)/sd(p8) * sqrt(12),
        mean(p9)/sd(p9) * sqrt(12),
        mean(p10)/sd(p10) * sqrt(12) ))

#############


##########################################################
## Random Forest 
rf <- read.csv("rf.csv")

##############

p1 = rf$P1 ; p2 = rf$P2
p3 = rf$P3 ; p4 = rf$P4
p5 = rf$P5 ; p6 = rf$P6
p7 = rf$P7 ; p8 = rf$P8
p9 = rf$P9 ; p10 = rf$P10


print(c(mean(p1)*100,mean(p2)*100,mean(p3)*100,
        mean(p4)*100,mean(p5)*100,mean(p6)*100,
        mean(p7)*100,mean(p8)*100,mean(p9)*100,
        mean(p10)*100))

print(c(sd(p1)*100,sd(p2)*100,
        sd(p3)*100,sd(p4)*100,
        sd(p5)*100,sd(p6)*100,
        sd(p7)*100,sd(p8)*100,
        sd(p9)*100,sd(p10)*100  )  )

print(c(mean(p1)/sd(p1) * sqrt(12),
        mean(p2)/sd(p2) * sqrt(12),
        mean(p3)/sd(p3) * sqrt(12),
        mean(p4)/sd(p4) * sqrt(12),
        mean(p5)/sd(p5) * sqrt(12),
        mean(p6)/sd(p6) * sqrt(12),
        mean(p7)/sd(p7) * sqrt(12),
        mean(p8)/sd(p8) * sqrt(12),
        mean(p9)/sd(p9) * sqrt(12),
        mean(p10)/sd(p10) * sqrt(12) ))




##############

r_10m1 <- rf$P10 * 100 - rf$P1 * 100

rf_IR <- mean(r_10m1)/sd(r_10m1)
rf_mean <- mean(r_10m1)

reg_rf <- lm( r_10m1 ~ FF6$RMRF + FF6$SMB + FF6$HML + FF6$RMW + FF6$CMA + FF6$UMD)
tidy(reg_rf)

reg_rf %>% coeftest(., vcov = NeweyWest(., lag = 6)) %>% tidy()
 
summary(reg_rf)$r.squared * 100
summary(reg_pcr)$r.squared * 100
summary(reg_pls)$r.squared * 100
summary(reg_NN3)$r.squared * 100
summary(reg_glm)$r.squared * 100

## 기초 통계량 잡기 

print(c(pls_IR,pcr_IR,enet_IR,glm_IR,rf_IR,NN3_IR) * sqrt(12))
print(c(pls_mean,pcr_mean,enet_mean,glm_mean,rf_mean,NN3_mean))

##############################################################3
## Equal Weight ###
FF6 <- read_xlsx("FF6.xlsx")
#####################################################
pls <- read.csv("pls_equal.csv")
remove()
r_10m1 <- pls$P10 * 100 - pls$P1 * 100

pls_mean <- mean(r_10m1)
pls_IR <- mean(r_10m1)/sd(r_10m1)

reg_pls <- lm( r_10m1 ~ FF6$RMRF + FF6$SMB + FF6$HML + FF6$RMW + FF6$CMA + FF6$UMD)
tidy(reg_pls)

NeweyWest(reg_pls, lag = 6)
library(lmtest)
coeftest(reg_pls, vcov = NeweyWest(reg_pls, lag = 6) ) %>%
    tidy

pcr <- read.csv("pcr_equal.csv")
remove()
r_10m1 <- pcr$P10 * 100 - pcr$P1 * 100
pcr_IR<- mean(r_10m1)/sd(r_10m1)
pcr_mean <- mean(r_10m1)

reg_pcr <- lm( r_10m1 ~ FF6$RMRF + FF6$SMB + FF6$HML + FF6$RMW + FF6$CMA + FF6$UMD)
tidy(reg_pcr)

coeftest(reg_pcr, vcov = NeweyWest(reg_pcr, lag = 6) ) %>% tidy()
####
enet <- read.csv("enet_equal.csv")
remove()

r_10m1 <- enet$P10 * 100 - enet$P1 * 100

enet_IR<- mean(r_10m1)/sd(r_10m1)
enet_mean <-  mean(r_10m1)
reg_enet <- lm( r_10m1 ~ FF6$RMRF + FF6$SMB + FF6$HML + FF6$RMW + FF6$CMA + FF6$UMD)
reg_enet %>% coeftest(.,  vcov = NeweyWest(reg_enet, lag = 12) ) %>% tidy()

#### GLM

glm<- read.csv("glm_equal.csv")
remove()
r_10m1 <- glm$P10 * 100 - glm$P1 * 100

glm_IR <- mean(r_10m1)/sd(r_10m1)
glm_mean <- mean(r_10m1)

reg_glm <- lm( r_10m1 ~ FF6$RMRF + FF6$SMB + FF6$HML + FF6$RMW + FF6$CMA + FF6$UMD)
reg_glm %>% coeftest(., vcov = NeweyWest(., lag = 6)) %>% tidy()

### NN3

NN3 <- read.csv("NN3_final_equal.csv")
remove()
r_10m1 <- NN3$P10 * 100 - NN3$P1 * 100

NN3_IR <- mean(r_10m1)/sd(r_10m1)
NN3_mean <- mean(r_10m1)

reg_NN3 <- lm( r_10m1 ~ FF6$RMRF + FF6$SMB + FF6$HML + FF6$RMW + FF6$CMA + FF6$UMD)
tidy(reg_NN3)

reg_NN3 %>% coeftest(., vcov = NeweyWest(., lag = 6)) %>% tidy()

## Random Forest 
rf <- read.csv("rf_equal.csv")
remove()
r_10m1 <- rf$P10 * 100 - rf$P1 * 100

rf_IR <- mean(r_10m1)/sd(r_10m1)
rf_mean <- mean(r_10m1)

reg_rf <- lm( r_10m1 ~ FF6$RMRF + FF6$SMB + FF6$HML + FF6$RMW + FF6$CMA + FF6$UMD)
tidy(reg_rf)

reg_rf %>% coeftest(., vcov = NeweyWest(., lag = 6)) %>% tidy()

summary(reg_rf)$r.squared * 100
summary(reg_pcr)$r.squared * 100
summary(reg_pls)$r.squared * 100
summary(reg_NN3)$r.squared * 100
summary(reg_glm)$r.squared * 100

print(c(pls_IR,pcr_IR,enet_IR,glm_IR,rf_IR,NN3_IR) * sqrt(12))
print(c(pls_mean,pcr_mean,enet_mean,glm_mean,rf_mean,NN3_mean))
