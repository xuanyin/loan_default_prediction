rm(list = ls())
setwd("C:/Users/xyin/Desktop/Manor Resource/test-loan-data")
getwd()

library(data.table)
library(stringr)

data = read.csv("loan.csv")

dim(data)[1]

str(data)

#turn the factor of state into numerical variable
##Median Household Income by State in 2015 from Census Bureau.  Households as of March of 2016.  Income in 2015 CPI-U-RS adjusted dollars.
##https://www.census.gov/data/tables/time-series/demo/income-poverty/historical-income-households.html
income = read.csv("income.csv")
data <- merge(data, income, by.x=c("addr_state"), by.y=c("state"))
data$addr_state <- NULL
data$state_median_household_income <- as.numeric(data$state_median_household_income)

############################################################
##Row operations

##Step 1: construct outcome variables: loan_status
levels(data$loan_status)
YN <- c("Issued")
YT <- c("Current","Fully Paid","In Grace Period","Does not meet the credit policy. Status:Fully Paid")
YF <- c("Charged Off","Default", "Late (16-30 days)","Late (31-120 days)", "Does not meet the credit policy. Status:Charged Off")
data <- data[!(data$loan_status %in% YN),]  #delete all status being "Issued"
data$loan_status <- (data$loan_status %in% YT) #Assigning the status in YT with True, the rest with FALSE

###########################################################
##Step 2: removing variables containing more than 50% missing values

#count NAs
data[data == ""] <- NA

countNA = function(x) {
  if (is.numeric(x) == TRUE) {
  return(sum(is.na(x)))
  } else
    return(0)
}

#The percentage of NAs in each covariate
pNA = function(x) {
  if (is.numeric(x) == TRUE) {
  return(100*countNA(x)/(length(x)))
  } else
    return(0)
}

a = sapply(data, pNA)
write.csv(a, file = "a.csv")
rm(a)

#Method 1
aa <- sapply(data, function(x) {pNA(x) < 50})
data2 <- data[aa]

#Method 2
data1 <- data[colSums(is.na(data))/nrow(data) < .5]

rm(data2, data, aa)

###########################################################
##Step 3: removing observations of missing values

str(data1)
sapply(data1, pNA)

data2 = na.omit(data1)
sapply(data2, pNA)

rm(data1)

#############################################
##Step 4: Outliers
# trim data based on iqr (inter quartile range)
for (i in 1:dim(data2)[2]) {
  if (is.numeric(data2[,i]) == TRUE) {
    q = quantile(data2[,i], prob = c(0.25, 0.75), na.rm = TRUE, names = TRUE, type = 1)
    iqr = q[2] - q[1]
    data2[(data2[,i] > q[2] + (6 * iqr)) | (data2[,i] < q[1] - (6 * iqr)), i] <- NA
  }
}

data2 <- na.omit(data2)

##########################################################
loan_status = as.factor(data2$loan_status)
data2$loan_status <- NULL

###########################################################
##########################################################
###Column Operations
##Step 5: seperate factors and num

##turn desc into numerical variable, the length of desc
data2$desc <- as.numeric(str_length(data2$desc)) 

str(data2)
factors = data2[sapply(data2, is.factor)]
str(factors)

##drop factor variables that have too many levels
sub_factors <- factors[sapply(factors, function(x) {nlevels(x) >= 5})] 
str(sub_factors)
####################
#$ grade             : Factor w/ 7 levels "A","B","C","D",..: 4 7 1 5 4 4 2 3 1 1 ...
#$ sub_grade         : Factor w/ 35 levels "A1","A2","A3",..: 17 32 2 24 18 17 9 15 3 3 ...
#$ emp_title         : Factor w/ 299273 levels "","'Property Manager",..: 207130 182905 236387 134494 30960 63614 178573 248249 293295 177784 ...
#$ emp_length        : Factor w/ 12 levels "< 1 year","1 year",..: 4 10 3 4 10 8 1 4 5 5 ...
#$ home_ownership    : Factor w/ 6 levels "ANY","MORTGAGE",..: 6 2 2 6 2 2 2 6 2 2 ...
#$ issue_d           : Factor w/ 103 levels "Apr-2008","Apr-2009",..: 51 17 85 93 85 94 76 8 42 58 ...
#$ url               : Factor w/ 887379 levels "https://www.lendingclub.com/browse/loanDetail.action?loan_id=1000007",..: 554602 612629 727554 237502 729336 674346 488050 430261 329286 606740 ...
#$ purpose           : Factor w/ 14 levels "car","credit_card",..: 3 3 5 3 3 3 2 9 2 3 ...
#$ title             : Factor w/ 63146 levels "","'08 & '09 Roth IRA Investments",..: 18723 18723 31768 18723 18723 18723 15530 41040 15530 18725 ...
#$ zip_code          : Factor w/ 935 levels "007xx","008xx",..: 931 935 932 932 932 934 933 933 931 931 ...
#$ earliest_cr_line  : Factor w/ 698 levels "","Apr-1955",..: 336 686 451 345 342 344 165 570 38 101 ...
#$ last_pymnt_d      : Factor w/ 99 levels "","Apr-2008",..: 43 26 43 43 43 43 43 43 43 43 ...
#$ next_pymnt_d      : Factor w/ 101 levels "","Apr-2008",..: 35 35 35 35 35 35 35 35 35 35 ...
#$ last_credit_pull_d: Factor w/ 104 levels "","Apr-2009",..: 43 43 43 43 43 43 43 43 43 43 ...
####################
droplist <- c("sub_grade", "emp_title", "issue_d", "url", "title", "zip_code", "earliest_cr_line", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d")

rm(sub_factors)
factors <- factors[,!(names(factors) %in% droplist)]

##create design matrix, because we can only put dummies generated by factor variables into models
str(factors)
dummies = data.frame(model.matrix(~ . , data = factors)) 

##drop factors with sparsity nearly 1
sparsity = function(x) {
  op = sort(table(x), decreasing = TRUE)[1]
  return(op/length(x))
}

dummies = dummies[sapply(dummies, function(x) {sparsity(x) < 0.9999})] #it will drop the intercept

factors = factors[sapply(factors, function(x) {sparsity(x) < 0.9999})] #it will drop the intercept

write.csv(factors, file = "factors.csv", row.names=FALSE)

###############################################################################################################
numbers = data2[sapply(data2, is.numeric)]
numbers$id <- NULL
numbers$member_id <- NULL

str(numbers)

##drop numerical variables with 0 variance
numbers = numbers[sapply(numbers, function(x) {var(x) != 0})]

##detect and drop collinear variables
data3 = data.frame(numbers, dummies)
cm = cor(data3, use = "pairwise.complete.obs")
collinear = apply(lower.tri(cm)& (abs(cm)>0.95), 1, sum)
data4 = data3[, (names(collinear)[collinear == 0])]

###################################################################################################################
x = data4
dummies <- x[, (names(x) %in% names(dummies))]
numericals <- x[, (names(x) %in% names(numbers))]

#######################################
# rescale all the numeric data into (0,1) interval

#for (ic in 1:ncol(numericals)) {
#  numericals[,ic] <- numericals[,ic] - min(numericals[,ic])
#  numericals[,ic] <- numericals[,ic]/max(numericals[,ic])
#}

# standardize the numeric data
for (ic in 1:ncol(numericals)) {
  numericals[,ic] <- (numericals[,ic] - mean(numericals[,ic]))/sd(numericals[,ic])
}

# measure the association between numericals and outcome, drop numericals that are not so correlated
names.numericals = names(numericals)
for (ic in 1:ncol(numericals)) {
  lmm <- lm(numericals[, ic] ~ loan_status)
  slmm <- summary(lmm)
  if (pf(slmm$fstatistic[1], slmm$fstatistic[2], slmm$fstatistic[3], lower.tail=FALSE) > 0.05) {
    names.numericals[ic] <- NA
  } 
}

numericals <- numericals[na.omit(names.numericals)]


###################################################################################################################
x = data.frame(dummies, numericals)
data5 = data.frame(x, loan_status)
###################################################################################################################


rm(data2, data3, data4, factors, numbers, cm)

write.csv(data5, file = "data5.csv", row.names=FALSE)
write.csv(dummies, file = "dummies.csv", row.names=FALSE)
write.csv(numericals, file = "numericals.csv", row.names=FALSE)
write.csv(x, file = "x.csv", row.names=FALSE)
