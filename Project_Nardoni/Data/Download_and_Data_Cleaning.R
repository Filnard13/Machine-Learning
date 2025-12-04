######################### MACHINE LEARNING PROJECT #############################
########################### DATASET PREPARATION ################################
#  Updated 16/10/2025
#  Filippo Nardoni 
#  ~ University of Bologna ~
################################################################################


###############################################################################
# ---------------------- SECTION 1 LIBRARIES AND DATA ----------------------- #
###############################################################################

rm(list=ls())

library(quantmod)
library(dplyr)
library(purrr)
library(xts)
library(lubridate)

# Countries Setting  and setting stocks selection
countries <- c("ES", "IT", "FR", "DE", "BE", "NL", "LU", "SE", "DK", "NO")

tickers_by_country <- list(
  
  PT = c( # Portugal – PSI
    "EDP.LS","GALP.LS","BCP.LS","NVG.LS","COR.LS","SON.LS","ALTR.LS","SCP.LS",
    "EGL.LS","RAM.LS","ECO.LS","IBS.LS","COIN.LS","MTX.LS","CPS.LS","SEM.LS",
    "RED.LS","GLS.LS","PHR.LS"
  ),
  
  ES = c( # Spain – IBEX 35
    "ITX.MC","IBE.MC","SAN.MC","BBVA.MC","CABK.MC","AENA.MC","REP.MC","FER.MC",
    "ACS.MC","ENG.MC","TEF.MC","REE.MC","END.MC","MAP.MC","COL.MC","MEL.MC",
    "GRF.MC","CIE.MC","CLNX.MC","ACC.MC"
  ),
  
  IT = c( # Italy – FTSE MIB
    "ENEL.MI","ISP.MI","ENI.MI","UCG.MI","RACE.MI","G.MI","PRY.MI","SRG.MI",
    "TRN.MI","MB.MI","BPE.MI","BAMI.MI","MONC.MI","LDO.MI","PST.MI","REC.MI",
    "BZU.MI","INW.MI","FBK.MI","CPR.MI"
  ),
  
  FR = c( # France – CAC 40
    "MC.PA","OR.PA","AIR.PA","BNP.PA","SU.PA","SAN.PA","GLE.PA","KER.PA",
    "VIE.PA","CAP.PA","SGO.PA","STM.PA","RNO.PA","ENGI.PA","PUB.PA","VIV.PA",
    "CS.PA","AI.PA","ACA.PA","UL.PA"
  ),
  
  DE = c( # Germany – DAX
    "SAP.DE","SIE.DE","ALV.DE","DTE.DE","VOW3.DE","MUV2.DE","BAS.DE","LIN.DE",
    "BMW.DE","BAYN.DE","DPW.DE","ADS.DE","RWE.DE","EOAN.DE","IFX.DE","HEN3.DE",
    "FRE.DE","DBK.DE","CON.DE","HEI.DE"
  ),
  
  BE = c( # Belgium – BEL 20 (expanded)
    "ABI.BR","GBLB.BR","KBC.BR","COFB.BR","UMI.BR","SOLB.BR","WDP.BR","ACKB.BR",
    "ELI.BR","PROX.BR","AGEAS.BR","APERB.BR","LOTB.BR","BPOST.BR","VGP.BR",
    "TESS.BR","ONTEX.BR","BARCO.BR","BEFB.BR","NYR.BR",
    # extras to balance
    "EURN.BR","TITR.BR","QRF.BR","DEME.BR","BEKB.BR"
  ),
  
  NL = c( # Netherlands – AEX (expanded)
    "ASML.AS","SHELL.AS","UNH.AS","AD.AS","PHIA.AS","NN.AS","IMCD.AS","DSM.AS",
    "ABN.AS","INGA.AS","AKZA.AS","URW.AS","RAND.AS","ASRNL.AS","HEIA.AS","KPN.AS",
    "SBMO.AS","GLPG.AS","WKL.AS","MT.AS",
    # extras
    "WHA.AS","TKWY.AS","ADYEN.AS","BESI.AS","PHAR.AS"
  ),
  
  LU = c( # Luxembourg – LuxX
    "SESF.LU","RTL.LU","SUB.LU","CLE.LU","CARG.LU","APG.LU","FBL.LU","SWLX.LU",
    "RRTL.LU","BIL.LU"
  ),
  
  SE = c( # Sweden – OMX Stockholm 30
    "ERIC-B.ST","VOLV-B.ST","ATCO-A.ST","SAND.ST","INVE-B.ST","SHB-A.ST","SEB-A.ST",
    "ASSA-B.ST","EVO.ST","ESSITY-B.ST","SKF-B.ST","ABB.ST","TELIA.ST","SKA-B.ST",
    "SWMA.ST","SCA-B.ST","SINCH.ST","HEX.ST","BOL.ST","HEXA-B.ST"
  ),
  
  DK = c( # Denmark – OMX Copenhagen 25 (expanded)
    "NOVO-B.CO","DANSKE.CO","VWS.CO","MAERSK-B.CO","ORSTED.CO","CARL-B.CO","COLR-B.CO",
    "GN.CO","DSV.CO","TRYG.CO","JYSK.CO","NZYM-B.CO","ROCK-B.CO","FLS.CO","NETC.CO",
    "CHR.CO","AMBUB.CO","LUN.CO","DEMANT.CO","BAVA.CO",
    # extras
    "PAND.CO","SNN.CO","ALK-B.CO","GEN.CO","WDH.CO"
  ),
  
  NO = c( # Norway – OBX
    "EQNR.OL","NHY.OL","TEL.OL","MOWI.OL","ORK.OL","AKRBP.OL","DNB.OL","YAR.OL",
    "TGS.OL","KOG.OL","STB.OL","SUBC.OL","PGS.OL","SCHB.OL","BWLPG.OL","FRO.OL",
    "LSG.OL","NAS.OL","HEX.OL","BAKKA.OL"
  )
)


# Start Date and End date selection
start_date <- as.Date("2003-01-01")
end_date   <- as.Date("2025-01-01")


# ----------- FUNCTION TO DOWNLOAD AND MERGE COUNTRY DATA -------------------- #

get_country_data <- function(tickers) {
  data_list <- list()
  ticker_names <- character()
  
  for (sym in tickers) {
    message("Downloading: ", sym)
    result <- tryCatch({
      data <- getSymbols(sym, src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)[,6]
      list(success = TRUE, data = data)
    }, error = function(e) {
      message("Failed: ", sym)
      list(success = FALSE)
    })
    
    if (result$success) {
      data_list[[length(data_list) + 1]] <- result$data
      ticker_names <- c(ticker_names, sym)
    }
  }
  
  if (length(data_list) == 0) return(NULL)
  
  merged_xts <- reduce(data_list, merge, all = TRUE)
  colnames(merged_xts) <- ticker_names
  
  return(merged_xts)
}


price_list <- map(tickers_by_country, get_country_data)


walk2(price_list, names(price_list), function(data, cname) {
  if (!is.null(data)) {
    assign(paste0(cname, "_data"), data, envir = .GlobalEnv)
  }
})



###############################################################################
# ---------------------- SECTION 2 ALIGNMENT and DATES----------------------- #
###############################################################################


datasets = lapply(countries, function(cty) {
  nm = paste0(cty, "_data")
  if (exists(nm)) get(nm) else NULL
})
names(datasets) = countries

# Remove NULL or empty datasets (ENTIRELY NULL)
datasets = datasets[!sapply(datasets, is.null)]

#  Compute union of all trading dates
all_dates = Reduce(union, lapply(datasets, index))
all_dates = sort(all_dates)

cat("Total unique trading days across all markets:", length(all_dates), "\n")
cat("From", min(all_dates), "to", max(all_dates), "\n")

# 2. Reindex each dataset to the full union of dates (preserving NA where missing)
datasets_aligned = lapply(datasets, function(x) {
  # Ensure the union index is of class Date
  all_dates = as.Date(all_dates)
  
  # Create an empty xts object with all desired dates
  template = xts(x = rep(NA, length(all_dates)), order.by = all_dates)
  template = template[, FALSE]  # keep structure but no columns
  
  # Merge ensures all_dates is used, and keeps original columns of x
  x_aligned = merge(template, x, all = TRUE)
  
  # Return the reindexed dataset (no need to drop anything)
  return(x_aligned)
})

# 3. Merge all datasets into one xts object (keeping all dates and NAs)
EU_data_all = do.call(merge, c(datasets_aligned, all = TRUE))

# 4. Final checks
cat("Final merged dataset dimensions:\n")
print(dim(EU_data_all))
cat("Time range:", start(EU_data_all), "to", end(EU_data_all), "\n")

na_count = sum(is.na(EU_data_all))
cat("Total missing values (expected with union merge):", na_count, "\n")




###############################################################################
# -------------------------- CLEANING DATA PROCESS -------------------------- #
###############################################################################



# 1] removing columns with first 100 rows reporting NAN
index_remove = c()
k = 1

for(i in 1:dim(EU_data_all)[2]){
  count = 0
  
  if(sum(is.na(EU_data_all[1:100,i]))){
    count = count + 1
  }
  
  if(count == 1){
    index_remove[k] = i
    k = k + 1
  }
}

data_cleaned = EU_data_all
if(length(index_remove) > 0) {
  data_cleaned = data_cleaned[,-index_remove]
}



#2] Count NAN
count_nan = sum(is.na(data_cleaned))
count_nan
which(is.na(data_cleaned), arr.ind = TRUE)
na_pos <- which(is.na(data_cleaned), arr.ind = TRUE)
dates_with_na <- index(data_cleaned)[na_pos[,"row"]]
columns_with_na <- colnames(data_cleaned)[na_pos[,"col"]]
data.frame(Date = dates_with_na, Column = columns_with_na)

col_missing = 0
for(i in 1:dim(data_cleaned)[2]){
  if( any(is.na(data_cleaned[,i]))){
    col_missing = col_missing + 1
  }
}
col_missing

## Option (1) Erase missing lines
# df = as.data.frame(data_cleaned)
# rownames(df)
# data_final = df[0,]
# k = 0
# unique_na = unique(dates_with_na)
# for(i in seq_len(nrow(df))){
#   if (!(rownames(df)[i] %in% unique_na)) {
#     k = k + 1
#     data_final[k, ] = df[i, ]
#   }
# }
# 
# is.na(data_final)


## Option (2) Imputation

for(i in 1:dim(data_cleaned)[2]){           
  for(j in 1:dim(data_cleaned)[1]){         
    
    if(is.na(data_cleaned[j,i])){           
      
      # Case 1: first observation → replace with next value
      if(j == 1){
        data_cleaned[j,i] = data_cleaned[j+1,i]
        
        # Case 2: last observation → replace with previous value
      } else if(j == dim(data_cleaned)[1]){
        data_cleaned[j,i] = data_cleaned[j-1,i]
        
        # Case 3: middle observation → average of previous and next values
      } else {
        data_cleaned[j,i] = mean(c(data_cleaned[j-1,i], data_cleaned[j+1,i]), na.rm = TRUE)
      }
    }
  }
}

data_final = data_cleaned




#3] Erase negative prices due to download problems
index_neg = c()
k = 1

for (i in 1:dim(data_final)[2]) {
  if (data_final[1, i] < 0) {
    index_neg[k] = i
    k = k + 1
  }
}

if(length(index_neg) > 0) {
  data_final = data_final[,-index_neg]
}








#4] Take Monthly data
if (inherits(data_final, "xts")) {
  df = data.frame(Date = index(data_final), coredata(data_final))
} else {
  df = data_final
  df$Date = as.Date(rownames(df))
}

df = df %>%
  mutate(Year = year(Date),
         Month = month(Date),
         Day = day(Date))

first_monthly_df = df %>%
  group_by(Year, Month) %>%
  arrange(Day) %>%
  filter(Day <= 5) %>%
  slice_head(n = 1) %>%
  ungroup()

data_monthly_first = xts(first_monthly_df %>% select(-Year, -Month, -Day, -Date),
                         order.by = first_monthly_df$Date)





###############################################################################
# ------------------- SECTION 4 CONVERTING TO RETURNS ----------------------- #
###############################################################################



# ---------------------------- Daily Data ------------------------------------#

dates <- as.character(index(data_final))
df <- as.matrix(data_final)

data_returns <- matrix(NA, nrow = nrow(df) - 1, ncol = ncol(df))

for (i in 1:ncol(df)) {
  for (j in 1:(nrow(df) - 1)) {
    if (!is.na(df[j, i]) && !is.na(df[j + 1, i]) && df[j, i] != 0) {
      data_returns[j, i] <- ((df[j + 1, i] - df[j, i]) / df[j, i]) * 100
    }
  }
}

data_returns <- as.data.frame(data_returns)
rownames(data_returns) <- dates[-1]
colnames(data_returns) <- colnames(data_final)


#### Check zeroes

zero_returns <- sum(data_returns == 0, na.rm = TRUE)

vect_zero <- numeric(ncol(data_returns))

for(i in 1:dim(data_returns)[2]){
  non_na_count <- sum(!is.na(data_returns[, i]))
  zero_count <- sum(data_returns[, i] == 0, na.rm = TRUE)
  
  if(non_na_count > 0){
    vect_zero[i] <- (zero_count / non_na_count) * 100
  } else {
    vect_zero[i] <- NA
  }
}

names(vect_zero) <- colnames(data_returns)
vect_zero_sorted <- sort(vect_zero, decreasing = TRUE)

vect_zero_sorted



# ---------------------------- Monthly Data ----------------------------------#

data_monthly <- data_monthly_first
dates_monthly <- as.character(index(data_monthly))
df_monthly <- as.matrix(data_monthly)

data_returns_monthly <- matrix(NA, nrow = nrow(df_monthly) - 1, ncol = ncol(df_monthly))

for (i in 1:ncol(df_monthly)) {
  for (j in 1:(nrow(df_monthly) - 1)) {
    if (!is.na(df_monthly[j, i]) && !is.na(df_monthly[j + 1, i]) && df_monthly[j, i] != 0) {
      data_returns_monthly[j, i] <- (df_monthly[j + 1, i] - df_monthly[j, i]) / df_monthly[j, i]
    }
  }
}

data_returns_monthly <- as.data.frame(data_returns_monthly)
rownames(data_returns_monthly) <- dates_monthly[-1]
colnames(data_returns_monthly) <- colnames(data_monthly)

zero_returns_monthly <- sum(data_returns_monthly == 0, na.rm = TRUE)

vect_zero_monthly <- numeric(ncol(data_returns_monthly))

for (i in 1:dim(data_returns_monthly)[2]) {
  non_na_count <- sum(!is.na(data_returns_monthly[, i]))
  zero_count <- sum(data_returns_monthly[, i] == 0, na.rm = TRUE)
  
  if(non_na_count > 0){
    vect_zero_monthly[i] <- (zero_count / non_na_count) * 100
  } else {
    vect_zero_monthly[i] <- NA
  }
}

names(vect_zero_monthly) <- colnames(data_returns_monthly)
vect_zero_sorted_monthly <- sort(vect_zero_monthly, decreasing = TRUE)

vect_zero_sorted_monthly





###############################################################################
# ----------------------- SECTION 5 SAVING SETTINGS ------------------------- #
###############################################################################

write.zoo(EU_data_all, file = "/Users/filipponardoni/Desktop/university/LMEC^2/2° Year/Machine Learning for Economists/Project/Data/EU_data_all_union_2.csv", sep = ",")
write.zoo(data_returns, file = "/Users/filipponardoni/Desktop/university/LMEC^2/2° Year/Machine Learning for Economists/Project/Data/EU_stocks_data_daily_2_new.csv", sep = ",")
write.zoo(data_returns_monthly, file = "/Users/filipponardoni/Desktop/university/LMEC^2/2° Year/Machine Learning for Economists/Project/Data/EU_stocks_data_monthly_2.csv_new", sep = ",")


dates_daily_vector <- rownames(data_returns)
dates_monthly_vector <- rownames(data_returns_monthly)

write.csv(data.frame(Date = dates_daily_vector), 
          file = "/Users/filipponardoni/Desktop/university/LMEC^2/2° Year/Machine Learning for Economists/Project/Data/dates_daily_new.csv",
          row.names = FALSE)