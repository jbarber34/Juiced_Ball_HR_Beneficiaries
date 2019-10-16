######################################
#### Juiced Ball HR Beneficiaries ####
######################################

library(dplyr)
library(mlr)

df <- readRDS("juiced_ball_HR_beneficiaries_df.rds")

# Determine mean values of 2019 HR to be used to find potential HR in other years
df2 <- df %>% 
  filter(!is.na(EXIT_SPEED),
         SEASON_KEY == 2019) %>% 
  summarise(MEAN_EV = mean(EXIT_SPEED),
            MIN_EV = min(EXIT_SPEED),
            MEAN_LA = mean(EXIT_ANGLE),
            MIN_LA = min(EXIT_ANGLE),
            MAX_LA = max(EXIT_ANGLE))

# Summary stats for average HR by year - filtered for 5+ HR/year
hr_grouped_by_player <- df %>% 
  group_by(BATTER_KEY, PLAYER_NAME, SEASON_KEY) %>% 
  summarise(HR = sum(HR),
            HIT_LOCATION_X = mean(HIT_LOCATION_X, na.rm = T),
            HIT_LOCATION_Y = mean(HIT_LOCATION_Y, na.rm = T),
            HIT_DISTANCE = mean(HIT_DISTANCE, na.rm = T),
            EXIT_SPEED = mean(EXIT_SPEED, na.rm = T),
            EXIT_ANGLE = mean(EXIT_ANGLE, na.rm = T),
            EXIT_DISTANCE = mean(EXIT_DISTANCE, na.rm = T),
            HANG_TIME = mean(HANG_TIME, na.rm = T)) %>% 
  filter(HR >= 5)

df_slim <- hr_grouped_by_player %>% 
  select("BATTER_KEY", "PLAYER_NAME", "SEASON_KEY", "EXIT_SPEED", "EXIT_ANGLE", "HR")


# First I will do simple exploration of the data to look at balls hit from '15-'18 that were NOT home runs and compare them to average 
# exit velocity, launch angle, and distance of home runs hit in 2019 to determine how many of these non-HR were hit above these levels.
# I am doing this just to get an idea of the data that I am working with.

non_HR <- readRDS("juiced_ball_HR_beneficiaries_non_HR_df.rds")

non_HR <- non_HR %>% 
  filter(HIT_DISTANCE > 250)

non_HR_df <- non_HR %>% 
  select("BATTER_KEY", "PLAYER_NAME", "SEASON_KEY", "VENUE_KEY", "HIT_LOCATION_X", "HIT_LOCATION_Y", "HIT_DISTANCE", "EXIT_SPEED", "EXIT_ANGLE",
         "EXIT_DISTANCE", "HANG_TIME","EXIT_SPIN_RATE","EXIT_BEARING") %>% 
  left_join(df_slim, by = c("BATTER_KEY", "PLAYER_NAME", "SEASON_KEY")) %>% 
  mutate(MAYBE_HR = ifelse(EXIT_SPEED.x >= EXIT_SPEED.y & EXIT_ANGLE.x >= EXIT_ANGLE.y, 1, 0)) %>% 
  group_by(BATTER_KEY, PLAYER_NAME, SEASON_KEY) %>% 
  summarise(ADDITIONAL_HR_2019_BALL = sum(MAYBE_HR))

potential_hr_performance <- df_slim %>% 
  left_join(non_HR_df, by = c("BATTER_KEY", "PLAYER_NAME", "SEASON_KEY")) %>% 
  select("BATTER_KEY", "PLAYER_NAME", "SEASON_KEY", "HR", "ADDITIONAL_HR_2019_BALL") %>% 
  mutate(TOTAL_HR = HR + ADDITIONAL_HR_2019_BALL)

#########################################################################################################

# The next step is to build a model to predict HR based on batted ball characteristics and train the model on 2019 data and test on 2015-2018 data

train_model <- readRDS("juiced_ball_HR_beneficiaries_train_model_df.rds")

train_model <- train_model %>% 
  select("EVENT_RESULT_KEY", "EXIT_SPEED", "EXIT_ANGLE", "EXIT_DIRECTION", "HANG_TIME","EXIT_SPIN_RATE","EXIT_BEARING")

# Build a Random forest model based on conditional inference trees 
class.task_RF <- makeClassifTask(data = train_model, target = "EVENT_RESULT_KEY")
lrn_RF <- makeLearner("classif.cforest", predict.type = "prob")
mod_RF <- train(lrn_RF, class.task_RF, NULL)
rdesc_RF <- makeResampleDesc("CV", iters = 2, predict = "both")
res_RF <- resample(learner = lrn_RF, task = class.task_RF, resampling = rdesc_RF, measures = list(acc, multiclass.au1u, logloss, kappa))
RF_pred <- res_RF$pred$data
RF_pred <- RF_pred %>% 
  filter(set == "test")

# Plot actual batted ball results vs predicted results
RF_truth <- RF_pred$truth
RF_response <- RF_pred$response
RF_bucket <- list(Actual = RF_truth, Predicted = RF_response)
ggplot(melt(RF_bucket), aes(x = value, fill = L1))+
  geom_histogram(position = "dodge", stat = "count")

# Test the model for Variance Importance
feature_importance <- getFeatureImportance(mod_RF)


# Use model to predict on 2015-2018 batted ball data to see how many additional home runs each hitter would have had from '15-'18, by year.

non_HR_pred <- predict(mod_RF$learner.model, newdata = non_HR[,mod_RF$features])

total_HR <- non_HR_pred %>% 
  mutate(CHANGED = ifelse(EVENT_RESULT_KEY == PREDICTED_2019_RESULT, 0, 1)) %>% 
  filter(CHANGED == 1,
         PREDICTED_2019_RESULT == 'home_run') %>% 
  group_by(BATTER_KEY, PLAYER_NAME, SEASON_KEY) %>% 
  summarise(ADDED_HR = sum(CHANGED)) %>% 
  left_join(hr_grouped_by_player, by = c("BATTER_KEY", "PLAYER_NAME", "SEASON_KEY")) %>% 
  select(BATTER_KEY, PLAYER_NAME, SEASON_KEY, HR, ADDED_HR) %>% 
  mutate(TOTAL_HR = HR + ADDED_HR)


##############################
#### TRIED MODEL FEATURES ####
##############################

# This is a list of model features that were tried before settling on V3, which produced the most accurate results. 

# V1
#train_model <- train_model %>% 
# select("EVENT_RESULT_KEY", "EXIT_SPEED", "EXIT_ANGLE", "EXIT_DIRECTION")

#V2
#train_model <- train_model %>% 
# select("EVENT_RESULT_KEY", "EXIT_SPEED", "EXIT_ANGLE", "EXIT_DIRECTION", "VENUE_KEY")

# V3
#train_model <- train_model %>% 
# select("EVENT_RESULT_KEY", "EXIT_SPEED", "EXIT_ANGLE", "EXIT_DIRECTION", "HANG_TIME","EXIT_SPIN_RATE","EXIT_BEARING")

