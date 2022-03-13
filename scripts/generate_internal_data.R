## code to prepare internal datasets go here
library(tidymodels)
library(workboots)

# ----------------------------penguins_preds------------------------------------

# load our dataset
data("penguins")
penguins <- penguins %>% drop_na()

# split data into testing & training sets
set.seed(123)
penguins_split <- initial_split(penguins)
penguins_test <- testing(penguins_split)
penguins_train <- training(penguins_split)

# create a workflow
penguins_wf <-
  workflow() %>%
  add_recipe(recipe(body_mass_g ~ ., data = penguins_train) %>% step_dummy(all_nominal())) %>%
  add_model(boost_tree("regression") %>% set_engine("xgboost"))

# generate predictions from 2000 bootstrap models
set.seed(345)
penguins_preds <-
  penguins_wf %>%
  predict_boots(
    n = 2000,
    training_data = penguins_train,
    new_data = penguins_test
  )

# ----------------------------ames_preds_boot-----------------------------------

# load and setup
data("ames")

ames_mod <-
  ames %>%
  select(First_Flr_SF, Sale_Price) %>%
  mutate(across(everything(), log10))

# split into train/test data
set.seed(918)
ames_split <- initial_split(ames_mod)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

# setup a workflow with a linear model
ames_wf <-
  workflow() %>%
  add_recipe(recipe(Sale_Price ~ First_Flr_SF, data = ames_train)) %>%
  add_model(linear_reg())

# generate bootstrap predictions on ames_test
set.seed(713)
ames_preds_boot <-
  ames_wf %>%
  predict_boots(
    n = 2000,
    training_data = ames_train,
    new_data = ames_test
  )

# ----------------------------------save----------------------------------------

ames_preds_boot %>% readr::write_rds("data/ames_preds_boot.rds")
penguins_preds %>% readr::write_rds("data/penguins_preds.rds")
