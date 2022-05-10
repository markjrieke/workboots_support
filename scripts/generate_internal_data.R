## code to prepare internal datasets go here
library(tidymodels)
library(workboots)

# ----------------------------------README--------------------------------------

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
  add_model(boost_tree("regression"))

# generate predictions for prediction interval summary
set.seed(345)
penguins_pred_int <-
  penguins_wf %>%
  predict_boots(
    n = 2000,
    training_data = penguins_train,
    new_data = penguins_test,
    verbose = TRUE
  )

# generate predictions for confidence interval summary
set.seed(456)
penguins_conf_int <-
  penguins_wf %>%
  predict_boots(
    n = 2000,
    training_data = penguins_train,
    new_data = penguins_test,
    interval = "confidence",
    verbose = TRUE
  )

penguins_train %>% readr::write_csv("data/penguins_train.csv")
penguins_test %>% readr::write_csv("data/penguins_test.csv")
penguins_pred_int %>% readr::write_rds("data/penguins_pred_int.rds")
penguins_conf_int %>% readr::write_rds("data/penguins_conf_int.rds")

# ------------------------Estimating-Linear-Intervals---------------------------

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

# generate bootstrap prediction intervals on ames test
set.seed(713)
ames_boot_pred_int <-
  ames_wf %>%
  predict_boots(
    n = 2000,
    training_data = ames_train,
    new_data = ames_test,
    verbose = TRUE
  )

# generate bootstrap confidence intervals on ames test
set.seed(867)
ames_boot_conf_int <-
  ames_wf %>%
  predict_boots(
    n = 2000,
    training_data = ames_train,
    new_data = ames_test,
    interval = "confidence",
    verbose = TRUE
  )

# save
ames_train %>% readr::write_csv("data/ames_train.csv")
ames_test %>% readr::write_csv("data/ames_test.csv")
ames_boot_pred_int %>% readr::write_rds("data/ames_boot_pred_int.rds")
ames_boot_conf_int %>% readr::write_rds("data/ames_boot_conf_int.rds")
