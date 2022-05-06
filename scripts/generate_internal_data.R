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
data("biomass")
biomass <-
  biomass %>%
  as_tibble() %>%
  select(carbon, HHV)

# split into train/test data
bio_split <- initial_split(biomass)
bio_train <- training(bio_split)
bio_test <- testing(bio_split)

# setup workflow with a linear model
bio_wf <-
  workflow() %>%
  add_recipe(recipe(HHV ~ carbon, data = bio_train)) %>%
  add_model(linear_reg())

# generate bootstrap predictions on the test set
set.seed(713)
bio_pred_int <-
  bio_wf %>%
  predict_boots(
    n = 2000,
    training_data = bio_train,
    new_data = bio_test,
    verbose = TRUE
  )

# generate confidence interval preds on test set
set.seed(867)
bio_conf_int <-
  bio_wf %>%
  predict_boots(
    n = 2000,
    training_data = bio_train,
    new_data = bio_test,
    interval = "confidence",
    verbose = TRUE
  )

# save
bio_train %>% readr::write_csv("data/bio_train.csv")
bio_test %>% readr::write_csv("data/bio_test.csv")
bio_pred_int %>% readr::write_rds("data/bio_pred_int.rds")
bio_conf_int %>% readr::write_rds("data/bio_conf_int.rds")

