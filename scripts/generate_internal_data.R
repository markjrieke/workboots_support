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

# -----------------------------Getting-Started----------------------------------

# setup data
data("car_prices")

# apply global transformations
car_prices <-
  car_prices %>%
  mutate(Price = log10(Price),
         Cylinder = as.character(Cylinder),
         Doors = as.character(Doors))

# split into testing and training
set.seed(999)
car_split <- initial_split(car_prices)
car_train <- training(car_split)
car_test <- testing(car_split)

# re-setup recipe with training dataset
car_rec <-
  recipe(Price ~ ., data = car_train) %>%
  step_BoxCox(Mileage) %>%
  step_dummy(all_nominal())

# setup model spec
car_spec <-
  boost_tree(
    mode = "regression",
    engine = "xgboost",
    mtry = tune(),
    trees = tune()
  )

# combine into workflow
car_wf <-
  workflow() %>%
  add_recipe(car_rec) %>%
  add_model(car_spec)

# setup cross-validation folds
set.seed(666)
car_folds <-vfold_cv(car_train)

# tune model
set.seed(555)
car_tune <-
  tune_grid(
    car_wf,
    car_folds,
    grid = 5
  )

# finalize workflow
car_wf_final <-
  car_wf %>%
  finalize_workflow(car_tune %>% select_best("rmse"))

# prediction interval
set.seed(444)
car_preds <-
  car_wf_final %>%
  predict_boots(
    n = 2000,
    training_data = car_train,
    new_data = car_test,
    verbose = TRUE
  )

# variable importances
set.seed(333)
car_importance <-
  car_wf_final %>%
  vi_boots(
    n = 2000,
    training_data = car_train,
    verbose = TRUE
  )

# save
car_train %>% readr::write_csv("data/car_train.csv")
car_test %>% readr::write_csv("data/car_test.csv")
car_tune %>% readr::write_rds("data/car_tune.rds")
car_preds %>% readr::write_rds("data/car_preds.rds")
car_importance %>% readr::write_rds("data/car_importance.rds")
