r_estimate_semipmetric_erf <- function(formula, family, data) {


  if (any(data$counter_weight < 0)){
    stop("Negative weights are not allowed.")
  }

  if (sum(data$counter_weight) == 0) {
    data$counter_weight <- data$counter_weight + 1
  }

  counter_weight <- data$counter_weight

  formula <- as.formula(formula)
  gam_model <- gam::gam(formula = formula,
                        family = family,
                        data = data,
                        weights = counter_weight)

  if (is.null(gam_model)) {
    stop("gnm model is null. Did not converge.")
  }

  fitted_values <- gam_model$fitted.values
  w_column <- gam_model$terms[[3]]
  
  if (is.null(fitted_values)) {
    vals <- data.frame()
  } else {
    vals <- data.frame(w_vals = data[[w_column]],
                       fitted_values = fitted_values)
    colnames(vals) <- c("w", "fitted_values")
  }

  return(vals)
}