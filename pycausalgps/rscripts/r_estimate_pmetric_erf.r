r_estimate_pmetric_erf <- function(formula, family, data) {

  if (any(data$counter_weight < 0)){
    stop("Negative weights are not allowed.")
  }

  if (sum(data$counter_weight) == 0) {
    data$counter_weight <- data$counter_weight + 1
  }

  counter_weight <- data$counter_weight

  formula <- as.formula(formula)
  gnm_model <- gnm::gnm(formula = formula,
                        family = family,
                        data = data,
                        weights = counter_weight)

  if (is.null(gnm_model)) {
    stop("gnm model is null. Did not converge.")
  }

  fitted_values <- gnm_model$fitted.values

  # get the w values based on the formula's second term
  w_column <- gnm_model$terms[[3]]

  if (is.null(fitted_values)) {
    vals <- data.frame()
  } else {
    vals <- data.frame(w_vals = data[[w_column]],
                       fitted_values = fitted_values)
    colnames(vals) <- c("w", "fitted_values")
  }

  return(vals)
}