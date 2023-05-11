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

  vals <- as.data.frame(gam_model$fitted.values)
  colnames(vals) <- c("fitted values")

  return(vals)
}