
r_locpol <- function(data, formula, bw, w_vals) {
  formula <- as.formula(formula)
  tmp_loc <- locpol::locpol(formula = formula,
                            data = data,
                            bw = bw,
                            weig = data$counter_weight,
                            xeval = w_vals,
                            kernel = locpol::gaussK)

  erf <- tmp_loc$lpFit$m_Y
  return(erf)
}