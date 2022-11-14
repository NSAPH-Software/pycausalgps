compute_density <- function(x0, x1){
    tmp_density <- stats::density(x0, na.rm = TRUE)
    dnst <- stats::approx(tmp_density$x, tmp_density$y, xout = x1,
                          rule = 2)$y
    return(dnst)
}