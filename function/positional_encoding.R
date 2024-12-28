library(torch)
# Function:
positional_encoding <- function(seq_leng, d_model, n = 10000) {
  if (missing(seq_leng) || missing(d_model)) {
    stop("'seq_leng' and 'd_model' must be provided.")
  }
  
  if (d_model %% 2 != 0) {
    stop("'d_model' must be even.")
  }
  
  P <- matrix(0, nrow = seq_leng, ncol = d_model)  
  
  for (k in 1:seq_leng) {
    for (i in 0:(d_model / 2 - 1)) {
      denominator <- n^(2 * i / d_model)
      P[k, 2 * i + 1] <- sin(k / denominator)
      P[k, 2 * i + 2] <- cos(k / denominator)
    }
  }
  
  return(P)
}

# Required parameters:
batch_size <- 3
seq_len <- 5
d_model <- 2

# Example:
set.seed(123)

input_tensor <- torch_randn(batch_size, seq_len, d_model)

pos_enc <- positional_encoding(seq_len, d_model)

pos_enc_tensor <- torch_tensor(pos_enc)


### Final output:
output_tensor <- input_tensor + pos_enc_tensor

output_tensor$shape