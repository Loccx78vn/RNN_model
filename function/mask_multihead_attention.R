# Function:
mask_self_attention <- nn_module(
  initialize = function(embed_dim, num_heads) {
    self$embed_dim <- embed_dim
    self$num_heads <- num_heads
    self$head_dim <- embed_dim / num_heads
    
    # Ensure that self$head_dim is a scalar
    if (self$head_dim %% 1 != 0) {
      stop("embed_dim must be divisible by num_heads")
    }
    
    # Linear layers for Q, K, V 
    self$query <- nn_linear(embed_dim, embed_dim, bias = FALSE)
    self$key <- nn_linear(embed_dim, embed_dim, bias = FALSE)
    self$value <- nn_linear(embed_dim, embed_dim, bias = FALSE)
    
    # Final linear layer after concatenating heads
    self$out <- nn_linear(embed_dim, embed_dim, bias = FALSE)
  },
  
  forward = function(x, mask = NULL) {
    batch_size <- x$size(1)
    seq_leng <- x$size(2)
    
    # Linear projections for Q, K, V
    Q <- self$query(x)  # (batch_size, seq_leng, embed_dim)
    K <- self$key(x)
    V <- self$value(x)
    
    # Reshape to separate heads: (batch_size, num_heads, seq_leng, head_dim)
    Q <- Q$view(c(batch_size, seq_leng, self$num_heads, self$head_dim))$transpose(2, 3)
    K <- K$view(c(batch_size, seq_leng, self$num_heads, self$head_dim))$transpose(2, 3)
    V <- V$view(c(batch_size, seq_leng, self$num_heads, self$head_dim))$transpose(2, 3)
    
    # Compute attention scores
    d_k <- self$head_dim
    attention_scores <- torch_matmul(Q, torch_transpose(K, -1, -2)) / sqrt(d_k)
    
    # Apply mask if provided
    mask <- torch_tril(torch_ones(c(seq_leng, seq_leng)))
    
    if (!is.null(mask)) {
      
      masked_attention_scores <- attention_scores$masked_fill(mask == 0, -Inf)
      
    } else {
      print("Warning: No mask provided")
    }
    
    # Compute attention weights
    weights <- nnf_softmax(masked_attention_scores, dim = -1)
    
    # Apply weights to V
    attn_output <- torch_matmul(weights, V)  # (batch_size, num_heads, seq_leng, head_dim)
    
    
    attn_output <- attn_output$transpose(2, 3)$contiguous()$view(c(batch_size, seq_leng, self$embed_dim))
    
    
    output <- self$out(attn_output)
    return(output)
  }
)

# Required parameters:
seq_length = 50
embed_dim = 32

# Example:
y_test <- matrix(runif(seq_length * embed_dim), nrow = seq_length, ncol = embed_dim)
y_test_tensor <- torch_tensor(y_test, dtype = torch_float())
y_test_tensor <- y_test_tensor$unsqueeze(2)

mask_self_layer <- mask_self_attention(embed_dim = embed_dim, num_heads = 1)


### Final output:
mask_output <- mask_self_layer(y_test_tensor)

mask_output$shape