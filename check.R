### Output of encoder:
library(torch)

sequence_length <- 27
n_lags <- 50
d_model <- 512
num_heads <- 8 

x_test <- torch_randn(c(sequence_length, n_lags, d_model))

encoder_layer <- nn_module(
  "TransformerEncoderLayer",
  
  initialize = function(d_model, num_heads, d_ff, dropout = 0.1) {
    self$multihead_attention <- nn_multihead_attention(embed_dim = d_model, num_heads = num_heads)
    self$feed_forward <- nn_sequential(
      nn_linear(d_model, d_ff),
      nn_relu(),
      nn_linear(d_ff, d_model)
    )
    self$dropout <- nn_dropout(dropout)
    self$layer_norm_1 <- nn_layer_norm(d_model)
    self$layer_norm_2 <- nn_layer_norm(d_model)
  },
  
  forward = function(x) {
    attn_output <- self$multihead_attention(x, x, x)
    x <- x + self$dropout(attn_output[[1]])
    x <- self$layer_norm_1(x)
    ff_output <- self$feed_forward(x)
    x <- x + self$dropout(ff_output)
    x <- self$layer_norm_2(x)
    return(x)
  }
)

model <- encoder_layer(d_model = d_model, 
                       num_heads = num_heads, 
                       d_ff = 2048, 
                       dropout = 0.1)

encoder_output <- model(x_test)

encoder_output$shape

### Output of mask multt-head attention:
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
batch_size = 1
seq_length = 27
embed_dim = 1

y_test <- matrix(runif(seq_length * embed_dim), nrow = seq_length, ncol = embed_dim)
y_test_tensor <- torch_tensor(y_test, dtype = torch_float())
y_test_tensor <- y_test_tensor$unsqueeze(2)

mask_self_layer <- mask_self_attention(embed_dim = embed_dim, num_heads = 1)

mask_output <- mask_self_layer(y_test_tensor)

mask_output$shape


### Cross attention:
library(torch)

cross_attention <- nn_module(
  initialize = function(embed_dim, num_heads) {
    self$embed_dim <- embed_dim
    self$num_heads <- num_heads
    self$head_dim <- embed_dim / num_heads
    
    if (self$head_dim %% 1 != 0) {
      stop("embed_dim must be divisible by num_heads")
    }
    
    self$query <- nn_linear(embed_dim, embed_dim, bias = FALSE)
    self$key <- nn_linear(embed_dim, embed_dim, bias = FALSE)
    self$value <- nn_linear(embed_dim, embed_dim, bias = FALSE)
    self$out <- nn_linear(embed_dim, embed_dim, bias = FALSE)
  },
  
  forward = function(decoder_input, encoder_output, mask = NULL) {
    batch_size <- decoder_input$size(1)
    seq_leng_dec <- decoder_input$size(2)
    seq_leng_enc <- encoder_output$size(2)
    
    decoder_input <- decoder_input$expand(c(batch_size, seq_leng_dec, self$embed_dim))
    Q <- self$query(decoder_input)
    K <- self$key(encoder_output)
    V <- self$value(encoder_output)
    
    Q <- Q$view(c(batch_size, seq_leng_dec, self$num_heads, self$head_dim))$transpose(2, 3)
    K <- K$view(c(batch_size, seq_leng_enc, self$num_heads, self$head_dim))$transpose(2, 3)
    V <- V$view(c(batch_size, seq_leng_enc, self$num_heads, self$head_dim))$transpose(2, 3)
    
    d_k <- self$head_dim
    attention_scores <- torch_matmul(Q, torch_transpose(K, -1, -2)) / sqrt(d_k)
    
    weights <- nnf_softmax(attention_scores, dim = -1)
    
    attn_output <- torch_matmul(weights, V)
    
    attn_output <- attn_output$transpose(2, 3)$contiguous()$view(c(batch_size, seq_leng_dec, self$embed_dim))
    
    output <- self$out(attn_output)
    return(output)
  }
)

cross_attention_layer <- cross_attention(embed_dim = 512, num_heads = 8)

output <- cross_attention_layer(encoder_output,mask_output)

output$shape

