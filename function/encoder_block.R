
### Encoder block:
encoder_layer <- nn_module(
  "TransformerEncoderLayer",
  
  initialize = function(d_model, num_heads, d_ff) {
    
    # Multi-Head Attention
    self$multihead_attention <- nn_multihead_attention(embed_dim = d_model, num_heads = num_heads)
    
    # Feedforward Network (Fully Connected)
    self$feed_forward <- nn_sequential(
      nn_linear(d_model, d_ff),
      nn_relu(),
      nn_linear(d_ff, d_model)
    )
    
    self$layer_norm <- nn_layer_norm(d_model)
    
  },
  
  forward = function(x) {
    
    attn_output <- self$multihead_attention(x, x, x) 
    x <- x + attn_output[[1]]
    x <- self$layer_norm(x) 
    
    # Feedforward network with residual connection
    ff_output <- self$feed_forward(x)
    x <- x + ff_output
    x <- self$layer_norm(x)
    
    return(x)
  }
)