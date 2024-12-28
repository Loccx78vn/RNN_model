### Decoder Layer
decoder_layer <- nn_module(
  "TransformerDecoderLayer",
  
  initialize = function(d_model, num_heads, d_ff) {
    self$mask_self_attention <- mask_self_attention(embed_dim = d_model, num_heads = num_heads)
    self$cross_attention <- cross_attention(embed_dim = d_model, num_heads = num_heads)
    self$feed_forward <- nn_sequential(
      nn_linear(d_model, d_ff),
      nn_relu(),
      nn_linear(d_ff, d_model)
    )
    
    self$layer_norm <- nn_layer_norm(d_model)
  },
  
  forward = function(x, encoder_output) {
    # Masked Self-Attention
    mask_output <- self$mask_self_attention(x)
    x <- x + mask_output
    x <- self$layer_norm(x)
    
    # Encoder-Decoder Multi-Head Attention
    cross_output <- self$cross_attention(x, encoder_output)
    x <- x + cross_output
    x <- self$layer_norm(x)
    
    # Feedforward Network
    ff_output <- self$feed_forward(x)
    x <- x + ff_output
    x <- self$layer_norm(x)
    
    return(x)
  }
)
