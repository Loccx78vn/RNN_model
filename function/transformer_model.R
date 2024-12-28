### Final transformer model: 
transformer <- nn_module(
  "Transformer",
  
  initialize = function(d_model, seq_leng, num_heads, d_ff, num_encoder_layers, num_decoder_layers) {
    self$d_model <- d_model
    self$num_heads <- num_heads
    self$d_ff <- d_ff
    self$num_encoder_layers <- num_encoder_layers
    self$num_decoder_layers <- num_decoder_layers
    self$seq_leng <- seq_leng
    self$en_pe <- en_pe
    self$de_pe <- de_pe
    
    # Encoder layers
    self$encoder_layers <- nn_module_list(
      lapply(1:num_encoder_layers, function(i) {
        encoder_layer(d_model, num_heads, d_ff)
      })
    )
    
    # Decoder layers
    self$decoder_layers <- nn_module_list(
      lapply(1:num_decoder_layers, function(i) {
        decoder_layer(d_model, num_heads, d_ff)
      })
    )
    
    # Final output layer
    self$output_layer <- nn_linear(d_model, 1)  # Output layer to predict a single value
    
  },
  
  forward = function(src, trg) {
    
    src <- src + self$en_pe  
    trg <- trg + self$de_pe
    
    # Encoder forward pass
    encoder_output <- src
    for (i in 1:self$num_encoder_layers) {
      encoder_output <- self$encoder_layers[[i]](encoder_output)
    }
    
    # Decoder forward pass
    decoder_output <- trg
    for (i in 1:self$num_decoder_layers) {
      decoder_output <- self$decoder_layers[[i]](decoder_output, encoder_output)
    }
    
    # Apply final output layer
    output <- self$output_layer(decoder_output)
    
    return(output)
  }
)