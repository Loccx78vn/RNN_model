library(torch)
# Function:
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

# Required parameters:
batch_size <- 2
seq_len_enc <- 3
seq_len_dec <- 2
embed_dim <- 4
num_heads <- 2

# Example
encoder_output <- torch_randn(c(batch_size, seq_len_enc, embed_dim))
decoder_input <- torch_randn(c(batch_size, seq_len_dec, embed_dim))

cross_attention_layer <- cross_attention(embed_dim = embed_dim, num_heads = num_heads)

output <- cross_attention_layer(decoder_input, encoder_output)

### Final output:
print(output$shape)


