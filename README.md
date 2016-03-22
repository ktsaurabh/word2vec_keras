# word2vec_keras

This Example is adapted from Keras Example (examples/skipgram_word_embeddings.py)
This program depends upon keras and theano. So make sure to install these with appropriate path provided.  

GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cbow_word_embeddings.py
        
CPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=cpu python cbow_word_embeddings.py