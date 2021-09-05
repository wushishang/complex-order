class Static_Config(object):
    model = 'Transformer_PE_reduce'
    N = 1 #6 in Transformer Paper
    d_model = 256 #512 in Transformer Paper
    d_ff = 512 #2048 in Transformer Paper
    h = 8
    dropout = 0.1

    output_size = 6
    lr = 0.00001
    max_epochs = 100
    n_fold=10
    batch_size = 32
    max_sen_len = 53

    data = 'TREC_transformer'
