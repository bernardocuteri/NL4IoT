class ItalianConfiguration:
    batch_size = 8
    hidden_layer_size = 300
    n_labels = 23
    iterations = 1000
    embeddings_size = 300
    max_seq_length = 20
    embeddings_model_file = 'word2vec_models/italian/fastText/wiki.it.bin'
    embedding_engine = 'fastText'
    # embeddings_model_file = 'word2vec_models/italian/glove_WIKI'
    # embedding_engine = 'Word2Vec'
    training_file = 'chqa_datasets/train.data'
    test_file = 'chqa_datasets/test.data'
    model_out_folder = 'models/italian'
    cells_type = 'lstm'
    task = 'intent_classification'


class ItalianConfigurationNer:
    batch_size = 8
    hidden_layer_size = 300
    n_labels = 23
    iterations = 1000
    embeddings_size = 300
    max_seq_length = 20
    embeddings_model_file = 'word2vec_models/italian/fastText/wiki.it.bin'
    embedding_engine = 'fastText'
    # embeddings_model_file = 'word2vec_models/italian/glove_WIKI'
    # embedding_engine = 'Word2Vec'
    training_file = 'chqa_datasets/ner/train.data'
    test_file = 'chqa_datasets/ner/test.data'
    model_out_folder = 'models/ner/italian'
    cells_type = 'lstm'
    task = 'ner'
    loss = 'log'


class EnglishConfiguration:
    batch_size = 64
    hidden_layer_size = 300
    n_labels = 7
    iterations = 4000
    embeddings_size = 300
    max_seq_length = 40
    embeddings_model_file = 'word2vec_models/english/fastText/wiki.en.bin'
    embedding_engine = 'fastText'
    # embeddings_model_file = 'word2vec_models/english/glove.6B.300d.w2v.txt'
    # embedding_engine = 'KeyedVectors'
    training_file = 'snipsco_english_data/train.data'
    test_file = 'snipsco_english_data/test.data'
    model_out_folder = 'models/english'
    cells_type = 'gru'
    task = 'intent_classification'



class EnglishConfigurationNer:
    batch_size = 8
    hidden_layer_size = 128
    n_labels = 72
    iterations = 64000
    embeddings_size = 300
    max_seq_length = 40
    embeddings_model_file = 'word2vec_models/english/fastText/wiki.en.bin'
    embedding_engine = 'fastText'
    # embeddings_model_file = 'word2vec_models/english/glove.6B.300d.w2v.txt'
    # embedding_engine = 'KeyedVectors'
    #training_file = 'english_ner_datasets/kaggle-entity-annotated-corpus/train.data'
    #test_file = 'english_ner_datasets/kaggle-entity-annotated-corpus/test.data'
    training_file = 'data/snips/train/ner'
    test_file = 'data/snips/test/ner'
    model_out_folder = 'models/ner/english'
    cells_type = 'lstm'
    task = 'ner'
    loss = 'crf'

class EnglishConfigurationNerCrf:
    batch_size = 64
    hidden_layer_size = 300
    n_labels = 17
    iterations = 2000
    embeddings_size = 300
    max_seq_length = 120
    embeddings_model_file = '../word2vec_models/english/fastText/wiki.en.bin'
    embedding_engine = 'fastText'
    # embeddings_model_file = 'word2vec_models/english/glove.6B.300d.w2v.txt'
    # embedding_engine = 'KeyedVectors'
    training_file = '../english_ner_datasets/kaggle-entity-annotated-corpus/train.data'
    test_file = '../english_ner_datasets/kaggle-entity-annotated-corpus/test.data'
    model_out_folder = '../models/ner/english'
    cells_type = 'lstm'
    task = 'ner'
    loss = 'crf'
