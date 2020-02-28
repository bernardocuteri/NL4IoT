import numpy as np
import tensorflow as tf
import datetime
import random
import commons
import re


class IntentClassificationTrainer:
    labels_set = []
    training_embeddings = []
    sentences_lengths = []
    one_hot_training_labels = []

    def get_train_batch(self, batch_size, max_seq_length):
        embeddings = []
        training_labels = []
        batch_lengths = []
        for i in range(batch_size):
            random_index = random.randint(0, len(self.training_embeddings) - 1)
            embeddings.append(self.training_embeddings[random_index])
            training_labels.append(self.one_hot_training_labels[random_index])
            # @TODO check why effective lengths does not work
            batch_lengths.append(max_seq_length)
        return embeddings, training_labels, batch_lengths

    def add_example_with_string_label(self, sentence, label, model, max_seq_length, embeddings_size, n_labels):
        if label not in self.labels_set:
            self.labels_set.append(label)
        embedding, length = commons.sentence2embeddings(sentence, max_seq_length, model, embeddings_size)
        self.training_embeddings.append(embedding)
        self.sentences_lengths.append(length)
        one_hot_label = np.zeros(n_labels)
        label_index = self.labels_set.index(label)
        one_hot_label[label_index] = 1
        self.one_hot_training_labels.append(one_hot_label)

    def build_training_set_from_file(self, train_filename, model, max_seq_length, embeddings_size, n_labels):
        file = open(train_filename, "r")
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            split = line.split("\t")
            self.add_example_with_string_label(split[0], split[1], model, max_seq_length, embeddings_size, n_labels)

    @staticmethod
    def build_net(hidden_layer_size, cells_type, input_data, n_labels, labels,
                  sequences_lengths, _, __):
        stacked_cells = tf.contrib.rnn.MultiRNNCell(
            [get_cell(hidden_layer_size, cells_type) for _ in range(2)])
            
        #[tf.contrib.rnn.DropoutWrapper(cell=get_cell(hidden_layer_size, cells_type), output_keep_prob=0.75) for _ in range(2)])

        value, _ = tf.nn.dynamic_rnn(stacked_cells, input_data, dtype=tf.float32, sequence_length=sequences_lengths)

        weight = tf.Variable(tf.truncated_normal([hidden_layer_size, n_labels]))
        bias = tf.Variable(tf.constant(0.1, shape=[n_labels]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = tf.add(tf.matmul(last, weight), bias, name="prediction")
        labels_pred = tf.argmax(prediction, 1, name="labels_prediction")
        correct_pred = tf.equal(labels_pred, tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))
        return loss, accuracy

    @staticmethod
    def get_labels_placeholder(batch_size, n_labels, _):
        return tf.placeholder(tf.float32, [batch_size, n_labels])


class NerTrainer:
    training_ner_labels = []
    labels_set = []
    training_embeddings = []
    sentences_lengths = []

    def get_train_batch(self, batch_size, _):
        embeddings = []
        batch_training_labels_ids = []
        batch_lengths = []
        for i in range(batch_size):
            random_index = random.randint(0, len(self.training_embeddings) - 1)
            embeddings.append(self.training_embeddings[random_index])
            batch_training_labels_ids.append(self.training_ner_labels[random_index])
            # @TODO check why effective lengths does not work
            # batch_lengths.append(max_seq_length)
            batch_lengths.append(self.sentences_lengths[random_index])
        return embeddings, batch_training_labels_ids, batch_lengths

    def add_ner_sentence(self, words, labels, model, max_seq_length, embeddings_size, _):
        embeddings = np.zeros([max_seq_length, embeddings_size])
        labels_vector = np.zeros(max_seq_length)
        for i in range(0, len(words)):
            word = words[i]
            label = labels[i]
            if label not in self.labels_set:
                self.labels_set.append(label)
            embeddings[i] = commons.get_word_embedding(word, model)
            labels_vector[i] = self.labels_set.index(label)
        self.training_embeddings.append(embeddings)
        self.training_ner_labels.append(labels_vector)
        self.sentences_lengths.append(len(words))

    def build_training_set_from_file(self, train_filename, model, max_seq_length, embeddings_size, n_labels):
        file = open(train_filename, "r")
        lines = file.readlines()
        words, tags = [], []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                # changing sentence
                if len(words) != 0:
                    self.add_ner_sentence(words, tags, model, max_seq_length, embeddings_size, n_labels)
                words, tags = [], []

            else:
                # @todo fix for tokens containint spaces
                split = re.split('[\t| ]+', line)
                if len(split) > 1:
                    words.append(split[0])
                    tags.append(split[1])
        file.close()

    @staticmethod
    def build_net(hidden_layer_size, cells_type, input_data, n_labels, labels,
                  sequences_lengths, batch_size, max_seq_length):
        fw_stacked_cells = tf.contrib.rnn.MultiRNNCell(
            [get_cell(hidden_layer_size, cells_type) for _ in range(2)])

        bw_stacked_cells = tf.contrib.rnn.MultiRNNCell(
            [get_cell(hidden_layer_size, cells_type) for _ in range(2)])

        # fw_stacked_cells = tf.contrib.rnn.LSTMCell(hidden_layer_size)
        # bw_stacked_cells = tf.contrib.rnn.LSTMCell(hidden_layer_size)

        (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(fw_stacked_cells, bw_stacked_cells, input_data,
                                                              dtype=tf.float32)

        value = tf.concat([out_fw, out_bw], axis=2)

        weight = tf.Variable(tf.truncated_normal([hidden_layer_size * 2, n_labels]))

        bias = tf.Variable(tf.constant(0.1, shape=[n_labels]))

        nsteps = tf.shape(value)[1]
        value = tf.reshape(value, [-1, 2 * hidden_layer_size])
        prediction = tf.add(tf.matmul(value, weight), bias, name="prediction")
        logits = tf.reshape(prediction, [-1, nsteps, n_labels])
        labels_pred = tf.reshape(tf.cast(tf.argmax(prediction, 1), tf.int32), [batch_size, max_seq_length],
                                 name="labels_prediction")
        correct_pred = tf.equal(labels_pred, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.sequence_mask(sequences_lengths)
        losses = tf.boolean_mask(losses, mask)
        loss = tf.reduce_mean(losses)

        # log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(logits, labels, sequences_lengths)
        # trans_params = trans_params  # need to evaluate it for decoding
        # loss = tf.reduce_mean(-log_likelihood)
        return loss, accuracy

    @staticmethod
    def get_labels_placeholder(batch_size, _, max_seq_length):
        return tf.placeholder(tf.int32, [batch_size, max_seq_length])


def train(configuration):
    batch_size = configuration.batch_size
    hidden_layer_size = configuration.hidden_layer_size
    n_labels = configuration.n_labels
    iterations = configuration.iterations
    embeddings_size = configuration.embeddings_size
    max_seq_length = configuration.max_seq_length
    training_file = configuration.training_file
    model_out_folder = configuration.model_out_folder
    cells_type = configuration.cells_type
    task = configuration.task

    model = commons.get_embeddings_model(configuration)

    if task == 'intent_classification':
        trainer = IntentClassificationTrainer()
    elif task == 'ner':
        trainer = NerTrainer()
    else:
        raise ValueError('Invalid task type ' + task)

    print('Start building training set')

    trainer.build_training_set_from_file(training_file, model, max_seq_length, embeddings_size, n_labels)

    print('Finished building training set')

    tf.reset_default_graph()

    labels = trainer.get_labels_placeholder(batch_size, n_labels, max_seq_length)

    input_data = tf.placeholder(tf.float32, [batch_size, max_seq_length, embeddings_size], name="input_data")

    sequences_lengths = tf.placeholder(tf.int32, [batch_size], name="sequences_lengths")

    # data = tf.Variable(tf.zeros([batchSize, maxSeqLength, embeddings_size]),dtype=tf.float32)
    # data = tf.nn.embedding_lookup(wordVectors,input_data)

    # lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    
    loss, accuracy = trainer.build_net(hidden_layer_size, cells_type, input_data, n_labels, labels,
                                       sequences_lengths, batch_size, max_seq_length)

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.InteractiveSession()

    tf.summary.scalar('Loss', loss)
    if accuracy is not None:
        tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    log_dir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        # Next Batch
        next_batch, next_batch_labels, next_batch_lengths = trainer.get_train_batch(batch_size, max_seq_length)
        # print("labels")
        # print(next_batch_labels)
        # print("embeddings")
        # print(next_batch)
        sess.run(optimizer, {input_data: next_batch, labels: next_batch_labels, sequences_lengths: next_batch_lengths})

        # Write summary to Tensorboard
        if i % 50 == 0:
            print(i)
            summary = sess.run(merged, {input_data: next_batch, labels: next_batch_labels,
                                        sequences_lengths: next_batch_lengths})
            writer.add_summary(summary, i)

        # Save the network every 10,000 training iterations
        if i == iterations - 1:
            save_path = saver.save(sess, model_out_folder + '/pretrained_lstm.ckpt', global_step=i)
            print("saved to %s" % save_path)
    writer.close()
    print(trainer.labels_set)


def get_cell(hidden_layer_size, cells_type):
    if cells_type == 'gru':
        return tf.contrib.rnn.GRUCell(hidden_layer_size)
    elif cells_type == 'lstm':
        return tf.contrib.rnn.BasicLSTMCell(hidden_layer_size)
    else:
        raise ValueError('Invalid cells type ' + cells_type)
