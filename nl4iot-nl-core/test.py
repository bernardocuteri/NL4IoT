import tensorflow as tf
import commons
import re

intents = []


class IntentClassificationTester:
    @staticmethod
    def read_test_file(test_filename):
        tests = list()
        labels = list()
        file = open(test_filename, "r")
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            split = line.split("\t")
            tests.append(split[0])
            labels.append(split[1])
        file.close()
        return tests, labels


class NerTester:
    @staticmethod
    def read_test_file(test_filename):
        tests = list()
        labels = list()
        file = open(test_filename, "r")
        lines = file.readlines()
        tags, words = [], []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                # changing sentence
                if len(words) != 0:
                    tests.append(words)
                    labels.append(tags)
                    tags, words = [], []
            else:
                split = re.split('[\t| ]+', line)
                # @todo fix for tokens containint spaces
                if len(split) > 1:
                    words.append(split[0])
                    tags.append(split[1])
        file.close()
        return tests, labels





def test(configuration):
    batch_size = configuration.batch_size
    iterations = configuration.iterations
    embeddings_size = configuration.embeddings_size
    max_seq_length = configuration.max_seq_length
    training_file = configuration.training_file
    model_out_folder = configuration.model_out_folder
    test_file = configuration.test_file
    task = configuration.task

    model = commons.get_embeddings_model(configuration)

    if task == 'intent_classification':
        tester = IntentClassificationTester()
    elif task == 'ner':
        tester = NerTester()
    else:
        raise ValueError('Invalid task type ' + task)

    recover_labels_from_training_file(training_file)
    test_sentences, labels = tester.read_test_file(test_file)
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_out_folder+'/pretrained_lstm.ckpt-'+str(iterations-1)+'.meta')
    saver.restore(sess, tf.train.latest_checkpoint(model_out_folder))
    graph = tf.get_default_graph()

    labels_prediction = graph.get_tensor_by_name("labels_prediction:0")
    input_data = graph.get_tensor_by_name("input_data:0")
    sequences_lengths = graph.get_tensor_by_name("sequences_lengths:0")

    total_tests = 0
    failed = 0
    # ner only
    total_sentences = 0
    failed_sentences = 0

    for i in range(0, len(test_sentences) // batch_size):
        test_batch_sentences = test_sentences[i * batch_size:(i + 1) * batch_size]
        test_batch_labels = labels[i * batch_size:(i + 1) * batch_size]

        test_batch_lenghts = list()
        test_batch_embeddings = list()
        test_batch_effective_lengths = list()
        for test_sentence in test_batch_sentences:
            embeddings = None
            word_counter = None
            if task == 'intent_classification':
                embeddings, word_counter = commons.sentence2embeddings(test_sentence, max_seq_length, model,
                                                                       embeddings_size)
            elif task == 'ner':
                embeddings, word_counter = commons.split_sentece2embeddings(test_sentence, max_seq_length, model,
                                                                            embeddings_size)
            test_batch_embeddings.append(embeddings)
            test_batch_effective_lengths.append(word_counter)
            # @TODO check why effective lentgh does not work
            test_batch_lenghts.append(max_seq_length)
        evaluation = sess.run(labels_prediction, {input_data: test_batch_embeddings,
                                                      sequences_lengths: test_batch_lenghts})

        if task == 'intent_classification':
            counter = 0
            for result in evaluation:
                # print(result)
                if test_batch_labels[counter] != intents[result]:
                    failed += 1
                    print(test_batch_sentences[counter])
                    print(test_batch_labels[counter],
                          intents[result])
                total_tests += 1
                counter += 1
        elif task == 'ner':
            sentence_counter = 0
            for result in evaluation:
                failed_sentence = False
                for j in range(0, test_batch_effective_lengths[sentence_counter]):
                    # print(sentence_counter, j, result[j], len(test_batch_labels), len(intents))
                    if test_batch_labels[sentence_counter][j] != intents[result[j]]:
                        failed += 1
                        failed_sentence = True
                        print(test_batch_labels[sentence_counter][j], intents[result[j]])
                    total_tests += 1
                if failed_sentence:
                    print(test_batch_sentences[sentence_counter])
                    for j in range(0, test_batch_effective_lengths[sentence_counter]):
                        print(test_batch_labels[sentence_counter][j], intents[result[j]])
                    failed_sentences += 1
                sentence_counter += 1
                total_sentences += 1

    if task == 'ner':
        print("sentece accuracy", 1 - failed_sentences / total_sentences)
    print(total_tests, failed)
    print("accuracy", 1 - failed / total_tests)


def add_example_with_string_label(label):

    global intents
    if label not in intents:
        intents.append(label)


def recover_labels_from_training_file(train_filename):
    file = open(train_filename, "r")
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        split = re.split('[\t| ]+', line)
        if len(split) > 1:
            add_example_with_string_label(split[len(split)-1])
    file.close()
