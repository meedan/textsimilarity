def read_text_file(path):
    with open(path) as fp:
        text = fp.readlines()
    text = [item.strip() for item in text]
    return text


def join_parallel_text(input_paths, first_language, second_language, output_path):
    parallel_corpus = []

    for path in input_paths:
        corpus = {
            first_language: read_text_file(path[first_language]),
            second_language: read_text_file(path[second_language])
        }

        if len(corpus[first_language]) != len(corpus[second_language]):
            print('error in parallel data: unequal number of rows')
            return None

        parallel_corpus.append(corpus)

    with open(output_path, 'a') as the_file:
        for corpus in parallel_corpus:
            corpus_len = len(corpus[first_language])
            for i in range(corpus_len):
                first_item = corpus[first_language][i]
                second_item = corpus[second_language][i]
                the_file.write(first_item + '\t' + second_item + '\n')


def create_english_hindi_parallel_corpus():
    first_language = 'english'
    second_language = 'hindi'
    train_input_paths = [
        {
            first_language: '../data/hindi_parallel_corpus/pruned_train.en',
            second_language: '../data/hindi_parallel_corpus/pruned_train.hi'
        },

        {
            first_language: '../data/hindi_parallel_corpus/bible-uedin.en-hi.en',
            second_language: '../data/hindi_parallel_corpus/bible-uedin.en-hi.hi'
        },

        {
            first_language: '../data/hindi_parallel_corpus/GlobalVoices.en-hi.en',
            second_language: '../data/hindi_parallel_corpus/GlobalVoices.en-hi.hi'
        },

        {
            first_language: '../data/hindi_parallel_corpus/QED.en-hi.en',
            second_language: '../data/hindi_parallel_corpus/QED.en-hi.hi'
        },

        {
            first_language: '../data/hindi_parallel_corpus/Ubuntu.en-hi.en',
            second_language: '../data/hindi_parallel_corpus/Ubuntu.en-hi.hi'
        },

        {
            first_language: '../data/hindi_parallel_corpus/WMT-News.en-hi.en',
            second_language: '../data/hindi_parallel_corpus/WMT-News.en-hi.hi'
        }
    ]

    join_parallel_text(train_input_paths, first_language, second_language, 'english_hindi_parallel_train.txt')

    dev_input_paths = [
        {
            first_language: '../data/hindi_parallel_corpus/dev.en',
            second_language: '../data/hindi_parallel_corpus/dev.hi'
        }
    ]

    join_parallel_text(dev_input_paths, first_language, second_language, 'english_hindi_parallel_dev.txt')

    test_input_paths = [
        {
            first_language: '../data/hindi_parallel_corpus/test.en',
            second_language: '../data/hindi_parallel_corpus/test.hi'
        }
    ]

    join_parallel_text(test_input_paths, first_language, second_language, 'english_hindi_parallel_test.txt')
