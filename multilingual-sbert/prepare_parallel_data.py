import random
from os import listdir
from os.path import isfile, join, isdir

random_seed = 72
random.seed(random_seed)


def read_text_file(path):
    with open(path) as fp:
        text = fp.readlines()
    text = [item.strip() for item in text]
    return text


def get_files_in_directory(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def get_directories_in_directory(mypath):
    return [f for f in listdir(mypath) if isdir(join(mypath, f))]


def join_parallel_text(input_paths, first_language, second_language, output_path):
    with open(output_path, 'a') as the_file:
        for path in input_paths:
            corpus = {
                first_language: read_text_file(path[first_language]),
                second_language: read_text_file(path[second_language])
            }

            if len(corpus[first_language]) != len(corpus[second_language]):
                print('error in parallel data: unequal number of rows')
                return None

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


def create_english_portuguese_parallel_corpus():
    first_language = 'english'
    second_language = 'portuguese'
    portuguese_parallel_corpus_path = '../data/portuguese_parallel_corpus/'

    train_file_names = get_files_in_directory(portuguese_parallel_corpus_path)
    train_file_names = sorted(train_file_names)
    train_input_paths = []
    for i in range(len(train_file_names) // 2):
        train_input_paths.append({
            first_language: portuguese_parallel_corpus_path + train_file_names[2 * i],
            second_language: portuguese_parallel_corpus_path + train_file_names[2 * i + 1]
        })

    join_parallel_text(train_input_paths, first_language, second_language, 'english_portuguese_parallel_train.txt')


def create_english_bengali_parallel_corpus():
    first_language = 'english'
    second_language = 'bengali'
    portuguese_parallel_corpus_path = '../data/english_bengali_parallel_data/'

    train_file_names = get_files_in_directory(portuguese_parallel_corpus_path)
    train_file_names = sorted(train_file_names)
    train_input_paths = []
    for i in range(len(train_file_names) // 2):
        train_input_paths.append({
            first_language: portuguese_parallel_corpus_path + train_file_names[2 * i + 1],
            second_language: portuguese_parallel_corpus_path + train_file_names[2 * i]
        })

    join_parallel_text(train_input_paths, first_language, second_language, 'english_bengali_parallel_train.txt')


def determine_filename_language(filenames):
    english = filenames[0] if filenames[0].endswith('en') else filenames[1]
    non_english = filenames[0] if filenames[0] != english else filenames[1]

    return english, non_english


def create_southeast_asian_parallel_corpus():
    southeast_asian_data_directory = '../data/Southeast Asian Parallel Corpus/'
    dataset_directories = get_directories_in_directory(southeast_asian_data_directory)
    output_path = 'southeast_asian_parallel_corpus.txt'

    for directory in dataset_directories:
        directory_full_path = southeast_asian_data_directory + directory + '/'
        dataset_filepaths = get_files_in_directory(directory_full_path)
        dataset_filepaths = sorted(dataset_filepaths)

        parallel_sentences = []
        with open(output_path, 'a') as the_file:
            reference_index = {}
            for j in range(len(dataset_filepaths) // 2):
                english_file_path, non_english_filepath = determine_filename_language(
                    dataset_filepaths[2 * j: 2 * (j + 1)])
                english_half = read_text_file(directory_full_path + english_file_path)
                non_english_half = read_text_file(directory_full_path + non_english_filepath)
                for i, english_sentence in enumerate(english_half):
                    if english_sentence in reference_index:
                        parallel_sentences[reference_index[english_sentence]].append(non_english_half[i])
                    else:
                        reference_index[english_sentence] = len(parallel_sentences)
                        parallel_sentences.append([english_sentence, non_english_half[i]])

            for texts in parallel_sentences:
                line = '\t'.join(texts) + '\n'
                the_file.write(line)

    all_data = read_text_file(output_path)
    indices = range(len(all_data))
    test_indices = set(random.sample(indices, len(indices)//1000))
    train_data = [line for i, line in enumerate(all_data) if i not in test_indices]
    test_data = [line for i, line in enumerate(all_data) if i in test_indices]
    with open('train_'+output_path, "w") as train_file:
        for line in train_data:
            train_file.write(line+'\n')
    with open('test_'+output_path, "w") as test_file:
        for line in test_data:
            test_file.write(line+'\n')


if __name__ == "__main__":
    # create_english_hindi_parallel_corpus()
    # create_english_portuguese_parallel_corpus()
    # create_english_bengali_parallel_corpus()
    create_southeast_asian_parallel_corpus()
