def read_text_file(path):
    with open(path) as fp:
        text = fp.readlines()
    text = [item.strip() for item in text]
    return text


parallel_corpus = []
iit = {
    'english': read_text_file('../data/hindi_parallel_corpus/pruned_train.en'),
    'hindi': read_text_file('../data/hindi_parallel_corpus/pruned_train.hi')
}

if len(iit['english']) != len(iit['hindi']):
    print('error in IIT parallel data')

parallel_corpus.append(iit)

bible = {
    'english': read_text_file('../data/hindi_parallel_corpus/bible-uedin.en-hi.en'),
    'hindi': read_text_file('../data/hindi_parallel_corpus/bible-uedin.en-hi.hi')
}

if len(bible['english']) != len(bible['hindi']):
    print('error in Bible parallel data')
parallel_corpus.append(bible)

global_voices = {
    'english': read_text_file('../data/hindi_parallel_corpus/GlobalVoices.en-hi.en'),
    'hindi': read_text_file('../data/hindi_parallel_corpus/GlobalVoices.en-hi.hi')
}

if len(global_voices['english']) != len(global_voices['hindi']):
    print('error in Global Voices parallel data')
parallel_corpus.append(global_voices)

qed = {
    'english': read_text_file('../data/hindi_parallel_corpus/QED.en-hi.en'),
    'hindi': read_text_file('../data/hindi_parallel_corpus/QED.en-hi.hi')
}

if len(qed['english']) != len(qed['hindi']):
    print('error in QED parallel data')
parallel_corpus.append(qed)

ubuntu = {
    'english': read_text_file('../data/hindi_parallel_corpus/Ubuntu.en-hi.en'),
    'hindi': read_text_file('../data/hindi_parallel_corpus/Ubuntu.en-hi.hi')
}

if len(ubuntu['english']) != len(ubuntu['hindi']):
    print('error in Ubuntu parallel data')
parallel_corpus.append(ubuntu)

WMT = {
    'english': read_text_file('../data/hindi_parallel_corpus/WMT-News.en-hi.en'),
    'hindi': read_text_file('../data/hindi_parallel_corpus/WMT-News.en-hi.hi')
}

if len(WMT['english']) != len(WMT['hindi']):
    print('error in WMT News parallel data')
parallel_corpus.append(WMT)

with open('english_hindi_parallel_train.txt', 'a') as the_file:
    for corpus in parallel_corpus:
        corpus_len = len(corpus['english'])
        for i in range(corpus_len):
            english_item = corpus['english'][i]
            hindi_item = corpus['hindi'][i]
            the_file.write(english_item + '\t' + hindi_item + '\n')

dev_corpus = {
    'english': read_text_file('../data/hindi_parallel_corpus/dev.en'),
    'hindi': read_text_file('../data/hindi_parallel_corpus/dev.hi')
}

if len(dev_corpus['english']) != len(dev_corpus['hindi']):
    print('error in dev parallel data')

with open('english_hindi_parallel_dev.txt', 'a') as the_file:
    corpus_len = len(dev_corpus['english'])
    for i in range(corpus_len):
        english_item = dev_corpus['english'][i]
        hindi_item = dev_corpus['hindi'][i]
        the_file.write(english_item + '\t' + hindi_item + '\n')

test_corpus = {
    'english': read_text_file('../data/hindi_parallel_corpus/test.en'),
    'hindi': read_text_file('../data/hindi_parallel_corpus/test.hi')
}

if len(test_corpus['english']) != len(test_corpus['hindi']):
    print('error in test parallel data')

with open('english_hindi_parallel_test.txt', 'a') as the_file:
    corpus_len = len(test_corpus['english'])
    for i in range(corpus_len):
        english_item = test_corpus['english'][i]
        hindi_item = test_corpus['hindi'][i]
        the_file.write(english_item + '\t' + hindi_item + '\n')
