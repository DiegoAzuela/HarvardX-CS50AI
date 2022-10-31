"""
SOURCE
    https://cs50.harvard.edu/ai/2020/projects/6/questions/

SOLVED BY
    Diego Arnoldo Azuela Rosas

BACKGROUND
Question Answering (QA) is a field within natural language processing focused on designing systems that can answer questions. Among the more famous question answering systems is Watson, the IBM computer that competed (and won) on Jeopardy!. A question answering system of Watson’s accuracy requires enormous complexity and vast amounts of data, but in this problem, we’ll design a very simple question answering system based on inverse document frequency.
Our question answering system will perform two tasks: document retrieval and passage retrieval. Our system will have access to a corpus of text documents. When presented with a query (a question in English asked by the user), document retrieval will first identify which document(s) are most relevant to the query. Once the top documents are found, the top document(s) will be subdivided into passages (in this case, sentences) so that the most relevant passage to the question can be determined.
How do we find the most relevant documents and passages? To find the most relevant documents, we’ll use tf-idf to rank documents based both on term frequency for words in the query as well as inverse document frequency for words in the query. Once we’ve found the most relevant documents, there many possible metrics for scoring passages, but we’ll use a combination of inverse document frequency and a query term density measure (described in the Specification).
More sophisticated question answering systems might employ other strategies (analyzing the type of question word used, looking for synonyms of query words, lemmatizing to handle different forms of the same word, etc.) but we’ll leave those sorts of improvements as exercises for you to work on if you’d like to after you’ve completed this project!

UNDERSTANDING
First, take a look at the documents in corpus. Each is a text file containing the contents of a Wikipedia page. Our goal is to write an AI that can find sentences from these files that are relevant to a user’s query. You are welcome and encouraged to add, remove, or modify files in the corpus if you’d like to experiment with answering queries based on a different corpus of documents. Just be sure each file in the corpus is a text file ending in .txt.
Now, take a look at questions.py. The global variable FILE_MATCHES specifies how many files should be matched for any given query. The global variable SENTENCES_MATCHES specifies how many sentences within those files should be matched for any given query. By default, each of these values is 1: our AI will find the top sentence from the top matching document as the answer to our question. You are welcome and encouraged to experiment with changing these values.
In the main function, we first load the files from the corpus directory into memory (via the load_files function). Each of the files is then tokenized (via tokenize) into a list of words, which then allows us to compute inverse document frequency values for each of the words (via compute_idfs). The user is then prompted to enter a query. The top_files function identifies the files that are the best match for the query. From those files, sentences are extracted, and the top_sentences function identifies the sentences that are the best match for the query.
The load_files, tokenize, compute_idfs, top_files, and top_sentences functions are left to you!

SPECIFICATION
    Complete the implementation of load_files, tokenize, compute_idfs, top_files, and top_sentences in questions.py.
        The load_files function should accept the name of a directory and return a dictionary mapping the filename of each .txt file inside that directory to the file’s contents as a string.
            Your function should be platform-independent: that is to say, it should work regardless of operating system. Note that on macOS, the / character is used to separate path components, while the \ character is used on Windows. Use os.sep and os.path.join as needed instead of using your platform’s specific separator character.
            In the returned dictionary, there should be one key named for each .txt file in the directory. The value associated with that key should be a string (the result of reading the corresonding file).
            Each key should be just the filename, without including the directory name. For example, if the directory is called corpus and contains files a.txt and b.txt, the keys should be a.txt and b.txt and not corpus/a.txt and corpus/b.txt.
        The tokenize function should accept a document (a string) as input, and return a list of all of the words in that document, in order and lowercased.
            You should use nltk’s word_tokenize function to perform tokenization.
            All words in the returned list should be lowercased.
            Filter out punctuation and stopwords (common words that are unlikely to be useful for querying). Punctuation is defined as any character in string.punctuation (after you import string). Stopwords are defined as any word in nltk.corpus.stopwords.words("english").
            If a word appears multiple times in the document, it should also appear multiple times in the returned list (unless it was filtered out).
        The compute_idfs function should accept a dictionary of documents and return a new dictionary mapping words to their IDF (inverse document frequency) values.
            Assume that documents will be a dictionary mapping names of documents to a list of words in that document.
            The returned dictionary should map every word that appears in at least one of the documents to its inverse document frequency value.
            Recall that the inverse document frequency of a word is defined by taking the natural logarithm of the number of documents divided by the number of documents in which the word appears.
        The top_files function should, given a query (a set of words), files (a dictionary mapping names of files to a list of their words), and idfs (a dictionary mapping words to their IDF values), return a list of the filenames of the the n top files that match the query, ranked according to tf-idf.
            The returned list of filenames should be of length n and should be ordered with the best match first.
            Files should be ranked according to the sum of tf-idf values for any word in the query that also appears in the file. Words in the query that do not appear in the file should not contribute to the file’s score.
            Recall that tf-idf for a term is computed by multiplying the number of times the term appears in the document by the IDF value for that term.
            You may assume that n will not be greater than the total number of files.
        The top_sentences function should, given a query (a set of words), sentences (a dictionary mapping sentences to a list of their words), and idfs (a dictionary mapping words to their IDF values), return a list of the n top sentences that match the query, ranked according to IDF.
            The returned list of sentences should be of length n and should be ordered with the best match first.
            Sentences should be ranked according to “matching word measure”: namely, the sum of IDF values for any word in the query that also appears in the sentence. Note that term frequency should not be taken into account here, only inverse document frequency.
            If two sentences have the same value according to the matching word measure, then sentences with a higher “query term density” should be preferred. Query term density is defined as the proportion of words in the sentence that are also words in the query. For example, if a sentence has 10 words, 3 of which are in the query, then the sentence’s query term density is 0.3.
            You may assume that n will not be greater than the total number of sentences.
    You should not modify anything else in questions.py other than the functions the specification calls for you to implement, though you may write additional functions, add new global constant variables, and/or import other Python standard library modules.

HINTS
    In the compute_idfs function, recall that the documents input will be represented as a dictionary mapping document names to a list of words in each of those documents. The document names themselves are irrelevant to the calculation of IDF values. That is to say, changing any or all of the document names should not change the IDF values that are computed.
    Different sources may use different formulas to calculate term frequency and inverse document frequency than the ones described in lecture and given in this specification. Be sure that the formulas you implement are the ones described in this specification.

IMPROVEMENTS
    - 

"""
# SOURCE: 

import nltk
import sys

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_dict = dict()
    # Iterate through .txt files in the given directory:
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                file_string = file.read()
                file_dict[filename] = file_string

    return file_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    cleaned_tokens = []
    # Tokenize document string using nltk
    tokens = nltk.tokenize.word_tokenize(document.lower())
    # Ensure all tokens are lowercase, non-stopwords, non-punctuation
    for token in tokens:
        if token in nltk.corpus.stopwords.words('english'):
            continue
        else:
            all_punct = True
            for char in token:
                if char not in string.punctuation:
                    all_punct = False
                    break
            if not all_punct:
                cleaned_tokens.append(token)
    


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Number of documents for idf:
    num_docs = len    # Dictionary to hold scores for files
    file_scores = {filename:0 for filename in files}

    # Iterate through words in query:
    for word in query:
        # Limit to words in the idf dictionary:
        if word in idfs:
            # Iterate through the corpus, update each texts tf-idf:
            for filename in files:
              tf = files[filename].count(word)
              tf_idf = tf * idfs[word]
              file_scores[filename] += tf_idf

    sorted_files = sorted([filename for filename in files], key = lambda x : file_scores[x], reverse=True)

    # Return best n files
    return sorted_files[:n]
    # Dictionary to count number of docs containing each word:
    docs_with_word = dict()
    # Iterate through documents looking at unique words in each:
    for document in documents:
        doc_words = set(documents[document])

        for word in doc_words:
            if word not in docs_with_word:
                docs_with_word[word] = 1
            else:
                docs_with_word[word] += 1
    # Calculate idfs for each word:
    word_idfs = dict()
    for word in docs_with_word:
        word_idfs[word] = math.log((num_docs / docs_with_word[word]))

    return word_idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Dictionary to hold scores for files
    file_scores = {filename:0 for filename in files}
    # Iterate through words in query:
    for word in query:
        # Limit to words in the idf dictionary:
        if word in idfs:
            # Iterate through the corpus, update each texts tf-idf:
            for filename in files:
              tf = files[filename].count(word)
              tf_idf = tf * idfs[word]
              file_scores[filename] += tf_idf
    sorted_files = sorted([filename for filename in files], key = lambda x : file_scores[x], reverse=True)
    # Return best n files
    return sorted_files[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Dict to score sentences:
    sentence_score = {sentence:{'idf_score': 0, 'length':0, 'query_words':0, 'qtd_score':0} for sentence in sentences}
    # Iterate through sentences:
    for sentence in sentences:
        s = sentence_score[sentence]
        s['length'] = len(nltk.word_tokenize(sentence))
        # Iterate through query words:
        for word in query:
            # If query word is in sentence word list, update its score
            if word in sentences[sentence]:
                s['idf_score'] += idfs[word]
                s['query_words'] += sentences[sentence].count(word)
        # Calculate query term density for each sentence:
        s['qtd_score'] = s['query_words'] / s['length']
    # Rank sentences by score and return n sentence
    sorted_sentences = sorted([sentence for sentence in sentences], key= lambda x: (sentence_score[x]['idf_score'], sentence_score[x]['qtd_score']), reverse=True)
    # Return n entries for sorted sentence:
    return sorted_sentences[:n]


if __name__ == "__main__":
    main()
