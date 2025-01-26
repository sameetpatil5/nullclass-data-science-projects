# import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer
import nltk

# nltk.download("punkt_tab") # Uncomment this line to download the required NLTK data


def create_frequency_table(text_string: str) -> dict:
    """
    This function takes in a string and returns a dictionary with the frequency
    of each word in the string. The words are stemmed and common stop words are
    excluded.

    Parameters
    ----------
    text_string : str
        The string to be processed

    Returns
    -------
    dict
        A dictionary of words to their frequencies
    """
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freq_table = dict()
    for word in words:
        word = ps.stem(word)
        if word in stop_words:
            continue
        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1

    return freq_table


def score_sentences(sentences: list, freq_table: dict) -> dict:
    """
    This function takes in a list of sentences and a frequency table and returns a dictionary
    with the score of each sentence. The score is calculated by summing up the frequency of
    all the words in a sentence and then dividing by the number of words in the sentence.

    Parameters
    ----------
    sentences : list[str]
        A list of sentences to be scored
    freq_table : dict
        A dictionary of words to their frequencies

    Returns
    -------
    dict
        A dictionary with the score of each sentence
    """

    sentence_value = dict()

    for sentence in sentences:
        word_count_in_sentence = len(word_tokenize(sentence))
        if word_count_in_sentence == 0:
            continue
        for word_value in freq_table:
            if word_value in sentence.lower():
                if sentence in sentence_value:
                    sentence_value[sentence] += freq_table[word_value]
                else:
                    sentence_value[sentence] = freq_table[word_value]

        sentence_value[sentence] = (
            sentence_value.get(sentence, 0) // word_count_in_sentence
        )

    return sentence_value


def find_average_score(sentence_value: dict) -> int:
    """
    Calculate the average score of sentences.

    This function takes a dictionary of sentences with their scores and computes the average score
    by summing up all the scores and dividing by the number of sentences.

    Parameters
    ----------
    sentence_value : dict
        A dictionary where keys are sentences and values are their respective scores.

    Returns
    -------
    int
        The average score of the sentences as an integer.
    """

    sum_values = 0
    for entry in sentence_value:
        sum_values += sentence_value[entry]

    # Average value of a sentence from original text
    average = int(sum_values / len(sentence_value))

    return average


def generate_summary(sentences, sentence_value, threshold) -> str:
    """
    Generate a summary by selecting important sentences.

    This function iterates over a list of sentences and selects those with a score
    above a given threshold to form a summary. The scores of the sentences are provided
    in a dictionary. Sentences that meet the criteria are concatenated to form the
    summary.

    Parameters
    ----------
    sentences : list
        A list of sentences from the text.
    sentence_value : dict
        A dictionary where keys are sentences and values are their scores.
    threshold : float
        The score threshold above which sentences are included in the summary.

    Returns
    -------
    str
        A string containing the generated summary.
    """

    sentence_count = 0
    summary = ""

    for sentence in sentences:
        if sentence in sentence_value and sentence_value[sentence] > (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def summarize_text(text) -> str:
    """
    Summarize a given text by scoring sentences and generating a summary with the top-scoring sentences.

    This function takes a text string and returns a string containing a summary of the text.
    The text is first tokenized into sentences, and then word frequencies are computed.
    The sentences are then scored based on their word frequencies and the sentences
    with the highest scores are selected to form the summary.

    Parameters
    ----------
    text : str
        The text to be summarized.

    Returns
    -------
    str
        A string containing the generated summary.
    """

    # Create the word frequency table
    freq_table = create_frequency_table(text)

    # Tokenize the sentences
    sentences = sent_tokenize(text)

    # Important Algorithm: score the sentences
    sentence_scores = score_sentences(sentences, freq_table)

    # Find the threshold
    threshold = 1.3 * find_average_score(sentence_scores)

    # Generate the summary
    summary = generate_summary(sentences, sentence_scores, threshold)

    return summary
