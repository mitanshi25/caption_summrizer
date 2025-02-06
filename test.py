from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.lsa import LsaSummarizer
from main import transcript_text
from sumy.nlp.tokenizers import Tokenizer


LANGUAGE = "english"
parser = PlaintextParser.from_string(transcript_text, Tokenizer(LANGUAGE))
summarizer = LsaSummarizer(Stemmer(LANGUAGE))
summarizer.stop_words = get_stop_words(LANGUAGE)
summary = summarizer(parser.document, 3)  # Number of sentences in the summary
