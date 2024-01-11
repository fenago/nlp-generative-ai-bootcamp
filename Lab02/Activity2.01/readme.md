
## Activity 2.01: Extracting Top Keywords from the News Article

In this activity, you will extract the most frequently occurring keywords from a sample news article using Python and the Natural Language Toolkit (NLTK).

### Prerequisites

- Basic understanding of Python programming.
- An environment to run Python code (like Jupyter Notebook or Google Colab).

### Data

The news article used in this activity is available at the following link: [news_article.txt](https://github.com/fenago/natural-language-processing-workshop/blob/master/Lab02/data/news_article.txt).

### Steps to Follow

1. **Set Up Your Environment**:
   - Open Jupyter Notebook or Google Colab.
   - Ensure Python is installed along with NLTK. Install NLTK if not already installed using `!pip install nltk`.

2. **Import Necessary Libraries**:
   - Import `nltk` and other necessary Python libraries.

3. **Define Helper Functions**:
   - Create functions to load the text file, convert text to lowercase, tokenize the text, remove stop words, perform stemming, and calculate word frequencies.

4. **Load the News Article**:
   - Use Python's file handling methods to load `news_article.txt` into a string.

5. **Preprocess the Text**:
   - Convert the text to lowercase.
   - Tokenize the text using a whitespace tokenizer.
   - Remove stop words from the tokens.
   - Perform stemming on the remaining tokens.

6. **Calculate Word Frequencies**:
   - Count the frequency of each word after stemming.
   - Display the most frequent keywords.

### Challenge for Students

Now that you've learned how to extract keywords from a news article, challenge yourself by applying these techniques to a different dataset. Here's what you can do:

- **Find a Unique Dataset**: Select a text dataset of your interest. This could be another news article, a blog post, or any textual data.
- **Implement the Keyword Extraction Process**: Apply the steps you've learned in this activity to your dataset. This includes text preprocessing, tokenization, stop word removal, stemming, and frequency analysis.
- **Analyze Your Results**: Look at the most frequent keywords in your dataset. Do they give you insights into the main themes or topics of the text?

**Contextualize Your Learning**: Reflect on how this process could be useful in real-world applications like search engine optimization, content analysis, or summarizing information.
