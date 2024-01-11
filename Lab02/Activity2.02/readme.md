## Activity 2.02: Text Visualization

In this activity, you will create a word cloud for the 50 most frequent words in a dataset. The dataset consists of random sentences that need to be cleaned and analyzed to identify frequently occurring words.

### Prerequisites

- Basic understanding of Python programming.
- Familiarity with text processing and visualization libraries in Python.

### Data

The dataset used in this activity is available at the following link: [text_corpus.txt](https://github.com/fenago/natural-language-processing-workshop/blob/master/Lab02/data/text_corpus.txt
).

### Steps to Follow

1. **Import Necessary Libraries**:
   - Import libraries required for data fetching, text processing, and visualization (like `pandas`, `nltk`, `matplotlib`, `wordcloud`, etc.).

2. **Fetch the Dataset**:
   - Retrieve the `text_corpus.txt` file and load its contents.

3. **Preprocess the Text**:
   - Perform text cleaning to remove unwanted characters and formats.
   - Tokenize the text.
   - Apply lemmatization to convert words to their base form.

4. **Identify Top 50 Words**:
   - Calculate the frequency of each word in the cleaned dataset.
   - Create a set of the top 50 most frequent words along with their frequencies.

5. **Create a Word Cloud**:
   - Use the word cloud library to visualize the top 50 words.
   - Customize the word cloud's appearance as needed.

6. **Analyze the Word Cloud**:
   - Compare the word cloud with the calculated word frequencies.
   - Justify the representation of words in the word cloud based on their frequencies.

### Challenge for Students

Now that you have created a word cloud for a given dataset, try extending your skills with these tasks:

- **Use a Different Dataset**: Find another text dataset that interests you. It could be a collection of social media posts, reviews, or any other textual content.
- **Apply Enhanced Text Processing**: Experiment with different preprocessing techniques like stop word removal, n-grams, or POS tagging.
- **Visualize Your Findings**: Create a word cloud for your chosen dataset. How does the word cloud reflect the key themes or sentiments in the data?
- **Draw Insights**: Reflect on how word clouds can aid in quick data analysis, highlighting key areas for deeper exploration.

**Explore Further**: Consider how word clouds can be used in areas like marketing analysis, sentiment analysis, or summarizing large volumes of text.
