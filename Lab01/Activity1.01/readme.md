# Activity 1.01: Preprocessing of Raw Text
We have a text corpus that is in an improper format. In this activity, we will perform all the preprocessing steps that were discussed earlier to get some meaning out of the text.

The text corpus, file.txt, can be found at this location: [https://github.com/fenago/natural-language-processing-workshop/blob/master/Lab01/data/file.txt](https://github.com/fenago/natural-language-processing-workshop/blob/master/Lab01/data/file.txt
)

After downloading the file, place it in the same directory as the notebook.

Follow these steps to implement this activity:

1. Import the necessary libraries.
2. Load the text corpus to a variable.
3. Apply the tokenization process to the text corpus and print the first 20 tokens.
4. Apply spelling correction on each token and print the initial 20 corrected tokens as well as the corrected text corpus.
5. Apply PoS tags to each of the corrected tokens and print them.
6. Remove stop words from the corrected token list and print the initial 20 tokens.
7. Apply stemming and lemmatization to the corrected token list and then print the initial 20 tokens.
8. Detect the sentence boundaries in the given text corpus and print the total number of sentences.
