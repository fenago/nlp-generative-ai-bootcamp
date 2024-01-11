# The Natural Language Processing 
Lab Environment: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fenago/natural-language-processing-workshop/HEAD)
### Requirements and Setup

To get started with the project files, you'll need to:
1. Install Jupyter on [Windows](https://www.python.org/downloads/windows/), [Mac](https://www.python.org/downloads/mac-osx/), [Linux](https://www.python.org/downloads/source/)
2. Install Anaconda on [Windows](https://www.anaconda.com/distribution/#windows), [Mac](https://www.anaconda.com/distribution/#macos), [Linux](https://www.anaconda.com/distribution/#linux)

### Prerequisites

1. Download and Install Python using [Anaconda Distribution](https://www.anaconda.com/distribution/)

2. Create a virtual environment by any of the following command:

   `conda create -n nlp-env python=3.7 (If using Anaconda distribution)`
   
   `conda activate nlp-env`
   
   or
   
   `python -m venv nlp-env`
   
   `.\nlp-env\Scripts\activate (Windows)`
   
   `source nlp-env/bin/activate    (Linux or macOS)`

3. Install all the required packages by running the following command 

   "pip install -r requirements.txt"
     
4. Download all the NLTK packages using the following command:
   nltk.download()
   
5. Download the SpaCy model using the following command:
   python -m spacy download en_core_web_sm

## What you will learn
* Obtain, verify, clean and transform text data into a correct format for use 
* Use methods such as tokenization and stemming for text extraction 
* Develop a classifier to classify comments in Wikipedia articles 
* Collect data from open websites with the help of web scraping 
* Train a model to detect topics in a set of documents using topic modeling 
* Discover techniques to represent text as word and document vectors 
