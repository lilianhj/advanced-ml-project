# CAPP 30255 Project: Advanced Machine Learning for Public Policy

### Project Description

We explore the possibilities of neural networks for predicting whether a Medicare/Medicaid adjudicatory decision can be successfully appealed through analyzing the the text from the original Administrative Law Judge (ALJ) court decision. This approach could potentially be applied to the numerous other policy domains with a similar structure of administrative law decisions and subsequent appeals.

### Team Members
* [Ben Fogarty](https://github.com/fogarty-ben)
* [Lilian Huang](https://github.com/lilianhj)
* [Katy Koenig](https://github.com/katykoenig)

### Libraries Used

See requirements.txt for our full list of dependencies. Major dependencies include:

* textract and BeautifulSoup for scraping and extracting text
* spaCy and torchtext for processing the data and preparing it for modeling
* PyTorch for modeling

### File Structure

Our code is organized into different directories as follows:

* /data_pipeline/: scraping and extracting text from the United States Department of Health and Human Services Departmental Appeals Board (DAB) website
    * download_data.py: scraper for extracting text from the HHS DAB website (552 lines, written for this project)
    * preprocessing.py: processing the raw text for use in PyTorch modeling (218 lines, written for this project)
    * db_connection.py: a template for the database connection information (5 lines, written for this project)
* /db_scripts/: scripts for storing our downloaded text data in a Postgres database
    * create_raw_table.sql: database table schema (12 lines, written for this project)
* /data_exploration/: exploratory analysis of our text data
    * textpreprocessing.ipynb: exploratory data analysis of our raw text (2061 lines, written for this project)
* /embeds/: downloading pre-trained embeddings for use in modeling
    * get_embeds.sh: shell script to download pre-trained word embeddings (2 lines, written for this project)
* /modeling/: using PyTorch to predict appeal outcomes from decision text
    * training.py: defining an overall TrainingModule class and functions to calculate model performance (158 lines, partly from CAPP 30255 HW2, partly written for this project)
    * SimpleNNs.ipynb: using simple neural networks with custom word embeddings (3780 lines, partly from CAPP 30255 HW2, partly written for this project)
    * SimpleNNs-Law2Vec.ipynb: using simple neural networks with pre-trained word embeddings (1250 lines, partly from CAPP 30255 HW2, partly written for this project)
    * RNNs.ipynb: using RNNs and LSTMs with custom word embeddings (3953 lines, partly from CAPP 30255 HW3, partly written for this project)
    * RNNs-Law2Vec.ipynb: using RNNs and LSTMs with pre-trained word embeddings (1309 lines, partly from CAPP 30255 HW3, partly written for this project)
* /results/: our model performance, the models identified as the best, and analysis of model performance
    * model_performance_results.ipynb: analyzing model performance and testing our identified best model (1586 lines, written for this project)
