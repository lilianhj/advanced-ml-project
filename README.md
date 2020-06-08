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

We also require the spaCy English language module.

### Getting started

To train our neural networks, our group used AWS EC2 instances. In particular,
we use the Deep Learning AMI (Ubuntu 18.04) Version 29.0 (ami-09aaaa836262a0c46)
Amazon Machine Image running on g4dn.2xlarge EC2 instances. These instances
have access to a single NVIDIA T4 GPU.

After spinning up the virtual machine, we uploaded the `ec2_setup.sh` file to
the root of the instance via `scp` and ran the following command to set-up our
workspace:

```
ngroktoken=<NGROK SECRET AUTHORIZATION TOKEN>
sudo sh ec2_setup.sh
```

This command cloned our repo, installed project dependencies, installed and
configured ngrok (a lightweight program allowing for easy remote access to our
Jupyter notebooks server), and configured our Jupyter notebooks server.

Next, we ran the following command inside a `screen` session to start the
Jupyter notebooks server and keep it running even if we disconnected from the
virtual machine:

```
jupyter notebook --no-browser
```

Lastly, in a separate `screen` session, we ran the following command to allow
for remote access to our Jupyter notebooks server:

```
ngrok http 8888
```

### File Structure

Our code is organized into different directories as follows:

```
| /data_exploration/: exploratory analysis of our text data
|--- textpreprocessing.ipynb: exploratory data analysis of our raw text (2061
|    lines, written for this project)
|
| /data_pipeline/: scraping and extracting text from the United States
|                  Department of Health and Human Services website
|--- download_data.py: scraper for extracting text from the HHS DAB website (552
|                      lines, written for this project)
|--- preprocessing.py: processing the raw text for use in PyTorch modeling (218
|                      lines, written for this project)
|--- db_connection.py.example: a template for the database connection
|                              information (5 lines, written for this project)
|
| /db_scripts/: scripts for setting up our Postgres database to store downloaded
|               cases
|--- create_raw_table.sql: database table schema (12 lines, written for this
|                          project)
|--- /embeds/: downloading pre-trained embeddings for use in modeling
|--- get_embeds.sh: shell script to download pre-trained word embeddings (2
|                   lines, written for this project)
|
| /modeling/: using PyTorch to predict appeal outcomes from decision text
|--- SimpleNNs.ipynb: using simple neural networks with custom word embeddings
|                     (3773 lines, partly from CAPP 30255 HW2, partly written
|                     for this project)
|--- SimpleNNs-Law2Vec.ipynb: using simple neural networks with pre-trained word
|                             embeddings (1243 lines, partly from CAPP 30255
|                             HW2, partly written for this project)
|--- RNNs.ipynb: using RNNs and LSTMs with custom word embeddings (3953 lines,
|                partly from CAPP 30255 HW3, partly written for this project)
|--- RNNs-Law2Vec.ipynb: using RNNs and LSTMs with pre-trained word embeddings
|                        (1309 lines, partly from CAPP 30255 HW3, partly written
|                        for this project)
|--- training.py: defining a general TrainingModule class and functions to
|                 calculate model performance (192 lines, partly from CAPP 30255
|                 HW2, partly written for this project)
|
|--- /results/: CSV datasets on model performance results, saved models
|               identified as the best, and analysis of model
|               performance
|--- README.md: describes the naming conventions of saved models/results in this
|               directory
|--- model_performance_results.ipynb: analyzing model performance and testing
|                                     our identified best model (1586 lines,
|                                     written for this project)
|
| ec2_setup.sh: shell script used to set-up AWS EC2 instance for training models
                (19 lines, written for this project)
| requirements.txt: model requirements
```
