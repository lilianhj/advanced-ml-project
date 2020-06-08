#!/bin/sh
git clone https://github.com/lilianhj/advanced-ml-project.git
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
unzip ngrok-stable-linux-amd64.zip
./ngrok authtoken $ngroktoken # must set ngroktoken before running script

# install dependencies
cd advanced-ml-project
pip3 install -r requirements.txt
python3 -m spacy download en
cd embeds
sh get_embeds.sh
cd ..

# set-up jupyter notebook
jupyter notebook --generate-config
echo "c.NotebookApp.allow_remote_access = True" >> ~/.jupyter/jupyter_notebook_config.py
ipython kernel install --name "adv-ml"
jupter notebook password