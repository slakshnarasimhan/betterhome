#!/bin/bash
cd /home/ec2-user/betterhome
source betterhome_env/bin/activate
streamlit run ask-questions.py --server.port=8501 --server.address=0.0.0.0
