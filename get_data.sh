cd data/
wget https://www.dropbox.com/s/hjvlciuabru9nbv/HAPT%20Data%20Set.zip?dl=1 -O HAPT_Data_Set.zip
unzip HAPT_Data_Set.zip
mv data/* .
rm -rf data/
rm HAPT_Data_Set.zip
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip UCI\ HAR\ Dataset.zip
mv UCI\ HAR\ Dataset HAR
rm UCI\ HAR\ Dataset.zip