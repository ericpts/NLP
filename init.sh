set -e pipefail

mkdir -p datasets
cd datasets
# wget -c http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
# unzip -o trainingandtestdata.zip
wget -c http://www.da.inf.ethz.ch/files/twitter-datasets.zip
unzip -o twitter-datasets.zip
cd ../

