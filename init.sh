#!/usr/bin/env bash

set -e pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
DATA_DIR=$DIR/datasets

function download_dataset() {
	echo "Downlading dataset..."
	mkdir -p $DATA_DIR

	if [ ! -f $DATA_DIR/twitter-datasets.zip ];
	then
		pushd $DATA_DIR

		wget -c http://www.da.inf.ethz.ch/files/twitter-datasets.zip
		unzip -o twitter-datasets.zip

		popd > /dev/null
	fi
	echo "Dataset downloaded."
}

function download_dependecies() {
	echo "Downlading dependecies..."
	pip3 install -r requirements.txt
	echo "Dependecies downloaded."
}

pushd $DIR > /dev/null
download_dataset
download_dependecies
popd > /dev/null
