#!/bin/bash
DOWNLOAD_TO=~/.pyannote
mkdir -p $DOWNLOAD_TO
bash ./download_ami.sh $DOWNLOAD_TO

bash ./download_musan.sh $DOWNLOAD_TO
cp ./database.yml $DOWNLOAD_TO

cp -r ./AMI $DOWNLOAD_TO
cp -r ./MUSAN $DOWNLOAD_TO
