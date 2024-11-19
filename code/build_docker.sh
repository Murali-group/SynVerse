#!/bin/bash

# Building docker for the different algorithms
echo "Building docker image for different pretrained models. This may take a while..."

BASEDIR=$(pwd)

# You may remove the -q flag if you want to see the docker build status
cd $BASEDIR/models/pretrained/mole
docker build -t mole:base .
if ([ $? = 0 ] && [ "$(docker images -q mole:base 2> /dev/null)" != "" ]); then
    echo "Docker container for MolE is built and tagged as mole:base"
elif [ "$(docker images -q mole:base 2> /dev/null)" != "" ]; then
    echo "Docker container failed to build, but an existing image exists at mole:base"
else
    echo "Oops! Unable to build Docker container for mole"
fi



# You may remove the -q flag if you want to see the docker build status
cd $BASEDIR/models/pretrained/kpgt
docker build -t kpgt:base .
if ([ $? = 0 ] && [ "$(docker images -q kpgt:base 2> /dev/null)" != "" ]); then
    echo "Docker container for kpgt is built and tagged as kpgt:base"
elif [ "$(docker images -q kpgt:base 2> /dev/null)" != "" ]; then
    echo "Docker container failed to build, but an existing image exists at kpgt:base"
else
    echo "Oops! Unable to build Docker container for kpgt"
fi
