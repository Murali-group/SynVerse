1. Put a Dockerfile to build image for MolE.
2. Put the requirement.txt file containing all the required package information for running MolE.
3. Put this following block in code/build_docker.sh.

`BASEDIR=$(pwd)
cd $BASEDIR/models/pretrained/mole
docker build -q -t mole:base .
if ([ $? = 0 ] && [ "$(docker images -q mole:base 2> /dev/null)" != "" ]); then
    echo "Docker container for MolE is built and tagged as mole:base"
elif [ "$(docker images -q mole:base 2> /dev/null)" != "" ]; then
    echo "Docker container failed to build, but an existing image exists at mole:base"
else
    echo "Oops! Unable to build Docker container for MolE"
fi
`