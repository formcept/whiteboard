#!/bin/bash

##################
set -e
export LEIN_SNAPSHOTS_IN_RELEASE=true
##################

echo -e "Building Onyx..."

##################

echo -e "\n---------------------------------------------------------------------"
echo "Building Onyx Utils"
echo -e "---------------------------------------------------------------------"
cd onyx-utils
lein do clean, compile, install, uberjar
cd ..

##################

echo -e "\n---------------------------------------------------------------------"
echo "Building Onyx LSH"
echo -e "---------------------------------------------------------------------"
cd locality-sensitive-hashing
lein with-profiles +ship do clean, compile, install, uberjar
cd ..

##################

