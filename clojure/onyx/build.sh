#!/bin/bash

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
lein do clean, compile, install, uberjar
cd ..

##################

