#!/bin/bash
# Install SC2 and add the custom maps

if [ -z "$EXP_DIR" ]
then
    EXP_DIR=~
fi

echo "EXP_DIR: $EXP_DIR"
cd $EXP_DIR/smacplus/pymarl

mkdir 3rdparty
cd 3rdparty

export SC2PATH=`pwd`'/StarCraftII'
echo 'SC2PATH is set to '$SC2PATH

if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.6.2.69232.zip
        unzip -P iagreetotheeula "./SC2.4.6.2.69232.zip"
        rm -rf SC2.4.6.2.69232.zip
else
        echo 'StarCraftII is already installed.'
fi
