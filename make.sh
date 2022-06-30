#!/bin/bash
echo "  _____                  _   _                               "
echo " /  __ \                | \ | |                              "
echo " | /  \/ __ _ ___  ___  |  \| | ___  _____      ____ _ _   _ "
echo " | |    / _\` / __|/ _ \ | . \` |/ _ \/ _ \ \ /\ / / _\` | | | |"
echo " | \__/\ (_| \__ \  __/ | |\  |  __/ (_) \ V  V / (_| | |_| |"
echo "  \____/\__,_|___/\___| \_| \_/\___|\___/ \_/\_/ \__,_|\__, |"
echo "                                                        __/ |"
echo "                                                       |___/ "
echo "Eduardo M. de Morais"

echo "Setting directories"

export PATH=$PATH:$(pwd)/cli

mkdir -p logs

mkdir -p data/interim

mkdir -p data/external

mkdir -p data/predicted

mkdir -p data/processed/