#!/bin/bash

arr=("$@")

for inferdir in "${arr[@]}";
    do 
        echo "Inferring $inferdir..."
        spk=$spk inferdir=$inferdir hparams=$hparams CUDA_VISIBLE_DEVICES="0,1" ./infer.sh
    done