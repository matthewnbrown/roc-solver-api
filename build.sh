#!/bin/bash

imageName=everspy/roccapsolver
imageTag=0.0.0

# build image
docker build -t $imageName:$imageTag .

# mark image as latest
docker tag $imageName:$imageTag $imageName:latest

# push image
docker push $imageName:$imageTag
docker push $imageName:latest