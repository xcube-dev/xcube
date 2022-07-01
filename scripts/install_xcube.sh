#!/bin/env bash

PACKAGE=$1;
PACKAGE_VERSION=$2;
INSTALL_MODE=$3

echo "############################################"
echo "INSTALLING ${PACKAGE}-${PACKAGE_VERSION}"
echo "############################################"

if [[ $INSTALL_MODE == "branch" ]]; then
  git clone https://github.com/dcs4cop/"${PACKAGE}"
  cd "${PACKAGE}" || exit
  git checkout "${PACKAGE_VERSION}"
  sed -i "s/- xcube/#- xcube/g" environment.yml || exit

  source activate xcube && mamba env update -n xcube
  source activate xcube && pip install .
  cd .. && rm -rf "${PACKAGE}"
elif [[ $INSTALL_MODE == "release" ]]; then
  # Receive version number if PACKAGE_VERSION is latest
  if [[ $PACKAGE_VERSION == "latest" ]]; then
    PACKAGE_VERSION=$(curl -sL https://api.github.com/repos/dcs4cop/"${PACKAGE}"/releases/latest | jq -r '.name')
  fi

  wget https://github.com/dcs4cop/"${PACKAGE}"/archive/v"${PACKAGE_VERSION}".tar.gz

  tar xvzf v"${PACKAGE_VERSION}".tar.gz

  cd "${PACKAGE}"-"${PACKAGE_VERSION}" || exit

  #sed -i "s/- xcube/#- xcube/g" environment.yml || exit
  cat environment.yml
  source activate xcube && mamba env update -n xcube
  source activate xcube && pip install .
  cd .. && rm v"${PACKAGE_VERSION}".tar.gz
else
  mamba install -y -n xcube -c conda-forge "${PACKAGE}"="${PACKAGE_VERSION}"
fi




