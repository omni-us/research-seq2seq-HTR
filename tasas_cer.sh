#!/bin/bash
source ./utils/htrsh.inc.sh
./utils/tasas <( htrsh_prep_tasas $1 $2 -f tab -c yes ) \
    -ie -s " " -f "|"
# $1: ground truth   $2: decoded
