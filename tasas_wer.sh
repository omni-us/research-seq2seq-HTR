#!/bin/bash
source /usr/local/bin/htrsh.inc.sh
tasas <( htrsh_prep_tasas $1 $2 -f tab) \
    -ie -s " " -f "|"
# $1: ground truth   $2: decoded
