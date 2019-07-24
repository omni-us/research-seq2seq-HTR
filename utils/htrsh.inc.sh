#!/bin/bash

##
## Collection of shell functions for Handwritten Text Recognition.
##
## @version $Version: 2019.07.24$
## @author Mauricio Villegas <mauricio_ville@yahoo.com>
## @copyright Copyright(c) 2014-present, Mauricio Villegas <mauricio_ville@yahoo.com>
## @license MIT License
##

##
## The MIT License (MIT)
##
## Copyright (c) 2015-present, Mauricio Villegas <mauricio_ville@yahoo.com>
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##

[ "${BASH_SOURCE[0]}" = "$0" ] &&
  echo "htrsh.inc.sh: error: script intended for sourcing, try: . htrsh.inc.sh" 1>&2 &&
  exit 1;
[ "$(type -t htrsh_version)" = "function" ] &&
  echo "htrsh.inc.sh: warning: library already loaded, to reload first use htrsh_unload" 1>&2 &&
  return 1;

run_parallel_path="${BASH_SOURCE%/*}/run_parallel.inc.sh";
if [ ! -e "$run_parallel_path" ]; then
  [ "$(which run_parallel.inc.sh)" = "" ] &&
    echo "htrsh.inc.sh: error: required run_parallel.inc.sh not found in same directory as htrsh.inc.sh or in path" 1>&2 &&
    return 1;
  run_parallel_path=$(which run_parallel.inc.sh);
fi
. "$run_parallel_path";

#-----------------------#
# Default configuration #
#-----------------------#

htrsh_keeptmp="0";

htrsh_xpath_regions='//_:TextRegion';
htrsh_xpath_lines='_:TextLine';
htrsh_xpath_words='_:Word';
htrsh_xpath_coords='_:Coords[@points and @points!="0,0 0,0"]';
htrsh_xpath_textequiv=$'_:TextEquiv[_:Unicode and translate(_:Unicode,"\n\r\t ","") != ""]/_:Unicode';
htrsh_extended_names="false";

htrsh_imgclean="prhlt"; # Image preprocessing technique, prhlt or ncsr
htrsh_clean_type="image"; htrsh_clean_type="line";

htrsh_imgtxtenh_regmask="yes";               # Whether to use a region-based processing mask
htrsh_imgtxtenh_opts="-r 0.16 -w 20 -k 0.1"; # Options for imgtxtenh tool
htrsh_imglineclean_opts="-V0 -m 99%";        # Options for imglineclean tool
htrsh_minres="50";                           # Minimum resolution for images

htrsh_warn_imgres="no";

htrsh_feat_deslope="yes"; # Whether to correct slope per line
htrsh_feat_deslant="yes"; # Whether to correct slant of the text
htrsh_feat_padding="1.0"; # Left and right white padding in mm for line images
htrsh_feat_contour="yes"; # Whether to compute connected components contours
htrsh_feat_dilradi="0.5"; # Dilation radius in mm for contours
htrsh_feat_normxheight="18"; # Normalize x-height (if in Page) to a fixed number of pixels
htrsh_feat_normheight="0";   # Normalize height to a fixed number of pixels

htrsh_feat="dotmatrix";    # Type of features to extract
htrsh_dotmatrix_shift="2"; # Sliding window shift in px @todo make it with respect to x-height
htrsh_dotmatrix_win="20";  # Sliding window width in px @todo make it with respect to x-height
htrsh_dotmatrix_W="8";     # Width of normalized frame in px
htrsh_dotmatrix_H="32";    # Height of normalized frame in px
htrsh_dotmatrix_mom="yes"; # Whether to add moments to features

htrsh_align_chars="no";             # Whether to align at a character level
htrsh_align_dilradi="0.5";          # Dilation radius in mm for contours
htrsh_align_contour="yes";          # Whether to compute contours from the image
htrsh_align_isect="yes";            # Whether to intersect parallelograms with line contour
htrsh_align_midbox="yes";           # Whether to use bounding box of parallelogram side means
htrsh_align_prefer_baselines="yes"; # Whether to always generate contours from baselines
htrsh_align_addtext="yes";          # Whether to add TextEquiv to word and glyph nodes
htrsh_align_words="yes";            # Whether to align at a word level when aligning regions
htrsh_align_wordsplit="no";         # Whether to split words when aligning regions

htrsh_hmm_states="6";  # Default number of HMM states (excluding special initial and final)
htrsh_hmm_ndstates=""; # Number of states for specific HMMs (name #states\n...) use 'd' for an expression involving the default
htrsh_hmm_nummix="4";  # Number of Gaussian mixture components per state
htrsh_hmm_iter="4";    # Number of training iterations
htrsh_hmm_type="char"; # HMM modelling type, currently among char and overlap

htrsh_HTK_HERest_opts="-m 2";      # Options for HERest tool
htrsh_HTK_HCompV_opts="-f 0.1 -m"; # Options for HCompV tool
htrsh_HTK_HHEd_opts="";            # Options for HHEd tool
htrsh_HTK_HVite_align_opts="";     # Options for HVite tool for alignments

htrsh_HTK_config='
HMMDEFFILTER   = "gzip -dc $"
HMMDEFOFILTER  = "gzip > $"
HNETFILTER     = "gzip -dc $"
HNETOFILTER    = "gzip > $"
NONUMESCAPES   = T
STARTWORD      = "<s>"
ENDWORD        = "</s>"
';

htrsh_symb_space="{space}";
htrsh_symb_eps="{eps}";
htrsh_symb_blank="{blank}";

htrsh_special_chars=$'
<gap/> {gap}
_ {_}
\x27 {squote}
" {dquote}
& {amp}
< {lt}
> {gt}
{ {lbrace}
} {rbrace}
';

htrsh_sed_tokenize_simplest='
  #s|$\.|$*|g;
  s|\([.,:;!¡?¿+\x27´`"“”„|(){}[—–_]\)| \1 |g;
  s|\x5D| ] |g;
  #s|$\*|$.|g;
  s|\([0-9]\)| \1 |g;
  s|^  *||;
  s|  *$||;
  s|   *| |g;
  s|\. \. \.|...|g;
  ';

htrsh_sed_translit_vowels='
  s|á|a|g; s|Á|A|g;
  s|é|e|g; s|É|E|g;
  s|í|i|g; s|Í|I|g;
  s|ó|o|g; s|Ó|O|g;
  s|ú|u|g; s|Ú|U|g;
  ';

htrsh_valschema="yes";
htrsh_pagexsd=$(readlink -f "$(which htrsh.inc.sh)" | sed 's|/htrsh.inc.sh$|/pagecontent_searchink.xsd|');
[ ! -e "$htrsh_pagexsd" ] &&
  htrsh_pagexsd="https://www.prhlt.upv.es/~mvillegas/xsd/pagecontent_prhlt.xsd";

htrsh_realpath="readlink -f";
[ $(realpath --help 2>&1 | grep relative | wc -l) != 0 ] &&
  htrsh_realpath="realpath --relative-to=.";

htrsh_infovars="XMLDIR IMDIR IMFILE XMLBASE IMBASE IMEXT IMSIZE IMRES RESSRC";

#---------------------------#
# Generic library functions #
#---------------------------#

##
## Function that prints the version of the library
##
htrsh_version () {
  echo '$Version: 2019.07.24$' \
    | sed -r 's|^\$Version[:] ([^$]+)\$|htrsh \1|' 1>&2;
}

##
## Function that unloads the library
##
htrsh_unload () {
  unset $(compgen -A variable htrsh_);
  unset -f $(compgen -A function htrsh_);
}

##
## Function that checks that all required commands are available
##
htrsh_check_dependencies () {
  local FN="htrsh_check_dependencies";
  local RC="0";
  local cmd;
  for cmd in xmlstarlet convert octave HVite dotmatrix imgtxtenh imglineclean imgccomp imgpolycrop imageSlant page_format_generate_contour; do
    local c=$(which $cmd 2>/dev/null | sed '/^alias /d; s|^\t||');
    [ ! -e "$c" ] && RC="1" &&
      echo "$FN: WARNING: unable to find command: $cmd" 1>&2;
  done

  [ $(dotmatrix -h 2>&1 | grep '\--htk' | wc -l) = 0 ] && RC="1" &&
    echo "$FN: WARNING: a dotmatrix with --htk option is required" 1>&2;

  for cmd in readhtk writehtk; do
    [ $(octave -q -H --eval "which $cmd" | wc -l) = 0 ] && RC="1" &&
      echo "$FN: WARNING: unable to find octave command: $cmd" 1>&2;
  done

  if [ "$RC" = 0 ]; then
    htrsh_version;
    echo run_parallel version $(run_parallel --version);
    for cmd in imgtxtenh imglineclean imgccomp; do
      $cmd --version;
    done
    { printf "xmlstarlet "; xmlstarlet --version;
      convert --version | sed -n '1{ s|^Version: ||; p; }';
      octave -q --version | head -n 1;
      HVite -V | grep HVite | cat;
    } 1>&2;
  fi

  return $RC;
}


#---------------------------------#
# XML Page manipulation functions #
#---------------------------------#

##
## Function that creates an empty Page file for a given image
##
htrsh_pagexml_create () {
  local FN="htrsh_pagexml_create";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Creates an empty Page file for a given image";
      echo "Usage: $FN IMAGE [pagereg]";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local IMG="$1";
  local SIZE=( $( identify -format "%w %h" "$IMG" 2>/dev/null ) );
  local DATE=$( date -u "+%Y-%m-%dT%H:%M:%S" );

  if [ ! -e "$IMG" ]; then
    echo "$FN: error: file not found: $IMG" 1>&2;
    return 1;
  elif [ "${#SIZE[@]}" != 2 ]; then
    echo "$FN: error: unable to determine size of image: $IMG" 1>&2;
    return 1;
  fi

  echo '<?xml version="1.0" encoding="utf-8"?>';
  echo '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">';
  echo "  <Metadata>";
  echo "    <Creator>htrsh_pagexml_create</Creator>";
  echo "    <Created>$DATE</Created>";
  echo "    <LastChange>$DATE</LastChange>";
  echo "  </Metadata>";
  echo "  <Page imageFilename=\"$IMG\" imageHeight=\"${SIZE[1]}\" imageWidth=\"${SIZE[0]}\">";
  if [ "$#" -gt 1 ] && [ "$2" = "pagereg" ]; then
    local X=$((SIZE[0]-1));
    local Y=$((SIZE[1]-1));
    echo '    <TextRegion id="r1">';
    echo "      <Coords points=\"0,0 $X,0 $X,$Y 0,$Y\"/>";
    echo '    </TextRegion>';
  fi
  echo "  </Page>";
  echo "</PcGts>";
}

##
## Function that creates an empty Page file for a given image
##
htrsh_pagexml_createmulti () {
  local FN="htrsh_pagexml_createmulti";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Creates an empty Page XML file for given images";
      echo "Usage: $FN IMAGE1 [IMAGE2 ...]";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local DATE=$( date -u "+%Y-%m-%dT%H:%M:%S" );

  local IMG="$1";
  local SIZES=( $( identify -format "%wx%h\n" "$@" ) );

  if [ "${#SIZES[@]}" != "$#" ]; then
    echo "$FN: error: unable to determine size of an image" 1>&2;
    return 1;
  fi

  echo '<?xml version="1.0" encoding="utf-8"?>';
  echo '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">';
  echo "  <Metadata>";
  echo "    <Creator>htrsh_pagexml_create</Creator>";
  echo "    <Created>$DATE</Created>";
  echo "    <LastChange>$DATE</LastChange>";
  echo "  </Metadata>";
  local n=0;
  while [ $# -gt 0 ]; do
    local SIZE=( ${SIZES[$n]/x/ } );
    echo "  <Page imageFilename=\"$1\" imageHeight=\"${SIZE[1]}\" imageWidth=\"${SIZE[0]}\"/>";
    n=$((n+1));
    shift;
  done
  echo "</PcGts>";
}

##
## Function that adds a region that covers the whole page area
##
htrsh_pagexml_addpagereg () {
  local FN="htrsh_pagexml_addpagereg";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Adds a region that covers the whole page area";
      echo "Usage: $FN XML";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";

  local xmledit=( xmlstarlet ed );

  for n in $(seq 1 $(xmlstarlet sel -t -v 'count(//_:Page)' "$1")); do
    local X=$(($(xmlstarlet sel -t -v "(//_:Page)[$n]/@imageWidth" "$1")-1));
    local Y=$(($(xmlstarlet sel -t -v "(//_:Page)[$n]/@imageHeight" "$1")-1));
    xmledit+=( -s "//_:Page[$n]" -t elem -n TMPNODE );
    xmledit+=( -i //TMPNODE -t attr -n id -v "page$n" );
    xmledit+=( -s //TMPNODE -t elem -n Coords );
    xmledit+=( -i //TMPNODE/Coords -t attr -n points -v "0,0 $X,0 $X,$Y 0,$Y" );
    xmledit+=( -r //TMPNODE -v TextRegion );
  done

  "${xmledit[@]}" "$XML";
}

##
## Function that creates a region for the whole page, moves all text lines to it and removes all other regions
##
htrsh_pagexml_to_single_region () {
  local FN="htrsh_pagexml_to_single_region";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Creates a region for the whole page, moves all text lines to it and removes all other regions";
      echo "Usage: $FN XML [regid]";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local RID="r1"; [ "$#" -gt 1 ] && [ "$2" != "" ] && RID="$2";
  local SIZE=( $( xmlstarlet sel -t -v //@imageWidth -o " " -v //@imageHeight "$XML" ) );
  local X=$((SIZE[0]-1));
  local Y=$((SIZE[1]-1));

  local xmledit=( xmlstarlet ed );
  xmledit+=( -m //_:TextLine //_:Page );
  xmledit+=( -d //_:TextRegion );
  xmledit+=( -s //_:Page -t elem -n TMPNODE );
  xmledit+=( -i //TMPNODE -t attr -n id -v "$RID" );
  xmledit+=( -s //TMPNODE -t elem -n Coords );
  xmledit+=( -i //TMPNODE/Coords -t attr -n points -v "0,0 $X,0 $X,$Y 0,$Y" );
  xmledit+=( -m //_:TextLine //TMPNODE );
  xmledit+=( -r //TMPNODE -v TextRegion );

  "${xmledit[@]}" "$XML";
}

##
## Function that applies a sed script on TextEquiv/Unicode nodes in a Page XML
##
htrsh_pagexml_sed_textequiv () {
  local FN="htrsh_pagexml_sed_textequiv";
  local XPATH="//*[$htrsh_xpath_textequiv]";
  local SEDOP="-r";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Applies a sed script on TextEquiv/Unicode nodes in a Page XML";
      echo "Usage: $FN XML SCRIPT [ Options ]";
      echo "Options:";
      echo " -x XPATH    Selector of elements for processing (def.=$XPATH)";
      echo " -o SEDOPS   Options for sed (def.=$SEDOP)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local SED="$2";
  shift 2;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-x" ]; then
      XPATH="$2";
    elif [ "$1" = "-o" ]; then
      SEDOP="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  ### Apply sed script to selection of TextEquivs ###
  local updatetext=();
  for tid in $( xmlstarlet sel -t -m "$XPATH" -o " " -v @id "$XML" ); do
    local text1=$( xmlstarlet sel -T -B -E utf-8 -t -v "//*[@id='$tid']/_:TextEquiv/_:Unicode" "$XML" );
    local text2=$( printf "%s" "$text1" | sed $SEDOP "$SED" );
    [ "$text2" != "" ] && [ "$text2" != "$text1" ] &&
      updatetext+=( "$tid" "$text2" );
  done
  if [ "${#updatetext[@]}" -gt 0 ]; then
    htrsh_pagexml_set_textequiv "$XML" "${updatetext[@]}";
  else
    cat "$XML";
  fi
}


##
## Function that sets TextEquiv/Unicode in a Page XML
##
htrsh_pagexml_set_textequiv () {
  local FN="htrsh_pagexml_set_textequiv";
  local INPLACE="";
  local CONF="no";
  if [ $# -lt 3 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Sets TextEquiv/Unicode in an XML Page";
      echo "Usage: $FN [--inplace --conf] XML ID TEXT [CONF] [ ID2 TEXT2 [CONF2] ... ]";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  while [ "${1:0:2}" = "--" ]; do
    if [ "$1" = "--inplace" ]; then
      INPLACE="$1";
    elif [ "$1" = "--conf" ]; then
      CONF="yes";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift;
  done

  local XML="$1";
  shift;

  ### Check XML file ###
  local $htrsh_infovars;
  htrsh_pageimg_info "$XML" noimg;
  [ "$?" != 0 ] && return 1;

  local ids=();
  local idmatch=( xmlstarlet sel -t );
  local xmledit=( xmlstarlet ed $INPLACE );

  while [ $# -gt 0 ]; do
    local text=$( echo "$2" | sed 's|&|\&amp;|g; s|<|\&lt;|g; s|>|\&gt;|g;' );
    ids+=( "$1" );
    idmatch+=( -m "//*[@id='$1']" -v @id -n );
    xmledit+=( -d "//*[@id='$1']/_:TextEquiv" );
    xmledit+=( -s "//*[@id='$1']" -t elem -n TMPNODE );
    xmledit+=( -s //TMPNODE -t elem -n Unicode -v "$text" );
    if [ "$CONF" = "yes" ]; then
      xmledit+=( -i //TMPNODE -t attr -n conf -v "$3" );
      shift;
    fi
    xmledit+=( -r //TMPNODE -v TextEquiv );
    shift 2;
  done

  ids=$( { printf "%s\n" "${ids[@]}"; "${idmatch[@]}" "$XML"; } \
           | sort | uniq -u | tr '\n' ',' );
  [ "$ids" != "" ] &&
    echo "$FN: error: some IDs not found ($ids): $XML" 1>&2 &&
    return 1;

  "${xmledit[@]}" "$XML";
}

##
## Function that propagates text in lines to the corresponding region
##
htrsh_pagexml_textequiv_lines2region () {
  local FN="htrsh_pagexml_textequiv_lines2region";
  local FILTER="cat";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Propagates text in lines to the corresponding region";
      echo "Usage: $FN XML [ Options ]";
      echo "Options:";
      echo " -F FILTER    Filtering pipe command, e.g. join broken words, etc. (def.=none)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  shift;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-F" ]; then
      FILTER="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  local TEXT=$( htrsh_pagexml_textequiv "$XML" -s region-lines -f tab \
    | sed -r 's|^[^ ]+*\.([^. ]+) |\1 |' );

  local updatetext=();
  for regid in $( echo "$TEXT" | sed 's| .*||' | sort -u ); do
    local text=$( echo "$TEXT" | sed -n "/^$regid /{ s|^[^ ]* ||; p; }" | "$FILTER" );
    updatetext+=( "$regid" "$text" );
  done
  htrsh_pagexml_set_textequiv --inplace "$XML" "${updatetext[@]}";
}

##
## Function that propagates text in words to the corresponding line
##
htrsh_pagexml_textequiv_words2line () {
  local FN="htrsh_pagexml_textequiv_words2line";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Propagates text in words to the corresponding line";
      echo "Usage: $FN XML";
    } 1>&2;
    return 1;
  fi

  local XML="$1";
  local TEXT=$( htrsh_pagexml_textequiv "$XML" -s line-words -f tab \
    | sed 's|^[^ ]*\.\([^. ]*\) |\1 |' );

  local updatetext=();
  for lineid in $( echo "$TEXT" | sed 's| .*||' | sort -u ); do
    local text=$( echo "$TEXT" | sed -n "/^$lineid /{ s|^[^ ]* ||; p; }" );
    updatetext+=( "$lineid" "$text" );
  done
  htrsh_pagexml_set_textequiv --inplace "$XML" "${updatetext[@]}";
}

##
## Function that fixes the line ID base of the word IDs
##
htrsh_pagexml_wordid_fix () {
  local FN="htrsh_pagexml_wordid_fix";
  if [ $# != 0 ]; then
    { echo "$FN: Error: Incorrect input arguments";
      echo "Description: Fixes the line ID base of the word IDs";
      echo "Usage: $FN < XML";
    } 1>&2;
    return 1;
  fi

  local XML=$(cat);
  local xmledit=( -d //@dummyattr
    $( xmlstarlet sel -t -m "//$htrsh_xpath_words" -v ../@id -o ' ' -v @id -n <( echo "$XML" ) \
         | gawk '
             { split( $2, wid, "_w" );
               if( $1 != wid[1] )
                 printf( " -u //_:Word[@id=\"%s\"]/@id -v %s_w%s", $2, $1, wid[2] );
             }' ) );

  xmlstarlet ed "${xmledit[@]}" <( echo "$XML" );
}

##
## Function that prints to stdout the TextEquiv from an XML Page file
##
# @todo hmms for different space types: word extremes, between numbers, etc.
htrsh_pagexml_textequiv () {
  local FN="htrsh_pagexml_textequiv";
  local SRC="lines";
  local FORMAT="raw";
  local FILTER="cat";
  local BASEXPATH="";
  local ESPACES="yes";
  local WSPACE="no";
  local WORDEND="no";
  local PRTHEAD="no";
  local PREPRINT="";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Prints to stdout the TextEquiv from an XML Page file";
      echo "Usage: $FN XMLFILE [ Options ]";
      echo "Options:";
      echo " -s SOURCE    Source of TextEquiv, either 'regions', 'lines' or 'words' (def.=$SRC)";
      echo " -f FORMAT    Output format among 'raw', 'mlf-chars', 'mlf-words', 'tab' and 'tab-chars' (def.=$FORMAT)";
      echo " -F FILTER    Filtering pipe command, e.g. tokenizer, transliteration, etc. (def.=none)";
      echo " -B XBASE     xpath to get the sample base name (def.=use the image basename)";
      echo " -E (yes|no)  For *-chars, whether to add spaces at start and end (def.=$ESPACES)";
      echo " -W (yes|no)  For mlf-words, whether to add start space (def.=$WSPACE)";
      echo " -w (yes|no)  For mlf-chars, whether to add word end marks (def.=$WORDEND)";
      echo " -H (yes|no)  Whether to print header (def.=$PRTHEAD)";
      echo " -p ARRAY     Pre-print xmlstarlet arguments array name (def.=$PREPRINT)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  shift;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-s" ]; then
      SRC="$2";
    elif [ "$1" = "-f" ]; then
      FORMAT="$2";
      if ! ( [ "$FORMAT" = 'raw' ] ||
             [ "$FORMAT" = 'mlf-chars' ] || [ "$FORMAT" = 'mlf-words' ] ||
             [ "$FORMAT" = 'tab' ] || [ "$FORMAT" = 'tab-chars' ] ); then
        echo "$FN: error: unexpected output format: $FORMAT" 1>&2;
        return 1;
      fi
    elif [ "$1" = "-F" ] || [ "$1" = "-D" ]; then
      FILTER="$2";
    elif [ "$1" = "-B" ]; then
      BASEXPATH="$2";
    elif [ "$1" = "-E" ]; then
      ESPACES="$2";
    elif [ "$1" = "-W" ]; then
      WSPACE="$2";
    elif [ "$1" = "-w" ]; then
      WORDEND="$2";
    elif [ "$1" = "-H" ]; then
      PRTHEAD="$2";
    elif [ "$1" = "-p" ]; then
      [ "$2" = "PREPRINT" ] &&
        echo "$FN: error: pre print array cannot be called PREPRINT" 1>&2 &&
        return 1;
      PREPRINT="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  ### Check page ###
  htrsh_pageimg_info "$XML" noinfo;
  [ "$?" != 0 ] && return 1;

  local idxmledit;
  local IDop;
  if [ "$BASEXPATH" != "" ]; then
    idxmledit=( cat "$XML" );
    IDop=( -v "$BASEXPATH" -o . );
  else
    idxmledit=( xmlstarlet ed 
                $(xmlstarlet sel -t -v //@imageFilename -n "$XML" \
                    | sed 's|.*/||; s|\.[^.]*$||; s|[\[ ()]|_|g; s|]|_|g;' \
                    | awk '{printf(" -u (//@imageFilename)[%d] -v %s",NR,$0);}' )
                "$XML" );
    IDop=( -v ancestor::_:Page/@imageFilename -o . );
  fi
  local XPATH;
  local PRINT=( -v . -n );
  [ "$PREPRINT" != "" ] &&
    eval "PRINT=( \"\${${PREPRINT}[@]}\" \"\${PRINT[@]}\" )";
  if [ "$SRC" = "regions" ]; then
    XPATH="$htrsh_xpath_regions/$htrsh_xpath_textequiv";
    IDop+=( -v ../../@id );
  elif [ "$SRC" = "line-words" ]; then
    XPATH="$htrsh_xpath_regions/$htrsh_xpath_lines[$htrsh_xpath_words/$htrsh_xpath_textequiv]";
    PRINT=( -m "$htrsh_xpath_words/$htrsh_xpath_textequiv" -o " " -v . -b -n );
    [ "$PREPRINT" != "" ] &&
      eval "PRINT=( -m \"\$htrsh_xpath_words/\$htrsh_xpath_textequiv\" \"\${${PREPRINT}[@]}\" -v . -b -n )";
    [ "$htrsh_extended_names" = "true" ] && IDop+=( -v ../@id -o . -v @id );
    [ "$htrsh_extended_names" != "true" ] && IDop+=( -v @id );
  elif [ "$SRC" = "solo-words" ]; then
    XPATH="$htrsh_xpath_regions/$htrsh_xpath_lines/$htrsh_xpath_words/$htrsh_xpath_textequiv";
    [ "$htrsh_extended_names" = "true" ] && IDop+=( -v ../../../../@id -o . -v ../../../@id -o . -v ../../@id );
    [ "$htrsh_extended_names" != "true" ] && IDop+=( -v ../../@id );
  elif [ "$SRC" = "lines" ]; then
    XPATH="$htrsh_xpath_regions/$htrsh_xpath_lines/$htrsh_xpath_textequiv";
    [ "$htrsh_extended_names" = "true" ] && IDop+=( -v ../../../@id -o . -v ../../@id );
    [ "$htrsh_extended_names" != "true" ] && IDop+=( -v ../../@id );
  elif [ "$SRC" = "region-lines" ]; then
    XPATH="$htrsh_xpath_regions[$htrsh_xpath_lines/$htrsh_xpath_textequiv]";
    PRINT=( -m "$htrsh_xpath_lines/$htrsh_xpath_textequiv" -o " " -v . -b -n );
    [ "$htrsh_extended_names" = "true" ] && IDop+=( -v ../@id -o . -v @id );
    [ "$htrsh_extended_names" != "true" ] && IDop+=( -v @id );
  elif [ "$SRC" = "all" ]; then
    XPATH="//$htrsh_xpath_textequiv";
    IDop+=( -v ../../@id );
  else
    echo "$FN: error: unexpected source type: $SRC" 1>&2;
    return 1;
  fi

  [ $(xmlstarlet sel -t -v "count($XPATH)" "$XML") = 0 ] &&
    echo "$FN: warning: zero matches for xpath $XPATH on file: $XML" 1>&2 &&
    return 0;

  [ "$PRTHEAD" = "yes" ] && [ "${FORMAT:0:3}" = "mlf" ] &&
    echo '#!MLF!#';

  paste \
      <( "${idxmledit[@]}" \
           | xmlstarlet sel -t -m "$XPATH" "${IDop[@]}" -n - ) \
      <( cat "$XML" \
           | tr '\t\n' '  ' \
           | xmlstarlet sel -T -B -E utf-8 -t -m "$XPATH" "${PRINT[@]}" \
           | perl -CSD -e 'use Unicode::Normalize; while ($line = <STDIN>){ print NFC($line) }' \
           | $FILTER ) \
    | sed '
        s|\t  *|\t|;
        s|  *$||;
        s|   *| |g;
        ' \
    | gawk -v FORMAT=$FORMAT -v hmmtype="$htrsh_hmm_type" \
           -v ESPACES="$ESPACES" -v WSPACE="$WSPACE" -v WORDEND="$WORDEND" \
           -v SPACE="$htrsh_symb_space" -v SPECIAL=<( echo "$htrsh_special_chars" ) \
        "$htrsh_gawk_func_word_to_chars"'
        BEGIN {
          load_special_chars( SPECIAL );
          FS = "\t";
          if( FORMAT == "tab" )
            OFS=" ";
        }
        { if( FORMAT == "raw" )
            print $2;
          else if( FORMAT == "tab" )
            print $1,$2;
          else if( FORMAT == "mlf-words" ) {
            printf("\"*/%s.lab\"\n",$1);
            if( WSPACE == "yes" )
              printf("\"%s\"\n",SPACE);
            gsub("\x22","\\\x22",$2);
            N = split($2,txt," ");
            for( n=1; n<=N; n++ )
              printf( "\"%s\"\n", txt[n] );
            printf(".\n");
          }
          else if( FORMAT == "tab-chars" ) {
            printf("%s",$1);
            if( ESPACES == "yes" )
              printf(" %s",SPACE);
            M = split( $2, words, " " );
            for( m=1; m<=M; m++ ) {
              N = word_to_chars( words[m], hmms, hmmtype, "no" );
              for( n=1; n<=N; n++ )
                printf( " %s", hmms[n] );
              if( WORDEND == "yes" )
                printf( " {wordend}" );
              if( m < M || ESPACES == "yes" )
                printf(" %s",SPACE);
            }
            printf("\n");
          }
          else if( FORMAT == "mlf-chars" ) {
            printf("\"*/%s.lab\"\n",$1);
            if( ESPACES == "yes" )
              printf("%s\n",SPACE);
            M = split( $2, words, " " );
            for( m=1; m<=M; m++ ) {
              N = word_to_chars( words[m], hmms, hmmtype, "no" );
              for( n=1; n<=N; n++ )
                printf( ( match(hmms[n],/^[.0-9]/) ? "\"%s\"\n" : "%s\n" ), hmms[n] );
              if( WORDEND == "yes" )
                printf( "{wordend}\n" );
              if( m < M || ESPACES == "yes" )
                printf("%s\n",SPACE);
            }
            printf(".\n");
          }
        }';
}

##
## Function that filters the TextEquiv in an XML Page file
##
htrsh_pagexml_textequiv_filter () {
  local FN="htrsh_pagexml_textequiv_filter";
  if [ $# != 2 ]; then
    { echo "$FN: Error: Incorrect input arguments";
      echo "Description: Filters the TextEquiv in an XML Page file";
      echo "Usage: $FN XMLFILE FILTER";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local FILTER="$2";

  type -t "$FILTER" >/dev/null;
  [ "$?" != 0 ] &&
    echo "$FN: error: filter not defined: $FILTER" 1>&2 &&
    return 1;

  ### Check page ###
  htrsh_pageimg_info "$XML" noinfo;
  local TEXT=$( htrsh_pagexml_textequiv "$XML" -s all -f tab | sed -r 's|^[^ ]+\.([^. ]+ )|\1|' );
  [ "$?" != 0 ] && return 1;

  local updatetext=();
  local N=$( echo "$TEXT" | wc -l );
  for n in $(seq 1 $N); do
    local text=$( echo "$TEXT" | sed -n ${n}p );
    local tid=$( echo "$text" | sed 's| .*||' );
    text=$( echo "$text" | sed 's|^[^ ]* ||' | "$FILTER" );
    updatetext+=( "$tid" "$text" );
  done
  htrsh_pagexml_set_textequiv --inplace "$XML" "${updatetext[@]}";
}

##
## Function that transforms plain text to character sequences
##
htrsh_text_to_chars () {
  local FN="htrsh_text_to_chars";
  local FORMAT="txt";
  local FILTER="cat";
  local ESPACES="yes";
  local WORDEND="no";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Transforms plain text to character sequences";
      echo "Usage: $FN TEXTFILE [ Options ]";
      echo "Options:";
      echo " -f FORMAT    Input format among 'txt' and 'tab' (def.=$FORMAT)";
      echo " -F FILTER    Filtering pipe command, e.g. tokenizer, transliteration, etc. (def.=none)";
      echo " -E (yes|no)  For *-chars, whether to add spaces at start and end (def.=$ESPACES)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local TEXT="$1"; [ "$TEXT" = "-" ] && TEXT="/dev/stdin";
  shift;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-f" ]; then
      FORMAT="$2";
    elif [ "$1" = "-F" ]; then
      FILTER="$2";
    elif [ "$1" = "-E" ]; then
      ESPACES="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  ### Process text ###
  text_to_chars () {
    gawk -v hmmtype="$htrsh_hmm_type" \
         -v ESPACES="$ESPACES" -v WORDEND="$WORDEND" \
         -v SPACE="$htrsh_symb_space" -v SPECIAL=<( echo "$htrsh_special_chars" ) \
      "$htrsh_gawk_func_word_to_chars"'
      BEGIN {
        load_special_chars( SPECIAL );
      }
      { SEP="";
        if( ESPACES == "yes" ) {
          printf("%s",SPACE);
          SEP=" ";
        }
        M = split( $0, words, " " );
        for( m=1; m<=M; m++ ) {
          N = word_to_chars( words[m], hmms, hmmtype, "no" );
          for( n=1; n<=N; n++ ) {
            printf( "%s%s", SEP, hmms[n] );
            SEP=" ";
          }
          if( WORDEND == "yes" )
            printf( " {wordend}" );
          if( m < M || ESPACES == "yes" )
            printf(" %s",SPACE);
        }
        printf("\n");
      }' "$@";
  }
  if [ "$FORMAT" = "tab" ]; then
    local TMP_TEXT=$(mktemp).txt;
    cat "$TEXT" > "$TMP_TEXT";
    paste -d " " \
      <( awk '{print $1}' "$TMP_TEXT" ) \
      <( sed 's|^[^ ]* *||' "$TMP_TEXT" | "$FILTER" | text_to_chars )
    rm "$TMP_TEXT";
  else
    "$FILTER" < "$TEXT" | text_to_chars;
  fi
}

##
## Function that generates a kaldi table of symbols for some given text
##
htrsh_text_get_symbol_tab () {
  local FN="htrsh_text_get_symbol_tab";
  local COUNT_FILE="cat";
  local FORMAT="text";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Generates a kaldi table of symbols for some given text";
      echo "Usage: $FN TRANSCIPT_TAB [ Options ]";
      echo "Options:";
      echo " -f FORMAT       Input format tab or text (def.=$FORMAT)";
      echo " -c COUNT_FILE   Save counts of symbols to given file (def.=false)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local TEXT="$1"; [ "$TEXT" = "-" ] && TEXT="/dev/stdin";
  shift;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-c" ]; then
      COUNT_FILE="$2";
    elif [ "$1" = "-f" ]; then
      FORMAT="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  [ "$COUNT_FILE" != "cat" ] && COUNT_FILE=( tee "$COUNT_FILE" );

  local n0="1"; [ "$FORMAT" = "tab" ] && n0="2";

  awk -v n0=$n0 '
      { for(n=n0;n<=NF;n++) char[$n]++; }
      END {
        for(c in char)
          printf("%d %s\n",char[c],c);
      }' "$TEXT" \
    | sort -k 1rn,1 \
    | "${COUNT_FILE[@]}" \
    | awk -v EPS="$htrsh_symb_eps" -v BLANK="$htrsh_symb_blank" -v N=1 '
        BEGIN {
          printf("%s 0\n",EPS);
          printf("%s 1\n",BLANK);
        }
        { if( !($NF in char) )
            printf("%s %d\n",$NF,++N);
          char[$NF]="";
        }';
}

##
## Function that concatenates symbols from consecutive samples having the same ID
##
htrsh_tab_concat () {
  local FN="htrsh_tab_concat";
  if [ $# -gt 0 ]; then
    { echo "$FN: Error: Incorrect number of input arguments";
      echo "Description: Concatenates symbols from consecutive samples having the same ID";
      echo "Usage: $FN < TAB";
    } 1>&2;
    return 1;
  fi

  gawk '
    { if ( $1 != PREV && VAL != "" ) {
        print(PREV" "VAL);
        VAL = "";
      }
      SAMP = $1;
      $1 = "";
      VAL = SAMP != PREV ? $0 : (VAL" "$0);
      PREV = SAMP;
    }
    END {
      if ( VAL != "" )
        print(PREV" "VAL);
    }' \
    | sed 's|   *| |g';
}

##
## Function that transforms a word MLF to character sequences using given dictionary
##
htrsh_mlf_word_to_chars () {
  local FN="htrsh_mlf_word_to_chars";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Transforms a word MLF to an hmm sequence using given dictionary";
      echo "Usage: $FN MLF DIC";
    } 1>&2;
    return 1;
  fi

  gawk -v RETVAL=0 '
    { w = $1;
      if( ARGIND == 1 ) {
        if( match(w,/^".+"$/) )
          w = gensub( /\\"/, "\"", "g", substr(w,2,length(w)-2) );
        w = gensub( /\\\x27/, "\x27", "g", w );
        if( ! ( w in dic ) )
          dic[w] = $0;
      }
      else {
        if( $0 == "#!MLF!#" || $0 == "." || match($0,/^".+\/.+\.[lr][ae][bc]"$/) )
          print;
        else {
          if( match(w,/^".+"$/) )
            w = gensub( /\\"/, "\"", "g", substr(w,2,length(w)-2) );
          if( ! ( w in dic ) ) {
            printf( "'"$FN"': error: word not in dictionary: %s\n", w ) > "/dev/stderr";
            RETVAL="1";
            #exit 1;
          }
          N = split( dic[w], chars );
          for( n=4; n<=N; n++ )
            if( match(chars[n],/^\.$/) || match(chars[n],/^[0-9]/) )
              printf( "\"%s\"\n", chars[n] );
            else
              print chars[n];
        }
      }
    }
    END {
      exit RETVAL;
    }' "$2" "$1";
}

##
## Function that transforms a lab/rec MLF to kaldi table format
##
htrsh_mlf_to_tab () {
  local FN="htrsh_mlf_to_tab";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Transforms a lab/rec MLF to kaldi table format";
      echo "Usage: $FN MLF";
    } 1>&2;
    return 1;
  fi

  gawk -v PROPERMLF="no" -v FN="$FN" '
    { if( $0 == "#!MLF!#" )
        PROPERMLF = "yes";
      else if( match($1,/^".+\/(.+)\.[lr][ae][bc]"$/) ) {
        printf( "%s", gensub( /^".+\/(.+)\.[lr][ae][bc]"$/, "\\1", 1, $1 ) );
        PROPERMLF = "yes";
      }
      else if( PROPERMLF == "no" ) {
        printf("%s: error: input file does not seem to be in MLF format\n",FN) > "/dev/stderr";
        exit(1);
      }
      else if( $0 == "." ) {
        PROPERMLF = "no";
        printf("\n");
      }
      else {
        if( NF > 1 )
          $1 = $3;
        if( match($1,/^".+"$/) )
          $1 = gensub( /\\"/, "\"", "g", substr($1,2,length($1)-2) );
        printf(" %s",$1);
      }
    }' "$1";
}

##
## Function that transforms a ground truth and a recognition file to the format used by tasas
##
htrsh_prep_tasas () {
  local FN="htrsh_prep_tasas";
  local FORMAT="mlf";
  local SEPCHARS="no";
  local TOKENIZER="";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Transforms a ground truth and a recognition file to the format used by tasas";
      echo "Usage: $FN GT REC [ Options ]";
      echo "Options:";
      echo " -f FORMAT    Input format, either 'mlf' or 'tab' (def.=$FORMAT)";
      echo " -c (yes|no)  Whether to separate characters for CER computation (def.=$SEPCHARS)";
      echo " -t TOKENIZER A pipe command to tokenize (def.=$TOKENIZER)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local GT="$1";
  local REC="$2";
  shift 2;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-f" ]; then
      FORMAT="$2";
    elif [ "$1" = "-c" ]; then
      SEPCHARS="$2";
    elif [ "$1" = "-t" ]; then
      TOKENIZER="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  if [ ! -e "$GT" ]; then
    echo "$FN: error: ground truth file not found: $GT" 1>&2;
    return 1;
  elif [ ! -e "$REC" ]; then
    echo "$FN: error: recognition file not found: $REC" 1>&2;
    return 1;
  fi

  if [ "$FORMAT" = "mlf" ]; then
    FORMAT="htrsh_mlf_to_tab";
  else
    FORMAT="cat";
  fi

  if [ "$TOKENIZER" = "" ]; then
    format_data () { $FORMAT "$1"; }
  else
    format_data () {
      paste -d " " \
        <( $FORMAT "$1" | sed 's| .*||' ) \
        <( $FORMAT "$1" | sed 's|^[^ ]* ||' | $TOKENIZER );
    }
  fi

  ### Create tasas file ###
  gawk -v SEPCHARS="$SEPCHARS" -v SPACE="$htrsh_symb_space" -v SPECIAL=<( echo "$htrsh_special_chars" ) \
    "$htrsh_gawk_func_word_to_chars"'
    BEGIN {
      load_special_chars( SPECIAL );
    }
    { if( ARGIND == 1 ) {
        if( $1 in GT )
          printf( "warning: duplicate ground truth %s\n", $1 ) > "/dev/stderr";
        else
          GT[$1] = $0;
      }
      else if( !( $1 in GT ) )
        printf( "warning: no ground truth for %s\n", $1 ) > "/dev/stderr";
      else {
        NWORDS = split( GT[$1], gt );
        for( w=2; w<=NWORDS; w++ ) {
          if( SEPCHARS == "no" )
            printf( w==2 ? "%s" : " %s", gt[w] );
          else {
            NCHARS = word_to_chars( gt[w], chars, "char", "no" );
            printf( w==2 ? "%s" : (" "SPACE" %s"), chars[1] );
            for( c=2; c<=NCHARS; c++ )
              printf( " %s", chars[c] );
          }
        }
        printf("|");
        for( w=2; w<=NF; w++ ) {
          if( SEPCHARS == "no" )
            printf( w==2 ? "%s" : " %s", $w );
          else {
            NCHARS = word_to_chars( $w, chars, "char", "no" );
            printf( w==2 ? "%s" : (" "SPACE" %s"), chars[1] );
            for( c=2; c<=NCHARS; c++ )
              printf( " %s", chars[c] );
          }
        }
        printf("\n");
      }
    }' <( format_data "$GT" ) <( format_data "$REC" );
}

##
## Function that checks and extracts basic info (XMLDIR, IMDIR, IMFILE, XMLBASE, IMBASE, IMEXT, IMSIZE, IMRES, RESSRC) from an XML Page file and respective image
##
htrsh_pageimg_info () {
  local FN="htrsh_pageimg_info";
  local XML="$1";
  #local VAL=( xmlstarlet val -q -e ); [ "$htrsh_valschema" = "yes" ] && VAL+=( -s "$htrsh_pagexsd" );
  local VAL=( xmllint --noout ); [ "$htrsh_valschema" = "yes" ] && VAL+=( --schema "$htrsh_pagexsd" );
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Checks and extracts basic info (XMLDIR, IMDIR, IMFILE, XMLBASE, IMBASE, IMEXT, IMSIZE, IMRES, RESSRC) from an XML Page file and respective image";
      echo "Usage: $FN XMLFILE";
    } 1>&2;
    return 1;
  elif [ ! -f "$XML" ]; then
    echo "$FN: error: page file not found: $XML" 1>&2;
    return 1;
  elif [ "$htrsh_valschema" = "yes" ] && [ ! -f "$htrsh_pagexsd" ]; then
    echo "$FN: error: unable to find schema file for validation: $htrsh_pagexsd";
    return 1;
  elif [ $("${VAL[@]}" "$XML" 2>/dev/null; echo "$?") != 0 ]; then
    echo "$FN: error: invalid page file: $XML" 1>&2;
    return 1;
  elif [ $(xmlstarlet sel -t -m '//*/@id' -v . -n "$XML" 2>/dev/null | sort | uniq -d | wc -l) != 0 ]; then
    echo "$FN: error: page file has duplicate IDs: $XML" 1>&2;
    return 1;
  fi

  if [ $# -eq 1 ] || [ "$2" != "noinfo" ]; then
    XMLDIR=$($htrsh_realpath $(dirname "$XML"));
    IMFILE="$XMLDIR/"$(xmlstarlet sel -t -v //@imageFilename "$XML");

    IMDIR=$($htrsh_realpath $(dirname "$IMFILE"));
    XMLBASE=$(echo "$XML" | sed 's|.*/||; s|\.[xX][mM][lL]$||;');
    IMBASE=$(echo "$IMFILE" | sed 's|.*/||; s|\.[^.]*$||;');
    IMEXT=$(echo "$IMFILE" | sed 's|.*\.||');

    if [ $# -eq 1 ] || [ "$2" != "noimg" ]; then
      local XMLSIZE=$(xmlstarlet sel -t -v //@imageWidth -o x -v //@imageHeight "$XML");
      IMSIZE=$(identify -format %wx%h "$IMFILE" 2>/dev/null);

      [ ! -f "$IMFILE" ] &&
        echo "$FN: error: image file not found: $IMFILE" 1>&2 &&
        return 1;
      [ "$IMSIZE" != "$XMLSIZE" ] &&
        echo "$FN: warning: image size discrepancy: image=$IMSIZE page=$XMLSIZE" 1>&2;

      RESSRC="xml";
      IMRES=$(xmlstarlet sel -t -v //_:Page/@custom "$XML" 2>/dev/null \
                | awk -F'[{}:; ]+' '
                    { for( n=1; n<=NF; n++ )
                        if( $n == "image-resolution" ) {
                          n++;
                          if( match($n,"dpcm") )
                            printf("%g",$n);
                          else if( match($n,"dpi") )
                            printf("%g",$n/2.54);
                        }
                    }');

      [ "$IMRES" = "" ] &&
      RESSRC="img" &&
      IMRES=$(
        identify -format "%x %y %U" "$IMFILE" \
          | awk '
              { if( NF > 3 ) {
                  $2 = $3;
                  $3 = $4;
                }
                if( $3 == "PixelsPerCentimeter" )
                  printf("%sx%s",$1,$2);
                else if( $3 == "PixelsPerInch" )
                  printf("%gx%g",$1/2.54,$2/2.54);
              }'
        );

      [ "$htrsh_warn_imgres" = "yes" ] &&
      if [ "$IMRES" = "" ]; then
        echo "$FN: warning: no resolution metadata for image: $IMFILE";
      elif [ $(echo "$IMRES" | sed 's|.*x||') != $(echo "$IMRES" | sed 's|x.*||') ]; then
        echo "$FN: warning: image resolution different for vertical and horizontal: $IMFILE";
      fi 1>&2

      IMRES=$(echo "$IMRES" | sed 's|x.*||');
    fi
  fi

  return 0;
}

##
## Function that resizes an XML Page file along with its corresponding image
##
htrsh_pageimg_resize () {
  local FN="htrsh_pageimg_resize";
  local INRES="";
  local OUTRES="118";
  local INRESCHECK="yes";
  local SFACT="";
  local OEXT="";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Resizes an XML Page file along with its corresponding image";
      echo "Usage: $FN XML OUTDIR [ Options ]";
      echo "Options:";
      echo " -i INRES    Input image resolution in ppc (def.=use image metadata)";
      echo " -o OUTRES   Output image resolution in ppc (def.=$OUTRES)";
      echo " -s SFACT    Scaling factor in % (def.=inferred from resolutions)";
      echo " -e OEXT     Output image format extension (def.=same as input)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local OUTDIR="$2";
  shift 2;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-i" ]; then
      INRES="$2";
    elif [ "$1" = "-o" ]; then
      OUTRES="$2";
    elif [ "$1" = "-s" ]; then
      SFACT="$2";
    elif [ "$1" = "-e" ]; then
      OEXT="$2";
    elif [ "$1" = "-c" ]; then
      INRESCHECK="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  ### Check XML file and image ###
  local $htrsh_infovars;
  htrsh_pageimg_info "$XML";
  [ "$?" != 0 ] && return 1;

  if [ "$SFACT" = "" ] && [ "$INRES" = "" ] && [ "$IMRES" = "" ]; then
    echo "$FN: error: resolution not given (-i option) and image does not specify resolution: $IMFILE" 1>&2;
    return 1;
  elif [ "$INRESCHECK" = "yes" ] && [ "$SFACT" = "" ] && [ "$INRES" = "" ] && [ $(echo $IMRES | awk '{printf("%.0f",$1)}') -lt "$htrsh_minres" ]; then
    echo "$FN: error: image resolution ($IMRES ppc) apparently incorrect since it is unusually low to be a text document image: $IMFILE" 1>&2;
    return 1;
  elif [ ! -d "$OUTDIR" ]; then
    echo "$FN: error: output directory does not exists: $OUTDIR" 1>&2;
    return 1;
  elif [ "$XMLDIR" = $($htrsh_realpath "$OUTDIR") ]; then
    echo "$FN: error: output directory has to be different from the one containing the input XML: $XMLDIR" 1>&2;
    return 1;
  fi

  [ "$INRES" = "" ] && INRES="$IMRES";
  [ "$OEXT" = "" ] && OEXT="$IMEXT";

  if [ "$SFACT" = "" ]; then
    SFACT=$(echo $OUTRES $INRES | awk '{printf("%g%%",100*$1/$2)}');
  else
    SFACT=$(echo $SFACT | sed '/%$/!s|$|%|');
    OUTRES=$(echo $SFACT $INRES | awk '{printf("%g",0.01*$1*$2)}');
  fi

  ### Resize image ###
  convert "$IMFILE" -units PixelsPerCentimeter -density $OUTRES -resize $SFACT "$OUTDIR/$IMBASE.$OEXT"; ### don't know why the density has to be set this way

  ### Resize XML Page ###
  # @todo change the sed to XSLT
  htrsh_pagexml_resize $SFACT < "$XML" \
    | xmlstarlet ed -u //@imageFilename -v "$IMBASE.$OEXT" \
    | sed '
        s|\( custom="[^"]*\)image-resolution:[^;]*;\([^"]*"\)|\1\2|;
        s| custom=" *"||;
        ' \
    > "$OUTDIR/$XMLBASE.xml";

  return 0;
}

##
## Function that horizontally inverts a Page XML and its image
##
htrsh_pageimg_flop () {
  local FN="htrsh_pageimg_flop";
  if [ $# != 2 ]; then
    { echo "$FN: Error: Incorrect input arguments";
      echo "Description: Horizontally inverts a Page XML and its image";
      echo "Usage: $FN XML_PAGE_FILE OUTDIR";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local OUTDIR="$2";

  ### Check XML file and image ###
  local $htrsh_infovars;
  htrsh_pageimg_info "$XML";
  [ "$?" != 0 ] && return 1;

  convert "$IMFILE" -flop "$OUTDIR/$IMBASE.$IMEXT";
  htrsh_pagexml_flop < "$XML" > "$OUTDIR/$XMLBASE.xml";
}

##
## Function that horizontally inverts a Page XML file
##
htrsh_pagexml_flop () {
  local FN="htrsh_pagexml_flop";
  if [ $# != 0 ]; then
    { echo "$FN: Error: Incorrect input arguments";
      echo "Description: Horizontally inverts a Page XML file";
      echo "Usage: $FN < XML_PAGE_FILE";
    } 1>&2;
    return 1;
  fi

  local XSLT='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns:str="http://exslt.org/strings"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  extension-element-prefixes="str"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:variable name="width" select="//_:Page/@imageWidth"/>

  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//*[@points]">
    <xsl:copy>
      <xsl:for-each select="@*[local-name() = '"'points'"' or local-name() = '"'fpgram'"']">
      <xsl:attribute name="{local-name()}">
        <xsl:for-each select="str:tokenize(.,'"', '"')">
          <xsl:choose>
            <xsl:when test="position() = 1">
              <xsl:value-of select="number($width)-number(.)"/>
            </xsl:when>
            <xsl:when test="position() mod 2 = 0">
              <xsl:text>,</xsl:text><xsl:value-of select="."/>
            </xsl:when>
            <xsl:otherwise>
              <xsl:text> </xsl:text><xsl:value-of select="number($width)-number(.)"/>
            </xsl:otherwise>
          </xsl:choose>
        </xsl:for-each>
      </xsl:attribute>
      </xsl:for-each>
      <xsl:apply-templates select="@*[local-name() != '"'points'"' and local-name() != '"'fpgram'"'] | node()" />
    </xsl:copy>
  </xsl:template>

</xsl:stylesheet>';

  xmlstarlet tr <( echo "$XSLT" );
}

##
## Function that resizes an XML Page file
##
htrsh_pagexml_resize () {
  local FN="htrsh_pagexml_resize";
  local newWidth newHeight scaleFact;
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Resizes an XML Page file";
      echo "Usage: $FN ( {newWidth}x{newHeight} | {scaleFact}% ) < XML_PAGE_FILE";
    } 1>&2;
    return 1;
  elif [ $(echo "$1" | grep '^[0-9][0-9]*x[0-9][0-9]*$' | wc -l) = 1 ]; then
    newWidth=$(echo "$1" | sed 's|x.*||');
    newHeight=$(echo "$1" | sed 's|.*x||');
  elif [ $(echo "$1" | grep '^[0-9.][0-9.]%$' | wc -l) = 1 ]; then
    scaleFact=$(echo "$1" | sed 's|%$||');
  fi

  local XSLT='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns:str="http://exslt.org/strings"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  extension-element-prefixes="str"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:variable name="oldWidth" select="//_:Page/@imageWidth"/>
  <xsl:variable name="oldHeight" select="//_:Page/@imageHeight"/>';

  if [ "$scaleFact" != "" ]; then
    XSLT="$XSLT"'
  <xsl:variable name="scaleWidth" select="number('${scaleFact}') div 100"/>
  <xsl:variable name="scaleHeight" select="$scaleWidth"/>
  <xsl:variable name="newWidth" select="round($oldWidth*$scaleWidth)"/>
  <xsl:variable name="newHeight" select="round($oldHeight*$scaleHeight)"/>';
  else
    XSLT="$XSLT"'
  <xsl:variable name="newWidth" select="'${newWidth}'"/>
  <xsl:variable name="newHeight" select="'${newHeight}'"/>
  <xsl:variable name="scaleWidth" select="number($newWidth) div number($oldWidth)"/>
  <xsl:variable name="scaleHeight" select="number($newHeight) div number($oldHeight)"/>';
  fi

  XSLT="$XSLT"'
  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//_:Page">
    <xsl:copy>
      <xsl:attribute name="imageWidth">
        <xsl:value-of select="$newWidth"/>
      </xsl:attribute>
      <xsl:attribute name="imageHeight">
        <xsl:value-of select="$newHeight"/>
      </xsl:attribute>
      <xsl:apply-templates select="@*[local-name() != '"'imageWidth'"' and local-name() != '"'imageHeight'"'] | node()" />
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//*[@points]">
    <xsl:copy>
      <xsl:for-each select="@*[local-name() = '"'points'"' or local-name() = '"'fpgram'"']">
      <xsl:attribute name="{local-name()}">
        <xsl:for-each select="str:tokenize(.,'"', '"')">
          <xsl:choose>
            <xsl:when test="position() = 1">
              <xsl:value-of select="number($scaleWidth)*number(.)"/>
            </xsl:when>
            <xsl:when test="position() mod 2 = 0">
              <xsl:text>,</xsl:text><xsl:value-of select="number($scaleHeight)*number(.)"/>
            </xsl:when>
            <xsl:otherwise>
              <xsl:text> </xsl:text><xsl:value-of select="number($scaleWidth)*number(.)"/>
            </xsl:otherwise>
          </xsl:choose>
        </xsl:for-each>
      </xsl:attribute>
      </xsl:for-each>
      <xsl:apply-templates select="@*[local-name() != '"'points'"' and local-name() != '"'fpgram'"'] | node()" />
    </xsl:copy>
  </xsl:template>

</xsl:stylesheet>';

  xmlstarlet tr <( echo "$XSLT" );
}

##
## Function that modifies an XML Page file by rounding coordinates and setting negatives to zero
##
htrsh_pagexml_round () {
  local FN="htrsh_pagexml_round";
  if [ $# != 0 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Modifies an XML Page file by rounding coordinates and setting negatives to zero";
      echo "Usage: $FN < XML_PAGE_FILE";
    } 1>&2;
    return 1;
  fi

  local XSLT='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns:str="http://exslt.org/strings"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  extension-element-prefixes="str"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//*[@points]">
    <xsl:copy>
      <xsl:for-each select="@*[local-name() = '"'points'"' or local-name() = '"'fpgram'"']">
        <xsl:attribute name="{local-name()}">
        <xsl:for-each select="str:tokenize(.,'"', '"')">
          <xsl:choose>
            <xsl:when test="position() mod 2 = 0">
              <xsl:text>,</xsl:text>
            </xsl:when>
            <xsl:when test="position() != 1">
              <xsl:text> </xsl:text>
            </xsl:when>
          </xsl:choose>
          <xsl:choose>
            <xsl:when test="round(number(.)) &lt; 0">
              <xsl:text>0</xsl:text>
            </xsl:when>
            <xsl:otherwise>
              <xsl:value-of select="round(number(.))"/>
            </xsl:otherwise>
          </xsl:choose>
        </xsl:for-each>
        </xsl:attribute>
      </xsl:for-each>
      <xsl:apply-templates select="@*[local-name() != '"'points'"' and local-name() != '"'fpgram'"'] | node()" />
    </xsl:copy>
  </xsl:template>

</xsl:stylesheet>';

  xmlstarlet tr <( echo "$XSLT" );
}

##
## Function that inserts XML Page nodes from an external XML Page
##
htrsh_pagexml_insertfrom () {
  local FN="htrsh_pagexml_insertfrom";
  if [ $# -lt 4 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Inserts XML Page nodes from an external XML Page";
      echo "Usage: $FN FILE_FROM FILE_TO XPATH_FROM XPATH_TO [XPATH_FROM2 XPATH_TO2 ...]";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local FILE_FROM=$(pwd)/$($htrsh_realpath "$1");
  local FILE_TO="$2";
  shift 2;

  ### Create XSLT ###
  local XSLT='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>';

  local XPATHS=("${@//\"/&quot;}"); XPATHS=("${XPATHS[@]//$'\x27'/&apos;}");
  local n="0";
  while [ "$n" -lt "${#XPATHS[@]}" ]; do
    if [ "${XPATHS[$n]}" = "" ]; then
      n=$((n+2));
      continue;
    fi
    local XPATH_FROM="${XPATHS[$n]}";
    local XPATH_TO="${XPATHS[$((n+1))]}";
    n=$((n+2));

    XSLT+='
  <xsl:template match="'"$XPATH_TO"'">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
      <xsl:copy-of select="document('"'$FILE_FROM'"')'"$XPATH_FROM"'"/>';
    local m="$n";
    while [ "$m" -lt "${#XPATHS[@]}" ]; do
      if [ "$XPATH_TO" = "${XPATHS[$((m+1))]}" ]; then
        XPATH_FROM="${XPATHS[$m]}";
        XPATHS[$m]="";
        XSLT+='
      <xsl:copy-of select="document('"'$FILE_FROM'"')'"$XPATH_FROM"'"/>';
      fi
      m=$((m+2));
    done
    XSLT+='
    </xsl:copy>
  </xsl:template>';
  done

  XSLT+='
</xsl:stylesheet>';

  xmlstarlet tr <( echo "$XSLT" ) "$FILE_TO";
}

##
## Function that replicates XML Page nodes
##
htrsh_pagexml_replicate_nodes () {
  local FN="htrsh_pagexml_replicate_nodes";
  if [ $# -lt 4 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Replicates XML Page nodes";
      echo "Usage: $FN XML PATTERN NUM ID [NUM ID ...]";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local PATTERN="$2";
  shift 2;

  ### Create XSLT ###
  local XSLT='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>';

  local n="0";
  local INFO=("$@");
  while [ "$n" -lt "${#INFO[@]}" ]; do
    local ID="${INFO[$((n+1))]}";
    XSLT+='
  <xsl:template match="//*[@id='"'$ID'"']">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>';
    local m;
    for m in $(seq 2 ${INFO[$n]}); do
      #local IDm=$(printf "$PATTERN" "$ID" "$m");
      local IDm=$(printf "$PATTERN" "$m");
      XSLT+='
    <xsl:copy>
      <!--<xsl:attribute name="id">'"$IDm"'</xsl:attribute>-->
      <xsl:attribute name="id">
        <xsl:value-of select="concat(@id,&quot;'"$IDm"'&quot;)"/>
      </xsl:attribute>
      <xsl:if test="@orig-id">
        <xsl:attribute name="orig-id">
          <xsl:value-of select="concat(@orig-id,&quot;'"$IDm"'&quot;)"/>
        </xsl:attribute>
      </xsl:if>
      <xsl:apply-templates select="@*[local-name()!='"'id'"' and local-name()!='"'orig-id'"'] | node()"/>
    </xsl:copy>';
    done
    XSLT+='
  </xsl:template>';

    n=$((n+2));
  done

  XSLT+='
</xsl:stylesheet>';

  xmlstarlet tr <( echo "$XSLT" ) "$XML";
}

##
## Function that sorts Words from left to right within TextLines in an XML Page file
## (based ONLY on (xmin+xmax)/2 of the word Coords)
##
htrsh_pagexml_sort_words () {
  local FN="htrsh_pagexml_sort_words";
  if [ $# != 0 ]; then
    { echo "$FN: Error: Incorrect input arguments";
      echo "Description: Sorts Words from left to right within TextLines in an XML Page file (based ONLY on (xmin+xmax)/2 of the word Coords)";
      echo "Usage: $FN < XMLIN";
    } 1>&2;
    return 1;
  fi

  local XML=$(cat);

  local SORTVALS=( $(
    xmlstarlet sel -t -m '//_:TextLine[_:Word]' -v @id -o " " -v 'count(_:Word)' \
        -m _:Word -o " | " -v @id -o " " -v _:Coords/@points -b -n <( echo "$XML" ) \
      | sed 's|,[0-9.]*||g' \
      | awk '
          { printf( "%s %s", $1, $2 );
            mn = 1e9;
            mx = 0;
            id = $4;
            for( n=5; n<=NF; n++ )
              if( $n == "|" ) {
                printf( " %s %g", id, (mn+mx)/2 );
                mn = 1e9;
                mx = 0;
                n ++;
                id = $n;
              }
              else {
                mn = mn > $n ? $n : mn ;
                mx = mx < $n ? $n : mx ;
              }
            printf( " %s %g\n", id, (mn+mx)/2 );
          }' \
      | awk '
          { if( $2 != (NF-2)/2 )
              printf( "parse error at line %s\n", $1 ) > "/dev/stderr";
            else if( $2 != 1 ) {
              for( n=6; n<=NF; n+=2 )
                if( $n <= $(n-2) )
                  break;
              if( n <= NF )
                for( n=3; n<=NF; n+=2 )
                  printf( " -i //_:Word[@id=\"%s\"] -t attr -n sortval -v %s", $n, $(n+1) );
            }
          }'
    ) );

  local XSLT='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//_:TextLine[_:Word/@sortval]">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()[not(self::_:Word) and not(self::_:TextEquiv)]" />
      <xsl:apply-templates select="_:Word">
        <xsl:sort select="@sortval" data-type="number" order="ascending"/>
      </xsl:apply-templates>
      <xsl:apply-templates select="node()[self::_:TextEquiv]" />
    </xsl:copy>
  </xsl:template>
</xsl:stylesheet>';

  if [ "${#SORTVALS[@]}" = 0 ]; then
    echo "$XML";
  else
    echo "$XML" \
      | xmlstarlet ed "${SORTVALS[@]}" \
      | xmlstarlet tr <( echo "$XSLT" ) \
      | xmlstarlet ed -d //@sortval;
  fi
}

##
## Function that sorts Words by its number, assuming the IDs are of the form .*_w###[_part.] also moving the words after a _part1 word to the line that contains the corresponding _part2
##
htrsh_pagexml_sort_words_bynum () {
  local FN="htrsh_pagexml_sort_words_bynum";
  if [ $# != 0 ]; then
    { echo "$FN: Error: Incorrect input arguments";
      echo "Description: Sorts Words from by its number, assuming the IDs are of the form .*_w###[_part.] also moving the words after a _part1 word to the line that contains the corresponding _part2";
      echo "Usage: $FN < XMLIN";
    } 1>&2;
    return 1;
  fi

  local XML=$(cat);

  local SORTVALS=( $(
    xmlstarlet sel -t -m '//_:TextLine[_:Word]' -v @id -m _:Word -o " " -v @id -b -n <( echo "$XML" ) \
      | awk '
          { partnum = 0;
            for( n=2; n<=NF; n++ )
              if( match($n,/_part1$/) ) {
                partnum = gensub( /.+_w0*/, "", 1, gensub(/_part[12]$/,"",1,$n) );
                parttwo = gensub( /.+_w/, "_w", 1, gensub(/_part1$/,"_part2",1,$n) );
                break;
              }

            for( n=2; n<=NF; n++ ) {
              num = gensub( /.+_w0*/, "", 1, gensub(/_part[12]$/,"",1,$n) );
              printf( " -i //_:Word[@id=\"%s\"] -t attr -n sortval -v %s", $n, num );
              if( partnum && num+0 > partnum+0 && ! match($n,/_part1$/) )
                printf( " -m //_:Word[@id=\"%s\"] //_:TextLine[_:Word[contains(@id,\"%s\")]]", $n, parttwo );
            }
          }' ) );

  local XSLT='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//_:TextLine[_:Word/@sortval]">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()[not(self::_:Word) and not(self::_:TextEquiv)]" />
      <xsl:apply-templates select="_:Word">
        <xsl:sort select="@sortval" data-type="number" order="ascending"/>
      </xsl:apply-templates>
      <xsl:apply-templates select="node()[self::_:TextEquiv]" />
    </xsl:copy>
  </xsl:template>
</xsl:stylesheet>';

  if [ "${#SORTVALS[@]}" = 0 ]; then
    echo "$XML";
  else
    echo "$XML" \
      | xmlstarlet ed "${SORTVALS[@]}" \
      | xmlstarlet tr <( echo "$XSLT" ) \
      | xmlstarlet ed -d //@sortval;
  fi
}

##
## Function that sorts TextLines within each TextRegion in an XML Page file
## (based on the rounded average of the Y coordinates of the baseline plus
## the width fraction of the smallest X coordinate of the baseline)
##
htrsh_pagexml_sort_lines () {
  local FN="htrsh_pagexml_sort_lines";
  if [ $# != 0 ]; then
    { echo "$FN: Error: Incorrect input arguments";
      echo "Description: Sorts TextLines within each TextRegion in an XML Page file (based on the rounded average of the Y coordinates of the baseline plus the width fraction of the smallest X coordinate of the baseline)";
      echo "Usage: $FN < XMLIN";
    } 1>&2;
    return 1;
  fi

  local XML=$( cat - );
  local WIDTH=$( echo "$XML" | xmlstarlet sel -t -v '//@imageWidth' - );
  local SORTVALS=( $( echo "$XML" \
          | xmlstarlet sel -t -m "//$htrsh_xpath_lines/_:Baseline[@points]" \
              -v ../@id -o ' ' -v 'translate(@points,","," ")' -n - \
          | awk -v W=$WIDTH '
              { s = 0;
                for( n=3; n<=NF; n+=2 )
                  s += $n;
                s = sprintf( "%.0f", 2*s/(NF-1) );
                xmin = $2;
                for( n=4; n<=NF; n+=2 )
                  xmin = xmin > $n ? $n : xmin;
                printf( " -i //_:TextLine[@id=\"%s\"] -t attr -n sortval -v %g", $1, s+xmin/W );
              }' ) );

  local XSLT='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:variable name="Width" select="//_:Page/@imageWidth"/>

  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//_:TextRegion[count(_:TextLine)=count(_:TextLine/_:Baseline)]">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()[not(self::_:TextLine or self::_:TextEquiv)]" />
      <xsl:apply-templates select="_:TextLine">
        <!--<xsl:sort select="number(substring-before(substring-after(_:Baseline/@points,&quot;,&quot;),&quot; &quot;))+(number(substring-before(_:Baseline/@points,&quot;,&quot;)) div number($Width))" data-type="number" order="ascending"/>-->
        <xsl:sort select="@sortval" data-type="number" order="ascending" />
      </xsl:apply-templates>
      <xsl:apply-templates select="_:TextEquiv" />
    </xsl:copy>
  </xsl:template>
</xsl:stylesheet>';

  if [ "${#SORTVALS[@]}" = 0 ]; then
    echo "$XML";
  else
    echo "$XML" \
      | xmlstarlet ed "${SORTVALS[@]}" \
      | xmlstarlet tr <( echo "$XSLT" ) \
      | xmlstarlet ed -d //@sortval;
  fi
}

##
## Function that sorts TextRegions in an XML Page file
## (based ONLY on the minimum Y coordinate of the region Coords)
##
# @todo reimplement like the word sort
htrsh_pagexml_sort_regions () {
  local FN="htrsh_pagexml_sort_regions";
  if [ $# != 0 ]; then
    { echo "$FN: Error: Incorrect input arguments";
      echo "Description: Sorts TextRegions in an XML Page file (sorts using only the minimum Y coordinate of the region Coords)";
      echo "Usage: $FN < XMLIN";
    } 1>&2;
    return 1;
  fi

  local XSLT='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  version="2.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:variable name="Width" select="//_:Page/@imageWidth"/>

  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//_:Page">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()[not(self::_:TextRegion)]" />
      <xsl:apply-templates select="_:TextRegion">
        <!--<xsl:sort select="min(for $i in tokenize(replace(_:Coords/@points,'"'\d+,'"','"''"'),'"' '"') return number($i))" data-type="number" order="ascending"/>-->
        <xsl:sort select="min(for $i in tokenize(replace(_:Coords/@points,'"'\d+,'"','"''"'),'"' '"') return number($i))+(min(for $i in tokenize(replace(_:Coords/@points,'"',\d+'"','"''"'),'"' '"') return number($i)) div number($Width))" data-type="number" order="ascending"/>
      </xsl:apply-templates>
    </xsl:copy>
  </xsl:template>

</xsl:stylesheet>';

  saxonb-xslt -s:- -xsl:<( echo "$XSLT" );
}

##
## Function that relabels ids of TextRegions, TextLines and Words in an XML Page file
##
htrsh_pagexml_relabel () {
  local FN="htrsh_pagexml_relabel";
  if [ $# != 0 ]; then
    { echo "$FN: error: function does not expect arguments";
      echo "Description: Relabels ids of TextRegions, TextLines and Words in an XML Page file";
      echo "Usage: $FN < XMLIN";
    } 1>&2;
    return 1;
  fi

  local XSLT1='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//_:TextRegion">
    <xsl:copy>
      <xsl:attribute name="id">
        <xsl:value-of select="'"'t'"'"/>
        <xsl:number count="//_:TextRegion" format="01"/>
      </xsl:attribute>
      <xsl:apply-templates select="@*[local-name() != '"'id'"'] | node()" />
    </xsl:copy>
  </xsl:template>

</xsl:stylesheet>';

  local XSLT2='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//_:TextRegion/_:TextLine">
    <xsl:variable name="pid" select="../@id"/>
    <xsl:copy>
      <xsl:attribute name="id">
        <xsl:value-of select="concat(../@id,&quot;_l&quot;)"/>
        <xsl:number count="//_:TextRegion/_:TextLine" format="01"/>
      </xsl:attribute>
      <xsl:apply-templates select="@*[local-name() != '"'id'"'] | node()" />
    </xsl:copy>
  </xsl:template>

</xsl:stylesheet>';

  local XSLT3='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//_:TextRegion/_:TextLine/_:Word">
    <xsl:variable name="pid" select="../@id"/>
    <xsl:copy>
      <xsl:attribute name="id">
        <xsl:value-of select="concat(../@id,&quot;_w&quot;)"/>
        <xsl:number count="//_:TextRegion/_:TextLine/_:Word"/>
      </xsl:attribute>
      <xsl:apply-templates select="@*[local-name() != '"'id'"'] | node()" />
    </xsl:copy>
  </xsl:template>

</xsl:stylesheet>';

  xmlstarlet tr <( echo "$XSLT1" ) \
    | xmlstarlet tr <( echo "$XSLT2" ) \
    | xmlstarlet tr <( echo "$XSLT3" );
}

##
## Function that relabels IDs of TextLines in an XML Page file
##
htrsh_pagexml_relabel_textlines () {
  local FN="htrsh_pagexml_relabel";
  if [ $# != 0 ]; then
    { echo "$FN: error: function does not expect arguments";
      echo "Description: Relabels IDs of TextLines in an XML Page file";
      echo "Usage: $FN < XMLIN";
    } 1>&2;
    return 1;
  fi

  local XSLT='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//_:TextRegion/_:TextLine">
    <xsl:variable name="pid" select="../@id"/>
    <xsl:copy>
      <xsl:attribute name="id">
        <xsl:value-of select="concat(../@id,&quot;_l&quot;)"/>
        <!--<xsl:number count="//_:TextRegion/_:TextLine" format="01"/>-->
        <xsl:number count="//_:TextRegion/_:TextLine"/>
      </xsl:attribute>
      <xsl:apply-templates select="@*[local-name() != '"'id'"'] | node()" />
    </xsl:copy>
  </xsl:template>

</xsl:stylesheet>';

  xmlstarlet tr <( echo "$XSLT" );
}

##
## Function that replaces Coords polygons by bounding boxes
##
htrsh_pagexml_points2bbox () {
  local FN="htrsh_pagexml_points2bbox";
  if [ $# != 0 ]; then
    { echo "$FN: Error: Incorrect input arguments";
      echo "Description: Replaces Coords polygons by bounding boxes";
      echo "Usage: $FN < XMLIN";
    } 1>&2;
    return 1;
  fi

  local XML=$(cat);

  local xmledit=( -d //@dummyattr $(
    xmlstarlet sel -t -m "//$htrsh_xpath_coords" -v ../@id -o " " \
        -v 'translate(@points,","," ")' -n <( echo "$XML" ) \
      | awk '
          { if( NF >= 3 ) {
              mn_x = mx_x = $2;
              mn_y = mx_y = $3;
              for( n=4; n<NF; n+=2 ) {
                mn_x = mn_x > $n ? $n : mn_x ;
                mx_x = mx_x < $n ? $n : mx_x ;
                mn_y = mn_y > $(n+1) ? $(n+1) : mn_y ;
                mx_y = mx_y < $(n+1) ? $(n+1) : mx_y ;
              }
              printf( " -u //_:Coords[../@id=\"%s\"]/@points", $1 );
              printf( " -v %s,%s;%s,%s;%s,%s;%s,%s", mn_x,mn_y, mx_x,mn_y, mx_x,mx_y, mn_x,mx_y );
            }
          }'
    ) );

  echo "$XML" \
    | xmlstarlet ed "${xmledit[@]//;/ }";
}

##
## Function that replaces @points with the respective @fpgram in an XML Page file
##
htrsh_pagexml_fpgram2points () {
  local FN="htrsh_pagexml_fpgram2points";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Replaces @points with the respective @fpgram in an XML Page file";
      echo "Usage: $FN XML";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";

  local xmledit=( ed -d //@dummyattr );
  local id;
  for id in $(xmlstarlet sel -t -m '//_:TextLine/_:Coords[@fpgram]' -v ../@id -n "$XML"); do
    xmledit+=( -d "//_:TextLine[@id='$id']/_:Coords/@points" );
    xmledit+=( -r "//_:TextLine[@id='$id']/_:Coords/@fpgram" -v points );
  done

  xmlstarlet "${xmledit[@]}" "$XML";
}

##
## Function that replaces region '0,0 0,0' coords with content bounding box or page size if empty and only one region
##
htrsh_pagexml_zeroreg_fix () {
  local FN="htrsh_pagexml_zeroreg_fix";
  local INPLACE="no";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Replaces region '0,0 0,0' coords with content bounding box or page size if empty and only one region";
      echo "Usage: $FN XML";
    } 1>&2;
    return 1;
  fi

  local XML="$1";
  local IDS=( $(xmlstarlet sel -t -m "//_:TextRegion[_:Coords/@points = '0,0 0,0']" -v @id -n "$XML") );

  [ "${#IDS[@]}" = 0 ] &&
    return 0;

  local xmledit=( xmlstarlet ed --inplace );
  for n in $(seq 1 ${#IDS[@]}); do
    local ID="${IDS[$((n-1))]}";

    local BBOX=$( xmlstarlet sel -t -m "//_:TextRegion[@id='$ID']//*[@points != '0,0 0,0']" -v @points -n "$XML" \
      | awk -F'[ ,]' '
          { if( NR == 1 ) {
              x_min = x_max = $1;
              y_min = y_max = $2;
            }
            for( n=1; n<=NF; n+=2 ) {
              if( x_min > $n ) x_min = $n;
              if( x_max < $n ) x_max = $n;
              if( y_min > $(n+1) ) y_min = $(n+1);
              if( y_max < $(n+1) ) y_max = $(n+1);
            }
          }
          END {
            if( NR > 0 )
              print( x_min "," y_min " " x_max "," y_min " " x_max "," y_max " " x_min "," y_max );
          }' );

    if [ "$BBOX" = "" ]; then
      if [ "$n" = 1 ]; then
        local SIZE=( $(xmlstarlet sel -t -v //@imageWidth -o " " -v //@imageHeight "$XML") );
        local W=$(( ${SIZE[0]} - 1 ));
        local H=$(( ${SIZE[1]} - 1 ));
        BBOX="0,0 $W,0 $W,$H 0,$H";
      else
        echo "$FN: error: unable to fix region coords: $XML" 1>&2;
        return 1;
      fi
    fi

    xmledit+=( -u "//_:TextRegion[@id='$ID']/_:Coords/@points" -v "$BBOX" );
  done

  "${xmledit[@]}" "$XML";
}

##
## Function that replaces new line characters in TextEquiv/Unicode with spaces
##
htrsh_pagexml_textequiv_rm_newlines () {
  local FN="htrsh_pagexml_textequiv_rm_newlines";
  if [ $# != 0 ]; then
    { echo "$FN: Error: Incorrect input arguments";
      echo "Description: Replaces new line characters in TextEquiv/Unicode with spaces";
      echo "Usage: $FN < XMLIN";
    } 1>&2;
    return 1;
  fi

  local XSLT='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//_:Unicode">
    <xsl:copy>
      <xsl:value-of select="translate(.,'"'&#10;'"','"' '"')"/>
    </xsl:copy>
  </xsl:template>

</xsl:stylesheet>';

  xmlstarlet tr <( echo "$XSLT" );
}

##
## Function that changes TextEquiv/Unicode to lowercase or uppercase
##
htrsh_pagexml_textequiv_case () {
  local FN="htrsh_pagexml_textequiv_case";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Changes TextEquiv/Unicode to lowercase or uppercase";
      echo "Usage: $FN (upper|lower) XMLIN";
    } 1>&2;
    return 1;
  fi

  local xpath_textequiv=$( echo "$htrsh_xpath_textequiv" | sed 's|"|\&quot;|g; s|\x27|\&apos;|g;' );
  local XSLT=$'<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  xmlns:_="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
  exclude-result-prefixes="_"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:variable name="lowercase" select="\x27abcdefghijklmnopqrstuvwxyzàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿžšœ\x27"/>
  <xsl:variable name="uppercase" select="\x27ABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞŸŽŠŒ\x27"/>

  <xsl:template match="@* | node()">
    <xsl:copy>
      <xsl:apply-templates select="@* | node()"/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="'"$xpath_textequiv"'">
    <Unicode>
      <xsl:value-of select="translate(., $uppercase, $lowercase)"/>
    </Unicode>
  </xsl:template>

</xsl:stylesheet>';

  if [ "$1" = "upper" ]; then
    XSLT="${XSLT/uppercase,/lowercase,}";
    XSLT="${XSLT/lowercase)/uppercase)}";
  fi

  xmlstarlet tr <( echo "$XSLT" ) "$2";
}

##
## Function that moves TextEquiv elements after all others as required by the Page schema
##
htrsh_pagexml_textequiv_fix_position () {
  local FN="htrsh_pagexml_textequiv_fix_position";
  if [ $# != 0 ]; then
    { echo "$FN: Error: Incorrect input arguments";
      echo "Description: Moves TextEquiv elements after all others as required by the Page schema";
      echo "Usage: $FN < XMLIN";
    } 1>&2;
    return 1;
  fi

  local XML=$(cat);
  local xmledit=( $( echo "$XML" \
    | xmlstarlet sel -t -m '//*[_:TextEquiv]/*[position() < last() and local-name()="TextEquiv"]' -v 'local-name(..)' -o ' ' -v ../@id -n \
    | awk '{ printf( " -m //_:%s[@id=\"%s\"]/_:TextEquiv //_:%s[@id=\"%s\"]", $1, $2, $1, $2 ); }' ) );

  if [ "${#xmledit[@]}" = 0 ]; then
    echo "$XML";
  else
    echo "$XML" \
      | xmlstarlet ed "${xmledit[@]}";
  fi
}

##
## Function that sorts XML attributes alphabetically
##
htrsh_pagexml_sortattr () {
  local FN="htrsh_pagexml_sortattr";
  if [ $# != 0 ]; then
    { echo "$FN: Error: Incorrect input arguments";
      echo "Description: Sorts XML attributes alphabetically";
      echo "Usage: $FN < XMLIN";
    } 1>&2;
    return 1;
  fi

  local XSLT='<?xml version="1.0"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  version="1.0">

  <xsl:output method="xml" indent="yes" encoding="utf-8" omit-xml-declaration="no"/>

  <xsl:template match="*">
    <xsl:copy>
      <xsl:apply-templates select="@*">
        <xsl:sort select="name()"/>
      </xsl:apply-templates>
      <xsl:apply-templates/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="@*|comment()|processing-instruction()">
    <xsl:copy/>
  </xsl:template>

</xsl:stylesheet>';

  xmlstarlet tr <( echo "$XSLT" );
}

##
## Function that sets the image size attributes of the XML
##
htrsh_pageimg_setsize () {
  local FN="htrsh_pageimg_setsize";
  if [ $# != 1 ]; then
    { echo "$FN: Error: Incorrect input arguments";
      echo "Description: Sets the image size attributes of the XML";
      echo "Usage: $FN XML";
    } 1>&2;
    return 1;
  fi

  local XML="$1";

  ### Check XML file ###
  local $htrsh_infovars;
  htrsh_pageimg_info "$XML" noimg;
  [ "$?" != 0 ] && return 1;

  local DATE=$( date -u "+%Y-%m-%dT%H:%M:%S" );
  local SIZE=( $(identify -format "%w %h" "$IMDIR/$IMBASE.$IMEXT") );
  ( [ "$?" != 0 ] || [ "${#SIZE[@]}" != 2 ] ) && return 1;

  local xmledit=( xmlstarlet ed --inplace );
  xmledit+=( -d //@imageWidth -i //_:Page -t attr -n imageWidth -v "${SIZE[0]}" );
  xmledit+=( -d //@imageHeight -i //_:Page -t attr -n imageHeight -v "${SIZE[1]}" );
  xmledit+=( -u //_:LastChange -v "$DATE" );
  if [ $(xmlstarlet sel -t -v 'count(//_:Created[.!=""])' "$XML") = 0 ]; then
    DATE=$( date -u "+%Y-%m-%dT%H:%M:%S" -d "$(stat -c %y "$XML")" );
    xmledit+=( -u //_:Created -v "$DATE" );
  fi

  "${xmledit[@]}" "$XML";
}

#--------------------------------------#
# Feature extraction related functions #
#--------------------------------------#

##
## Function that cleans and enhances a text image based on regions defined in an XML Page file
##
htrsh_pageimg_clean () {
  local FN="htrsh_pageimg_clean";
  local INRES="";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Cleans and enhances a text image based on regions defined in an XML Page file";
      echo "Usage: $FN XML OUTDIR [ Options ]";
      echo "Options:";
      echo " -i INRES    Input image resolution in ppc (def.=use image metadata)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local OUTDIR="$2";
  shift 2;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-i" ]; then
      INRES="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  ### Check XML file and image ###
  local $htrsh_infovars;
  htrsh_pageimg_info "$XML";
  [ "$?" != 0 ] && return 1;

  if [ ! -d "$OUTDIR" ]; then
    echo "$FN: error: output directory does not exists: $OUTDIR" 1>&2;
    return 1;
  elif [ "$INRES" = "" ] && [ $(echo $IMRES | awk '{printf("%.0f",$1)}') -lt "$htrsh_minres" ]; then
    echo "$FN: error: image resolution ($IMRES ppc) apparently incorrect since it is unusually low to be a text document image: $IMFILE" 1>&2;
    return 1;
  elif [ "$XMLDIR" = $($htrsh_realpath "$OUTDIR") ]; then
    echo "$FN: error: output directory has to be different from the one containing the input XML: $XMLDIR" 1>&2;
    return 1;
  fi

  [ "$INRES" = "" ] && [ "$IMRES" != "" ] && INRES=$IMRES;
  [ "$INRES" != "" ] && INRES="-d $INRES";

  ### Enhance image ###
  local RC="0";
  if [ "$htrsh_clean_type" != "image" ]; then
    #convert "$IMFILE" -units PixelsPerCentimeter -density $(echo "$INRES" | sed 's|.* ||') "$OUTDIR/$IMBASE.png";
    #RC="$?";
    cp -p "$XML" "$IMFILE" "$OUTDIR";
    return 0;

  elif [ "$htrsh_imgclean" = "ncsr" ]; then
    EnhanceGray "$IMFILE" "$OUTDIR/$IMBASE.EnhanceGray.$IMEXT" 0 &&
    binarization "$OUTDIR/$IMBASE.EnhanceGray.$IMEXT" "$OUTDIR/$IMBASE.png" 2;
    RC="$?";
    rm -r "$OUTDIR/$IMBASE.EnhanceGray.$IMEXT";

  elif [ "$htrsh_imgclean" = "ncsr_b" ]; then
    binarization "$IMFILE" "$OUTDIR/$IMBASE.png" 2;
    RC="$?";

  elif [ "$htrsh_imgclean" != "prhlt" ]; then
    echo "$FN: error: unexpected preprocessing type: $htrsh_imgclean" 1>&2;
    return 1;

  elif [ "$htrsh_imgtxtenh_regmask" != "yes" ]; then
    imgtxtenh $htrsh_imgtxtenh_opts $INRES "$IMFILE" "$OUTDIR/$IMBASE.png" 2>&1;
    RC="$?";

  else
    local drawreg=( $( xmlstarlet sel -t -m "$htrsh_xpath_regions/$htrsh_xpath_coords" \
                         -o ' -fill gray(' -v '256-position()' -o ')' \
                         -o ' -stroke gray(' -v '256-position()' -o ')' \
                         -o ' -draw polygon_' -v 'translate(@points," ","_")' "$XML"
                         2>/dev/null ) );
    if [ $(echo "$htrsh_xpath_regions" | grep -F '[' | wc -l) != 0 ]; then
      local IXPATH=$(echo "$htrsh_xpath_regions" | sed 's|\[\([^[]*\)]|[not(\1)]|');
      drawreg+=( -fill black -stroke black );
      drawreg+=( $( xmlstarlet sel -t -m "$IXPATH/$htrsh_xpath_coords" \
                      -o ' -draw polygon_' -v 'translate(@points," ","_")' "$XML" \
                      2>/dev/null ) );
    fi

    ### Create mask and enhance selected text regions ###
    convert -size $IMSIZE xc:black +antialias "${drawreg[@]//_/ }" \
        -alpha copy "$IMFILE" +swap -compose copy-opacity -composite miff:- \
      | imgtxtenh $htrsh_imgtxtenh_opts $INRES - "$OUTDIR/$IMBASE.png" 2>&1;
    RC="$?";
  fi

  [ "$RC" != 0 ] &&
    echo "$FN: error: problems enhancing image: $IMFILE" 1>&2 &&
    return 1;

  ### Create new XML with image in current directory and PNG extension ###
  xmlstarlet ed -P -u //@imageFilename -v "$IMBASE.png" "$XML" \
    > "$OUTDIR/$XMLBASE.xml";
}

##
## Function that removes noise from borders of a quadrilateral region defined in an XML Page file
##
# @todo remove evals
htrsh_pageimg_quadborderclean () {
  local FN="htrsh_pageimg_quadborderclean";
  local TMPDIR=".";
  local CFG="";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Removes noise from borders of a quadrilateral region defined in an XML Page file";
      echo "Usage: $FN XML OUTIMG [ Options ]";
      echo "Options:";
      echo " -c CFG      Options for imgpageborder (def.=$CFG)";
      echo " -d TMPDIR   Directory for temporal files (def.=$TMPDIR)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local OUTIMG="$2";
  shift 2;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-c" ]; then
      CFG="$2";
    elif [ "$1" = "-d" ]; then
      TMPDIR="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  ### Check XML file and image ###
  local $htrsh_infovars;
  htrsh_pageimg_info "$XML";
  [ "$?" != 0 ] && return 1;

  local IMW=$(echo "$IMSIZE" | sed 's|x.*||');
  local IMH=$(echo "$IMSIZE" | sed 's|.*x||');

  ### Get quadrilaterals ###
  local QUADs=$(xmlstarlet sel -t -m "$htrsh_xpath_regions/$htrsh_xpath_coords" -v @points -n "$XML");
  local N=$(echo "$QUADs" | wc -l);

  local comps="";
  local n;
  for n in $(seq 1 $N); do
    local quad=$(echo "$QUADs" | sed -n ${n}p);
    [ $(echo "$quad" | wc -w) != 4 ] &&
      echo "$FN: error: region not a quadrilateral: $XML" 1>&2 &&
      return 1;

    local persp1=$(
      echo "$quad" \
        | awk -F'[ ,]' -v imW=$IMW -v imH=$IMH '
            { w = $3-$1;
              if( w > $5-$7 )
                w = $5-$7;
              h = $6-$4;
              if( h > $8-$2 )
                h = $8-$2;

              printf("-distort Perspective \"");
              printf("%d,%d %d,%d  ",$1,$2,0,0);
              printf("%d,%d %d,%d  ",$3,$4,w-1,0);
              printf("%d,%d %d,%d  ",$5,$6,w-1,h-1);
              printf("%d,%d %d,%d"  ,$7,$8,0,h-1);
              printf("\" -crop %dx%d+0+0\n",w,h);

              printf("-extent %dx%d+0+0 ",imW,imH);
              printf("-distort Perspective \"");
              printf("%d,%d %d,%d  ",0,0,$1,$2);
              printf("%d,%d %d,%d  ",w-1,0,$3,$4);
              printf("%d,%d %d,%d  ",w-1,h-1,$5,$6);
              printf("%d,%d %d,%d"  ,0,h-1,$7,$8);
              printf("\"\n");
            }');

    local persp0=$(echo "$persp1" | sed -n 1p);
    persp1=$(echo "$persp1" | sed -n 2p);

    eval convert "$IMFILE" $persp0 "$TMPDIR/${IMBASE}~${n}-persp.$IMEXT";

    imgpageborder $CFG -M "$TMPDIR/${IMBASE}~${n}-persp.$IMEXT" "$TMPDIR/${IMBASE}~${n}-pborder.$IMEXT";
    [ $? != 0 ] &&
      echo "$FN: error: problems estimating border: $XML" 1>&2 &&
      return 1;

    #eval convert -virtual-pixel white -background white "$TMPDIR/${IMBASE}~${n}-pborder.$IMEXT" $persp1 -white-threshold 1% "$TMPDIR/${IMBASE}~${n}-border.$IMEXT";
    eval convert -virtual-pixel black -background black "$TMPDIR/${IMBASE}~${n}-pborder.$IMEXT" $persp1 -white-threshold 1% -stroke white -strokewidth 3 -fill none -draw \"polygon $quad $(echo $quad | sed 's| .*||')\" "$TMPDIR/${IMBASE}~${n}-border.$IMEXT";
    #eval convert -virtual-pixel black -background black "$TMPDIR/${IMBASE}~${n}-pborder.$IMEXT" $persp1 -white-threshold 1% "$TMPDIR/${IMBASE}~${n}-border.$IMEXT";

    comps="$comps $TMPDIR/${IMBASE}~${n}-border.$IMEXT -composite";

    if [ "$htrsh_keeptmp" -lt 2 ]; then
      rm "$TMPDIR/${IMBASE}~${n}-persp.$IMEXT" "$TMPDIR/${IMBASE}~${n}-pborder.$IMEXT";
    fi
  done

  eval convert -compose lighten "$IMFILE" $comps "$OUTIMG";

  if [ "$htrsh_keeptmp" -lt 1 ]; then
    rm "$TMPDIR/${IMBASE}~"*"-border.$IMEXT";
  fi
  return 0;
}

##
## Function that extracts lines from an image given its XML Page file
##
htrsh_pageimg_extract_lines () {
  local FN="htrsh_pageimg_extract_lines";
  local OUTDIR=".";
  local IMFILE="";
  local SRC="lines";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Extracts lines from an image given its XML Page file";
      echo "Usage: $FN XMLFILE [ Options ]";
      echo "Options:";
      echo " -d OUTDIR   Output directory for images (def.=$OUTDIR)";
      echo " -i IMFILE   Extract from provided image (def.=@imageFilename in XML)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  shift;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-d" ]; then
      OUTDIR="$2";
    elif [ "$1" = "-i" ]; then
      IMFILE="$2";
    elif [ "$1" = "-s" ]; then
      SRC="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  ### Check page and obtain basic info ###
  local $htrsh_infovars;
  htrsh_pageimg_info "$XML";
  [ "$?" != 0 ] && return 1;

  local BASE=$(echo "$OUTDIR/$IMBASE" | sed 's|[\[ ()]|_|g; s|]|_|g;');
  local XPATH="$htrsh_xpath_regions/$htrsh_xpath_lines/$htrsh_xpath_coords";
  local IDop=( -o "$BASE." );
  if [ "$SRC" = "solo-words" ]; then
    XPATH="$htrsh_xpath_regions/$htrsh_xpath_lines/$htrsh_xpath_words/$htrsh_xpath_coords";
    [ "$htrsh_extended_names" = "true" ] && IDop=( -v ../../../@id -o . -v ../../@id -o . -v ../@id );
    [ "$htrsh_extended_names" != "true" ] && IDop=( -v ../@id );
  else
    [ "$htrsh_extended_names" = "true" ] && IDop+=( -v ../../@id -o "." -v ../@id );
    [ "$htrsh_extended_names" != "true" ] && IDop+=( -v ../@id );
  fi

  [ $(xmlstarlet sel -t -v "count($XPATH)" "$XML") = 0 ] &&
    echo "$FN: error: zero lines match xpath for extraction: $XML :: xpath: $XPATH" 1>&2 &&
    return 1;

  if [ "$RESSRC" = "xml" ]; then
    IMRES="-d $IMRES";
  else
    IMRES="";
  fi

  xmlstarlet sel -t -m "$XPATH" "${IDop[@]}" -o ".png " -v @points -n "$XML" \
    | imgpolycrop $IMRES "$IMFILE";

  [ "$?" != 0 ] &&
    echo "$FN: error: line image extraction failed" 1>&2 &&
    return 1;

  return 0;
}

##
## Function that discretizes a list of features using a given codebook
##
htrsh_feats_discretize () {
  local FN="htrsh_feats_discretize";
  if [ $# -lt 3 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Discretizes a list of features using a given codebook";
      echo "Usage: $FN FEATLST CBOOK OUTDIR";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local FEATLST="$1";
  local CBOOK="$2";
  local OUTDIR="$3";

  if [ ! -e "$FEATLST" ]; then
    echo "$FN: error: features list file does not exists: $FEATLST" 1>&2;
    return 1;
  elif [ ! -e "$CBOOK" ]; then
    echo "$FN: error: codebook file does not exists: $CBOOK" 1>&2;
    return 1;
  elif [ ! -d "$OUTDIR" ]; then
    echo "$FN: error: output directory does not exists: $OUTDIR" 1>&2;
    return 1;
  fi

  local CFG="$htrsh_HTK_config"'
TARGETKIND     = USER_V
VQTABLE        = '"$CBOOK"'
SAVEASVQ       = T
';

  local LST=$(sed 's|\(.*/\)\(.*\)|\1\2 '"$OUTDIR"'/\2|; t; s|\(.*\)|\1 '"$OUTDIR"'/\1|;' "$FEATLST");

  HCopy -C <( echo "$CFG" ) $LST;
  [ "$?" != 0 ] &&
    echo "$FN: error: problems discretizing features" 1>&2 &&
    return 1;

  return 0;
}

##
## Function that extracts features from an image
##
htrsh_extract_feats () {
  local FN="htrsh_extract_feats";
  local HEIGHT="";
  local XHEIGHT="";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Extracts features from an image";
      echo "Usage: $FN IMGIN FEAOUT [ Options ]";
      echo "Options:";
      echo " -h HEIGHT    Line height for size normalization (def.=false)";
      echo " -xh XHEIGHT  Line x-height for size normalization based on htrsh_feat_normxheight, ignored if -h given (def.=false)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local IMGIN="$1";
  local FEAOUT="$2";
  shift 2;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-h" ]; then
      HEIGHT="$2";
    elif [ "$1" = "-xh" ]; then
      XHEIGHT="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  local IMGPROC="$IMGIN";
  if [ "$HEIGHT" != "" ]; then
    IMGPROC=$(mktemp).png;
    convert "$IMGIN" -resize x"$HEIGHT" "$IMGPROC";
  elif [ "$XHEIGHT" != "" ] && [ "$htrsh_feat_normxheight" != "" ]; then
    IMGPROC=$(mktemp).png;
    convert "$IMGIN" -resize $(echo "100*$htrsh_feat_normxheight/$XHEIGHT" | bc -l)% "$IMGPROC";
  fi

  ### Extract features ###
  if [ "$htrsh_feat" = "dotmatrix" ]; then
    local featcfg="-S --htk --width $htrsh_dotmatrix_W --height $htrsh_dotmatrix_H --shift=$htrsh_dotmatrix_shift --win-size=$htrsh_dotmatrix_win -i";
    if [ "$htrsh_dotmatrix_mom" = "yes" ]; then
      dotmatrix -m $featcfg "$IMGPROC";
    else
      dotmatrix $featcfg "$IMGPROC";
    fi > "$FEAOUT";

  elif [ "$htrsh_feat" = "prhlt" ]; then
    local TMP=$(mktemp);
    convert "$IMGPROC" $TMP.pgm;
    pgmtextfea -F 2.5 -i $TMP.pgm > $TMP.fea;
    pfl2htk $TMP.fea "$FEAOUT" 2>/dev/null;
    rm $TMP $TMP.{pgm,fea};

  elif [ "$htrsh_feat" = "fki" ]; then
    local TMP=$(mktemp);
    convert "$IMGPROC" -threshold 50% $TMP.pbm;
    fkifeat $TMP.pbm > $TMP.fea;
    pfl2htk $TMP.fea "$FEAOUT" 2>/dev/null;
    rm $TMP $TMP.{pbm,fea};

  else
    echo "$FN: error: unknown features type: $htrsh_feat" 1>&2;
    return 1;
  fi

  [ "$IMGIN" != "$IMGPROC" ] && [ "$htrsh_keeptmp" = 0 ] &&
    rm "$IMGPROC" "${IMGPROC%.png}";

  return 0;
}

##
## Function that concatenates line features for regions defined in an XML Page file
##
htrsh_feats_catregions () {(
  local FN="htrsh_feats_catregions";
  local FEATLST="/dev/null";
  local RMORIG="yes";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Concatenates line features for regions defined in an XML Page file";
      echo "Usage: $FN XML FEATDIR [ Options ]";
      echo "Options:";
      echo " -l FEATLST  Output list of features to file (def.=$FEATLST)";
      echo " -r (yes|no) Whether to remove original features (def.=$RMORIG)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local FEATDIR="$2";
  shift 2;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-l" ]; then
      FEATLST="$2";
    elif [ "$1" = "-r" ]; then
      RMORIG="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  ### Check page and obtain basic info ###
  local $htrsh_infovars;
  htrsh_pageimg_info "$XML";
  [ "$?" != 0 ] && return 1;

  [ ! -e "$FEATDIR" ] &&
    echo "$FN: error: features directory not found: $FEATDIR" 1>&2 &&
    return 1;

  local FBASE=$(echo "$FEATDIR/$IMBASE" | sed 's|[\[ ()]|_|g; s|]|_|g;');

  #xmlstarlet sel -t -m "$htrsh_xpath_regions/$htrsh_xpath_lines/$htrsh_xpath_coords" \
  #    -o "$FBASE." -v ../../@id -o "." -v ../@id -o ".fea" -n "$XML" \
  xmlstarlet sel -t -m "$htrsh_xpath_regions/$htrsh_xpath_lines" \
      -o "$FBASE." -v ../@id -o "." -v @id -o ".fea" -n "$XML" \
    | xargs --no-run-if-empty ls >/dev/null;
  [ "$?" != 0 ] &&
    echo "$FN: error: some line feature files not found" 1>&2 &&
    return 1;

  local IFS=$'\n';
  local id feats f;
  #for id in $( xmlstarlet sel -t -m "$htrsh_xpath_regions[$htrsh_xpath_lines/$htrsh_xpath_coords]" -v @id -n "$XML" ); do
  for id in $( xmlstarlet sel -t -m "$htrsh_xpath_regions[$htrsh_xpath_lines]" -v @id -n "$XML" ); do
    #feats=( $( xmlstarlet sel -t -m "//*[@id='$id']/$htrsh_xpath_lines[$htrsh_xpath_coords]" -o "$FBASE.$id." -v @id -o ".fea" -n "$XML" | sed '2,$ s|^|+\n|' ) );
    feats=( $( xmlstarlet sel -t -m "//*[@id='$id']/$htrsh_xpath_lines" -o "$FBASE.$id." -v @id -o ".fea" -n "$XML" | sed '2,$ s|^|+\n|' ) );

    HCopy "${feats[@]}" "$FBASE.$id.fea";

    echo "$FBASE.$id.fea" >> "$FEATLST";

    #feats=( $( xmlstarlet sel -t -m "//*[@id='$id']/$htrsh_xpath_lines[$htrsh_xpath_coords]" -o "$FBASE.$id." -v @id -o ".fea" -n "$XML" ) );
    feats=( $( xmlstarlet sel -t -m "//*[@id='$id']/$htrsh_xpath_lines" -o "$FBASE.$id." -v @id -o ".fea" -n "$XML" ) );

    for f in "${feats[@]}"; do
      echo \
        $( echo "$f" | sed 's|.*\.\([^.][^.]*\)\.fea$|\1|' ) \
        $( HList -h -z "$f" | sed -n '/Num Samples:/{ s|.*Num Samples: *||; s| .*||; p; }' );
    done > "$FBASE.$id.nfea";

    [ "$RMORIG" = "yes" ] &&
      rm "${feats[@]}";
  done

  return 0;
)}

##
## Function that computes a PCA base for a given list of HTK features
##
htrsh_feats_pca () {(
  local FN="htrsh_feats_pca";
  local EXCL="[]";
  local RDIM="";
  local RNDR="no";
  local THREADS="1";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Computes a PCA base for a given list of HTK features";
      echo "Usage: $FN FEATLST OUTMAT [ Options ]";
      echo "Options:";
      echo " -e EXCL     Dimensions to exclude in matlab range format (def.=false)";
      echo " -r RDIM     Return base of RDIM dimensions (def.=all)";
      echo " -R (yes|no) Random rotation (def.=$RNDR)";
      echo " -T THREADS  Threads for parallel processing (def.=$THREADS)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local FEATLST="$1";
  local OUTMAT="$2";
  shift 2;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-e" ]; then
      EXCL="$2";
    elif [ "$1" = "-r" ]; then
      RDIM="$2";
    elif [ "$1" = "-R" ]; then
      RNDR="$2";
    elif [ "$1" = "-T" ]; then
      THREADS="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  if [ ! -e "$FEATLST" ]; then
    echo "$FN: error: feature list not found: $FEATLST" 1>&2;
    return 1;
  elif [ $(wc -l < "$FEATLST") != $(xargs --no-run-if-empty ls < "$FEATLST" | wc -l) ]; then
    echo "$FN: error: some files in list not found: $FEATLST" 1>&2;
    return 1;
  fi

  #local htrsh_fastpca="no";
  #if [ "$htrsh_fastpca" = "yes" ]; then
  #  local DIMS=$(HList -h -z $(head -n 1 < "$FEATLST") \
  #          | sed -n '/^  Num Comps:/{s|^[^:]*: *||;s| .*||;p;}');
  #  tail -qc +13 $(< "$FEATLST") | swap4bytes | fast_pca -C -e $EXCL -f binary -b 500 -p $DIMS -m "$OUTMAT";
  #  RC="$?";
  #else

  local RC;
  local xEXCL=""; [ "$EXCL" != "[]" ] && xEXCL="se = se + sum(x(:,$EXCL)); x(:,$EXCL) = [];";
  local xxEXCL=""; [ "$EXCL" != "[]" ] && xxEXCL="se = se + cse;";
  local nRDIM="D"; [ "$RDIM" != "" ] && nRDIM="min(D,$RDIM)-DE";

  htrsh_comp_csgma () {
    { local f;
      echo "
        DE = length($EXCL);
        se = zeros(1,DE);
      ";
      for f in $(<"$1"); do
        echo "
          x = readhtk('$f'); $xEXCL
          if ~exist('cN','var')
            cN = size(x,1);
            cmu = sum(x);
            csgma = x'*x;
          else
            cN = cN + size(x,1);
            cmu = cmu + sum(x);
            csgma = csgma + x'*x;
          end
        ";
      done
      echo "
        cse = se;
        save('-z','$2','cN','cmu','csgma','cse');
      ";
    } | octave -q -H;
  }

  run_parallel -T "$THREADS" -n split -l "$FEATLST" htrsh_comp_csgma "{@}" "$OUTMAT.csgma{%}.mat.gz";
  [ "$?" != 0 ] &&
    echo "$FN: error: problems computing PCA" 1>&2 &&
    return 1;

  { local f;
    echo "
      DE = length($EXCL);
      se = zeros(1,DE);
    ";
    for f in "$OUTMAT.csgma"*.mat.gz; do
      echo "
        load('$f'); $xxEXCL
        if ~exist('N','var')
          N = cN;
          mu = cmu;
          sgma = csgma;
        else
          N = N + cN;
          mu = mu + cmu;
          sgma = sgma + csgma;
        end
      ";
    done
    echo "
      mu = (1/N)*mu;
      sgma = (1/N)*sgma - mu'*mu;
      sgma = 0.5*(sgma+sgma');
      [ B, V ] = eig(sgma);
      V = real(diag(V));
      [ srt, idx ] = sort(-1*V);
      V = V(idx);
      B = B(:,idx);
      D = size(sgma,1);
      DR = $nRDIM;
      B = B(:,1:DR);
    ";
    if [ "$EXCL" != "[]" ]; then
      echo "
        sel = true(DE+D,1);
        sel($EXCL) = false;
        selc = [ false(DE,1) ; true(DR,1) ];
        BB = zeros(DE+D,DE+DR);
        BB(sel,selc) = B;
        BB(~sel,~selc) = eye(DE);
        B = BB;
        mmu = zeros(1,DE+D);
        mmu(sel) = mu;
        mmu(~sel) = (1/N)*se;
        mu = mmu;
      ";
    fi
    if [ "$RNDR" = "yes" ]; then
      echo "
        rand('state',1);
        [ R, ~ ] = qr(rand(size(B,2)));
        B = B*R;
      ";
    fi
    echo "save('-hdf5','$OUTMAT','B','V','mu');";
  } | octave -q -H;

  RC="$?";

  rm "$OUTMAT.csgma"*.mat.gz;

  #fi

  [ "$RC" != 0 ] &&
    echo "$FN: error: problems computing PCA" 1>&2;

  return $RC;
)}

##
## Function that projects a list of features for a given base
##
htrsh_feats_project () {(
  local FN="htrsh_feats_project";
  local RDIM="";
  local UNPROJ="no";
  local THREADS="1";
  if [ $# -lt 3 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Projects a list of features for a given base";
      echo "Usage: $FN FEATLST PBASE OUTDIR";
      echo " -r RDIM     Project to RDIM dimensions (def.=all)";
      echo " -T THREADS  Threads for parallel processing (def.=$THREADS)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local FEATLST="$1";
  local PBASE="$2";
  local OUTDIR="$3";
  shift 3;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-r" ]; then
      RDIM="$2";
    elif [ "$1" = "-u" ]; then
      UNPROJ="$2";
    elif [ "$1" = "-T" ]; then
      THREADS="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  if [ ! -e "$FEATLST" ]; then
    echo "$FN: error: features list file does not exists: $FEATLST" 1>&2;
    return 1;
  elif [ ! -e "$PBASE" ]; then
    echo "$FN: error: projection base does not exists: $PBASE" 1>&2;
    return 1;
  elif [ ! -d "$OUTDIR" ]; then
    echo "$FN: error: output directory does not exists: $OUTDIR" 1>&2;
    return 1;
  fi

  feats_project () {
    { echo "load('$PBASE');"
      [ "$RDIM" != "" ] &&
        echo "B = B(:,1:min($RDIM,size(B,2)));";
      local f ff;
      local proj="x = (x-repmat(mu,size(x,1),1))*B;";
      [ "$UNPROJ" = "yes" ] && proj="x = x*B'+repmat(mu,size(x,1),1);";
      for f in $(<"$1"); do
        ff=$(echo "$f" | sed "s|.*/||; s|^|$OUTDIR/|;");
        echo "
          [x,FP,DT,TC] = readhtk('$f');
          %x = (x-repmat(mu,size(x,1),1))*B;
          $proj
          writehtk('$ff',x,FP,TC);
          ";
      done
    } | octave -q -H;
  }

  if [ "$THREADS" = 1 ]; then
    feats_project "$FEATLST";
  else
    run_parallel -T $THREADS -n balance -l "$FEATLST" feats_project '{@}';
  fi

  [ "$?" != 0 ] &&
    echo "$FN: error: problems projecting features" 1>&2 &&
    return 1;

  return 0;
)}

##
## Function that converts a list of features in HTK format to Kaldi ark,scp
##
htrsh_feats_htk_to_kaldi () {
  local FN="htrsh_feats_htk_to_kaldi";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Converts a list of features in HTK format to Kaldi ark,scp";
      echo "Usage: $FN OUTBASE < FEATLST";
    } 1>&2;
    return 1;
  fi

  sed 's|^\([^/]*\)\.fea$|\1 \1.fea|;
       s|^\(.*/\)\([^/]*\)\.fea$|\2 \1\2.fea|;' \
    | copy-feats --print-args=false --htk-in scp:- ark,scp:$1.ark,$1.scp;

  return $?;
}


htrsh_pageimg_extract_linefeats_fast () {
  local FN="htrsh_pageimg_extract_linefeats_fast";
  local OUTDIR=".";
  local FEATLST="/dev/null";
  local PBASE="";
  #local HEIGHT="";
  local REPLC="yes";
  #local SRC="lines";
  local THREADS="1";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Extracts line features from an image given its XML Page file";
      echo "Usage: $FN XMLIN XMLOUT [ Options ]";
      echo "Options:";
      echo " -d OUTDIR   Output directory for features (def.=$OUTDIR)";
      echo " -l FEATLST  Output list of features to file (def.=$FEATLST)";
      echo " -b PBASE    Project features using given base (def.=false)";
      echo " -h HEIGHT   Normalize line height (def.=false)";
      echo " -c (yes|no) Whether to replace Coords/@points with the features contour (def.=$REPLC)";
      echo " -T THREADS  Threads for parallel processing (def.=$THREADS)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local XMLOUT="$2";
  shift 2;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-d" ]; then
      OUTDIR="$2";
    elif [ "$1" = "-l" ]; then
      FEATLST="$2";
    elif [ "$1" = "-b" ]; then
      PBASE="$2";
    elif [ "$1" = "-c" ]; then
      REPLC="$2";
    #elif [ "$1" = "-h" ]; then
    #  HEIGHT="$2";
    #elif [ "$1" = "-s" ]; then
    #  SRC="$2";
    elif [ "$1" = "-T" ]; then
      THREADS="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  ### Check XML file ###
  local $htrsh_infovars;
  htrsh_pageimg_info "$XML";
  [ "$?" != 0 ] && return 1;

  local XPATH="$htrsh_xpath_regions/$htrsh_xpath_lines/$htrsh_xpath_coords";
  local TMP="${TMPDIR:-/tmp}";
  TMP=$(mktemp -d --tmpdir="$TMP" ${FN}_XXXXX);

  local enh_win=$( echo "$htrsh_imgtxtenh_opts" | sed -r 's|^.*-w ([0-9]+).*|\1|' );
  local enh_prm=$( echo "$htrsh_imgtxtenh_opts" | sed -r 's|^.*-k ([.0-9]+).*|\1|' );
  local featype="dotm";
  local feaformat="htk";
  if [ "${htrsh_feat:0:3}" = "th:" ]; then
    featype="raw";
    feaformat="img";
  fi

  local CFG="
  PageXML: {
    extended_names = $htrsh_extended_names;
  }
  TextFeatExtractor: {
    type = \"$featype\";
    format = \"$feaformat\";
    normheight = $htrsh_feat_normheight;
    enh_win = $enh_win;
    enh_prm = $enh_prm;
    padding = $htrsh_feat_padding;
    deslope = \"$htrsh_feat_deslope\";
    deslant = \"$htrsh_feat_deslant\";
    slide_shift = $htrsh_dotmatrix_shift;
    slide_span = $htrsh_dotmatrix_win;
    sample_width = $htrsh_dotmatrix_W;
    sample_height = $htrsh_dotmatrix_H;
    fcontour = \"$htrsh_feat_contour\";
    fpgram = \"$htrsh_feat_contour\";"$'\n';
  #[ "$HEIGHT" != "" ] && CFG+="    normheight = $HEIGHT;"$'\n';
  [ "$PBASE" != "" ] && CFG+="    projfile = \"$PBASE\";"$'\n';
  CFG+=$'  }\n';

  echo "$CFG" > "$OUTDIR/textFeats.cfg";

  textFeats -V --featlist --saveclean --savexml --cfg <( echo "$CFG" ) \
    --outdir "$TMP" -T "$THREADS" --xpath "$XPATH" --fpoints="$REPLC" \
    "$XML" > "$TMP/feats.lst" 2> "$OUTDIR/textFeats.log";
  [ "$?" != 0 ] &&
    echo "$FN: error: problems extracting features, more info in file $OUTDIR/textFeats.log" 1>&2 &&
    return 1;

  if [ "${htrsh_feat:0:3}" = "th:" ]; then
    sed 's|^|'"$TMP"'/|; s|$|.png|;' "$TMP/feats.lst" >> "$TMP/imgs.lst";
    th "$HOME/work/prog/HTR/htr-exps/code/Laia/net_output.lua" \
      "${htrsh_feat:3}" "$TMP/imgs.lst" "$TMP" -batch_size 1 -htk #-convout
  fi

  mv "$TMP"/*.fea "$TMP"/*.png "$OUTDIR";
  mv "$TMP/$IMBASE.xml" "$XMLOUT";
  sed 's|^|'"$OUTDIR"'/|; s|$|.fea|;' "$TMP/feats.lst" >> "$FEATLST";

  #xmlstarlet ed --inplace -d //@slope -d //@slant "$XMLOUT";
  #mv "$TMP" "$OUTDIR";
  rm -r "$TMP";
}

##
## Function that extracts line features from an image given its XML Page file
##
# @todo option(s) for random variations (size, slant, extremes white space, etc.)
htrsh_pageimg_extract_linefeats () {
  local FN="htrsh_pageimg_extract_linefeats";
  local OUTDIR=".";
  local FEATLST="/dev/null";
  local PBASE="";
  local HEIGHT="";
  local REPLC="yes";
  local SRC="lines";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Extracts line features from an image given its XML Page file";
      echo "Usage: $FN XMLIN XMLOUT [ Options ]";
      echo "Options:";
      echo " -d OUTDIR   Output directory for features (def.=$OUTDIR)";
      echo " -l FEATLST  Output list of features to file (def.=$FEATLST)";
      echo " -b PBASE    Project features using given base (def.=false)";
      echo " -h HEIGHT   Normalize line height (def.=false)";
      echo " -c (yes|no) Whether to replace Coords/@points with the features contour (def.=$REPLC)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local XMLOUT="$2";
  shift 2;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-d" ]; then
      OUTDIR="$2";
    elif [ "$1" = "-l" ]; then
      FEATLST="$2";
    elif [ "$1" = "-b" ]; then
      PBASE="-b $2";
    elif [ "$1" = "-c" ]; then
      REPLC="$2";
    elif [ "$1" = "-h" ]; then
      HEIGHT="$2";
    elif [ "$1" = "-s" ]; then
      SRC="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  ### Check page and obtain basic info ###
  local $htrsh_infovars;
  htrsh_pageimg_info "$XML";
  [ "$?" != 0 ] && return 1;

  ### Extract lines from line coordinates ###
  local LINEIMGS=$(htrsh_pageimg_extract_lines "$XML" -d "$OUTDIR" -s "$SRC");
  ( [ "$?" != 0 ] || [ "$LINEIMGS" = "" ] ) && return 1;

  local xmledit=( ed );
  local FEATS="";

  ### Process each line ###
  local oklines="0";
  local n;
  for n in $(seq 1 $(echo "$LINEIMGS" | wc -l)); do
    local ff=$(echo "$LINEIMGS" | sed -n $n'{s|\.png$||;p;}');
    local id=$(echo "$ff" | sed 's|.*\.||');

    echo "$FN: processing line image ${ff}.png";

    ### Clean and trim line image ###
    if [ "$htrsh_clean_type" = "none" ]; then
      cp -p ${ff}.png ${ff}_clean.png;
    elif [ "$htrsh_clean_type" = "line" ]; then
      imgtxtenh $htrsh_imgtxtenh_opts -a ${ff}.png miff:- \
        | imglineclean $htrsh_imglineclean_opts - ${ff}_clean.png;
    else
      imglineclean $htrsh_imglineclean_opts ${ff}.png ${ff}_clean.png;
    fi 2>&1;
    [ "$?" != 0 ] &&
      echo "$FN: error: problems cleaning line image: ${ff}.png" 1>&2 &&
      continue;

    local bbox=$(identify -format "%wx%h%X%Y" ${ff}_clean.png);
    local bboxsz=$(echo "$bbox" | sed 's|x| |; s|+.*||;');
    local bboxoff=$(echo "$bbox" | sed 's|[0-9]*x[0-9]*||; s|+| |g;');

    ### Estimate slope, slant and affine matrices ###
    local slope="";
    local slant="";
    [ "$htrsh_feat_deslope" = "yes" ] &&
      slope=$(convert ${ff}_clean.png +repage -flatten \
               -deskew 40% -print '%[deskew:angle]\n' \
               -trim +repage ${ff}_deslope.png);
      #slope=$(imageSlope -i ${ff}_clean.png -o ${ff}_deslope.png -v 1 -s 10000 2>&1 \
      #         | sed -n '/slope medio:/{s|.* ||;p;}');

    [ "$htrsh_feat_deslant" = "yes" ] &&
      slant=$(imageSlant -v 1 -g -i ${ff}_deslope.png -o ${ff}_deslant.png 2>&1 \
                | sed -n '/Slant medio/{s|.*: ||;p;}');

    [ "$slope" = "" ] && slope="0";
    [ "$slant" = "" ] && slant="0";

    local affine=$(echo "
      h = [ $bboxsz ];
      w = h(1);
      h = h(2);
      co = cos(${slope}*pi/180);
      si = sin(${slope}*pi/180);
      s = tan(${slant}*pi/180);
      R0 = [ co,  si, 0 ; -si, co, 0; 0, 0, 1 ];
      R1 = [ co, -si, 0 ;  si, co, 0; 0, 0, 1 ];
      S0 = [ 1, 0, 0 ;  s, 1, 0 ; 0, 0, 1 ];
      S1 = [ 1, 0, 0 ; -s, 1, 0 ; 0, 0, 1 ];
      A0 = R0*S0;
      A1 = S1*R1;

      %mn = round(min([0 0 1; w-1 h-1 1; 0 h-1 1; w-1 0 1]*A0))-1; % Jv3pT: incorrect 5 out of 1117 = 0.45%

      save('${ff}_affine.mat','A0','A1');

      printf('%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n',
        A0(1,1), A0(1,2), A0(2,1), A0(2,2), A0(3,1), A0(3,2) );
      " | octave -q -H);

    ### Apply affine transformation to image ###
    local mn;
    #if [ "$affine" = "1,0,0,1,0,0" ]; then
    # @todo This doesn't work since offset of _clean.png and _affine.png differs, is off(3:4) still necessary?
    #  ln -s $(echo "$ff" | sed 's|.*/||')_clean.png ${ff}_affine.png;
    #  mn="0,0";
    #  #mn="-1,-1";
    #else
      mn=$(convert ${ff}_clean.png +repage -flatten \
             -virtual-pixel white +distort AffineProjection ${affine} \
             -shave 1x1 -format %X,%Y -write info: \
             +repage -trim ${ff}_affine.png);
      [ $(identify -format "%wx%h" ${ff}_affine.png) = "1x1" ] &&
        mn=$(convert ${ff}_clean.png +repage -flatten \
               -virtual-pixel white +distort AffineProjection ${affine} \
               -shave 1x1 -format %X,%Y -write info: \
               +repage ${ff}_affine.png);
    #fi

    ### Add left and right padding ###
    local PADpx=$(echo $IMRES $htrsh_feat_padding | awk '{printf("%.0f",$1*$2/10)}');
    convert ${ff}_affine.png +repage \
      -bordercolor white -border ${PADpx}x \
      +repage ${ff}_fea.png;

    ### Compute features parallelogram ###
    local fpgram=$(echo "
      load('${ff}_affine.mat');

      off = [ $(identify -format %w,%h,%X,%Y ${ff}_affine.png) ];
      w = off(1);
      h = off(2);

      mn = [ $mn ];
      off = off(3:4) + mn(1:2);

      feaWin = $htrsh_dotmatrix_win;
      feaShift = $htrsh_dotmatrix_shift;

      numFea = size([-feaWin-${PADpx}:feaShift:w+${PADpx}+1],2);
      xmin = -feaWin/2-${PADpx};
      xmax = xmin+(numFea-1)*feaShift;

      pt0 = [ $bboxoff 0 ] + [ [ xmin 0   ]+off 1 ] * A1 ;
      pt1 = [ $bboxoff 0 ] + [ [ xmax 0   ]+off 1 ] * A1 ;
      pt2 = [ $bboxoff 0 ] + [ [ xmax h-1 ]+off 1 ] * A1 ;
      pt3 = [ $bboxoff 0 ] + [ [ xmin h-1 ]+off 1 ] * A1 ;

      printf('%g,%g %g,%g %g,%g %g,%g\n',
        pt0(1), pt0(2),
        pt1(1), pt1(2),
        pt2(1), pt2(2),
        pt3(1), pt3(2) );
      " | octave -q -H);

    ### Prepare information to add to XML ###
    #xmledit+=( -i "//*[@id='$id']/_:Coords" -t attr -n bbox -v "$bbox" );
    #xmledit+=( -i "//*[@id='$id']/_:Coords" -t attr -n slope -v "$slope" );
    #[ "$htrsh_feat_deslant" = "yes" ] &&
    #xmledit+=( -i "//*[@id='$id']/_:Coords" -t attr -n slant -v "$slant" );
    #xmledit+=( -i "//*[@id='$id']/_:Coords" -t attr -n fpgram -v "$fpgram" );
    xmledit+=( -d "//*[@id='$id']/_:Property[@key='fpgram']" );
    xmledit+=( -s "//*[@id='$id']" -t elem -n TMPNODE );
    xmledit+=( -i //TMPNODE -t attr -n key -v fgram );
    xmledit+=( -i //TMPNODE -t attr -n value -v "$fgram" );
    xmledit+=( -r //TMPNODE -v Property );

    ### Compute detailed contours if requested ###
    if [ "$htrsh_feat_contour" = "yes" ]; then
      local pts=$(imgccomp -V1 -NJS -A 0.5 -D $htrsh_feat_dilradi -R 5,2,2,2 ${ff}_clean.png);
      [ "$pts" = "" ] && pts="$fpgram";
      #xmledit+=( -i "//*[@id='$id']/_:Coords" -t attr -n fcontour -v "$pts" );
      xmledit+=( -d "//*[@id='$id']/_:Property[@key='fcontour']" );
      xmledit+=( -s "//*[@id='$id']" -t elem -n TMPNODE );
      xmledit+=( -i //TMPNODE -t attr -n key -v fcontour );
      xmledit+=( -i //TMPNODE -t attr -n value -v "$pts" );
      xmledit+=( -r //TMPNODE -v Property );
    fi 2>&1;

    local FEATOP="";
    if [ "$HEIGHT" != "" ]; then
      FEATOP="-h $HEIGHT";
    elif [ "$htrsh_feat_normxheight" != "" ]; then
      FEATOP=$(xmlstarlet sel -t -v "//*[@id='$id']/@custom" "$XML" 2>/dev/null \
        | sed -n '/x-height:/ { s|.*x-height:\([^;]*\).*|\1|; s|px$||; p; }' );
      [ "$FEATOP" != "" ] && FEATOP="-xh $FEATOP";
    fi

    ### Extract features ###
    htrsh_extract_feats "${ff}_fea.png" "$ff.fea" $FEATOP;
    [ "$?" != 0 ] && return 1;

    echo "$ff.fea" >> "$FEATLST";

    oklines=$((oklines+1));

    [ "$PBASE" != "" ] && FEATS=$( echo "$FEATS"; echo "${ff}.fea"; );

    ### Remove temporal files ###
    #rm "${ff}_clean.png";
    [ "$htrsh_keeptmp" -lt 1 ] &&
      rm -f "${ff}.png" "${ff}_fea.png";
    [ "$htrsh_keeptmp" -lt 2 ] &&
      rm -f "${ff}_affine.png" "${ff}_affine.mat";
    [ "$htrsh_keeptmp" -lt 3 ] &&
      rm -f "${ff}_deslope.png" "${ff}_deslant.png";
  done

  [ "$oklines" = 0 ] &&
    echo "$FN: error: extracted features for zero lines: $XML" 1>&2 &&
    return 1;

  ### Project features if requested ###
  if [ "$PBASE" != "" ]; then
    htrsh_feats_project <( echo "$FEATS" | sed '/^$/d' ) "$PBASE" "$OUTDIR";
    [ "$?" != 0 ] && return 1;
  fi

  ### Generate new XML Page file ###
  xmlstarlet "${xmledit[@]}" "$XML" > "$XMLOUT";
  [ "$?" != 0 ] &&
    echo "$FN: error: problems generating XML file: $XMLOUT" 1>&2 &&
    return 1;

  if [ "$htrsh_feat_contour" = "yes" ] && [ "$REPLC" = "yes" ]; then
    xmledit=( ed --inplace );
    local id;
    #for id in $(xmlstarlet sel -t -m '//*/_:Coords[@fcontour]' -v ../@id -n "$XMLOUT"); do
    for id in $(xmlstarlet sel -t -m '//*/_:Property[@key="fcontour"]' -v ../@id -n "$XMLOUT"); do
      xmledit+=( -d "//*[@id='${id}']/_:Coords/@points" "//*[@id='${id}']/_:Coords/@value" );
      #xmledit+=( -r "//*[@id='${id}']/_:Coords/@fcontour" -v points );
      xmledit+=( -m "//*[@id='${id}']/_:Property[@key='fcontour']/@value" "//*[@id='${id}']/_:Coords" );
      xmledit+=( -r "//*[@id='${id}']/_:Coords/@value" "//*[@id='${id}']/_:Coords/@points" );
    done
    xmlstarlet "${xmledit[@]}" "$XMLOUT";
  fi

  return 0;
}


#----------------------------------#
# Model training related functions #
#----------------------------------#

##
## GAWK functions to convert a word to an array of hmm names
##
htrsh_gawk_func_word_to_chars='
  function load_special_chars( SPECIAL,   n,c,line,sline ) {
    delete SCHAR;
    delete SWORD;
    delete SMARK;
    NSPECIAL = 0;
    while( ( getline line<SPECIAL ) > 0 )
      if( line != "" ) {
        n = split(line,sline);
        c = substr( sline[1], 1, 1 );
        SCHAR[c] = "";
        NSPECIAL ++;
        SWORD[NSPECIAL] = sline[1];
        SMARK[NSPECIAL] = n == 1 ? sline[1] : sline[2] ;
      }
    close( SPECIAL );
  }

  function word_to_chars( word, hmms, hmmtype, endspace,   C,N,n,m,w,txt,cprev ) {
    delete hmms;
    C = 0;
    N = split( word, txt, "" );
    cprev = SPACE;
    for( n=1; n<=N; n++ ) {
      c = txt[n];
      if( c in SCHAR ) {
        for( m=1; m<=NSPECIAL; m++ ) {
          w = SWORD[m];
          if( w == substr(word,n,length(w)) ) {
            if( hmmtype == "overlap" )
              hmms[++C] = ( cprev SMARK[m] );
            cprev = SMARK[m];
            hmms[++C] = SMARK[m];
            n += length(w)-1;
            break;
          }
        }
        if( m <= NSPECIAL )
          continue;
      }
      if( n+1 <= N ) {
        cc = txt[n+1];
        if( ( cc >= "\xcc\x80"     && cc <= "\xcd\xaf" )     ||  # Combining Diacritical Marks 0300-036F
            ( cc >= "\xe1\xaa\xb0" && cc <= "\xe1\xab\xbf" ) ||  # Combining Diacritical Marks Extended 1AB0-1AFF
            ( cc >= "\xe1\xb7\x80" && cc <= "\xe1\xb7\xbf" ) ||  # Combining Diacritical Marks Supplement 1DC0-1DFF
            ( cc >= "\xe2\x83\x90" && cc <= "\xe2\x83\xbf" ) ||  # Combining Diacritical Marks for Symbols 20D0-20FF
            ( cc >= "\xef\xb8\xa0" && cc <= "\xef\xb8\xaf" ) ) { # Combining Half Marks FE20-FE2F
          c = ( c cc );
          n++;
        }
      }
      if( hmmtype == "overlap" )
        hmms[++C] = ( cprev c );
      cprev = c;
      hmms[++C] = c;
    }
    if( hmmtype == "overlap" )
      hmms[++C] = ( cprev SPACE );
    if( endspace == "yes" )
      hmms[++C] = SPACE;
    return C;
  }';

##
## GAWK functions to convert chars to words
##
htrsh_gawk_func_chars_to_word='
  function load_special_chars( SPECIAL,   n,c,line,sline ) {
    delete SCHAR;
    SCHAR[SPACE] = " ";
    while( ( getline line<SPECIAL ) > 0 )
      if( line != "" ) {
        n = split(line,sline);
        SCHAR[ n == 1 ? sline[1] : sline[2] ] = sline[1];
      }
    close( SPECIAL );
  }

  function chars_to_word( text ) {
    N = split(text,stext);
    txt = "";
    for( n=1; n<=N; n++ ) {
      s = stext[n];
      if( s in SCHAR )
        txt = (txt SCHAR[s]);
      else
        txt = (txt s);
    }
    return txt;
  }';

##
## Function that creates a dictionary from file lists with the representations of the words
##
htrsh_create_dict () {
  local FN="htrsh_create_dict";
  local EXTR="no";
  local CHARSEQ="no";
  local ENDSPACE="no";
  local WSPACE="no";
  if [ $# -lt 3 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Creates a dictionary from file lists with the representations of the words";
      echo "Usage: $FN ORIGINAL CANONIC DIPLOMATIC [ Options ]";
      echo "Options:";
      echo " -S (yes|no)     Whether to add start/end labels to dictionary (def.=$EXTR)";
      echo " -H (yes|no)     Whether to transform diplomatic to a character sequence (def.=$CHARSEQ)";
      echo " -E (yes|no)     Whether to add space at end of words (def.=$ENDSPACE)";
      echo " -W (yes|no)     Whether to add space as a word (def.=$WSPACE)";
      #echo " --[no-]extr-labs  Whether to add start/end labels to dictionary (def.=$EXTR)";
      #echo " --[no-]charseq    Whether to transform diplomatic to a character sequence (def.=$CHARSEQ)";
      #echo " --[no-]endspace   Whether to add space at end of words (def.=$ENDSPACE)";
      #echo " --[no-]space      Whether to add space as a word (def.=$WSPACE)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  #local OPTS=$( getopt --long extr-labs,no-extr-labs,charseq,no-charseq,endspace,no-endspace,space,no-space -n "$FN" -- "$@" );
  #eval set -- "$OPTS";
  #while true; do
  #  case "$1" in
  #    --extr-labs )    EXTR="yes";     ;;
  #    --no-extr-labs ) EXTR="no";      ;;
  #    --charseq )      CHARSEQ="yes";  ;;
  #    --no-charseq )   CHARSEQ="no";   ;;
  #    --endspace )     ENDSPACE="yes"; ;;
  #    --no-endspace )  ENDSPACE="no";  ;;
  #    --spase )        WSPACE="yes";   ;;
  #    --no-spase )     WSPACE="yes";   ;;
  #    -- ) shift; break ;;
  #    * )
  #      echo "$FN: error: unexpected input argument: $1" 1>&2;
  #      return 1;
  #  esac
  #  shift;
  #done

  local ORIGINAL="$1";
  local CANONIC="$2";
  local DIPLOMATIC="$3";
  shift 3;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-S" ]; then
      EXTR="$2";
    elif [ "$1" = "-H" ]; then
      CHARSEQ="$2";
    elif [ "$1" = "-E" ]; then
      ENDSPACE="$2";
    elif [ "$1" = "-W" ]; then
      WSPACE="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  ### Create dictionary ###
  paste "$CANONIC" "$ORIGINAL" "$DIPLOMATIC" \
    | gawk -v hmmtype="$htrsh_hmm_type" -v charseq="$CHARSEQ" -v extr="$EXTR" \
           -v endspace="$ENDSPACE" -v wspace="$WSPACE" \
           -v SPACE="$htrsh_symb_space" -v SPECIAL=<( echo "$htrsh_special_chars" ) \
        "$htrsh_gawk_func_word_to_chars"'
        BEGIN {
          load_special_chars( SPECIAL );
          FS = "\t";
        }
        { canonic_count[$1] ++;
          variant_count[$1][$2] ++;
          variant_diplom[$1][$2] = $3;
          if( NF != 3 ) {
            printf( "'"$FN"': error: at line %d, expected three tokens: %s", FNR, $0 ) > "/dev/stderr";
            exit 1;
          }
        }
        END {
          if( wspace == "yes" )
            printf( "\"%s\"\t[]\t1\t%s\n", SPACE, SPACE );
          if( extr == "yes" ) {
            printf( "\"<s>\"\t[]\t1\t%s\n", SPACE );
            printf( "\"</s>\"\t[]\n" );
          }
          for( canonic in canonic_count ) {
            wcanonic = canonic;
            gsub( "\x22", "\\\x22", wcanonic );
            gsub( "\x27", "\\\x27", wcanonic );
            if( wcanonic == "" ) {
              printf( "'"$FN"': warning: ignored empty cannonic word" ) > "/dev/stderr";
              continue;
            }
            for( variant in variant_count[canonic] ) {
              vprob = sprintf("%g",variant_count[canonic][variant]/canonic_count[canonic]);
              if( ! match(vprob,/\./) )
                vprob = ( vprob ".0" );
              printf( "\"%s\"\t[%s]\t%s\t", wcanonic, variant, vprob );
              diplom = variant_diplom[canonic][variant];
              if( charseq == "no" )
                printf( "%s\n", diplom );
              else {
                N = word_to_chars( diplom, hmms, hmmtype, endspace );
                for( n=1; n<=N; n++ )
                  printf( n==1 ? "%s" : " %s", hmms[n] );
                printf( "\n" );
              }
            }
          }
        }';
}

##
## Function that trains a language model and creates related files
##
# @todo dictionary without hmm models list, new function to add hmm models list, only create hmm dictionary when training
htrsh_langmodel_train () {
  local FN="htrsh_langmodel_train";
  local OUTDIR=".";
  local ORDER="2";
  local TOKENIZER="cat";
  local CANONIZER="cat";
  local DIPLOMATIZER="cat";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Trains a language model and creates related files";
      echo "Usage: $FN TEXTFILE [ Options ]";
      echo "Options:";
      echo " -o ORDER        Order of the language model (def.=$ORDER)";
      echo " -d OUTDIR       Directory for output models and temporal files (def.=$OUTDIR)";
      echo " -T TOKENIZER    Tokenizer pipe command (def.=none)";
      echo " -C CANONIZER    Word canonization pipe command, e.g. convert to upper (def.=none)";
      echo " -D DIPLOMATIZER Word diplomatizer pipe command, e.g. remove expansions (def.=none)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local TXT="$1"; [ "$TXT" = "-" ] && TXT="/dev/stdin";
  shift;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-o" ]; then
      ORDER="$2";
    elif [ "$1" = "-d" ]; then
      OUTDIR="$2";
    elif [ "$1" = "-T" ]; then
      TOKENIZER="$2";
    elif [ "$1" = "-C" ]; then
      CANONIZER="$2";
    elif [ "$1" = "-D" ]; then
      DIPLOMATIZER="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  local ORDEROPTS=( -order $ORDER );
  local n;
  for n in $(seq 1 $ORDER); do
    ORDEROPTS+=( -ukndiscount$n );
  done

  ### Tokenize training text ###
  cat "$TXT" \
    | $TOKENIZER \
    > "$OUTDIR/text_tokenized.txt";

  ### Create dictionary ###
  { htrsh_create_dict \
      <( cat "$OUTDIR/text_tokenized.txt" \
           | tr ' ' '\n' | sed '/^$/d' ) \
      <( cat "$OUTDIR/text_tokenized.txt" \
           | $CANONIZER \
           | tee "$OUTDIR/text_canonized.txt" \
           | tr ' ' '\n' | sed '/^$/d' ) \
      <( cat "$OUTDIR/text_tokenized.txt" \
           | $DIPLOMATIZER \
           | tr ' ' '\n' | sed '/^$/d' ) \
      -S yes -H yes -E yes;
    [ "$?" != 0 ] &&
      echo "$FN: error: problems creating dictionary" 1>&2 &&
      return 1;
  } | LC_ALL=C.UTF-8 sort \
    > "$OUTDIR/dictionary.txt";

  ### Create vocabulary ###
  awk '
    { if( $1 != "\"<s>\"" && $1 != "\"</s>\"" )
        print $1;
    }' "$OUTDIR/dictionary.txt" \
    | sed 's|^"\(.*\)"$|\1|; s|\\\(["\x27]\)|\1|g;' \
    > "$OUTDIR/vocabulary.txt";

  ### Create n-gram ###
  ngram-count -text "$OUTDIR/text_canonized.txt" -vocab "$OUTDIR/vocabulary.txt" \
      -lm - "${ORDEROPTS[@]}" \
    | sed 's|\(["\x27]\)|\\\1|g' \
    > "$OUTDIR/langmodel_${ORDER}-gram.arpa";

  HBuild -n "$OUTDIR/langmodel_${ORDER}-gram.arpa" -s "<s>" "</s>" \
    "$OUTDIR/dictionary.txt" "$OUTDIR/langmodel_${ORDER}-gram.lat";

  [ "$htrsh_keeptmp" -lt 1 ] &&
    rm "$OUTDIR/vocabulary.txt";

  return 0;
}

##
## Function that prints to stdout HMM prototype(s) in HTK format
##
htrsh_hmm_proto () {
  local FN="htrsh_hmm_proto";
  local PNAME="proto";
  local GLOBAL="";
  local MEAN="";
  local VARIANCE="";
  local DISCR="no";
  local RAND="no";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Prints to stdout HMM prototype(s) in HTK format";
      echo "Usage: $FN (DIMS|CODES) STATES [ Options ]";
      echo "Options:";
      echo " -n PNAME     Proto names (optionally with #states), if several separated by '\n' (def.=$PNAME)";
      echo " -g GLOBAL    Include given global options string (def.=none)";
      echo " -m MEAN      Use given mean vector (def.=zeros)";
      echo " -v VARIANCE  Use given variance vector (def.=ones)";
      echo " -D (yes|no)  Whether proto should be discrete (def.=$DISCR)";
      echo " -R (yes|no)  Whether to randomize (def.=$RAND)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local DIMS="$1";
  local STATES="$2";
  shift 2;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-n" ]; then
      PNAME="$2";
    elif [ "$1" = "-g" ]; then
      GLOBAL="$2";
    elif [ "$1" = "-m" ]; then
      MEAN="$2";
    elif [ "$1" = "-v" ]; then
      VARIANCE="$2";
    elif [ "$1" = "-D" ]; then
      DISCR="$2";
    elif [ "$1" = "-R" ]; then
      RAND="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  ### Print global options ###
  if [ "$DISCR" = "yes" ]; then
    echo '~o <DISCRETE> <STREAMINFO> 1 1';
  else
    if [ "$GLOBAL" != "off" ]; then
      echo "~o";
      echo "<STREAMINFO> 1 $DIMS";
      echo "<VECSIZE> $DIMS<NULLD><USER><DIAGC>";
    fi

    [ "$MEAN" = "" ] &&
      MEAN=$(echo $DIMS | awk '{for(d=$1;d>0;d--)printf(" 0.0")}');

    [ "$VARIANCE" = "" ] &&
      VARIANCE=$(echo $DIMS | awk '{for(d=$1;d>0;d--)printf(" 1.0")}');
  fi

  [ "$GLOBAL" != "" ] && [ "$GLOBAL" != "off" ] &&
    echo "$GLOBAL";

  ### Print prototype(s) ###
  echo "$PNAME" \
    | awk -v D=$DIMS -v SS=$STATES \
          -v MEAN="$MEAN" -v VARIANCE="$VARIANCE" \
          -v DISCR=$DISCR -v RAND=$RAND '
        BEGIN { srand('$RANDOM'); }
        { S = NF > 1 ? $2 : SS;
          printf("~h \"%s\"\n",$1);
          printf("<BEGINHMM>\n");
          printf("<NUMSTATES> %d\n",S+2);
          for(s=1;s<=S;s++) {
            printf("<STATE> %d\n",s+1);
            if(DISCR=="yes") {
              printf("<NUMMIXES> %d\n",D);
              printf("<DPROB>");
              if(RAND=="yes") {
                tot=0;
                for(d=1;d<=D;d++)
                  tot+=rnd[d]=rand();
                for(d=1;d<=D;d++) {
                  v=int(sprintf("%.0f",-2371.8*log(rnd[d]/tot)));
                  printf(" %d",v>32767?32767:v);
                }
                delete rnd;
              }
              else
                for(d=1;d<=D;d++)
                  printf(" %.0f",-2371.8*log(1/D));
              printf("\n");
            }
            else {
              printf("<MEAN> %d\n",D);
              if(RAND=="yes") {
                for(d=1;d<=D;d++)
                  printf(d==1?"%g":" %g",(rand()-0.5)/10);
                printf("\n");
                printf("<VARIANCE> %d\n",D);
                for(d=1;d<=D;d++)
                  printf(d==1?"%g":" %g",1+(rand()-0.5)/10);
                printf("\n");
              }
              else {
                printf("%s\n",MEAN);
                printf("<VARIANCE> %d\n",D);
                printf("%s\n",VARIANCE);
              }
            }
          }
          printf("<TRANSP> %d\n",S+2);
          printf(" 0.0 1.0");
          for(a=2;a<=S+1;a++)
            printf(" 0.0");
          printf("\n");
          for(aa=1;aa<=S;aa++) {
            for(a=0;a<=S+1;a++)
              if(RAND=="yes") {
                if( a == aa ) {
                  pr=rand();
                  pr=pr<1e-9?1e-9:pr;
                  printf(" %g",pr);
                }
                else if( a == aa+1 )
                  printf(" %g",1-pr);
                else
                  printf(" 0.0");
              }
              else {
                if( a == aa )
                  printf(" 0.6");
                else if( a == aa+1 )
                  printf(" 0.4");
                else
                  printf(" 0.0");
              }
            printf("\n");
          }
          for(a=0;a<=S+1;a++)
            printf(" 0.0");
          printf("\n");
          printf("<ENDHMM>\n");
        }';

  return 0;
}

##
## Function that trains HMMs for a given feature list and mlf
##
htrsh_hmm_train () {
  local FN="htrsh_hmm_train";
  local OUTDIR=".";
  local CODES="0";
  local PROTO="";
  local EDITS="";
  local DIC="";
  local EXCLREALIGN="";
  local KEEPITERS="yes";
  local RESUME="yes";
  local RAND="no";
  local THREADS="1";
  local NUMELEM="balance";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Trains HMMs for a given feature list and mlf";
      echo "Usage: $FN FEATLST MLF [ Options ]";
      echo "Options:";
      echo " -d OUTDIR    Directory for output models and temporal files (def.=$OUTDIR)";
      echo " -c CODES     Train discrete model with given codebook size (def.=false)";
      echo " -P PROTO     Use PROTO as initialization prototype (def.=false)";
      echo " -e EDITS     File with HHEd commands to execute to proto before training (def.=false)";
      echo " -D DICT      Realign using given dictionary, requires word MLF (def.=false)";
      echo " -E MLF       HMM-based MLF for training only features, i.e. no realigning (def.=false)";
      echo " -k (yes|no)  Whether to keep models per iteration, including initialization (def.=$KEEPITERS)";
      echo " -r (yes|no)  Whether to resume previous training, looks for models per iteration (def.=$RESUME)";
      echo " -R (yes|no)  Whether to randomize initialization prototype (def.=$RAND)";
      echo " -T THREADS   Threads for parallel processing (def.=$THREADS)";
      echo " -n NUMELEM   Elements per instance for parallel (see run_parallel) (def.=$NUMELEM)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local FEATLST="$1";
  local MLF="$2";
  shift 2;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-d" ]; then
      OUTDIR="$2";
    elif [ "$1" = "-c" ]; then
      CODES="$2";
    elif [ "$1" = "-P" ]; then
      PROTO="$2";
    elif [ "$1" = "-e" ]; then
      EDITS="$2";
    elif [ "$1" = "-D" ]; then
      DIC="$2";
    elif [ "$1" = "-E" ]; then
      EXCLREALIGN="$2";
    elif [ "$1" = "-k" ]; then
      KEEPITERS="$2";
    elif [ "$1" = "-r" ]; then
      RESUME="$2";
    elif [ "$1" = "-R" ]; then
      RAND="$2";
    elif [ "$1" = "-T" ]; then
      THREADS="$2";
    elif [ "$1" = "-n" ]; then
      NUMELEM="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  if [ ! -e "$FEATLST" ]; then
    echo "$FN: error: feature list not found: $FEATLST" 1>&2;
    return 1;
  elif [ ! -e "$MLF" ]; then
    echo "$FN: error: MLF file not found: $MLF" 1>&2;
    return 1;
  elif [ "$PROTO" != "" ] && [ ! -e "$PROTO" ]; then
    echo "$FN: error: initialization prototype not found: $PROTO" 1>&2;
    return 1;
  elif [ "$EDITS" != "" ] && [ ! -e "$EDITS" ]; then
    echo "$FN: error: HHEd script not found: $HHEd" 1>&2;
    return 1;
  elif [ "$DIC" != "" ] && [ ! -e "$DIC" ]; then
    echo "$FN: error: realigning dictionary not found: $DIC" 1>&2;
    return 1;
  elif [ "$EXCLREALIGN" != "" ] && [ ! -e "$EXCLREALIGN" ]; then
    echo "$FN: error: MLF file for training only features not found: $EXCLREALIGN" 1>&2;
    return 1;
  elif [ "$CODES" != 0 ] && [ $(HList -z -h "$(head -n 1 "$FEATLST")" | grep DISCRETE_K | wc -l) = 0 ]; then
    echo "$FN: error: features are not discrete" 1>&2;
    return 1;
  fi

  ### Prepare for realignment ###
  if [ "$DIC" != "" ]; then
    local WMLF="$MLF";
    MLF="$OUTDIR/train.mlf";
    htrsh_mlf_word_to_chars "$WMLF" "$DIC" > "$MLF";
    [ "$?" != 0 ] &&
      echo "$FN: error: MLF does not appear to be word based according to dictionary" 1>&2 &&
      return 1;
    local REALIGNLST="$FEATLST";
    if [ "$EXCLREALIGN" != "" ]; then
      REALIGNLST="$OUTDIR/realign.lst";
      sed -n '/\.lab"$/{ s|.*/||; s|\.lab"$||; p; }' "$EXCLREALIGN" \
        | gawk '
            { if( ARGIND == 1 )
                excl[$0] = "";
              else {
                fea = gensub( /^.*\//, "", 1, gensub(/\.fea$/,"",1,$0) );
                if( ! ( fea in excl ) )
                  print;
              }
            }' - "$FEATLST" \
            > "$REALIGNLST";
      sed '/^#!MLF!#/d' "$EXCLREALIGN" >> "$MLF";
    fi

    local HMMLST=$(awk '{for(n=4;n<=NF;n++)print $n}' "$DIC" \
                     | LC_ALL=C.UTF-8 sort -u);
  fi

  ### Auxiliary variables ###
  local DIMS=$(HList -z -h $(head -n 1 "$FEATLST") | sed -n '/Num Comps:/{s|.*Num Comps: *||;s| .*||;p;}');

  [ "$DIC" = "" ] &&
  local HMMLST=$(cat "$MLF" \
                   | sed '/^#!MLF!#/d; /^"\*\//d; /^\.$/d; s|^"\(.*\)"$|\1|;' \
                   | LC_ALL=C.UTF-8 sort -u);

  local STATES=$( gawk -v d="$htrsh_hmm_states" '
      { if( ARGIND == 2 )
          printf( "%s %s\n", $1, ( ($1 in states) ? states[$1] : d ) );
        else if( NF > 1 ) {
          expr = gensub( /d/, d, "g", $2 );
          expr = sprintf( "gawk \x27 BEGIN { print %s; } \x27", expr );
          expr | getline s;
          if( s == "" )
            printf( "'"$FN"': error: unable to interpret number of states: %s\n", $0 ) >> "/dev/stderr";
          states[$1] = s;
        }
      }' <( echo "$htrsh_hmm_ndstates" ) <( echo "$HMMLST" ) );

  ### Discrete training ###
  if [ "$CODES" -gt 0 ]; then
    ### Initialization ###
    if [ "$PROTO" != "" ]; then
      cp -p "$PROTO" "$OUTDIR/Macros_hmm.gz";
    else
      htrsh_hmm_proto "$CODES" "$htrsh_hmm_states" -D yes -n "$STATES" -R $RAND \
        | gzip > "$OUTDIR/Macros_hmm.gz";
    fi

    [ "$KEEPITERS" = "yes" ] &&
      cp -p "$OUTDIR/Macros_hmm.gz" "$OUTDIR/Macros_hmm_i00.gz";

    ### Iterate ###
    local i;
    for i in $(seq -f %02.0f 1 $htrsh_hmm_iter); do
      echo "$FN: info: HERest iteration $i" 1>&2;
      HERest $htrsh_HTK_HERest_opts -C <( echo "$htrsh_HTK_config" ) \
        -S "$FEATLST" -I "$MLF" -H "$OUTDIR/Macros_hmm.gz" <( echo "$HMMLST" ) 1>&2;
      if [ "$?" != 0 ]; then
        echo "$FN: error: problem with HERest" 1>&2;
        mv "$OUTDIR/Macros_hmm.gz" "$OUTDIR/Macros_hmm_i${i}_err.gz";
        return 1;
      fi
      [ "$KEEPITERS" = "yes" ] &&
        cp -p "$OUTDIR/Macros_hmm.gz" "$OUTDIR/Macros_hmm_i$i.gz";
    done
    [ "$KEEPITERS" = "yes" ] &&
      cp -p "$OUTDIR/Macros_hmm.gz" "$OUTDIR/Macros_hmm_i$i.gz";

  ### Continuous training ###
  else
    ### Initialization ###
    if [ "$PROTO" != "" ]; then
      cp -p "$PROTO" "$OUTDIR/Macros_hmm.gz";

      [ "$KEEPITERS" = "yes" ] &&
        cp -p "$OUTDIR/Macros_hmm.gz" "$OUTDIR/Macros_hmm_g001_i00.gz";

    elif [ "$RESUME" != "no" ] && [ -e "$OUTDIR/Macros_hmm_g001_i00.gz" ]; then
      RESUME="Macros_hmm_g001_i00.gz";
      cp -p "$OUTDIR/Macros_hmm_g001_i00.gz" "$OUTDIR/Macros_hmm.gz";

    else
      RESUME="no";

      htrsh_hmm_proto "$DIMS" 1 | gzip > "$OUTDIR/proto";
      HCompV $htrsh_HTK_HCompV_opts -C <( echo "$htrsh_HTK_config" ) \
        -S "$FEATLST" -M "$OUTDIR" "$OUTDIR/proto" 1>&2;

      local GLOBAL=$(< "$OUTDIR/vFloors");
      local MEAN=$(gzip -dc "$OUTDIR/proto" | sed -n '/<MEAN>/{N;s|.*\n||;p;q;}');
      local VARIANCE=$(gzip -dc "$OUTDIR/proto" | sed -n '/<VARIANCE>/{N;s|.*\n||;N;p;q;}');

      htrsh_hmm_proto "$DIMS" "$htrsh_hmm_states" -n "$STATES" \
          -g "$GLOBAL" -m "$MEAN" -v "$VARIANCE" \
        | gzip \
        > "$OUTDIR/Macros_hmm.gz";

      [ "$EDITS" != "" ] &&
        HHEd $htrsh_HTK_HHEd_opts -C <( echo "$htrsh_HTK_config" ) -H "$OUTDIR/Macros_hmm.gz" \
            -M "$OUTDIR" "$EDITS" <( echo "$HMMLST" ) 1>&2;

      [ "$KEEPITERS" = "yes" ] &&
        cp -p "$OUTDIR/Macros_hmm.gz" "$OUTDIR/Macros_hmm_g001_i00.gz";
    fi

    local TS=$(($(date +%s%N)/1000000));

    ### Training loop ###
    local g="1";
    local gg i;
    while [ "$g" -le "$htrsh_hmm_nummix" ]; do
      ### Duplicate Gaussians ###
      if [ "$g" -gt 1 ] && ! ( [ "$RESUME" != "no" ] && [ -e "$OUTDIR/Macros_hmm_g${gg}_i$i.gz" ] ); then
        echo "$FN: info: duplicating Gaussians to $g" 1>&2;
        HHEd $htrsh_HTK_HHEd_opts -C <( echo "$htrsh_HTK_config" ) -H "$OUTDIR/Macros_hmm.gz" \
          -M "$OUTDIR" <( echo "MU $g {*.state[2-$((htrsh_hmm_states-1))].mix}" ) \
          <( echo "$HMMLST" ) 1>&2;
      fi

      ### Re-estimation iterations ###
      local gg=$(printf %.3d $g);
      for i in $(seq -f %02.0f 1 $htrsh_hmm_iter); do
        if [ "$RESUME" != "no" ] && [ -e "$OUTDIR/Macros_hmm_g${gg}_i$i.gz" ]; then
          RESUME="Macros_hmm_g${gg}_i$i.gz";
          cp -p "$OUTDIR/Macros_hmm_g${gg}_i$i.gz" "$OUTDIR/Macros_hmm.gz";
          continue;
        fi

        [ "$RESUME" != "no" ] && [ "$RESUME" != "yes" ] &&
          echo "$FN: info: resuming from $RESUME" 1>&2;
        RESUME="no";

        echo "$FN: info: $g Gaussians HERest iteration $i" 1>&2;

        ### Multi-thread ###
        if [ "$THREADS" -gt 1 ]; then
          run_parallel -T $THREADS -n $NUMELEM -l "$FEATLST" \
            HERest $htrsh_HTK_HERest_opts -C <( echo "$htrsh_HTK_config" ) -p '{#}' \
            -S '{@}' -I "$MLF" -H "$OUTDIR/Macros_hmm.gz" -M "$OUTDIR" <( echo "$HMMLST" ) 1>&2;
          [ "$?" != 0 ] &&
            echo "$FN: error: problem with parallel HERest" 1>&2 &&
            return 1;
          HERest $htrsh_HTK_HERest_opts -C <( echo "$htrsh_HTK_config" ) \
            -p 0 -H "$OUTDIR/Macros_hmm.gz" <( echo "$HMMLST" ) "$OUTDIR/"*.acc 1>&2;
          [ "$?" != 0 ] &&
            echo "$FN: error: problem with accumulation HERest" 1>&2 &&
            return 1;
          rm "$OUTDIR/"*.acc;

        ### Single thread ###
        else
          HERest $htrsh_HTK_HERest_opts -C <( echo "$htrsh_HTK_config" ) \
            -S "$FEATLST" -I "$MLF" -H "$OUTDIR/Macros_hmm.gz" <( echo "$HMMLST" ) 1>&2;
          [ "$?" != 0 ] &&
            echo "$FN: error: problem with HERest" 1>&2 &&
            return 1;
        fi

        local TE=$(($(date +%s%N)/1000000)); echo "$FN: time g=$g i=$i: $((TE-TS)) ms" 1>&2; TS="$TE";

        [ "$KEEPITERS" = "yes" ] &&
          cp -p "$OUTDIR/Macros_hmm.gz" "$OUTDIR/Macros_hmm_g${gg}_i$i.gz";

        ### Realign using given dictionary ###
        if [ "$DIC" != "" ]; then
          echo "$FN: realigning using dictionary" 1>&2;
          local k=$(ls "$OUTDIR/realigned.mlf~"* 2>/dev/null | wc -l);
          local kk=$(ls "$MLF"~* 2>/dev/null | wc -l);
          htrsh_hvite_parallel "$THREADS" \
            HVite $htrsh_HTK_HVite_align_opts -C <( echo "$htrsh_HTK_config" ) -H "$OUTDIR/Macros_hmm.gz" -S "$REALIGNLST" -a -m -I "$WMLF" -i "$OUTDIR/realigned.mlf~$k" "$DIC" <( echo "$HMMLST" );
          [ "$?" != 0 ] &&
            echo "$FN: error: problems realigning with HVite" 1>&2 &&
            return 1;

          ln -fs "realigned.mlf~$k" "$OUTDIR/realigned.mlf";

          awk '
            { if( ARGIND == 1 )
                realigned[$1] = "";
              else if( ! ( $1 in realigned ) )
                print;
            }' <( sed -n '/^".*\/.*\.rec"$/{ s|.*/||; s|\.rec"$||; p; }' "$OUTDIR/realigned.mlf~$k" ) \
               <( sed 's|.*/||; s|\.fea$||;' "$REALIGNLST" ) \
            > "$OUTDIR/failrealign.lst~$k";

          mv "$MLF" "$MLF~$kk";

          gawk '
            { if( match($0,/^".+\/[^/]+\.rec"$/) )
                $0 = gensub( /^".+\/([^/]+)\.rec"$/, "\"*/\\1.lab\"", 1, $0 );
              else if( NR > 1 && $0 != "." )
                $0 = $3;
              print;
            }' "$OUTDIR/realigned.mlf~$k" \
            | htrsh_fix_mlf_quotes - \
            > "$MLF";
          if [ -s "$OUTDIR/failrealign.lst~$k" ]; then
            htrsh_mlf_filter "$OUTDIR/failrealign.lst~$k" "$MLF~$kk" >> "$MLF";
            ln -fs "failrealign.lst~$k" "$OUTDIR/failrealign.lst";
          else
            rm -f "$OUTDIR/failrealign.lst~$k" "$OUTDIR/failrealign.lst";
          fi
          [ "$EXCLREALIGN" != "" ] &&
            sed '/^#!MLF!#/d' "$EXCLREALIGN" >> "$MLF";

          local TE=$(($(date +%s%N)/1000000)); echo "$FN: realign time g=$g i=$i: $((TE-TS)) ms" 1>&2; TS="$TE";
        fi
      done

      cp -p "$OUTDIR/Macros_hmm.gz" "$OUTDIR/Macros_hmm_g${gg}_i$i.gz";
      g=$((g+g));
    done

    [ "$RESUME" != "no" ] && [ "$RESUME" != "yes" ] &&
      echo "$FN: warning: model already trained: $RESUME" 1>&2;

    echo "$OUTDIR/Macros_hmm_g${gg}_i$i.gz";
  fi

  rm -f "$OUTDIR/proto" "$OUTDIR/vFloors" "$OUTDIR/Macros_hmm.gz";

  return 0;
}


#----------------------------#
# Decoding related functions #
#----------------------------#

##
## Function that executes N parallel threads of HVite or HLRescore for a given feature list
##
# @todo When not in alignment mode (i.e. no -a) enable beam search retry for -t f i l
htrsh_hvite_parallel () {
  local FN="htrsh_hvite_parallel";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Executes N parallel threads of HVite or HLRescore for a given feature list";
      echo "Usage: $FN THREADS (HVite|HLRescore) OPTIONS";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local THREADS="$1";
  shift;
  if [ "$THREADS" -le 1 ]; then
    "$@";
    return $?;
  fi
  local CMD=( "$1" );
  shift;

  local TMP="${TMPDIR:-/tmp}";
  TMP=$(mktemp -d --tmpdir="$TMP" ${FN}_XXXXX);
  [ ! -d "$TMP" ] &&
    echo "$FN: error: failed to create temporal directory" 1>&2 &&
    return 1;

  local FEATLST="";
  local MLF="";
  while [ $# -gt 0 ]; do
    if [ -p "$1" ]; then
      ### Pipes fail within run_parallel in CentOS bash 4.1.2(1)-release ###
      local num=$(ls "$TMP/pipe"* 2>/dev/null | wc -l);
      cat "$1" > "$TMP/pipe$num";
      CMD+=( "$TMP/pipe$num" );
    else
      CMD+=( "$1" );
    fi
    if [ "$1" = "-S" ]; then
      FEATLST="$2";
      CMD+=( "{@}" );
      shift 1;
    elif [ "$1" = "-i" ]; then
      MLF="$2";
      CMD+=( "$TMP/mlf_{#}" );
      shift 1;
    fi
    shift 1;
  done

  if [ "$CMD" != "HVite" ] && [ "$CMD" != "HLRescore" ]; then
    echo "$FN: error: command has to be either HVite or HLRescore" 1>&2;
    return 1;
  elif [ "$FEATLST" = "" ]; then
    echo "$FN: error: a feature list using option -S must be given" 1>&2;
    return 1;
  elif [ ! -e "$FEATLST" ]; then
    echo "$FN: error: feature list file not found: $FEATLST" 1>&2;
    return 1;
  fi

  sort -R "$FEATLST" \
    | run_parallel -T $THREADS -n balance -l - -d "$TMP" "${CMD[@]}" 1>&2;
  [ "$?" != 0 ] &&
    echo "$FN: error: problems executing $CMD ($TMP)" 1>&2 &&
    return 1;

  [ "$MLF" != "" ] &&
    { echo "#!MLF!#";
      sed '/^#!MLF!#/d' "$TMP/mlf_"*;
    } > "$MLF";

  [ "$htrsh_keeptmp" -lt 1 ] &&
    rm -r "$TMP";
  return 0;
}

##
## Function that fixes the quotes of an MLF file
##
htrsh_fix_mlf_quotes () {
  local FN="htrsh_fix_mlf_quotes";
  local COL="1";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Fixes the quotes of rec MLFs";
      echo "Usage: $FN MLF [ Options ]";
      echo "Options:";
      echo " -c COLUMN    The column number to fix (def.=$COL)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local MLF="$1";
  shift 1;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-c" ]; then
      COL="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  gawk -v pdot=0 -v col=$COL '
    { if( $0 == "#!MLF!#" || match($0,/^".+\/.+\.[lr][ae][bc]"$/) ) {
        if( pdot )
          print ".";
        pdot = 0;
        print;
      }
      else {
        if( pdot )
          print "\".\"";
        pdot = 0;
        if( $0 == "." )
          pdot = 1;
        else {
          if( match($col,/^\x27.+\x27$/) )
            $col = gensub( /^\x27(.+)\x27$/, "\\1", 1, $col );
          if( ! match($col,/^".+"$/) )
            $col = ("\"" gensub( /\x22/, "\\\\\x22", "g", $col ) "\"");
          print;
        }
      }
    }
    END {
      if( pdot )
        print ".";
    }' "$MLF";
}

##
## Function that fixes the quotes of rec MLFs
##
htrsh_fix_rec_mlf_quotes () { htrsh_fix_mlf_quotes "$1" -c 3; }

##
## Function that replaces special character names with corresponding characters
##
# @todo should modify this so that the replacement is only in TextEquiv/Unicode, not all the XML
htrsh_fix_rec_names () {
  local FN="htrsh_fix_rec_names";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Replaces special HMM model names with corresponding characters";
      echo "Usage: $FN XMLIN";
    } 1>&2;
    return 1;
  fi

  local SED_REP="s|$htrsh_symb_space| |g;"$(
    echo "$htrsh_special_chars" \
      | sed '
          s/^\([^ ]*\) \([^ ]*\)/s|\2|\1|g;/;
          s/&/\\\&amp;/g;
          s/</\\\&lt;/g;
          s/>/\\\&gt;/g;' );

  sed -i "$SED_REP" "$1";

  return 0;
}

##
## Function that extracts MLF samples for a given list
##
htrsh_mlf_filter () {
  local FN="htrsh_mlf_filter";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Extracts MLF samples for a given list";
      echo "Usage: $FN LIST MLF";
    } 1>&2;
    return 1;
  fi

  gawk -v PRNT=0 '
    { if( ARGIND == 1 )
        filter[$1] = "";
      else {
        if( match($1,/^".+\.[lr][ae][bc]"$/) ) {
          samp = gensub( /^"*(.+)\.[lr][ae][bc]"$/, "\\1", 1, gensub(/.*\//,"",1,$1) );
          PRNT = samp in filter ? 1 : 0 ;
        }
        if( PRNT )
          print;
      }
    }' <( if [ "$1" = "-" ] || [ -e "$1" ]; then cat "$1"; else echo "$1"; fi ) "$2";
}


#-----------------------------#
# Alignment related functions #
#-----------------------------#

##
## Function that prepares a rec MLF for inserting alignment information in a Page XML
##
htrsh_mlf_prepalign () {
  local FN="htrsh_mlf_prepalign";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Prepares a rec MLF for inserting alignment information in a Page XML";
      echo "Usage: $FN MLF";
    } 1>&2;
    return 1;
  fi

  gawk -v SPACE="$htrsh_symb_space" '
    { if( $0 != "#!MLF!#" && ! match( $0, /\.rec"$/ ) ) {
        if( $0 != "." ) {
          printf( "%s %s %s\n", $1, $1, SPACE );
          PE = $2;
        }
        else
          printf( "%s %s %s\n", PE, PE, SPACE );
        if( match( $3, /^".*"$/ ) )
          $3 = gensub( /\\"/, "\"", "g", gensub( /^"(.+)"$/, "\\1", 1, $3 ) );
     }
     print;
   }' "$1";

  return $?;
}

##
## Function that prepares an MLF result of diplomatization for inserting alignment information in a Page XML
##
htrsh_mlf_diplom_prepalign () {
  local FN="htrsh_mlf_diplom_prepalign";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Prepares an MLF result of diplomatization for inserting alignment information in a Page XML";
      echo "Usage: $FN MLF DIC";
    } 1>&2;
    return 1;
  fi

  local MLF="$1";
  local DICT="$2";

  cat "$MLF" \
    | awk -v SPACE="$htrsh_symb_space" '
      { if( NF == 1 || $3 == SPACE )
          printf("%s\n",$0);
        else
          printf("%s ",$0);
      }' \
    | gawk -v SPACE="$htrsh_symb_space" '
      { if( ARGIND == 1 ) {
          full = substr($2,2,length($2)-2);
          canon = match($1,/^".+"$/) ?
            gensub( /\\"/, "\"", "g", substr($1,2,length($1)-2) ) : $1 ;
          $1 = $2 = $3 = "";
          if( $NF == SPACE )
            NF--;
          dipl = gensub( /^ +/, "", 1, $0 );
          #printf("full=%s canon=%s dipl=%s\n",full,canon,dipl);
          if( (canon,dipl) in dict )
            printf("warning: duplicate dictionary entry: %s : %s : %s\n",canon,dipl,full) >> "/dev/stderr";
          else
            dict[canon,dipl] = full;
        }
        else if( NF == 1 || NF == 5 || match($0,/^".+\/[^/]+\.rec"$/) ) {
          if( match($0,/^".+\/[^/]+\.rec"$/) )
            $0 = gensub( /^".+\/([^/]+)\.rec"$/, "\"*/\\1.rec\"", 1, $0 );
          else if( NF == 5 )
            NF -= 2;
          print;
        }
        else {
          canon = $5;
          space = "";
          if( $(NF-1) == SPACE ) {
            space = ( $(NF-3) " " $(NF-2) " " $(NF-1) );
            NF = NF-4;
          }
          dipl = $3;
          for( n=8; n<NF; n+=4 )
            dipl = ( dipl " " $n );
          if( (canon,dipl) in dict )
            printf("%s %s %s\n",$1,NF==5?$2:$(NF-2),dict[canon,dipl]);
          else
            printf("error: not in dict: %s %s %s :: %s\n",$1,$(NF-2),canon,dipl) >> "/dev/stderr";
          if( space )
            print space;
        }
      }' "$DICT" - ;
}


##
## Function that inserts alignment information in an XML Page given a rec MLF
##
htrsh_pagexml_insertalign_lines () {(
  local FN="htrsh_pagexml_insertalign_lines";
  #local SRC="lines";
  local MAP="";
  local LINETXT="no";
  local HTKALI="yes";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Inserts alignment information in an XML Page given a rec MLF";
      echo "Usage: $FN XML MLF [ Options ]";
      echo "Options:";
      #echo " -s SOURCE      Source of TextEquiv, either 'regions' or 'lines' (def.=$SRC)";
      echo " -M MAPFILE     Diplomatic text mapping file (def.=none)";
      echo " -l (yes|no)    Whether to set the TextEquiv of the lines (def.=$LINETXT)";
      echo " -htk (yes|no)  MLF aligments as HTK *100k values (def.=$HTKALI)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local MLF="$2";
  shift 2;
  while [ $# -gt 0 ]; do
    #if [ "$1" = "-s" ]; then
    #  SRC="$2";
    #elif [ "$1" = "-M" ]; then
    if [ "$1" = "-M" ]; then
      MAP="$2";
    elif [ "$1" = "-l" ]; then
      LINETXT="$2";
    elif [ "$1" = "-htk" ]; then
      HTKALI="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  if [ ! -e "$XML" ]; then
    echo "$FN: error: XML Page file not found: $XML" 1>&2;
    return 1;
  elif [ ! -e "$MLF" ]; then
    echo "$FN: error: MLF file not found: $MLF" 1>&2;
    return 1;
  elif [ "$MAP" != "" ] && [ ! -e "$MAP" ]; then
    echo "$FN: error: MAP file not found: $MAP" 1>&2;
    return 1;
  fi

  ### Check XML file ###
  local $htrsh_infovars;
  htrsh_pageimg_info "$XML" noimg;
  [ "$?" != 0 ] && return 1;

  ### Prepare alignment information for each line to align ###
  echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): generating Page XML with alignments ..." 1>&2;

  local ids=$( sed -nr '/\/'"$IMBASE"'\.(.+\.)?[^.]+\.rec"$/{ s|.*\.([^.]+)\.rec"$|\1|; p; }' "$MLF" );

  [ "$ids" = "" ] &&
    echo "$FN: error: unable to extract sample IDs from MLF, expected pattern $IMBASE\.(.+\.)?[^.]+\.rec" 1>&2 &&
    return 1;

  local fpgram=( xmlstarlet sel -t );
  local id;
  for id in $ids; do
    #fpgram+=( -o " " -v "//*[@id='$id']/_:Coords/@fpgram" -o " ;" );
    #fpgram+=( -m "//_:TextLine[@id='$id' and _:Coords/@fpgram]" -v @id -o $'\t' -v _:Coords/@fpgram -n -b );
    fpgram+=( -m "//_:TextLine[@id='$id' and _:Property[@key='fpgram']" -v @id -o $'\t' -v "_:Property[@key='fpgram']/@value" -n -b );
  done
  fpgram=$( "${fpgram[@]}" "$XML" );

  [ "$fpgram" = "" ] &&
    echo "$FN: error: unable to extract feature parallelograms: $XML" 1>&2 &&
    return 1;

  if [ $(echo "$ids" | wc -l) != $(echo "$fpgram" | wc -l) ]; then
    local missing=$(( $(echo "$ids" | wc -l) - $(echo "$fpgram" | wc -l) ));
    echo "$FN: warning: not aligning $missing lines due to missing parallelograms" 1>&2 &&
    ids=$(echo "$fpgram" | cut -f 1);
  fi

  #local TS=$(($(date +%s%N)/1000000));

  local aligns=$(
    sed -nr '/\/'"$IMBASE"'\.(.+\.)?[^.]+\.rec"$/,/^\.$/p' "$MLF" \
      | awk -v HTKALI="$HTKALI" '
          { if( FILENAME != "-" )
              ids[$0] = "";
            else {
              if( match( $0, /\.rec"$/ ) )
                id = gensub(/.*\.([^.]+)\.rec"$/, "\\1", 1, $0 );
              else if( $0 != "." && id in ids ) {
                #NF = 3;
                if( NF > 4 ) NF = 4;
                if( HTKALI == "yes" ) {
                  $2 = sprintf( "%.0f", $2/100000-1 );
                  $1 = sprintf( "%.0f", $1==0 ? 0 : $1/100000-1 );
                }
                $1 = ( id " " $1 );
                print;
              }
            }
          }' <( printf %s "$ids" ) -
      );

  [ "$aligns" = "" ] &&
    echo "$FN: error: unable to extract alignments from MLF" 1>&2 &&
    return 1;

  local acoords=$(
    echo "
      fpgram = [ "$( echo "$fpgram" | cut -f 2 | paste -sd ';' )" ];
      aligns = [ "$(
        echo "$aligns" \
          | awk '
              { if( FILENAME != "-" )
                  rid[$1] = FNR;
                else
                  printf("%s,%s\n",rid[$1],$3);
              }' <( printf %s "$ids" ) - \
          | sed '$!s|$|;|' \
          | tr -d '\n'
          )" ];

      for l = unique(aligns(:,1))'
        a = [ aligns( aligns(:,1)==l, 2 ) ];
        a = [ 0 a(1:end-1)'; a' ]';
        f = reshape(fpgram(l,:),2,4)';

        dx = ( f(2,1)-f(1,1) ) / a(end) ;
        dy = ( f(2,2)-f(1,2) ) / a(end) ;

        xup = f(1,1) + dx*a;
        yup = f(1,2) + dy*a;
        xdown = f(4,1) + dx*a;
        ydown = f(4,2) + dy*a;

        for n = 1:size(a,1)
          printf('%d %g,%g %g,%g %g,%g %g,%g\n',
            l,
            xdown(n,1), ydown(n,1),
            xup(n,1), yup(n,1),
            xup(n,2), yup(n,2),
            xdown(n,2), ydown(n,2) );
        end
      end" \
    | octave -q -H \
    | awk '
        { if( FILENAME != "-" )
            rid[FNR] = $1;
          else {
            $1 = rid[$1];
            print;
          }
        }' <( printf %s "$ids" ) - ;
    );

  #local TE=$(($(date +%s%N)/1000000)); echo "time 0: $((TE-TS)) ms" 1>&2; TS="$TE";

  ( [ "$htrsh_align_contour" = "yes" ] || [ "$htrsh_align_isect" = "yes" ] ) &&
    local size=$(xmlstarlet sel -t -v //@imageWidth -o x -v //@imageHeight "$XML");

  local prevreg="";
  local wbreak="no";

  insert_nondipl_word () {
    wid="${id}_w$(printf %.3d $rw)";
    xmledit+=( -s "//*[@id='$id']" -t elem -n TMPNODE );
    xmledit+=( -i //TMPNODE -t attr -n id -v "$wid" );
    xmledit+=( -s //TMPNODE -t elem -n Coords );
    xmledit+=( -i //TMPNODE/Coords -t attr -n points -v "0,0 0,0" );
    xmledit+=( -r //TMPNODE -v Word );
    xmledit+=( -s "//*[@id='$wid']" -t elem -n TextEquiv );
    xmledit+=( -s "//*[@id='$wid']/TextEquiv" -t elem -n Unicode -v "${wmap[0]}" );
    rw=$((rw+1));
  }

  ### Insert alignment for each sample ###
  local n=0;
  for id in $ids; do
    n=$((n+1));
    echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): alignments for line $n (id=$id) ..." 1>&2;

    ### Prepare command to add alignments to XML ###
    local xmledit=( -d "//*[@id='$id']/_:Word" );

    if [ "$htrsh_align_contour" = "yes" ]; then
      local LIMG="$XMLDIR/$IMBASE.${id}_clean.png";
      [ "$htrsh_extended_names" = "true" ] &&
        LIMG="$XMLDIR/$IMBASE."$(xmlstarlet sel -t -v "//*[@id='$id']/../@id" "$XML")".${id}_clean.png";
      local LGEO=( $(identify -format "%w %h %X %Y %x %U" "$LIMG" | sed 's|+||g') );
    fi
    [ "$htrsh_align_isect" = "yes" ] &&
      local contour=$(xmlstarlet sel -t -v "//*[@id='$id']/_:Coords/@points" "$XML");

    local align=$(echo "$aligns" | sed -n "/^$id /{ s|^$id ||; p; }");
    [ "$align" = "" ] && continue;
    local coords=$(echo "$acoords" | sed -n "/^$id /{ s|^$id ||; p; }");

    #TE=$(($(date +%s%N)/1000000)); echo "time 1: $((TE-TS)) ms" 1>&2; TS="$TE";

    local reg=$(xmlstarlet sel -t -v "//*[@id='$id']/../@id" "$XML");

    #[ "$reg" != "$prevreg" ] && local rw="1";
    if [ "$reg" != "$prevreg" ]; then
      if [ "$MAP" != "" ]; then
        [ "$prevreg" != "" ] && [ "$rw" -le "${#WMAP[@]}" ] &&
          echo "$FN: error: unaligned words accoring to MAP file: regid=$reg word=#$rw" 1>&2 &&
          return 1;
        local pIFS="$IFS";
        local IFS=$'\n';
        local WMAP=($(grep "^$reg " "$MAP" | sed 's|^[^ ]* ||'));
        IFS="$pIFS";
      fi
      local rw="1";
    fi

    prevreg="$reg";

    local linetxt="";

    ### Word level alignments ###
    local W=$(echo "$align" | awk '{if($3=="'"${htrsh_symb_space}"'")n++;}END{print n;}'); W=$((W-1));
    #local W=$(echo "$align" | grep " ${htrsh_symb_space}$" | wc -l); W=$((W-1));
    local w;
    for w in $(seq 1 $W); do
      #TE=$(($(date +%s%N)/1000000)); echo "time 2: $((TE-TS)) ms" 1>&2; TS="$TE";

      [ "$MAP" != "" ] &&
        while true; do
          [ "$rw" -gt "${#WMAP[@]}" ] &&
            echo "$FN: error: found more words than defined in the MAP file: regid=$reg word=#$rw" 1>&2 &&
            return 1;
          wmap=( ${WMAP[$((rw-1))]} );
          [ "${#wmap[@]}" = 2 ] && break;
          insert_nondipl_word;
        done

      local wid="${id}_w$(printf %.3d $rw)";
      #local pS=$(echo "$align" | grep -n " ${htrsh_symb_space}$" | sed -n "$w{s|:.*||;p;}"); pS=$((pS+1));
      #local pE=$(echo "$align" | grep -n " ${htrsh_symb_space}$" | sed -n "$((w+1)){s|:.*||;p;}"); pE=$((pE-1));
      local pS=$(echo "$align" | awk -v s=$w '{if($3=="'"${htrsh_symb_space}"'"){n++;if(s==n)print NR;}}'); pS=$((pS+1));
      local pE=$(echo "$align" | awk -v s=$((w+1)) '{if($3=="'"${htrsh_symb_space}"'"){n++;if(s==n)print NR;}}'); pE=$((pE-1));
      local pts;
      if [ "$pS" = "$pE" ]; then
        pts=$(echo "$coords" | sed -n "${pS}p");
      else
        pts=$(echo "$coords" \
          | sed -n "$pS{s| [^ ]* [^ ]*$||;p;};$pE{s|^[^ ]* [^ ]* ||;p;};" \
          | tr '\n' ' ' \
          | sed 's| $||');
      fi
      #TE=$(($(date +%s%N)/1000000)); echo "time 3: $((TE-TS)) ms" 1>&2; TS="$TE";

      local cpts="";

      if [ "$htrsh_align_contour" = "yes" ]; then
        cpts=$( echo $pts \
                 | awk -F'[, ]' -v oX=${LGEO[2]} -v oY=${LGEO[3]} '
                     { for( n=1; n<NF; n+=2 )
                         printf( " %s,%s", $n-oX, $(n+1)-oY );
                     }' );
        cpts=$( convert -fill black -stroke black -size ${LGEO[0]}x${LGEO[1]} \
                    xc:white +antialias -draw "polygon$cpts" "$LIMG" \
                    -compose lighten -composite -page $size+${LGEO[2]}+${LGEO[3]} \
                    -units ${LGEO[5]} -density ${LGEO[4]} miff:- \
                  | imgccomp -V0 -NJS -A 0.1 -D $htrsh_align_dilradi -R 2,2,2,2 - 2>/dev/null );
        [ "$cpts" = "" ] && echo "warning: failed to obtain contour for word $wid" 1>&2;
      fi

      local AWK_ISECT='
        BEGIN {
          printf( "convert -fill white -stroke white +antialias" );
        }
        { if( NR == 1 ) {
            mn_x=$1; mx_x=$1;
            mn_y=$2; mx_y=$2;
            for( n=3; n<=NF; n+=2 ) {
              if( mn_x > $n ) mn_x = $n;
              if( mx_x < $n ) mx_x = $n;
              if( mn_y > $(n+1) ) mn_y = $(n+1);
              if( mx_y < $(n+1) ) mx_y = $(n+1);
            }
            w = mx_x-mn_x+1;
            h = mx_y-mn_y+1;
          }
          printf( " ( -size %dx%d xc:black -draw polygon", w, h );
          for( n=1; n<=NF; n+=2 )
            printf( "_%d,%d", $n-mn_x, $(n+1)-mn_y );
          printf( " )" );
        }
        END {
          printf( " -compose darken -composite -page %s+%d+%d miff:-", sz, mn_x, mn_y );
        }';
      if [ "$cpts" = "" ] && [ "$htrsh_align_isect" = "yes" ]; then
        local polydraw=( $(
          { echo "$pts";
            echo "$contour";
          } | awk -F'[ ,]' -v sz=$size "$AWK_ISECT" ) );
        cpts=$( "${polydraw[@]//_/ }" | imgccomp -V0 -JS - );
        [ "$cpts" = "" ] && echo "warning: failed to obtain intersection for word $wid" 1>&2;
      fi

      if [ "$cpts" = "" ] && [ "$htrsh_align_midbox" = "yes" ]; then
        cpts=$( echo $pts \
                  | awk -F'[, ]' '
                      function ceil( v ) { return v == int(v) ? v : int(v)+1 ; }
                      { x0 = int( 0.5*($1+$3) );
                        x1 = ceil( 0.5*($5+$7) );
                        y0 = int( 0.5*($4+$6) );
                        y1 = ceil( 0.5*($2+$8) );
                        printf("%d,%d %d,%d %d,%d %d,%d", x0,y0, x1,y0, x1,y1, x0,y1 );
                      }' );
      fi

      [ "$cpts" != "" ] && pts="$cpts";

      pts=$(echo "$pts" | sed '/^[^ ]*,[^ ]*$/s|\(.*\)|\1 \1|');
      local wpts="$pts";

      #TE=$(($(date +%s%N)/1000000)); echo "time 4: $((TE-TS)) ms" 1>&2; TS="$TE";

      ### Region word numbering and broken word handling ###
      #local text=$(echo "$align" | sed -n "$pS,$pE{s|.* ||;p;}" | tr -d '\n');
      local text=$(echo "$align" | awk "{if(NR>=$pS&&NR<=$pE)"'printf("%s",$3);}');

      if [ $(echo "$text" | grep -c $'\xC2\xAD$') != 0 ]; then
        [ wbreak = "yes" ] &&
          echo "$FN: error: encountered first part of broken word $rw ($text...) while expecting second part of broken word $numpart1 ($textpart1...) in region $reg: $XML" 1>&2;
        wid+="_part1";
        wbreak="yes";
        local numpart1="$rw";
        local textpart1="$text";
      elif [ $(echo "$text" | grep -c $'^\xC2\xAD') != 0 ]; then
        wid="${id}_w$(printf %.3d $numpart1)_part2";
        wbreak="no";
        rw=$((rw-1));
      fi
      rw=$((rw+1));

      xmledit+=( -s "//*[@id='$id']" -t elem -n TMPNODE );
      xmledit+=( -i //TMPNODE -t attr -n id -v "$wid" );
      xmledit+=( -s //TMPNODE -t elem -n Coords );
      xmledit+=( -i //TMPNODE/Coords -t attr -n points -v "$pts" );
      xmledit+=( -r //TMPNODE -v Word );

      ### Character level alignments ###
      # @todo fails when doing region alignment
      if [ "$htrsh_align_chars" = "yes" ]; then
        local g=1;
        local c;
        for c in $(seq $pS $pE); do
          local gg=$(printf %.2d $g);
          local pts=$(echo "$coords" | sed -n "${c}p");
          if [ "$htrsh_align_isect" = "yes" ]; then
            local polydraw=( $(
              { echo "$pts";
                echo "$wpts";
              } | awk -F'[ ,]' -v sz=$size "$AWK_ISECT" ) );
            pts=$( "${polydraw[@]//_/ }" | imgccomp -V0 -JS - );
            [ "$pts" = "" ] && echo "failed to obtain intersection for character ${wid}_g${gg}" 1>&2;
            # @todo character polygons overlap slightly, possible solution: reduce width of parallelograms by 1 pixel in each side
          fi
          pts=$(echo "$pts" | sed '/^[^ ]*,[^ ]*$/s|\(.*\)|\1 \1|');

          xmledit+=( -s "//*[@id='$wid']" -t elem -n TMPNODE );
          xmledit+=( -i //TMPNODE -t attr -n id -v "${wid}_g${gg}" );
          xmledit+=( -s //TMPNODE -t elem -n Coords );
          xmledit+=( -i //TMPNODE/Coords -t attr -n points -v "$pts" );
          if [ "$htrsh_align_addtext" = "yes" ]; then
            #local text=$(echo "$align" | sed -n "$c{s|.* ||;p;}" | tr -d '\n');
            local text=$(echo "$align" | awk "{if(NR==$c)"'printf("%s",$3);}');
            xmledit+=( -s //TMPNODE -t elem -n TextEquiv );
            xmledit+=( -s //TMPNODE/TextEquiv -t elem -n Unicode -v "$text" );
          fi
          xmledit+=( -r //TMPNODE -v Glyph );

          g=$((g+1));
        done
      fi

      #TE=$(($(date +%s%N)/1000000)); echo "time 5: $((TE-TS)) ms" 1>&2; TS="$TE";

      if [ "$htrsh_align_addtext" = "yes" ]; then
        #local text=$(echo "$align" | sed -n "$pS,$pE{s|.* ||;p;}" | tr -d '\n');
        local text=$(echo "$align" | awk "{if(NR>=$pS&&NR<=$pE)"'printf("%s",$3);}');
        local score=$(echo "$align" | awk "{if(NR>=$pS&&NR<=$pE)"'{s+=$4;n++;}}END{s/=n;if(s>=0&&s<=1)print s;}');
        [ "$MAP" != "" ] && text="${wmap[0]}";
        xmledit+=( -s "//*[@id='$wid']" -t elem -n TextEquiv );
        xmledit+=( -s "//*[@id='$wid']/TextEquiv" -t elem -n Unicode -v "$text" );
        [ "$score" != "" ] &&
          xmledit+=( -i "//*[@id='$wid']/TextEquiv" -t attr -n conf -v "$score" );
        #TE=$(($(date +%s%N)/1000000)); echo "time 6: $((TE-TS)) ms" 1>&2; TS="$TE";
        linetxt+=" $text";
      fi

      [ "$MAP" != "" ] &&
        while [ "$rw" -le "${#WMAP[@]}" ]; do
          wmap=( ${WMAP[$((rw-1))]} );
          [ "${#wmap[@]}" = 2 ] && break;
          insert_nondipl_word;
        done
    done # for w in $(seq 1 $W); do

    if [ "$htrsh_align_addtext" = "yes" ] && [ "$LINETXT" = "yes" ]; then
      local score=$(echo "$align" | awk '{s+=$4;}END{s/=NR;if(s>=0&&s<=1)print s;}');
      xmledit+=( -d "//*[@id='$id']/_:TextEquiv" );
      xmledit+=( -s "//*[@id='$id']" -t elem -n TextEquiv );
      xmledit+=( -s "//*[@id='$id']/TextEquiv" -t elem -n Unicode -v "${linetxt/ /}" );
      [ "$score" != "" ] &&
        xmledit+=( -i "//*[@id='$id']/TextEquiv" -t attr -n conf -v "$score" );
    fi

    xmledit+=( -m "//*[@id='$id']/_:TextEquiv" "//*[@id='$id']" );
    xmledit+=( -m "//*[@id='$id']/_:TextStyle" "//*[@id='$id']" );

    xmlstarlet ed --inplace "${xmledit[@]}" "$XML";
    [ "$?" != 0 ] &&
      echo "$FN: error: problems creating XML file: $XMLOUT" 1>&2 &&
      return 1;
  done

  [ "$MAP" != "" ] && [ "$rw" -le "${#WMAP[@]}" ] &&
    echo "$FN: error: unaligned words accoring to MAP file: regid=$reg word=#$rw" 1>&2 &&
    return 1;

  [ "$MAP" = "" ] &&
    htrsh_fix_rec_names "$XML";

  return 0;
)}

##
## Function that sorts an MLF alphabetically or using a given order list
##
htrsh_mlf_sort () {
  local FN="htrsh_mlf_sort";
  if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    { echo "$FN: Error: Incorrect input arguments";
      echo "Description: Sorts an MLF alphabetically or using a given order list";
      echo "Usage: $FN [ORDERLST] MLF";
    } 1>&2;
    return 1;
  fi

  local SRT=""; [ $# = 2 ] && SRT="-k 1n,1";

  gawk -v NUM="$#" '
    { if( NUM == 2 && ARGIND == 1 ) {
        ORDER[$1] = FNR;
        N = FNR;
      }
      else {
        if( NUM == 2 && match($0,/^".*\/.+\.[lr][ae][bc]"$/) ) {
          samp = gensub( /^".*\/(.+)\.[lr][ae][bc]"$/, "\\1", 1, $0 );
          num = samp in ORDER ? ORDER[samp] : N+1;
          printf( "%d %s", num, $0 );
        }
        else if( $0 != "#!MLF!#" )
          printf( $0 == "." ? "\t%s\n" : "\t%s", $0 );
      }
    }' "$@" \
    | sort $SRT \
    | sed 's|^\t||; s|^[0-9]* ||;' \
    | awk '
        BEGIN {
          FS="\t";
          print("#!MLF!#");
        }
        { for( n=1; n<=NF; n++ )
            printf( "%s\n", $n );
        }';
}

##
## Function that blindly creates an alignment MLF by assuming all characters have equal width
##
htrsh_mlf_align_blind () {
  local FN="htrsh_mlf_align_blind";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Blindly creates an alignment MLF by assuming all characters have equal width";
      echo "Usage: $FN MLF FEATDIR [NAME1 ...]";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local MLF="$1";
  local FEATDIR="$2";
  shift 2;

  if [ ! -e "$MLF" ]; then
    echo "$FN: error: MLF file not found: $MLF" 1>&2;
    return 1;
  elif [ ! -e "$FEATDIR" ]; then
    echo "$FN: error: features directory not found: $FEATDIR" 1>&2;
    return 1;
  fi

  local FEATS=("$@");
  if [ "${#FEATS[@]}" = 0 ]; then
    FEATS=( $( sed -n '/\.lab"$/{ s|.*/||; s|\.lab"$||; p }' ) );
  else
    local MLFNUM=$( awk '
      { if( ARGIND == 1 )
          FEATS[$0] = "";
        else if( $0 in FEATS )
          print;
      }' <( printf "%s\n" "${FEATS[@]}" ) \
         <( sed -n '/\.lab"$/{ s|.*/||; s|\.lab"$||; p }' "$MLF" ) \
      | wc -l );
    if [ "$MLFNUM" != "${#FEATS[@]}" ]; then
      echo "$FN: error: some sample names not found in MLF file" 1>&2;
      return 1;
    fi
  fi
  local FEATLST=("${FEATS[@]/%/.fea}");
  FEATLST=("${FEATLST[@]/#/$FEATDIR/}");

  ls "${FEATLST[@]}" >/dev/null;
  if [ "$?" != 0 ]; then
    echo "$FN: error: some .fea files not found" 1>&2;
    return 1;
  elif [ "${#FEATS[@]}" = 0 ]; then
    echo "$FN: error: no sample names for aligning" 1>&2;
    return 1;
  fi

  ### Blindly align each sample ###
  local n;
  for n in $(seq 0 $((${#FEATS[@]}-1))); do
    local FRAMES=$( HList -h -z "${FEATLST[$n]}" \
      | sed -n '/Num Samples:/{ s|.*Samples: *||; s| .*||; p; }' );
    local CHARS=$( awk -v INFEAT=0 -v FEAT="${FEATS[$n]}" '
      { if( INFEAT ) {
          if( $0 == "." ) {
            printf( "\n" );
            exit 0;
          }
          w = gensub( /\\\x27/, "\x27", "g", $0 );
          if( match(w,/^".+"$/) )
            w = gensub( /\\"/, "\"", "g", substr(w,2,length(w)-2) );
          printf( " %s", w );
        }
        if( match($0,/\/'"${FEATS[$n]//./\\.}"'\.lab"$/) )
          INFEAT = 1;
      }' "$MLF" );
    echo "${FEATS[$n]} $FRAMES$CHARS";
  done \
    | awk  '
        { printf( "\"*/%s.rec\"\n", $1 );
          frames_per_char = $2 / (NF-2) ;
          pos = 0;
          #pword = 0;
          for( n=3; n<=NF; n++ ) {
            ppos = pos;
            pos += frames_per_char ;
            #word = n==3 && $n == "'"$htrsh_symb_space"'" ? " - '"$htrsh_symb_space"'" : "" ;
            #if( pword ) {
            #  word = ( " - " $n );
            #  for( m=n+1; m<=NF; m++ ) {
            #    if( $m == "'"$htrsh_symb_space"'" )
            #      break;
            #    word = ( word $m );
            #  }
            #}
            #pword = $n == "'"$htrsh_symb_space"'" ? 1 : 0 ;
            #printf( "%.0f %.0f %s%s\n", 100000*ppos, 100000*pos, $n, word );
            printf( "%.0f %.0f %s\n", 100000*ppos, 100000*pos, $n );
          }
          printf( ".\n" );
        }';
}

##
## Function that blindly creates Words from TextLines by assuming all characters have equal width
##
htrsh_pagexml_align_blind () {
  local FN="htrsh_pagexml_align_blind";
  if [ $# -lt 1 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Blindly creates Words from TextLines by assuming all characters have equal width";
      echo "Usage: $FN XMLIN [XMLOUT]";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local XMLOUT="-"; [ "$#" -gt 1 ] && XMLOUT="$2";
  shift 2;

  if [ "$XML" = "$XMLOUT" ]; then
    echo "$FN: error: input and output files must be different" 1>&2;
    return 1;
  fi

  ### Check page ###
  local $htrsh_infovars;
  htrsh_pageimg_info "$XML" noimg;
  [ "$?" != 0 ] && return 1;

  local IDS=$(xmlstarlet sel -t -m "$htrsh_xpath_regions/$htrsh_xpath_lines[$htrsh_xpath_textequiv]" -v @id -n "$XML");

  local xmledit=( ed -d //@dummyattr );

  local id n TEXT LGTH COORD;
  for id in $IDS; do
    TEXT=( $(xmlstarlet sel -T -B -E utf-8 -t -v "//*[@id='$id']/_:TextEquiv/_:Unicode" "$XML") );
    LGTH=( $( echo "${TEXT[@]}" \
      | awk '
          { N=M=0;
            for(n=1;n<=NF;n++)
              N += length($n) + ( n==1 ? 0 : 1 );
            for(n=1;n<=NF;n++) {
              M += ( length($n) + ( (n==1||n==NF) ? 0.5 : 1 ) )/N;
              print M;
            }
          }') );
    if [ "${#TEXT[@]}" != "${#LGTH[@]}" ]; then
      echo "error: unexpected state: @id :: $XML" 1>&2;
      return 1;
    fi
    COORDS=( $(
        { echo "${LGTH[@]}";
          xmlstarlet sel -t -v "//*[@id='$id']/_:Coords/@points" -n "$XML";
        } | awk -F'[, ]' '
          { if( NR == 1 ) {
              N = NF;
              LGTH[1] = 0;
              for(n=1;n<=NF;n++)
                LGTH[n+1] = $n;
            }
            else {
              Xtl=$1; Ytl=$2;
              Xtr=$3; Ytr=$4;
              Xbr=$5; Ybr=$6;
              Xbl=$7; Ybl=$8;
              dX = Xtr - Xtl;
              dY = Ytr - Ytl;
              for(n=1;n<=N;n++)
                printf( "%s,%s_%s,%s_%s,%s_%s,%s\n",
                    Xtl+LGTH[n]*dX,   Ytl+LGTH[n]*dY,
                    Xtl+LGTH[n+1]*dX, Ytl+LGTH[n+1]*dY,
                    Xbl+LGTH[n+1]*dX, Ybl+LGTH[n+1]*dY,
                    Xbl+LGTH[n]*dX,   Ybl+LGTH[n]*dY );
            }
          }'
      ) );

    for n in $(seq 1 ${#TEXT[@]}); do
      wid="${id}_w$n";
      xmledit+=( -s "//*[@id='$id']" -t elem -n TMPNODE );
      xmledit+=( -i //TMPNODE -t attr -n id -v "$wid" );
      xmledit+=( -s //TMPNODE -t elem -n Coords );
      xmledit+=( -i //TMPNODE/Coords -t attr -n points -v "${COORDS[$((n-1))]//_/ }" );
      xmledit+=( -s //TMPNODE -t elem -n TextEquiv );
      xmledit+=( -s //TMPNODE/TextEquiv -t elem -n Unicode -v "${TEXT[$((n-1))]}" );
      xmledit+=( -r //TMPNODE -v Word );
    done
    xmledit+=( -m "//*[@id='$id']/_:TextEquiv" "//*[@id='$id']" );
    xmledit+=( -m "//*[@id='$id']/_:TextStyle" "//*[@id='$id']" );
  done

  if [ "$XMLOUT" = "-" ]; then
    xmlstarlet "${xmledit[@]}" "$XML" | xmlstarlet fo -e utf-8 -;
  else
    xmlstarlet "${xmledit[@]}" "$XML" | xmlstarlet fo -e utf-8 - > "$XMLOUT";
  fi
}

##
## Function that does a forced alignment at a line level for a given XML Page, feature list and model
##
htrsh_pageimg_forcealign_lines () {
  local FN="htrsh_pageimg_forcealign_lines";
  local TMP="${TMPDIR:-/tmp}";
  local DIPLOMATIZER=();
  if [ $# -lt 3 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Does a forced alignment at a line level for a given XML Page, feature list and model";
      echo "Usage: $FN XML FEATDIR MODEL [ Options ]";
      echo "Options:";
      echo " -d TMPDIR       Directory for temporal files (def.=$TMP)";
      echo " -D DIPLOMATIZER Diplomatizer pipe command, e.g. remove expansions (def.=none)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local FEATDIR="$2";
  local MODEL="$3";
  shift 3;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-d" ]; then
      TMP="$2";
    elif [ "$1" = "-D" ]; then
      DIPLOMATIZER=( -D "$2" );
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  if [ ! -e "$XML" ]; then
    echo "$FN: error: Page XML file not found: $XML" 1>&2;
    return 1;
  elif [ ! -e "$FEATDIR" ]; then
    echo "$FN: error: features directory not found: $FEATDIR" 1>&2;
    return 1;
  elif [ ! -e "$MODEL" ]; then
    echo "$FN: error: model file not found: $MODEL" 1>&2;
    return 1;
  fi

  ### Check XML file and image ###
  local $htrsh_infovars;
  htrsh_pageimg_info "$XML";
  [ "$?" != 0 ] && return 1;
  local B=$(echo "$XMLBASE" | sed 's|[\[ ()]|_|g; s|]|_|g;');
  echo "$FN: aligning $B" 1>&2;

  ### Check feature files ###
  local pIFS="$IFS";
  local IFS=$'\n';
  #local FBASE="$FEATDIR/"$(echo "$IMFILE" | sed 's|.*/||; s|\.[^.]*$||;');
  local FBASE="$FEATDIR/$IMBASE";
  local IDop=( -o "$FBASE." );
  [ "$htrsh_extended_names" = "true" ] && IDop+=( -v ../@id -o . -v @id );
  [ "$htrsh_extended_names" != "true" ] && IDop+=( -v @id );
  local FEATLST=( $( xmlstarlet sel -t -m "$htrsh_xpath_regions/$htrsh_xpath_lines" "${IDop[@]}" -o ".fea" -n "$XML" ) );
  IFS="$pIFS";

  ls "${FEATLST[@]}" >/dev/null;
  [ "$?" != 0 ] &&
    echo "$FN: error: some .fea files not found" 1>&2 &&
    return 1;

  ### Create MLF from XML ###
  htrsh_pagexml_textequiv "$XML" -f mlf-chars "${DIPLOMATIZER[@]}" -H yes > "$TMP/$B.mlf";
  [ "$?" != 0 ] &&
    echo "$FN: error: problems creating MLF file: $XML" 1>&2 &&
    return 1;

  ### Create auxiliary files: HMM list and dictionary ###
  local HMMLST=$(gzip -dc "$MODEL" | sed -n '/^~h "/{ s|^~h "||; s|"$||; p; }');
  local DIC=$(echo "$HMMLST" | awk '{printf("\"%s\" [%s] 1.0 %s\n",$1,$1,$1)}');

  ### Do forced alignment with HVite ###
  printf "%s\n" "${FEATLST[@]}" > "$TMP/$B.lst";
  HVite $htrsh_HTK_HVite_align_opts -C <( echo "$htrsh_HTK_config" ) -H "$MODEL" -S "$TMP/$B.lst" -a -I "$TMP/$B.mlf" -i "$TMP/${B}_aligned.mlf" <( echo "$DIC" ) <( echo "$HMMLST" );
  [ "$?" != 0 ] &&
    echo "$FN: error: problems aligning with HVite: $XML" 1>&2 &&
    return 1;

  sed '/\.rec"$/s|^".*/|"*/|;' -i "$TMP/${B}_aligned.mlf";

  ### Blindly align failed lines ###
  local missing=( $(
    { sed 's|.*/||; s|\.fea$||;' "$TMP/$B.lst";
      sed -n '
        /\/'"$IMBASE"'\..\+\.rec"$/ {
          s|.*/||; s|\.rec"$||;
          p;
        }' "$TMP/${B}_aligned.mlf";
    } | sort | uniq -u ) );

  if [ "${#missing[@]}" = "${#FEATLST[@]}" ]; then
    echo "$FN: error: HVite failed to align all lines, aborting: $XML" 1>&2 &&
    return 1;
  elif [ "${#missing[@]}" != 0 ]; then
    echo "$FN: warning: ${#missing[@]} failed lines will be blindly aligned: $B :: $(echo ${missing[@]//$IMBASE./})" 1>&2;
    mv "$TMP/${B}_aligned.mlf" "$TMP/${B}_aligned.mlf-";
    { cat "$TMP/${B}_aligned.mlf-";
      htrsh_mlf_align_blind "$TMP/$B.mlf" "$FEATDIR" "${missing[@]}";
    } > "$TMP/${B}_aligned.mlf";
  fi

  mv "$TMP/${B}_aligned.mlf" "$TMP/${B}_aligned.mlf-";
  htrsh_mlf_sort <( sed 's|.*/||; s|\.fea$||;' "$TMP/$B.lst" ) "$TMP/${B}_aligned.mlf-" > "$TMP/${B}_aligned.mlf";

  ### Create diplomatic mapping ###
  if [ "${#DIPLOMATIZER[@]}" != 0 ]; then
    htrsh_pagexml_textequiv "$XML" -f tab \
      | sed -r 's|^[^ ]+\.([^. ]+)\.[^. ]+ |\1 |' \
      | awk '{ for( n=2; n<=NF; n++ ) printf("%s %s\n",$1,$n); }' \
      > "$TMP/$B.txt";
    sed -r 's|^[^ ]+ ||' "$TMP/${B}.txt" \
      | "${DIPLOMATIZER[1]}" \
      | paste -d " " "$TMP/${B}.txt" - \
      > "$TMP/${B}_map.txt";
    DIPLOMATIZER=( -M "$TMP/${B}_map.txt" );
  fi

  ### Insert alignment information in XML ###
  htrsh_pagexml_insertalign_lines "$XML" "$TMP/${B}_aligned.mlf" "${DIPLOMATIZER[@]}";
  [ "$?" != 0 ] &&
    return 1;

  [ "$htrsh_keeptmp" -lt 1 ] &&
    rm -f "$TMP/$B.mlf" "$TMP/$B.lst" "$TMP/${B}_aligned.mlf"{,-} "$TMP/$B"{,_map}.txt;

  return 0;
}

##
## Function that adds HMM models for missing characters given a model and character list
##
htrsh_hmm_add_missing () {
  local FN="htrsh_hmm_add_missing";
  local TMP="";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Adds HMM models for missing characters given a model and character list";
      echo "Usage: $FN CHARLIST FEATLST MODELIN MODELOUT [ Options ]";
      echo "Options:";
      echo " -d TMPDIR    Directory for temporal files (def.=${TMPDIR:-/tmp})";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local CHARLIST="$1";
  local FEATLST="$2";
  local MODELIN="$3";
  local MODELOUT="$4";
  shift 4;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-d" ]; then
      TMP=$(echo "$2" | sed '/^[./]/!s|^|./|');
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  if [ ! -e "$CHARLIST" ]; then
    echo "$FN: error: character list not found: $CHARLIST" 1>&2;
    return 1;
  elif [ ! -e "$FEATLST" ]; then
    echo "$FN: error: feature list not found: $FEATLST" 1>&2;
    return 1;
  elif [ ! -e "$MODELIN" ]; then
    echo "$FN: error: input model file not found: $MODELIN" 1>&2;
    return 1;
  fi

  local DIMS=$( gzip -dc "$MODELIN" \
    | sed -n '/^<VECSIZE>/ { s|^<VECSIZE> *\([0-9][0-9]*\).*|\1|; p; q; }' );
  local MODELCHARS=$( gzip -dc "$MODELIN" \
    | sed -n '/^~h ".*"$/ { s|^~h "\(.*\)"$|\1|; p; }' );
  if [ "$DIMS" = "" ] || [ "$MODELCHARS" = "" ]; then
    echo "$FN: error: unexpected input model format: $MODELIN" 1>&2;
    return 1;
  fi

  local CHARCHECK=$( awk '
          { if( ARGIND == 1 )
              model[$1] = "";
            else if( ! ( $1 in model ) )
              print;
          }' <( echo "$MODELCHARS" ) <( < "$CHARLIST" ) );

  if [ "$CHARCHECK" = "" ]; then
    echo "$MODELIN";
    return 0;
  fi

  local NUMCHARS=$( echo "$CHARCHECK" | wc -l );

  echo "$FN: adding $NUMCHARS missing characters ($(echo $CHARCHECK | tr ' ' ',')) to model" 1>&2;

  [ "$TMP" = "" ] &&
    TMP="${TMPDIR:-/tmp}";
  [ ! -d "$TMP" ] &&
    echo "$FN: error: temporal files directory does not exist: $TMP" 1>&2 &&
    return 1;

  htrsh_hmm_proto "$DIMS" 1 | gzip > "$TMP/proto";
  HCompV $htrsh_HTK_HCompV_opts -C <( echo "$htrsh_HTK_config" ) \
    -S "$FEATLST" -M "$TMP" "$TMP/proto" 1>&2;

  local MEAN=$(gzip -dc "$TMP/proto" | sed -n '/<MEAN>/{N;s|.*\n||;p;q;}');
  local VARIANCE=$(gzip -dc "$TMP/proto" | sed -n '/<VARIANCE>/{N;s|.*\n||;N;p;q;}');

  { gzip -dc "$MODELIN";
    htrsh_hmm_proto "$DIMS" "$htrsh_hmm_states" -n "$CHARCHECK" \
      -g off -m "$MEAN" -v "$VARIANCE";
  } | gzip \
    > "$MODELOUT";
  echo "$MODELOUT";
}

##
## Function that does a line by line forced alignment given only a page with baselines or contours and optionally a model
##
# @todo Option to give directory of already extracted features
htrsh_pageimg_forcealign () {
  local FN="htrsh_pageimg_forcealign";
  local TS=$(date +%s);
  local TMP="";
  local INRES="";
  local MODEL="";
  local PBASE="";
  local ENHIMG="yes";
  local DOPCA="yes";
  local ADAPT="no";
  local KEEPTMP="no";
  local KEEPAUX="no";
  local QBORD="no";
  local DIPLOMATIZER=();
  local SFACT="";
  if [ $# -lt 2 ]; then
    { echo "$FN: Error: Not enough input arguments";
      echo "Description: Does a line by line forced alignment given only a page with baselines or contours and optionally a model";
      echo "Usage: $FN XMLIN XMLOUT [ Options ]";
      echo "Options:";
      echo " -d TMPDIR    Directory for temporal files (def.=${TMPDIR:-/tmp}/${FN}_XXXXX)";
      echo " -i INRES     Input image resolution in ppc (def.=use image metadata)";
      echo " -m MODEL     Use given model for aligning (def.=train model for page)";
      echo " -b PBASE     Project features using given base (def.=false)";
      echo " -e (yes|no)  Whether to enhance the image using imgtxtenh (def.=$ENHIMG)";
      echo " -p (yes|no)  Whether to compute PCA for image and project features (def.=$DOPCA)";
      echo " -A (yes|no)  Whether to adapt provided model (def.=$ADAPT)";
      echo " -t (yes|no)  Whether to keep temporal directory and files (def.=$KEEPTMP)";
      echo " -a (yes|no)  Whether to keep auxiliary attributes in XML (def.=$KEEPAUX)";
      #echo " -q (yes|no)  Whether to clean quadrilateral border of regions (def.=$QBORD)";
      echo " -D DIPLOM    Diplomatizer pipe command, e.g. remove expansions (def.=none)";
      echo " -s RES|FACT% Rescale image to RES dpcm or by FACT% for processing (def.=orig.)";
    } 1>&2;
    return 1;
  fi

  ### Parse input arguments ###
  local XML="$1";
  local XMLOUT="$2";
  shift 2;
  while [ $# -gt 0 ]; do
    if [ "$1" = "-d" ]; then
      TMP=$(echo "$2" | sed '/^[./]/!s|^|./|');
    elif [ "$1" = "-i" ]; then
      INRES="$2";
    elif [ "$1" = "-m" ]; then
      MODEL="$2";
    elif [ "$1" = "-b" ]; then
      PBASE="$2";
    elif [ "$1" = "-e" ]; then
      ENHIMG="$2";
    elif [ "$1" = "-p" ]; then
      DOPCA="$2";
    elif [ "$1" = "-A" ]; then
      ADAPT="$2";
    elif [ "$1" = "-t" ]; then
      KEEPTMP="$2";
    elif [ "$1" = "-a" ]; then
      KEEPAUX="$2";
    elif [ "$1" = "-q" ]; then
      QBORD="$2";
    elif [ "$1" = "-D" ]; then
      DIPLOMATIZER=( -D "$2" );
    elif [ "$1" = "-s" ]; then
      SFACT="$2";
    else
      echo "$FN: error: unexpected input argument: $1" 1>&2;
      return 1;
    fi
    shift 2;
  done

  ### Create temporal directory ###
  if [ "$TMP" = "" ]; then
    TMP="${TMPDIR:-/tmp}";
    TMP=$(mktemp -d --tmpdir="$TMP" ${FN}_XXXXX);
    [ ! -d "$TMP" ] &&
      echo "$FN: error: failed to create temporal directory" 1>&2 &&
      return 1;
  elif [ -d "$TMP" ]; then
    echo -n "$FN: temporal directory ($TMP) already exists, current contents will be deleted, continue? " 1>&2;
    local RMTMP="";
    read -n 1 RMTMP;
    [ "${RMTMP:0:1}" != "y" ] &&
      printf "\n$FN: aborting ...\n" 1>&2 &&
      return 1;
    rm -r "$TMP";
    echo 1>&2;
  fi

  ### Check page ###
  local $htrsh_infovars;
  htrsh_pageimg_info "$XML";
  [ "$?" != 0 ] && return 1;

  local RCNT=$(xmlstarlet sel -t -v "count($htrsh_xpath_regions/$htrsh_xpath_textequiv)" "$XML");
  #local RCNT="0";
  local LCNT=$(xmlstarlet sel -t -v "count($htrsh_xpath_regions/$htrsh_xpath_lines/$htrsh_xpath_textequiv)" "$XML");
  [ "$RCNT" = 0 ] && [ "$LCNT" = 0 ] &&
    echo "$FN: error: no TextEquiv/Unicode nodes for processing: $XML" 1>&2 &&
    return 1;

  local WGCNT=$(xmlstarlet sel -t -v 'count(//_:Word)' -o ' ' -v 'count(//_:Glyph)' "$XML");
  [ "$WGCNT" != "0 0" ] &&
    echo "$FN: warning: input already contains Word and/or Glyph information: $XML" 1>&2;

  local AREG=( -s lines );
  if [ "$LCNT" = 0 ]; then
    AREG[1]="regions";
    echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): no text in lines, so aligning regions";
  fi

  local B=$(echo "$XMLBASE" | sed 's|[\[ ()]|_|g; s|]|_|g;');

  echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): processing page: $XML";
  echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): temporal directory: $TMP";

  mkdir -p "$TMP/proc";
  cp -p "$XML" "$IMFILE" "$TMP/proc";
  sed 's|\(imageFilename="\)[^"/]*/|\1|' -i "$TMP/proc/$XMLBASE.xml";

  ### Generate contours from baselines ###
  if [ $(xmlstarlet sel -t -v \
           "count($htrsh_xpath_regions/$htrsh_xpath_lines/_:Baseline)" \
           "$XML") -gt 0 ] && (
       [ "$htrsh_align_prefer_baselines" = "yes" ] ||
       [ $(xmlstarlet sel -t -v \
             "count($htrsh_xpath_regions/$htrsh_xpath_lines/$htrsh_xpath_coords)" \
             "$XML") = 0 ] ); then
    echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): generating line contours from baselines ...";
    page_format_generate_contour -a 75 -d 25 \
      -p "$TMP/proc/$XMLBASE.xml" \
      -o "$TMP/proc/$XMLBASE.xml";
    [ "$?" != 0 ] &&
      echo "$FN: error: page_format_generate_contour failed" 1>&2 &&
      return 1;
  fi

  ### Rescale image for processing ###
  if [ "$SFACT" != "" ]; then
    echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): rescaling image ...";
    SFACT=$(echo "$SFACT" "$IMRES" | awk '{printf("%g",match($1,"%$")?$1:100*$1/$2)}');
    mkdir "$TMP/scaled";
    mv "$TMP/proc/"* "$TMP";
    htrsh_pageimg_resize "$TMP/$XMLBASE.xml" "$TMP/proc" -s "$SFACT";
  fi

  ### Clean page image ###
  if [ "$ENHIMG" = "yes" ]; then
    echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): enhancing page image ...";
    [ "$INRES" != "" ] && INRES="-i $INRES";
    htrsh_pageimg_clean "$TMP/proc/$XMLBASE.xml" "$TMP" $INRES \
      > "$TMP/${XMLBASE}_pageclean.log";
    [ "$?" != 0 ] && [ -s "$TMP/${XMLBASE}_pageclean.log" ] &&
      echo "$FN: error: more info might be in file $TMP/${XMLBASE}_pageclean.log" 1>&2 &&
      return 1;
  else
    mv "$TMP/proc/"* "$TMP";
  fi

  ### Clean quadrilateral borders ###
  if [ "$QBORD" = "yes" ]; then
    echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): cleaning quadrilateral borders ...";
    htrsh_pageimg_quadborderclean "$TMP/${XMLBASE}.xml" "$TMP/${IMBASE}_nobord.png" -d "$TMP";
    [ "$?" != 0 ] && return 1;
    mv "$TMP/${IMBASE}_nobord.png" "$TMP/$IMBASE.png";
  fi

  ### Extract line features ###
  echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): extracting line features ...";
  local xpath_regions="//_:TextRegion";
  local xpath_lines="_:TextLine[$htrsh_xpath_textequiv and $htrsh_xpath_coords]";
  [ "${AREG[1]}" = "regions" ] &&
    xpath_regions="//_:TextRegion[$htrsh_xpath_textequiv]" &&
    xpath_lines="_:TextLine[$htrsh_xpath_coords]";
  htrsh_xpath_regions="$xpath_regions" htrsh_xpath_lines="$xpath_lines" \
    htrsh_pageimg_extract_linefeats \
      "$TMP/$XMLBASE.xml" "$TMP/${XMLBASE}_feats.xml" \
      -d "$TMP" -l "$TMP/${B}_feats.lst" \
      > "$TMP/${XMLBASE}_linefeats.log";
  [ "$?" != 0 ] && [ -s "$TMP/${XMLBASE}_linefeats.log" ] &&
    echo "$FN: error: more info might be in file $TMP/${XMLBASE}_linefeats.log" 1>&2 &&
    return 1;

  ### Compute PCA and project features ###
  [ "$htrsh_feat" != "dotmatrix" ] && DOPCA="no";
  if [ "$PBASE" = "" ] && [ "$DOPCA" = "yes" ]; then
    echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): computing PCA for page ...";
    PBASE="$TMP/pcab.h5";
    htrsh_feats_pca "$TMP/${B}_feats.lst" "$PBASE" -e 1:4 -r 24;
    [ "$?" != 0 ] && return 1;
  fi #| sed '/^$/d';
  if [ "$PBASE" != "" ]; then
    [ ! -e "$PBASE" ] &&
      echo "$FN: error: projection base file not found: $PBASE" 1>&2 &&
      return 1;
    echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): projecting features ...";
    htrsh_feats_project "$TMP/${B}_feats.lst" "$PBASE" "$TMP";
    [ "$?" != 0 ] && return 1;
  fi | sed '/^$/d';

  ### Concatenate features ###
  [ "${AREG[1]}" = "regions" ] &&
    > "$TMP/${B}_feats.lst" &&
    htrsh_xpath_regions="$xpath_regions" htrsh_xpath_lines="$xpath_lines" \
      htrsh_feats_catregions "$TMP/${XMLBASE}_feats.xml" "$TMP" -l "$TMP/${B}_feats.lst";

  ### Compute frames per char stat ###
  local PER_CHAR=$( awk '
    { if ( ARGIND == 1 )
        numchar[$1] = 2 + length( gensub( /^[^ ]* /, "", 1, $0 ) );
      else if( $2 in numchar )
        printf( "%g %s\n", $1/numchar[$2], $2 );
    }' <( htrsh_pagexml_textequiv "$TMP/${XMLBASE}_feats.xml" -f tab "${AREG[@]}" "${DIPLOMATIZER[@]}" ) \
       <( for f in $(<"$TMP/${B}_feats.lst"); do
            echo \
              $(HList -z -h "$f" | awk '{if($2=="Samples:")print $3;}') \
              $(echo "$f" | sed 's|.*/||; s|\.fea$||;');
          done ) \
    | awk '{s+=$1}END{print s/NR}' );
  echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): feature frames per character: $PER_CHAR";

  ### Get list of required HMM character models ###
  htrsh_pagexml_textequiv "$TMP/${XMLBASE}_feats.xml" -f mlf-chars -H yes "${AREG[@]}" "${DIPLOMATIZER[@]}" \
    > "$TMP/${B}_page.mlf";
  [ "$?" != 0 ] &&
    echo "$FN: error: problems extracting text for page" 1>&2 &&
    return 1;
  local REQHMMS=$(
          cat "$TMP/${B}_page.mlf" \
            | sed '/^#!MLF!#/d; /^"\*\/.*"$/d; /^\.$/d; s|^"\(.*\)"|\1|;' \
            | sort -u );

  ### Prepare given model for adaptation, adding missing models if required ###
  [ "$MODEL" = "" ] && [ "$ADAPT" = "yes" ] && ADAPT="no";
  local HMMTYPE="train";
  if [ "$MODEL" != "" ] && [ "$ADAPT" = "yes" ]; then
    HMMTYPE="adapt";
    { ADAPT=$( htrsh_hmm_add_missing <( echo "$REQHMMS" ) "$TMP/${B}_feats.lst" "$MODEL" "$TMP"/$(echo "$MODEL" | sed 's|.*/||') ); } 2>&1;
    [ "$?" != 0 ] &&
      echo "$FN: error: problems adding missing character models" 1>&2 &&
      return 1;
    MODEL="";
  fi

  ### Train or adapt model for this single page ###
  if [ "$MODEL" = "" ]; then
    echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): ${HMMTYPE}ing model for page ...";
    if [ "$ADAPT" != "no" ]; then
      MODEL=$( htrsh_hmm_nummix=1 htrsh_hmm_train "$TMP/${B}_feats.lst" "$TMP/${B}_page.mlf" -d "$TMP" -P "$ADAPT" );
    else
      MODEL=$( htrsh_hmm_train "$TMP/${B}_feats.lst" "$TMP/${B}_page.mlf" -d "$TMP" );
    fi 2> "$TMP/${XMLBASE}_hmm${HMMTYPE}.log"
    [ "$?" != 0 ] && [ -s "$TMP/${XMLBASE}_hmm${HMMTYPE}.log" ] &&
      echo "$FN: error: problems ${HMMTYPE}ing model, more info might be in file $TMP/${XMLBASE}_hmm${HMMTYPE}.log" 1>&2 &&
      return 1;

  ### Check that given model has all characters, otherwise add protos for these ###
  else
    MODEL=$( htrsh_hmm_add_missing <( echo "$REQHMMS" ) "$TMP/${B}_feats.lst" "$MODEL" "$TMP"/$(echo "$MODEL" | sed 's|.*/||') );
    [ "$?" != 0 ] &&
      echo "$FN: error: problems adding missing character models" 1>&2 &&
      return 1;
  fi
  [ ! -e "$MODEL" ] &&
    echo "$FN: error: model file not found: $MODEL" 1>&2 &&
    return 1;

  ### Do forced alignment using model ###
  echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): doing forced alignment ...";
  cp "$TMP/${XMLBASE}_feats.xml" "$TMP/${XMLBASE}_align.xml";
  local forcealign="htrsh_pageimg_forcealign_lines";
  [ "${AREG[1]}" = "regions" ] &&
    forcealign="htrsh_pageimg_forcealign_regions";
  htrsh_xpath_regions="$xpath_regions" htrsh_xpath_lines="$xpath_lines" \
    $forcealign "$TMP/${XMLBASE}_align.xml" "$TMP" "$MODEL" -d "$TMP" "${DIPLOMATIZER[@]}" \
      > "$TMP/${XMLBASE}_forcealign.log";
  [ "$?" != 0 ] && [ -s "$TMP/${XMLBASE}_forcealign.log" ] &&
    echo "$FN: error: more info might be in file $TMP/${XMLBASE}_forcealign.log" 1>&2 &&
    return 1;
  mv "$TMP/${XMLBASE}_align.xml" "$XMLOUT";

  [ "$KEEPTMP" != "yes" ] && rm -r "$TMP";

  if [ "$SFACT" != "" ]; then
    SFACT=$(echo "10000/$SFACT" | bc -l);
    cat "$XMLOUT" \
      | htrsh_pagexml_resize "$SFACT"% \
      | xmlstarlet ed \
          -u //@imageWidth -v ${IMSIZE%x*} \
          -u //@imageHeight -v ${IMSIZE#*x} \
      > "$XMLOUT"~;
    mv "$XMLOUT"~ "$XMLOUT";
  fi

  local I=$(xmlstarlet sel -t -v //@imageFilename "$XML");
  local xmledit=( -u //@imageFilename -v "$I" );
  #[ "$KEEPAUX" != "yes" ] && xmledit+=( -d //@fpgram -d //@fcontour );
  [ "$KEEPAUX" != "yes" ] && xmledit+=( -d "//_:Property[@key='fpgram']" -d "//_:Property[@key='fcontour']" );

  xmlstarlet ed "${xmledit[@]}" "$XMLOUT" \
    | htrsh_pagexml_round \
    > "$XMLOUT"~;
  mv "$XMLOUT"~ "$XMLOUT";

  echo "$FN ($(date -u '+%Y-%m-%d %H:%M:%S')): finished, $(( $(date +%s)-TS )) seconds";

  return 0;
}
