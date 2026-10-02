#!/bin/bash

find `pwd` -depth -name '*' -exec sh -c '
    for file do
      dir=${file%/*};
      file=${file##*/};
      without_spaces=$(printf %s "$file." | sed "s/[() ]/_/g")
      echo "$dir/$file" "->" "$dir/${without_spaces%.}";
      mv "$dir/$file" "$dir/${without_spaces%.}";
    done
' _ {} +
