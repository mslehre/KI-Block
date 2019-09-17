URLs to animal photos obtained from http://image-net.org.

shepherd: German shepherd, German shepherd dog, German police dog, alsatian
wolf: timber wolf, grey wolf, gray wolf, Canis lupus. Definition: a wolf with a brindled grey coat living in forested northern regions of North America.
tabby: tabby cat. Definition: a cat with a grey or tawny coat mottled with black.

Plant dataset: https://www.kaggle.com/vbookshelf/a-simple-keras-solution

# Some commands that were used in data preparation
# download the images from the URL lists obtained from IMAGENET
for s in shepherd wolf tabby; do
   cd $s
   wget --timeout=10 --tries=1 -i ../$s.urls
   cd -
done

# delete a small number of files with spaces in file name
find . -name "* *" | while read LINE; do echo "rm \"$LINE\""; done

# Use ImageMagick to resize by keeping aspect ratio so that the smaller dimension is 96, then crop the center 96x96 square
# convert in.jpg -resize 96x96^  -gravity center -crop 96x96+0+0 out.jpg
I=1
for f in `ls`; do
  I=$((I+1))
  convert $f -resize 96x96^  -gravity center -crop 96x96+0+0 resized-$I.jpg
done


# obtain a tab separated file with all image paths and classes in a specific order (animals first)
for s in  cat shepherd wolf CommonWheat Maize Small-floweredCranesbill Charlock FatHen ScentlessMayweed SugarBeet CommonChickweed LooseSilky-bent; do
    ls $s/*
done | perl -pe 's/\//\t/' > classes-and-fnames.txt

