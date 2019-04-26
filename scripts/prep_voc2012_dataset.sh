

mkdir /tmp/voc
cd /tmp/voc && wget "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
cd /tmp/voc && tar xf "VOCtrainval_11-May-2012.tar"


cd /tmp/voc/VOCdevkit/VOC2012/


all_lines=`cat ImageSets/Segmentation/train.txt`
out_dir="./voc_prepped"
mkdir $out_dir
mkdir $out_dir/images_prepped_train
mkdir $out_dir/images_prepped_test
mkdir $out_dir/annotations_prepped_train
mkdir $out_dir/annotations_prepped_test


for item in $all_lines;
do
  cp SegmentationClass/$item.png $out_dir/annotations_prepped_train
  cp JPEGImages/$item.jpg $out_dir/images_prepped_train
  inp="\"$out_dir/annotations_prepped_train/$item.png\""
  python -c "import numpy as np ; from PIL import Image; import cv2 ; hoho = np.array(Image.open( $inp )) ; hoho[hoho == 255 ] = 0 ; hoho = np.repeat(hoho[: , : , None] , 3 , 2 ) ; cv2.imwrite($inp , hoho );print($inp)"
done


all_lines=`cat ImageSets/Segmentation/train.txt`


for item in $all_lines;
do
  cp SegmentationClass/$item.png $out_dir/annotations_prepped_test
  cp JPEGImages/$item.jpg $out_dir/images_prepped_test
  inp="\"$out_dir/annotations_prepped_test/$item.png\""
  python -c "import numpy as np ; from PIL import Image; import cv2 ; hoho = np.array(Image.open( $inp )) ; hoho[hoho == 255 ] = 0 ; hoho = np.repeat(hoho[: , : , None] , 3 , 2 ) ; cv2.imwrite($inp , hoho );print($inp)"
done




