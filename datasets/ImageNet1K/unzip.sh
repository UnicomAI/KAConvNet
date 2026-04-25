dir=./train 
for x in `ls $dir/*tar` 
do     
  filename=`basename $x .tar`     
  mkdir $dir/$filename     
  tar -xvf $x -C $dir/$filename 
done 
rm $dir/*.tar
