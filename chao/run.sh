
EXE=./knn

n=1024


# benchmark for typical configuration

s="1 2 4 8 128 1024"
d=256
k=128
m=128

for i in $s
do
  $EXE -B -n $n -d $d -k $k -s $i -m $m
done

s="1 2 4 8 128 1024 4096"
d=64
k=64
m=64

for i in $s
do
  $EXE -B -n $n -d $d -k $k -s $i -m $m
done

s="1 2 4 8 128 1024 16384"
d=16
k=16
m=16

for i in $s
do
  $EXE -B -n $n -d $d -k $k -s $i -m $m
done


# benchmark for blocking sizes

$EXE -B -d 256 -k 256 -m 512 -s 1024
$EXE -B -d 256 -k 128 -m 256 -s 2048
$EXE -B -d 256 -k 64 -m 64 -s 4096

$EXE -B -d 64 -k 64 -m 128 -s 4096
$EXE -B -d 64 -k 32 -m 64 -s 8192
$EXE -B -d 64 -k 16 -m 16 -s 16384

$EXE -B -d 16 -k 16 -m 32 -s 16384
$EXE -B -d 16 -k 8 -m 16 -s 32768
$EXE -B -d 16 -k 4 -m 4 -s 65536

