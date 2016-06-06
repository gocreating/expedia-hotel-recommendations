hadoop fs -get result
echo 'id,hotel_cluster' > result/head
cat result/head result/part-* > result.csv
