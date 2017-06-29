make install
#../install/bin/image_encoder -z seq -i passau.pgm -o out_seq
../install/bin/image_encoder -z cpu -i passau.pgm -o out_cpu
diff out_cpu out_seq