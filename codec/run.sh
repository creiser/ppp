make install
#../install/bin/image_encoder -c -z cpu -i passau.pgm -o out_cpu
srun --gres=gpu -n 1 ../install/bin/image_encoder -c -z gpu -i passau.pgm -o out_gpu
#../install/bin/image_encoder -d -z seq -i passau.pgm -o out_seq

#../install/bin/viewer out_seq
#../install/bin/viewer out_gpu