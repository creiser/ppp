make install
srun --gres=gpu -n 1 ../install/bin/image_encoder -s -c -z gpu -i Alps.pgm -o out_gpu
#../install/bin/image_encoder -d -z cpu -i passau.pgm -o out_cpu
#../install/bin/image_encoder -s -c -z seq -i Alps.pgm -o out_seq

#../install/bin/viewer out_seq
#../install/bin/viewer out_gpu