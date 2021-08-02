
#!/usr/bin/env bash


for block in P2.2 #A1.1  A1.2  A1.3  A1.4  A2.1  A2.2  A2.3  A2.4  A3.1  A3.2  A3.3  A4.1  A4.2  A5.1  A5.2  A6.1  P1.1  P1.2  P1.3  P1.4  P2.1  P2.2  P2.3  P3.1  P3.2  P3.3  P4.1  P4.2  P4.3  P5.1  P5.2  P6.1  P6.2  P7.1  P8.1
do
    python train.py --stain MRI --block $block
#    python train.py --stain LFB --block $block
done