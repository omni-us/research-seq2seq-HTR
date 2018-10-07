# Convolve, Attend and Spell: An Attention-based Sequence-to-Sequence Model for Handwritten Word Recognition.

## Paper:

[Convolve, Attend and Spell: An Attention-based Sequence-to-Sequence Model for Handwritten Word Recognition.
L. Kang, J.I. Toledo, P. Riba, M. Villegas, A. Fornés and M. Rusiñol.
To appear in Proceedings of the German Conference on Pattern Recognition, GCPR18 2018.](http://www.cvc.uab.es/~marcal/pdfs/GCPR18.pdf)

## Software environment:

- Ubuntu 16.04 x64
- Python 3.5
- PyTorch 0.3

## Architecture:

![](https://user-images.githubusercontent.com/9562709/43207634-c8028e9a-9028-11e8-80e2-b4e8f8b309e5.png)

Figure 1. Architecture of Seq2Seq model with attention mechanism.

## How to run:

### Dataset preparation:

1. Download the [IAM datasets](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) into <your_folder>, copy all the word-level images into folder <your_folder>/words/, remember to make sure that in the words/ directory, there should be no subdirectory structure. And then copy the groundtruth files from RWTH_partition/ to <your_folder>. You can also find the original RWTH Aachen partition [here](https://github.com/jpuigcerver/Laia/tree/master/egs/iam/data/part/lines/aachen), which is in line-level.

e.g. In <your_folder>:

- words/

- RWTH.iam_word_gt_final.train.thresh

- RWTH.iam_word_gt_final.valid.thresh

- RWTH.iam_word_gt_final.test.thresh

In <your_folder>/words/:

- c04-028-05-07.png

- g06-011f-06-12.png  

- l01-023-03-03.png   

- r06-143-03-10.png

- c04-028-05-08.png

- etc.


2. Open file datasetConfig.py and replace the dataset location in the first row with <your_folder>. Remember to put a "/" to the end of <your_folder>. It's not guaranteed that the model can work fine with the line-level dataset, but if you are interested, you are welcome to have it a try:-)

### Training process:

Just type ./run_train.sh in your terminal, then it will work. If you have multiple GPUs, you can also open file run_train.sh and change the number in row 3 to specify the GPU to use. 

It is also available for you to run the training process again from specific epoch, which allows you to change the learning rate and see the result without training from the beginning. Open file run_train.sh, comment the row 1 while uncomment the row 2, then change the number of both row 2 and row 4 to the epoch you want to run from. But keep in mind, you must run over that epoch before, because there are no pre-trained weights in this repo.

### Evaluation process:

Just type ./run_test.sh <epoch> in your terminal, and you can get the result soon. You are using the specific weights at <epoch>. It is the same that you can specify the GPU to use in the run_test.sh.


### Calculating the CER and WER:

1. Run python3.5 pytasas_words.py <final_epoch> <flag>, and it will calculate all the CERs from first epoch to <final_epoch>. If you set <flag> "si", the test CER will also be calculated, while if the <flag> is "no", only the training CER and validation CER can be obtained. E.g. "python3.5 62 si".

2. When calculating the WER, you need to do the step 1 and 2 again, but it is the file pytasas_words_wer.py.

## Results:

![](https://user-images.githubusercontent.com/9562709/43208467-cfb127d0-902a-11e8-9295-96e0717ca784.png)

Figure 2. Training curves and some attention samples for test set.

The scripts to generate this result can be found in folder others/.

## References:
