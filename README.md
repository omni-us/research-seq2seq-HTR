
## Software environment:

- Ubuntu 16.04 x64
- Python 3.5
- PyTorch 0.3

## Structure:

- **main.py** The main file of the encoder-decoder model with Bahdanau attention (Figure 1)
- **mnist.py** The MNIST digits sequence generator, after which you can get sequence datasets (Figure 2)
- **tasas_cer.sh** The tool to calculate character error rate based on [https://github.com/mauvilsa/htrsh](https://github.com/mauvilsa/htrsh)
- **pytasas.py** The python wrapper to access the CER calculated by the shell above
- **drawCER.py** Display the CER for both training data and testing data
- **drawLoss.py** Display the Loss for both training data and testing data
- **clear.sh** Clean the pred_logs directory

![](https://user-images.githubusercontent.com/9562709/32320622-b86b0e5c-bfbe-11e7-8c10-a37c534dba34.png)

Figure 1. Encoder-decoder model with Bahdanau attention

