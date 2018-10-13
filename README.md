# Generating Reviews and Discovering Sentiment
Code for [Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/abs/1704.01444) (Alec Radford, Rafal Jozefowicz, Ilya Sutskever).
Docker image adapted by [Loreto Parisi](https://github.com/loretoparisi/)

## How To Build the Docker image

To build the Docker image for CPU only

```
git clone https://github.com/loretoparisi/generating-reviews-discovering-sentiment.git
cd generating-reviews-discovering-sentiment
docker build -t sentiment-neuron -f Dockerfile .
```

or execute `./build.sh`

while to build the Docker image for GPU

```
cd generating-reviews-discovering-sentiment
docker build -t sentiment-neuron -f Dockerfile.gpu .
```

or you execute `./build.sh GPU`

## How To Run the Docker image

To run for CPU

```
cd generating-reviews-discovering-sentiment
docker run --rm -it sentiment-neuron bash
```

or  execute `./run.sh`

while to run for GPU you have to attach the nvidia-docker driver and device (here we attach the device 0, that is the first GPU as default):

```
docker run -it --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia0 --volume-driver nvidia-docker -v nvidia_driver_367.57:/usr/local/nvidia:ro $IMAGE $CMD

```

or  execute `./run.sh GPU`

## How To Use it

As soon as you run the image you will be in the `/sentiment` folder.
Then you can run the provided examples `test_sentiment.py`:

```
root@718644c454d5:/sentiment# python test_sentiment.py 
7.592 seconds to transform 8 examples
it was a nice day 0.012658
it was a great day 0.371533
it was a bad day -0.499269
It was a wonderful day 0.503395
It was an excellent day 0.44557
It was a super excellent day 0.623401
It was such a bad bad day  -0.858701
It was such a bad bad bad day -1.04497
```

and the `test_generative.py` example, adapted from [this](https://github.com/ahirner/generating-reviews-discovering-sentiment) fork.

```
root@e713b094abb6:/sentiment# python test_generative.py 
'I couldn't figure out'... --> (argmax sampling):
Positive sentiment (1 sentence): 
 I couldn't figure out how to use the stand and the stand but I love it and it is so easy to use.

Negative sentiment (+100 chars):
 I couldn't figure out how to get the product to work and the company would not even try to resolve the problem.  I would ...


'I couldn't figure out'... --> (weighted samples after each word):
Positive sentiment (3 examples, 2 sentences each):
(0) I couldn't figure out what was going on with the characters from page one. I was so engrossed in the story that I read all day.
(1) I couldn't figure out how to install the installation video that came with it but I am so glad I did. My son was so excited to put this together for me.
(2) I couldn't figure out what it was until finding this book by accident.  Every time I encounter a book from this trilogy I enjoy it as much now as I did when I was a child.

Negative sentiment (3 examples, 2 sentences each):
(0) I couldn't figure out how to get the stupid thing to play youtube videos.  I should have never bought this product.
...
```

## Notes

- I had to merge the PR [here](https://github.com/openai/generating-reviews-discovering-sentiment/pull/20) to support the generative test that adds the `generate_sequence` method.
- To enable the Nvidia GPU on the host machine, you need to have `nvidia-docker` installed. To check the nvidia toolkit installation please run the `nvidia-smi` command to list the available connected gpu.
- To address some python language compatibility issues, I'm using the `tensorflow` latest python3 docker image  -
 `tensorflow:latest-py3` and `tensorflow:latest-gpu-py3` for the gpu.
- I'm adding the `tqdm` module via pip in the Dockerfile.


## How to run Feature Extractor

```
from encoder import Model

model = Model()
text = ['demo!']
text_features = model.transform(text)
```

A demo of using the features for sentiment classification as reported in the paper for the binary version of the Stanford Sentiment Treebank (SST) is included as `sst_binary_demo.py`. Additionally this demo visualizes the distribution of the sentiment unit like Figure 3 in the paper.

![Sentiment Unit Visualization](/data/sst_binary_sentiment_unit_vis.png)

Additionally there is a [PyTorch port](https://github.com/guillitte/pytorch-sentiment-neuron) made by @guillitte which demonstrates how to train a model from scratch.

This repo also contains the parameters of the multiplicative LSTM model with 4,096 units we trained on the Amazon product review dataset introduced in McAuley et al. (2015) [1]. The dataset in de-duplicated form contains over 82 million product reviews from May 1996 to July 2014 amounting to over 38 billion training bytes. Training took one month across four NVIDIA Pascal GPUs, with our model processing 12,500 characters per second.

[1] McAuley, Julian, Pandey, Rahul, and Leskovec, Jure. Inferring networks of substitutable and complementary products. In *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794. ACM, 2015.
