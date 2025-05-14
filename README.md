# AI Upskilling Workshop: Bioacoustics track

Welcome! In this workshop, we will be learning the basics of transfer learning by training frog species classifiers using audio embeddings from open-source bird species classifiers. Cool!

## 0. Environment setup

We will provide you with a login to a virtual machine managed by CyVerse. Once logged in, run the following commands to get an environment up and running:

**For easiest debugging, copy+paste one command into your terminal at a time (rather than the whole block at once).**

**1. Install miniconda**, which we will use to manage Python environments:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
Say "yes" to everything, and use all the defaults. If you accidentally say no to something, you can run it again with `./Miniconda3-latest-Linux-x86_64.sh -u` to fix it.

**2. Install perch-hoplite and dependencies**
```
source ~/.bashrc
conda create -n hoplite python=3.10
conda activate hoplite
sudo apt-get update
sudo apt-get install libsndfile1 ffmpeg
pip install absl-py requests 'tensorflow[and-cuda]<2.16' tensorflow-hub ipywidgets etils
pip install git+https://github.com/google-research/perch-hoplite.git
git clone https://github.com/google-research/perch-hoplite.git
```

**3. Clone this codebase!**
```
git clone https://github.com/CV4EcologySchool/ai-upskilling-audio.git
```

Once this is good to go, we recommend installing VS Code locally and connecting via SSH. See slides 19, 20, 33, and 34 [here](https://docs.google.com/presentation/d/1VFkmj5dvtlnziBOFM9GO4PEcfhcPrv7tfBaSo6uEOWg/edit#slide=id.g2680127c5bb_0_88) for a walkthrough.

## 1. Perch-hoplite notebook 1: Create embeddings

We will first run through an example of *agile modeling* using the `perch-hoplite` codebase.

We have already downloaded the AnuraSet dataset to a read-only shared storage drive at `/mnt/class_data/anuraset/`. To work with thie data, first make a copy of a subset of the AnuraSet dataset (specifically, data from site INCT17) on your VM's local storage:

```
mkdir data
cp -r /mnt/class_data/anuraset/audio/INCT17/ data
cp /mnt/class_data/anuraset/metadata.csv data
```

Then, run through the steps in `perch_hoplite/agile/1_embed_audio_v2.ipynb`.

**Your task:** You will need to **update the file and directory paths** in the notebook to point to your copy of the data. Once you do this, you will be able to extract embeddings. This assignment is complete as soon as you get the embedding creation process running -- you don't actually need to wait for it to finish. For instance, once you see something like this:
```
Embedding dataset: inct17
  8%|▊         | 1704/20532 [00:40<03:07, 100.47it/s]
```
Feel free to interrupt the kernel and move on.

## 2. Data exploration and human learning

Now you will prepare yourself for some agile modeling by (1) Doing some basic data exploration and analysis in Python, and (2) Training yourself to identify frog species from their vocalizations! Once you complete this "human learning" process, you will be ready to train a machine learning model to replicate your classification ability.

Complete the `01_anuraset_explore.ipynb` notebook in this repo.

Note: in this notebook you will **not need to change the default path** (you won't be modifying anything, so you can just use the read-only copy on the shared drive `/mnt/class_data`).

## 3. Perch-hoplite notebook 2:

Now you will use your knowledge to quickly create a frog species classifier with agile modeling. This classifer will map from embeddings to species, using the embeddings from step 1. We told you to interrupt your database creation in step 1, because we've done it for you for the entire dataset!

Copy over the full database with:
```
cp /mnt/class_data/anuraset/*sqlite* data/
cp /mnt/class_data/anuraset/usearch.index data/
```

Then, run through the steps in `perch_hoplite/agile/2_agile_modeling_v2.ipynb` to create your own classifer via agile modeling. 

Agile modeling starts with a "query": a single example of the thing you are looking for. You may need to do some sleuthing on your own to find a good query clip for your species of interest (you may have already done this in the previous notebook) -- look for a single WAV file where your target species is clearly audible. In the "Search" cell, change `query_uri` to be a filepath of a WAV file containing your species of interest, and `query_label` to your species label (e.g. 'SPHSUR').

Run the search; you should be able to click on the label for each example to indicate whether it matches your target sound. Green for yes, orange for no, gray for unsure.

Search options:
* `exact_search` will search all embeddings and give a plot of the histogram of all similarity scores.
* `target_sampling` allows you to search for examples with a particular score. Set the score with `target_score` to something lower int he distribution to get some negative examples.
  (note - there is, buggily, a line which sets `target_score=None` in the `exact_search` code block. Delete this line.)

After running the search and labeling some examples, you can train a classifier. You don't need many examples!
After training the classifier, you can display some high-scoring results, and (optionally) label them in the same way as we did for the search results. After labeling more examples and saving them to the DB, we can re-train the classifier.

In active learning, *margin sampling* is when we surface samples with a lower score for labeling (eg, near logit=0, corresponding to 50% model confidence). This is similar to target sampling that we did with search. After a couple rounds of labeling things with score near 0, you should see that the distribution of scores becomes more bi-modal. And you should find that the scores near 0 become much more ambiguous...

## 4. Intro to Numpy

Practice basic array operations in Numpy with `02_numpy_intro.ipynb`.

## 5. Training linear models

First, copy over a portion of the raw data from AnuraSet that we will process:

```
mkdir data/raw_data
cp -r /mnt/class_data/anuraset/raw_data/ data/raw_data
cp -r /mnt/class_data/anuraset/strong_labels/ data
```

Then, complete the `03_anuraset_train.ipynb` notebook.

## 6. Go fasterer

Wanna go fast? Soup up your data loading pipeline with multithreading in `03b_threads.ipynb`.

## 7. Learn more about vector databases

`04_vector_search.ipynb`

## 8. Additional mini-projects

You've successfully trained a simple binary classifier with Perch embeddings with `03_anuraset_train.ipynb`. Nice work! Here are some additional things you can try to gain more experience training and evaluating classifiers:

- [Evaluation] Think about different ways you can stratify your test set. How does the model perform on different sites? How about on clips with different numbers of species present? From different days or times of day? Make some plots to break down performance.

- [Training & Evaluation] Try using different embedding models. How does the choice of embedding change the performance of your classifier? How does this change across the strata you came up with before?

- [Training] Try training a multiclass classifier, rather than a binary classifier.

- [Evaluation] Try recreating the t-SNE plot from [this paper, Figure 3](https://www.nature.com/articles/s41598-023-49989-z/figures/3). Can you think of other ways to color-code the points (hint: what about your strata above?)


