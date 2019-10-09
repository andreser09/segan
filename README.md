## Notes to run this repository

Since there are some bugs to run the original SEGAN due to the tensorflow version used I've added these notes to ease the process and also share some tips to improve the results.

The easiest way that I found to train SEGAN from scratch was using tensorflow's docker images, mainly because of problems with CUDA and cudnn.

First pull the docker image:

	docker pull tensorflow/tensorflow:1.4.1-gpu 

Must have installed nvidia-docker in your computer

Inside the container clone this repository.

Install the libraries in the requirements.txt 

Authorize reading/writing for prepare\_data.sh and train\_segan.sh

	chmod 755 prepare_data.sh train_segan.sh

prepare\_data.sh creates the required folders to run SEGAN and converts the training set to 16khz.

	./prepare_data.sh
	
The dataset used originally was deprecated so this file will download the dateset found in

	https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/

I used the 56spk dataset because it provides better results than the used originally by the authors of the paper.

train_segan.sh will train SEGAN with the standard parameters.

	./train_segan.sh

The default parameters work just fine but I found that moving towards bigger batch size in training improves the result a lot (with longer training time though). Smaller learning rate (0.00008) for both the generator and the discriminator and also more epochs (100-150) will give noticeable improvements.

Don't forget to change the appropriate flags in the script.

To run inference on noisy files you can run 

	./clean_wav.sh
	
You should pass as arguments the name of the audio file and also the name of the directory to save the cleaned audio. 

Don't forget to change the --save\_path flag with the name of the folder with the weights and the --weights flag with the name of the weight file. If you changed the --batch\_size in training you should also change it for inference.

I also found that the quality of the results decrease with the length of the audio file. That's why I made a small script to split the audio in smaller chunks to be denoised separately and then merged again. The script uses the pydub library.

In order to function properly you need to setup a noise threshold for the splitting and a minimum value to be considered silence (in ms). There is also a function to change the playback speed that can be useful sometimes.

You need to create a folder named /files with two folders inside it: /chunks and /segan
Inside of /files place the audios you want to denoise. The output will appear on /segan after denoising. /chunks is a placeholder for temporary files

	python split.py


## (Original Notes) SEGAN: Speech Enhancement Generative Adversarial Network

### Introduction

This is the repository of the SEGAN project. Our original paper can be found [here](https://arxiv.org/abs/1703.09452), and test samples are available [here](http://veu.talp.cat/segan/).

In this work a Generative Adversarial approach has been taken to do speech enhancement (i.e. removing noise from corrupted speech signals) with a fully convolutional architecture schematized as follows:

![SEGAN_G](assets/segan_g.png)

This model deals with raw speech waveforms on many noise conditions at different SNRs (40 at training time and 20 during test). It also models the speech characteristics from many speakers mixed within the same structure (without any supervision of identities), which makes the generative structure generalizable in the noise and speaker dimensions.

**All the project is developed with TensorFlow**. There are two repositories that were good references on how GANs are defined and deployed:

* [improved-gan](https://github.com/openai/improved-gan): implementing improvements to train GANs in a more stable way
*  [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow): implementation of the DCGAN in tensorflow

### Dependencies

* Python 2.7
* TensorFlow 0.12

You can install the requirements either to your virtualenv or the system via pip with:

```
pip install -r requirements.txt
```

### Data

The speech enhancement dataset used in this work [(Valentini et al. 2016)](http://ssw9.net/papers/ssw9_PS2-4_Valentini-Botinhao.pdf) can be found in [Edinburgh DataShare](http://datashare.is.ed.ac.uk/handle/10283/1942). However, **the following script downloads and prepares the data for TensorFlow format**:

```
./prepare_data.sh
```

Or alternatively download the dataset, convert the wav files to 16kHz sampling and set the `noisy` and `clean` training files paths in the config file `e2e_maker.cfg` in `cfg/`. Then run the script:

```
python make_tfrecords.py --force-gen --cfg cfg/e2e_maker.cfg
```

### Training

Once you have the TFRecords file created in `data/segan.tfrecords` you can simply run the training process with:

```
./train_segan.sh
```

By default this will take all the available GPUs in your system, if any. Otherwise it will just take the CPU.

**NOTE:** If you want to specify a subset of GPUs to work on, you can do so with the `CUDA_VISIBLE_DEVICES="0, 1, <etc>"` flag in the python execution within the training script. In the case of having two GPUs they'll be identified as 0 and 1, so we could just take the first GPU with: `CUDA_VISIBLE_DEVICES="0"`.

A sample of G losses is interesting to see as stated in the paper, where L1 follows a minimization with a `100` factor and the adversarial loss gets to be equilibrated with low variance:

**L1 loss (smoothing 0.5)**

![G_L1](assets/g_l1_loss.png)

**Adversarial loss (smoothing 0.5)**

![G_ADV](assets/g_adv_loss.png)

### Loading model and prediction

First, the trained weights will have to be downloaded from [here](http://veu.talp.cat/segan/release_weights/segan_v1.1.tar.gz) and uncompressed.

Then the `main.py` script has the option to process a wav file through the G network (inference mode), where the user MUST specify the trained weights file and the configuration of the trained network. In the case of the v1 SEGAN presented in the paper, the options would be:

```
CUDA_VISIBLE_DEVICES="" python main.py --init_noise_std 0. --save_path segan_v1.1 \
                                       --batch_size 100 --g_nl prelu --weights SEGAN_full \
                                       --test_wav <wav_filename> --clean_save_path <clean_save_dirpath>
```

To make things easy, there is a bash script called `clean_wav.sh` that accepts as input argument the test filename and
the save path.

### Authors

* **Santiago Pascual** (TALP-UPC, BarcelonaTech)
* **Antonio Bonafonte** (TALP-UPC, BarcelonaTech)
* **Joan Serrà** (Telefónica Research, Barcelona)

### Reference

```
@article{pascual2017segan,
  title={SEGAN: Speech Enhancement Generative Adversarial Network},
  author={Pascual, Santiago and Bonafonte, Antonio and Serr{\`a}, Joan},
  journal={arXiv preprint arXiv:1703.09452},
  year={2017}
}
```

### Contact

e-mail: santi.pascual@upc.edu

### Notes

* If using this code, parts of it, or developments from it, please cite the above reference.
* We do not provide any support or assistance for the supplied code nor we offer any other compilation/variant of it.
* We assume no responsibility regarding the provided code.
