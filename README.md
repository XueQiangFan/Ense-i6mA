Ense-i6mA: *Identification of DNA N6-methyladenine Sites Using XGB-RFE Feature Selection and Ensemble Machine Learning.*
====

Contents
----
  * [Abstract](#abstract)
  * [System Requirments](#system-requirments)
  * [Installation](#installation)
  * [Usage](#usage)
  * [Datasets](#datasets)
  * [Citation guide](#citation-guide)
  * [Licence](#licence)
  * [Contact](#contact)


Abstract
----
DNA N6-methyladenine (6mA) is an important epigenetic modification that plays a vital role in various cellular processes. Accurate identification of the 6mA sites is fundamental to elucidate the biological functions and mechanisms of modification. However, experimental methods for detecting 6mA sites are high-priced and time-consuming. In this study, we propose a novel computational method, called Ense-i6mA, to predict 6mA sites. Firstly, five encoding schemes, i.e., one-hot encoding, gcContent, Z-Curve, K-mer nucleotide frequency, and K-mer nucleotide frequency with gap, are employed to extract DNA sequence features. Secondly, to our knowledge, it is the first time that eXtreme gradient boosting coupled with recursive feature elimination is applied to 6mA sites prediction domain to remove noisy features for avoiding over-fitting, reducing computing time and complexity. Then, the best subset of features is fed into base-classifiers composed of Extra Trees, eXtreme Gradient Boosting, Light Gradient Boosting Machine, and Support Vector Machine. Finally, to minimize generalization errors, the prediction probabilities of the base-classifiers are aggregated by averaging for inferring the final 6mA sites results. We conduct experiments on two species, i.e., Arabidopsis thaliana and Drosophila melanogaster, to compare the performance of Ense-i6mA against the recent 6mA sites prediction methods. The experimental results demonstrate that the proposed Ense-i6mA achieves area under the receiver operating characteristic curve values of 0.967 and 0.968, accuracies of 91.4% and 92.0%, and Mathew’s correlation coefficient values of 0.829 and 0.842 on two benchmark datasets, respectively, and outperforms several existing state-of-the-art methods. 

System Requirments
----

**Hardware Requirments:**
Ense-i6mA requires only a standard computer with around 16 GB RAM to support the in-memory operations.

**Software Requirments:**
* [Python3.7](https://docs.python-guide.org/starting/install3/linux/)
* [Pytorch](https://pytorch.org/)
* [Anaconda](https://anaconda.org/anaconda/virtualenv)
* [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) (Optional If using GPU)
* [cuDNN (>= 7.4.1)](https://developer.nvidia.com/cudnn) (Optional If using GPU)

Ense-i6mA has been tested on Ubuntu 18.04 and Window10 operating systems

Installation
----

To install Ense-i6mA and it's dependencies following commands can be used in terminal:

1. `git clone https://github.com/XueQiangFan/Ense-i6mA.git`
2. `cd Ense-i6mA`

Either follow **virtualenv** column steps or **conda** column steps to create virtual environment and to install Ense-i6mA dependencies given in table below:<br />

|  | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; conda |
| :- | :--- |
| 3. |  `conda create -n venv python=3.7` |
| 4. |  `conda activate venv` | 
| 5. |  *To run Ense-i6mA on CPU:*<br /> <br /> `conda install pytorch torchvision torchaudio cpuonly -c pytorch` <br /> <br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *or* <br /> <br />*To run Ense-i6mA on GPU:*<br /> <br /> `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch` |
| 6. | `while read p; do conda install --yes $p; done < requirements.txt` | 

Usage
----

**To run the Ense-i6mA**
### run: python main.py -train_data train path -test_data test path
~~~
    For example:
    python main.py -train_data /DNAN6mAsites/dataset/A.thaliana_test.xlsx -test_data /DNAN6mAsites/dataset/A.thaliana_test.xlsx
~~~

Datasets
----

The following benchmark datasets ware used for Ense-i6mA:
[Datasets](https://github.com/XueQiangFan/Ense-i6mA/tree/main/Dataset)

Citation guide
----

**If you use Ense-i6mA for your research please cite the following papers:**

@article{fan2024ense,
  title={Ense-i6mA: Identification of DNA N6-methyl-adenine Sites Using XGB-RFE Feature Se-lection and Ensemble Machine Learning},
  author={Fan, Xue-Qiang and Lin, Bing and Hu, Jun and Guo, Zhong-Yi},
  journal={IEEE/ACM Transactions on Computational Biology and Bioinformatics},
  year={2024},
  publisher={IEEE}
}

Licence
----
Mozilla Public License 2.0

Contact
----
Thanks for your attention. If you have any questions, please contact my email: xstrongf@163.com
