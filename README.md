# Search Engine for Pretrained Neural Network Architecture

## Code implementation of Neural Architectures Neural Architecture Retrieval.

#### To discover similar neural architectures in an efficient and automatic manner, we define a new problem Neural Architecture Retrieval which retrieves a set of existing neural architectures which have similar designs to the query neural architecture. Existing graph pre-training strategies
cannot address the computational graph in neural architectures due to the graph size and motifs. To fulfill this potential, we propose to divide the graph into motifs which are used to rebuild the macro graph to tackle these issues, and introduce multi-level contrastive learning to achieve accurate graph representation learning. Extensive evaluations on both human-designed and synthesized neural architectures demonstrate the superiority of our algorithm. Such a dataset which contains 12k real-world network architec-
tures, as well as their embedding, is built for neural architecture retrieval.

![LICENSE](https://img.shields.io/github/license/TerryPei/NNSearchUI)
![VERSION](https://img.shields.io/badge/version-v1.01-blue)
![PYTHON](https://img.shields.io/badge/python-3.8-orange)
![MODEL](https://img.shields.io/badge/NNSearchUI-v1.01-red)
![search engine](/demos/demo1.gif)

This project is a robust search engine (vectors database) dedicated to pretrained neural network architectures. Utilizing Flask and Javascript, the search engine provides users with easy access to a multitude of neural network architectures for a variety of purposes.


## Functions:

- Search by the name or purpose of the neural network.
- Retrieval of pretrained models for various applications.
- Seamless user experience with a clean and interactive UI.

## Related Skills
> Pytorch, Flask, Javascript, CSS, HTML

## Installation

```bash
# clone the repository
git clone https://github.com/yourusername/NNSearchUI.git
cd NNSearchUI

# install dependencies
pip install -r requirements.txt
npm install

# start the application
python app.py
