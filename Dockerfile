FROM jupyter/datascience-notebook

RUN python -m pip install --upgrade pip
RUN pip install image
RUN pip install addict
RUN pip install future
RUN pip install lmdb
RUN pip install numpy
RUN pip install opencv-python
RUN pip install Pillow
RUN pip install pyyaml
RUN pip install requests
RUN pip install scikit-image
RUN pip install scipy
RUN pip install tb-nightly
RUN pip install tqdm
RUN pip install yapf
RUN pip install image_slicer
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install --upgrade tensorflow

