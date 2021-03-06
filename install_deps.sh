sudo apt-get install python
sudo apt-get install python-pip python-dev build-essential cmake git pkg-config
sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install libfreetype6-dev libpng-dev
sudo apt-get install unzip

unzip opencv-2.4.12.zip -d .
cd opencv-2.4.12
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D BUILD_PYTHON_SUPPORT=ON \
      -D WITH_CUDA=OFF\
      -D WITH_XINE=ON \
      -D WITH_OPENGL=ON \
      -D WITH_TBB=ON \
      -D BUILD_EXAMPLES=ON \
      -D BUILD_NEW_PYTHON_SUPPORT=ON \
      -D WITH_V4L=ON \
      -D CMAKE_INSTALL_PREFIX=~/.opencv-2.4.6.1 \
      -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
      .
make -j4
sudo make install
export PYTHONPATH=~/.opencv-2.4.6.1/lib/python2.7/dist-packages

pip install numpy
pip install scipy
pip install matplotlib
pip install -U scikit-learn
pip install ipython[notebook]

sudo apt-get install ipython ipython-notebook
