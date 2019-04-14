# Installing Sudo
apt-get update && apt-get install -y sudo && rm -rf /var/lib/apt/lists/*
# Installing C++ and make tools
apt-get update && apt-get install -y gcc g++ make libgtest-dev cmake
# Installing Google test
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp *.a /usr/lib

# Getting Datasets

# MNIST
git clone https://github.com/wichtounet/mnist.git 