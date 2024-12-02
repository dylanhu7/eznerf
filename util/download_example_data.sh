# Credit to https://github.com/bmild/nerf
mkdir -p data
cd data || exit
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
unzip nerf_example_data.zip
rm nerf_example_data.zip
cd ..
