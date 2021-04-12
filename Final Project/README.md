# INM431-Machine Learning Final Project

- Software version: MATLAB2020b

1. nbc.m
2. nbcCatdogbase.m
3. nbcCatdogpca.m
4. lrcCatdogbase.m (estimated runtime ~36hrs, result for this not included in analysis)
5. lrcCatdogpca.m

# INM427-Neural Computing Final Project

- Anaconda Environment Setup command line using .yaml; conda env create --file yaml_file.yaml
- Alternatively, following steps is also available for creating the same conda environment as .yaml

1. Open the Windows PowerShell
2. Move to the directory where requirements file are placed
3. Check conda environment status and confirm the base is activated; conda info --envs
4. Add two conda channels;
conda config --add channels conda-forge
conda config --add channels pytorch
5. Create a test environment; conda create -n test_envs --file .\modelling_req_conda.txt
6. Proceed installation [y]
python.exe installation is possible to skip for this stage (if the server doesn't allow any new program)
7. Activate the test_envs; conda activate test_envs
8. Install pip packages; pip install -r .\modelling_req_pip.txt
9. Check GPU version; nvcc --version
and run the modified command line regarding on Cuda compilation tools version; conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -c conda-forge
10. Launch jupyter notebook; jupyter notebook
open file in a browser or copy & paste one of URLs
11. Run All
