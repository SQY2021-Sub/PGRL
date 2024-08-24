export CUDA_VISIBLE_DEVICES=1

# starting program

python -u main.py \

#Start TensorBoard and view the model structure diagram:

tensorboard --logdir=runs

#Open your browser and go to http://localhost:6006 and you will see the TensorBoard interface

#If your environment supports it, you can also start TensorBoard directly in Jupyter Notebook:

%load_ext tensorboard
%tensorboard --logdir runs


