
# 3 Commands to this working on your machine
1. `python -m venv [nameofenvironment]`
2. `source ./[env-name]/bin/activate`
3. `pip install -r requirements.txt`

# Useful Links
- [google's dataset for training models](https://ai.google.com/research/ConceptualCaptions/download)
- [microsofts oscar model](https://github.com/microsoft/Oscar)

# Instructions to use coco.sh
- make sure you have 'wget' installed
- run the coco.sh file
    - on windows just type 'coco.sh' in the repo's main dir
- wait for the download

# Running the ImageCaptioning pre-trained model
1. Clone the repo
    - `git clone --recursive https://github.com/ruotianluo/ImageCaptioning.pytorch`
2. Install
    - `python -m pip install -e .`
3. Download a pretrained resnet101 model
    - https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/MODEL_ZOO.md
4. Install transformers
    - `pip install transformers` OR `conda install -c huggingface transformers`
5. Install gensim
    - `pip install gensim`
6. Install lmdbdict
    - `pip install lmdbdict`
7. Update `eval_utils.py` to copy the images properly
    - `captioning/utils/eval_utils.py`
    - Add `import shutil`
    - Replace
        ```python
        cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
        print(cmd)
        os.system(cmd)
        ```
        with
        ```python
        source = os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path'])
        dest = 'vis/imgs/img' + str(len(predictions)) + '.jpg'
        print('Copying ' + source + ' to ' + dest)
        shutil.copyfile(source, dest)
        ```
8. Run the model
    - Running from repo root, assuming pretrained model is in `model` and the images to run are in `images`
    - `python tools/eval.py --model .\model\model-best.pth --infos_path .\model\infos_fc-best.pkl --image_folder images --num_images 10`
9. Start the web server
    - From `vis` directory, `python -m http.server`
