
##----------------------------------------SETUP_KAGGLE------------------------------------------------
from google.colab import files
import os
import shutil
def setup_kaggle(upload=False):
  """
  This function helps you to setup kaggle and user kaggle command in colab notebook.

  Parameters
  ----------
  upload (bool): if you want to upload kaggle.json file then, default True
  """
  
  if upload:
    uploaded = files.upload()
    for fn in uploaded.keys():
      print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
  if not os.path.exists('/root/.kaggle'):
    os.mkdir('/root/.kaggle')
  
  shutil.move("kaggle.json", "/root/.kaggle/kaggle.json")
  os.chmod('/root/.kaggle',600 )
  
##----------------------------------------UNZIP------------------------------------------------
import zipfile

def unzip(filepath):
  """
  This function unzip a zip file

  Parameters
  ----------
  filepath (str): provide filepath you want to extract all data
  """
  
  zip_ref = zipfile.ZipFile(filepath)
  zip_ref.extractall()
  zip_ref.close()
  
##----------------------------------------WALK_THROUGH_DIR------------------------------------------------
def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

