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
