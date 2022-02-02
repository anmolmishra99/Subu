
from google.colab import files
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
  
  !mkdir ~/.kaggle && mv kaggle.json ~/.kaggle && chmod 600 ~/.kaggle/kaggle.json