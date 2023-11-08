import gdown 

def downloadGDriveFile(id: str, dest_file_path: str):
    prefix = 'https://drive.google.com/uc?/export=download&id='
    url = prefix + id
    return gdown.download(url = url, output=dest_file_path)