import shutil
from pathlib import Path

from lib.utils import download_file


URL_LINKS = {
    'model': 'https://www.googleapis.com/drive/v3/files/1Fsknyk8DWbI1nASTISscU2Kwxbr3gKYl?alt=media&key=AIzaSyBAigZjTwo8uh77umBBmOytKc_qjpTfRjI',
    'encoder': 'https://www.googleapis.com/drive/v3/files/1vOVg500jLVlqTKhXeeNeL2H5DTtn_Wiv?alt=media&key=AIzaSyBAigZjTwo8uh77umBBmOytKc_qjpTfRjI',
}

INFERENCE_DIRNAME = 'inference'


def main():
    # download model
    checkpoint_filename = 'checkpoint.pth'
    checkpoint_filepath = Path(INFERENCE_DIRNAME) / checkpoint_filename
    if not checkpoint_filepath.exists():
        print('Downloading LM checkpoint...')
        download_file(URL_LINKS['model'], Path(INFERENCE_DIRNAME), checkpoint_filename)

    # downloading encoder
    encoder_dirpath = Path(INFERENCE_DIRNAME) / 'encoder'
    if not encoder_dirpath.exists():
        print('Downloading encoder')
        encoder_dirpath.mkdir(parents=True)
        encoder_arch_path = encoder_dirpath / 'encoder.zip'
        if not encoder_arch_path.exists():
            download_file(URL_LINKS['encoder'], encoder_dirpath, 'encoder.zip')
        shutil.unpack_archive(encoder_arch_path, encoder_dirpath)


if __name__ == '__main__':
    main()
