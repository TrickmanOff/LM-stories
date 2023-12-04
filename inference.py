import json
import shutil
from pathlib import Path

import torch

from lib.encoder import BPETextEncoder
from lib.model import SimpleTransformer
from lib.text_generator import RandomGenerator
from lib.utils import download_file, get_device


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

    encoder = BPETextEncoder(name='tiny_stories_encoder_4k',
                             encoder_dirpath=encoder_dirpath)

    device = get_device()
    model_config = json.load(open(Path(__file__).parent / 'model_configs/model1.json', 'r'))
    model = SimpleTransformer(vocab_size=encoder.vocab_size, max_len=512, **model_config)
    state_dict = torch.load(checkpoint_filepath, map_location=device)
    model.load_state_dict(state_dict['model'])
    print('Model weights loaded')

    text_generator = RandomGenerator(temp=0.2, top_k=35, model=model, encoder=encoder, max_len=512)
    while True:
        prompt = input('Enter a prompt')
        print('Processing...')
        result = text_generator.generate_text(prompt)
        print(result)
        print('='*10)


if __name__ == '__main__':
    main()
