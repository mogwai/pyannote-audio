#TODO REMOVE THIS BEFORE PR
from pyannote.audio.cli import run_train

config = {
'--batch': None,
 '--cpu': False,
 '--debug': False,
 '--duration': None,
 '--every': '1',
 '--from': None,
 '--gpu': False,
 '--help': False,
 '--parallel': '4',
 '--pretrained': None,
 '--step': '0.25',
 '--subset': 'train',
 '--to': '5',
 '--version': False,
 '-h': False,
 '<protocol>': 'AMI.SpeakerDiarization.MixHeadset',
 '<root>': 'tutorials/models/speech_activity_detection',
 '<train>': None,
 '<validate>': None,
 'apply': False,
 'extract': False,
 'train': True,
 'validate': False
}

run_train(config)
