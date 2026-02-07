



`setup_speakers` takes all files from `/speakers_samples` and trains the voice and labels it as the filename of the .wav

dirration: the "enrollment" (training in voices) script is freely available from HuggingFace, the live


tech
- `pyannote/embedding` converts arbitrary lengthed audio clips of a single person into a vector repsentation#
- - 
- - (we could theoretically both speakers and create a hybrid of both)
- `pyannote/speaker-diarization-3.1` is multiple models stacked
- - segmentation model of speech vs non-speech, general speaker changes and overlaps
- - embedding model of the same kind as used in enrollment
- - clustering algorithm which takes the vectors created from the segmentation model with the ones of the embedding model to label them




setup
- brew install portaudio