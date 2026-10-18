LlamaPartialSpoof Dataset v1.0.b
This dataset is made available under Creative Commons Attribution 4.0 International License (CC BY 4.0). 

For any inquiries, please email me at contact @ hieuthi.com

v1.0.b includes two parts:
- R01TTS.0.a contains bonafide, fully fake (TTS001--006), partially fake (TTS001--006) using crossfade
- R01TTS.0.b contains partially fake (TTS001--006) using cut/paste or overlap/add 

Labels of each part is in text files with corresponding names while speech samples are in the archives.

Each line in the label files is formatted as follows:
<id> <utterance-duration> <utterance-label> <segment1> <segment2> ... <segmentN>

Each segment is formatted as follows:
<start>-<end>-<label>

Label is either "bonafide" or "spoof".
The id is also the name of the wave files. It explicitly states the name of the TTS model used for generation.

The dataset also includes a metadata file (metadata_crossfade.csv) contains information about the fading function used to create partially fake samples (crossfade). With t=triangle (linear), q=quarter of sinewave, h=half of sinewave, l=logarithmic, p=inverted parabola.

DOI: https://doi.org/10.5281/zenodo.14214149