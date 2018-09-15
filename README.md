# Detect video character relations using face recognition

**Goal**: create visual characters / actors relationship graph by detecting which characters appear in the same video frames.

Popular [The Office (U.S.)](https://www.imdb.com/title/tt0386676/) season 1 series were chosen for processing.

Result:

![Relations graph and red rectangles highlighting actors faces](https://raw.githubusercontent.com/tautrimas/video-face-recognition/master/web_assets/office_s01_face_recognition.png)

## Workflow

1. Use [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition) (uses [dlib](http://dlib.net/)) to detect faces and extract facial features from video into JSON.
    1. Decodes video frames using [OpenCV](https://opencv.org/)
    1. Uses 1 frame per second from the video input
    1. Streams faces features as JSON output
1. Match extracted faces with known faces using [euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance).
    1. Face features are 128-th dimensional vectors of floats.
    1. I had to add manually inspect which actor's faces were not detected and add face sample and character name mappings for each pair.
1. Record which faces appeared together in the same frames
1. Draw a graph where two frequently together appearing characters are close in the graph too.
    1. Used [networkx](https://networkx.github.io/) and [matplotlib](https://matplotlib.org/) libraries
