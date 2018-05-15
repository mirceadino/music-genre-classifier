# music-genre-classifier
:notes:

## Refinements
- [ ] Write an installation file:
  * pip requirements
  * brew/apt-get packages: ffmpeg
  * something with the encoding from youtube-dl
- [x] On training mode, load the model (if one exists) and continue training on it.
- [x] On predicting mode, if the file doesn't exist then consider it as an URL or a Youtube video ID and download it. Also modularize that part.
- [ ] Write a module for downloading songs, in batch from file or individually from url.
- [ ] Add a re-download mode that downloads the songs from an existing info.csv. 
- [ ] Consider better modularization of the code.
- [ ] Consider testing various known architectures.

## TODO - application
- [x] Run remotely with GPU and write requirements to make the code portable. __It's now running on AWS!__
- [x] Prepare structure of the dataset paths.
- [x] Write song processing and dataset creation.
- [x] Fix the reversed label bug.
- [ ] Add statistics on accuracy: confusion matrix
- [ ] Collect more songs. Possible ideas:
  * download more songs; consider moving the downloader to the main program or to write a proper downloader
  * find an online dataset; advatange: I can compare results with other approaches
- [ ] Extend classifier to more genres.
- [ ] Classify a slice as "other" rather than classying it with a low confidence.
- [ ] Implement alternative method to classify a full song. Possible ideas:
  * maintain voting method
  * use a NN and feed it with the predicted genres for all the slices
  * use a recurrent NN and feed it with slices
- [ ] Add a minimal GUI: something with an address bar where the user can type a Youtube url
- [ ] (stretch) Experiment with subgenres:
  * try subgenres of latino: bachata, salsa etc.
  * try subgenres of multiple genres
  
## TODO - thesis
- [ ] Consider writing a paper for SCSS
- [ ] Write content summary.
- ...
