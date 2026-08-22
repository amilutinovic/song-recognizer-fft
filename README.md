# song-recognizer-fft

An app that uses Fourier transformation to recongnize songs from a short audio recording of a song's clip.

[Songs used](data/ATTRIBUTIONS.md)

## Build and run locally


#### 1. Install necessary packages
```bash 
pip install -r requirements.txt
```
If you are using Linux you may need to also install this:
```bash 
sudo apt-get install libportaudio2 portaudio19-dev
```
#### 2. Build the database
```bash 
unzip data/songs.zip -d data/
python3 main/build_database.py
```
#### 3. Run the app
##### 3.1 Running it from the terminal:


You can give the song clip via listening from the device's microphone, in which case you run the command with flag `--mic SECONDS`

Example:
```bash
python3 recognize.py --mic 8
```

Or you can specify the audio file directly, in which xase you run the command with flag `--file FILE`


Example:
```bash
python3 recognize.py --file "data/songs/LZYBY - Baby Bird.mp3" --start 30 --duration 8
```

##### 3.2 Running it with UI:
```bash
python3 ui/main_window.py
```

## Authors:
Lana Matić 

Anja Milutinović
