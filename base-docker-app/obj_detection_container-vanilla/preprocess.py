# includes all necessary preprocessing steps for speech recognition
# based on https://www.tensorflow.org/tutorials/audio/simple_audio
# returns: train, test and validation sets + commands and spectrogram_ds
 
import numpy as np
import tensorflow as tf
import os


###### HELPER FUNCTIONS ######
# decode raw WAV into audio tensors
def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)

# create labels for each file
def get_label(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    return parts[-2]

# outputs tuple containing audio + label tensors
def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

# convert to spectograms
def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

###### MAIN PREPROCESSING ######
def preprocess(data_dir):
    # extract the commands (no, yes, down, go, left, up, right, stop)
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']
    print('Commands:', commands)

    #extract the audio clips
    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    print('Number of examples per label:', len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
    print('Example file tensor:', filenames[0])

    # split into train, test and validation sets (80:10:10)
    train_files = filenames[:int(num_samples*0.8)]
    val_files = filenames[int(num_samples*0.8): int(num_samples*0.8 + num_samples*0.1)]
    test_files = filenames[-int(num_samples*0.1):]

    print('Training set size', len(train_files))
    print('Validation set size', len(val_files))
    print('Test set size', len(test_files))

    AUTOTUNE = tf.data.AUTOTUNE


    ###### TRAIN SET ######
    files_ds = tf.data.Dataset.from_tensor_slices(train_files)

    waveform_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE)

    # transform waveform into spectogram + corresponding label
    def get_spectrogram_and_label_id(audio, label):
        spectrogram = get_spectrogram(audio)
        label_id = tf.math.argmax(label == commands)
        return spectrogram, label_id

    # map across the dataset
    spectrogram_ds = waveform_ds.map(
    map_func=get_spectrogram_and_label_id,
    num_parallel_calls=AUTOTUNE)


    ###### VALIDATION + TEST SET ######
    def preprocess_dataset(files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(
            map_func=get_waveform_and_label,
            num_parallel_calls=AUTOTUNE)
        output_ds = output_ds.map(
            map_func=get_spectrogram_and_label_id,
            num_parallel_calls=AUTOTUNE)
        return output_ds

    train_ds = spectrogram_ds
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)

    batch_size = 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    # reduce latency by caching and prefetching
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, test_ds, val_ds, spectrogram_ds, commands
