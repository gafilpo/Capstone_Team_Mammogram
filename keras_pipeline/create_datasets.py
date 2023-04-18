import tensorflow as tf

# Create tensorflow dataset for fast Keras feeding
class create_datasets():

    def __init__(self, dfs, fold, target_var, img_var, exclude_cols=[], double_input=False):

        self.dfs = dfs
        self.fold = fold
        self.target_var = target_var
        self.img_var = img_var
        self.exclude_cols = exclude_cols
        self.double_input = double_input

    def process_path(self, file_path):
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = tf.io.decode_png(img, channels=1)
        # Standardize - done when reading now
        img = tf.cast(img, dtype=tf.float32) * tf.cast(1./255., dtype=tf.float32)
        
        return img
    
    def create_ds(self, train_val_test, resizing=False):
        # Create tensorflow datasets

        # Images
        img_ds = tf.data.Dataset.from_tensor_slices(self.dfs[self.fold][train_val_test][self.img_var])
        img_ds = img_ds.map(self.process_path, num_parallel_calls=tf.data.AUTOTUNE)
        if resizing:
            img_ds = img_ds.map(lambda x: tf.image.resize(x, resizing), num_parallel_calls=tf.data.AUTOTUNE)
        
        # Labels
        label_ds = tf.data.Dataset.from_tensor_slices(self.dfs[self.fold][train_val_test][self.target_var])

        # If we want to also feed data from the df, we create a new ds
        if self.double_input:
            feed_df = self.dfs[self.fold][train_val_test].drop(self.exclude_cols, axis=1).to_numpy()
            data_ds = tf.data.Dataset.from_tensor_slices(feed_df.astype('int'))

            # Zip the two input datasets together, train,
            input_ds = tf.data.Dataset.zip((img_ds, data_ds))

        else:
            input_ds = img_ds

        # Zip input and labels dataset

        input_ds = tf.data.Dataset.zip((input_ds, label_ds))

        return input_ds

    def create_ds_autoencoder(self, train_val, resizing=False):
        # Create tensorflow datasets

        # Images
        img_ds = tf.data.Dataset.from_tensor_slices(self.dfs[self.fold][train_val][self.img_var])
        img_ds = img_ds.map(self.process_path, num_parallel_calls=tf.data.AUTOTUNE)
        if resizing:
            img_ds = img_ds.map(lambda x: tf.image.resize(x, resizing), num_parallel_calls=tf.data.AUTOTUNE)
            
        # Zip input images as X and y dataset

        input_ds = tf.data.Dataset.zip(((img_ds), (img_ds)))
        
        return input_ds
    
    def configure_for_performance(self, ds, batch_size=10, train=True, disk_cache=False):
        if disk_cache:
            ds = ds.cache('cache')
        else:
            ds = ds.cache()

        # Only shuffle the training dataset
        if train:
            ds = ds.shuffle(buffer_size=batch_size)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return ds