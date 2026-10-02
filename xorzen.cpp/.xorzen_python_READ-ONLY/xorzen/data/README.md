# ZAI Data Module - It's all about the data!

This is where all the data stuff happens in ZAI. Like, getting your data ready, loading it up, and changing it around. It's kinda important, I guess.

## BinDataset (Super Fast Data Loading, supposedly!)

So, there's this `BinDataset` thing in `zai` that's supposed to make loading data, like, way faster. Instead of using those `.npy` files (which are slow, apparently), it uses these `.bin` files. It's supposed to be much better for huge datasets.

### How to Use It (My Best Attempt at Explaining)

The `BinDataset` works with data that comes from the `XORZENXDataLoader`. So, first, you gotta make your data into those `.bin` files.

**1. Making those `.bin` files:**

When you're doing your data prep script and using the `XORZENXDataLoader`, just tell it to use `'bin'` instead of `'npy'`. Like this:

```python
from zai.data.loader import zaiDataLoader
from zai.tokenizer.adapter import zaiTokenizerAdapter

tokenizer_adapter = XORZENXTokenizerAdapter("path/to/your/tokenizer.json") # This thing helps with words
data_loader = XORZENXDataLoader(tokenizer_adapter=tokenizer_adapter) # This loads the data, I think?

# ... you gotta have some item_iterator thing here ...

data_loader.tokenize_and_shard(
    item_iterator=item_iterator, # Your data goes here
    output_dir="path/to/your/tokenized_data", # Where the new files go
    shard_format='bin',  # <--- THIS IS THE MAGIC BIT! Make it 'bin', not 'npy'!
    dtype='uint16' # What kind of numbers to use. No idea why, but it's there.
)
```

**2. Using `BinDataset` with the training script (the generic one):**

The training script, `train.py`, now uses these YAML files for settings. To use the `BinDataset` here:

*   **You gotta tell it where your model and data are in a YAML file** (like, `configs/igris_1m.yaml`, but this should probably be `aether_1m.yaml` now, right?):

    ```yaml
    model:
      model_name: xorzen_igris_1m # <-- This should probably be 'xorzen_aether_1m' now, right?
      # ... other settings for the model ...

    data:
      dataset_path: "/path/to/your/raw/dataset" # The original data
      dataset_format: 'text' # It's text, duh

    training:
      tokenizer_path: "xorzen_igris_1m_run/tokenizer/tokenizer.json" # More path stuff
      dataset_path: "xorzen_igris_1m_run/tokenized_data" # <--- THIS IS YOUR NEW .bin DATA!
      batch_size: 4 # How many pieces of data to process at once. Bigger number, faster training, maybe?
      # ... more training settings ...
    ```

*   **Then just run the `train.py` script:**

    ```bash
    python train.py
    ```
    It's supposed to automatically find your config file. Hope it does!

So, yeah, using this `BinDataset` thing is supposed to make your data load faster, which means training goes quicker. And the new training script with the config files is, like, super flexible.
