# ZAI Pretrained Tokenizers Folder - Where the Magic Words Live!

So, this folder right here has all those, like, pre-made word-stuff-getters (they call 'em tokenizers) for ZAI. They make computers understand words better, I guess.

## How to use 'em (Easy Peasy, kinda)

Just type this Python code in your thingy:

```python
from zai.tokenizer import load_pretrained, list_pretrained

# Wanna see what tokenizers are here?
print(list_pretrained())

# Load one up! I'm loading 'xorzen_65k' here, which sounds like a big one.
tokenizer = load_pretrained('xorzen_65k')
# Then you give it words, and it turns them into numbers!
tokens = tokenizer.encode("Hello world!")
```

## What Tokenizers We Got (Like, My Collection)
- **xorzen_bpe_627h**: This is a tiny one, only 627 words it knows. Good for, like, when you just wanna test small stuff. **Status**: It's trained and ready to go!
- **xorzen_10k**: This one knows 10,000 words. It's supposed to be, like, for everything. **Status**: Not trained yet, and not even registered. Oops!
- **xorzen_opmi_65k**: This is a big one, 65,000 words, trained on some 'openmathinstruct' dataset (whatever that is). It's for thinking words, I think? **Status**: Trained and ready!
- **xorzen_65k**: Also 65,000 words. This is the main one for, like, the old 'xorzen_igris' thing. (I should probably rename this too, huh?)

## Adding Your Own Special Tokenizers (If You're Feeling Brave)

1. Make your own tokenizer using `zai.tokenizer.trainer`.
2. Save it as a `.json` file in THIS folder.
3. Update that `metadata.json` file with your tokenizer's info.
4. Then it'll just, like, magically show up!

Here's some code (looks scary, but it's fine):
```python
from zai.tokenizer import train_tokenizer

# Train your own!
tokenizer = train_tokenizer(
    files=['train.txt'], # Gotta tell it what words to learn from
    vocab_size=32000, # How many words you want it to know
    output_path='pretrained/my_tokenizer.json' # Where to save it
)

# Remember to go open 'metadata.json' and, like, add your tokenizer's stuff there!
```

## How the Files Are Organized (Like, a Map)

```
pretrained/
├── metadata.json              # This file knows all about the tokenizers
├── xorzen_bpe_627h.json          # One of the tokenizer files
├── xorzen_10k.json               # Another tokenizer file
├── xorzen_opmi_65k.json          # You guessed it, another one!
├── xorzen_65k.json               # And another!
└── README.md                  # This very file you're reading!
```

## Just Some Random Thoughts (Notes)

- The tokenizer files are, like, made with some 'HuggingFace tokenizers' stuff.
- Each tokenizer needs its own little entry in `metadata.json`. Don't forget!
- They just show up automatically when you import them. Cool, right?

## My Super Scientific Tests!
So, I tried this `xorzen_627h` tokenizer on some text it never ever saw before. And guess what? It worked! I was, like, totally shocked! Before, I thought you needed a gazillion words in your tokenizer for it to be good, but maybe not? Check out my proof below:

```python
from zai.tokenizer import load_pretrained

tok = load_pretrained('xorzen_bpe_627h')
data = "Hello, World! nice to meet everyone. Tui ki koros bro? khaisos?" # My totally random test sentence

encoded = tok.encode(data)
print(f"Data encoded.\nData: {data}\nEncoded IDs: {encoded}\n") # Showing off the numbers it made
decoded = tok.decode(encoded)
print(f"Decoded: {decoded}") # Turning the numbers back into words! It works!
```
**Output from My Computer**
```log
python tokenizer_test_from_xorzen.py
[2025-12-21T15:23:07.140672] [INFO] [global_logger] Global logger initialized

Data encoded.
Data: Hello, World! nice to meet everyone. Tui ki koros bro? khaisos?

Encoded IDs: [2, 587, 19, 228, 62, 568, 83, 75, 8, 305, 447, 391, 502, 283, 299, 89, 96, 86, 285, 21, 228, 560, 80, 228, 82, 80, 228, 82, 568, 86, 90, 302, 89, 86, 38, 228, 82, 79, 72, 272, 86, 90, 38, 3]

Decoded: Hello, World! nice to meet everyone. Tui ki koros bro? khaisos?

[2025-12-21T15:23:12.385755] [INFO] [tokenizer.loader] Loading pretrained tokenizer: xorzen_bpe_627h from zai\tokenizer\pretrained\xorzen_bpe_627h.json
[2025-12-21T15:23:12.387439] [INFO] [tokenizer.loader] Loaded tokenizer from zai\tokenizer\pretrained\xorzen_bpe_627h.json
```
### That 10k model.
Okay, so after seeing how that small 627h model did its thing on totally new text, my brain, like, exploded! I always thought "bigger vocab size = better tokenizer = better AI." But now? I'm not so sure. So, I'm gonna train a 10k word model using that 'gutenberg_text' data. Wish me luck!

**Training script (still figuring this out)**
```python

```
---
**Test script (also needs work)**
```python

```
**Output (nothing yet, obviously)**
```log

```
