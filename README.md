
# Character based RNN
This is a repository for a character based RNN, used for classification of short bits of text. The basis for this was taken from a Pytorch tutorial ['NLP From Scratch: Classifying Names with a Character-Level RNN'](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html). This type of network is useful for NLP where the snippets of text are short (1-2 words), so you can't really do word embeddings. Or for example where word embeddings would be too rigid at prediction time (names that don't appear in the training set for example).

I have built upon this by:
- Creating pytorch dataset and dataloaders that are generalisable to words -> classification task.
- Dataloader handles batching with pad_sequences (tutorial does single sample batches), and also provides validation split (missing from tutorial)
- Adds character embeddings before feeding to the RNN
- Uses pytorchs LSTM module, rather than constructing our own RNN class
- Learner class to train the model, save and reload weights.

## Example
Here is an example showing the result of a trained model (name->nationality). The model obtains an overall accuracy of 85% (across 252 nationalites and principalities - Some more accurate than others). The validation set used consisted of full names that do not appear in the training set, although first and last names may appear in the training set but never in the same combination as those in the validation set. 

**No model weights or data are included in this repository though.**


```python
from dataset import WordDataset
from dataloader import WordDataLoader
from model import CharRNN
from learn import Learner
import torch

# put on gpu if available
dev = "cuda" if torch.cuda.is_available() else "cpu"

# dataset - get words and labels by index of column name of a csv
dataset = WordDataset(path_to_csv='names_nationality.csv', word_col=3, label_col=4)
dl = WordDataLoader(ds=dataset, device=dev, batch_size=(500, 6000), validation_split=0.05)

# embedding size of 40 worked best for the given data
# got to add 1 to vocab size as 0 is used in padding batches to same length
# can try regularising with classification dropout + rnn dropout if needed
model = CharRNN(vocab_size=len(dataset.all_chars) + 1, embed_size=40, 
                 hidden=250, layers=2, output_size=len(dataset.all_labels), fc_size=500,
                clas_drop=[0, 0], rnn_drop=0)
model.to(dev)

# create a learner
learn = Learner(model=model, dataloader=dl, all_chars=dataset.all_chars, all_labels=dataset.all_labels)

# can use learning rate finder to find a good starting learning rate
learn = Learner(model=model, dataloader=dl, all_chars=dataset.all_chars, all_labels=dataset.all_labels)
# adjust weight decay accordingly, if you need to regularise
log, losses = learn.find_lr(start_lr=1e-5, end_lr=1e1, wd=0)
# fit
learn.fit_one_cycle(10, base_lr=2e-4, max_lr=2e-3, wd=0)
```


```python
# load in the weights
learn.load(f='40_emb_250_hid_2_layer_500_fc.pth')
```
## Some Predictions

```python
# hey that's me
learn.predict('jack curtis')
```

    
     > jack curtis
    (0.66) united kingdom
    (0.14) ireland
    (0.06) united states of america



```python
# Current President of Afghanistan
learn.predict('Ashraf Ghani Ahmadzai')
```

    
     > Ashraf Ghani Ahmadzai
    (0.95) afghanistan
    (0.01) iran
    (0.01) pakistan



```python
# Current Chairman of the Council of Ministers of Bosnia and Herzegovina
learn.predict('Denis Zvizdic')
```

    
     > Denis Zvizdic
    (0.27) croatia
    (0.22) serbia
    (0.22) slovenia



```python
# Currently serving as the 9th Prime Minister of Cameroon
learn.predict('Joseph Dion Ngute')
```

    
     > Joseph Dion Ngute
    (0.32) kenya
    (0.18) cameroon
    (0.09) philippines



```python
# Prime minister of Yemen
learn.predict('Maeen Abdulmalik Saeed')
```

    
     > Maeen Abdulmalik Saeed
    (0.63) iraq
    (0.19) yemen
    (0.06) united arab emirates



```python
# Current Chancellor of Germany
learn.predict('Angela Dorothea Merkel')
```

    
     > Angela Dorothea Merkel
    (0.86) germany
    (0.09) switzerland
    (0.01) netherlands



```python
# Birth name of Marie Curie
learn.predict('Maria Salomea Sklodowska')
```

    
     > Maria Salomea Sklodowska
    (0.82) poland
    (0.08) germany
    (0.04) sweden



```python
# Current Primeminister of Japan
learn.predict('Shinzo Abe')
```

    
     > Shinzo Abe
    (0.98) japan
    (0.00) philippines
    (0.00) brazil

