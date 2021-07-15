# Deep Text Complexity Metric

Classifying the complexity of German texts on a 7-step Likert Scale using the german BERT model [gbert](https://huggingface.co/deepset/gbert-base). 

---

## Downloading the Model

Executing 

```
$ python3 get_model.py
```

will download the model and unpack its contents into the `model` folder.



## Expected Dataset Format

This implementation expects a certain dataset format. The dataset has to be a `.csv` file and needs to have the following two columns:
`sent_id` and `sentence`.



## Using the Model

After downloading the model you can use it by executing 

```
$ python eval.py  --input dataset.csv
```

 in your terminal, where `dataset.csv` is your dataset file.

## License

MIT License

Copyright 2021 Max Reinhard & Faraz Maschhur

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
