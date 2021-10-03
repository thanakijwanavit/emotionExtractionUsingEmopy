# Emotion Extraction Using A Wrapped Emopy Convolutional Neural Network
emotion extraction using emopy algorithm

## Overall brief diagram of the emotion extraction system and a binary classifier attached
<img width="1279" alt="image" src="https://user-images.githubusercontent.com/29235787/131147577-9bf5be98-f1ac-4fe8-ab6b-81a42e7ee50f.png">

1. raw image data in RGB color space and dimension 48x48
2-4. Emopy [pre-trained Convolutional network](https://github.com/thoughtworksarts/EmoPy/tree/master/EmoPy/models) with emotion output 
```python
  ['anger', 'fear', 'surprise', 'calm'],
  ['happiness', 'disgust', 'surprise'],
  ['anger', 'fear', 'surprise'],
  ['anger', 'fear', 'calm'],
  ['anger', 'happiness', 'calm'],
  ['anger', 'fear', 'disgust'],
  ['calm', 'disgust', 'surprise'],
  ['sadness', 'disgust', 'surprise'],
  ['anger', 'happiness']
```
5. Output from [Emotion Accumulator](#Emotion Accumulator)
6. Probablility output from the Classifier model to forecast whether the person is an Apple employee or not.


## Training algorithm
used to train the classifier model using the training dataset. This is turned off during testing run.


## Emotion Accumulator
take an average output of each occurance of the emotion cumulatively. The total score is then normalized so that the sum is 100
```python
def combineResult(result_dict_list:List[dict]):
  final_result = {}
  for result_dict in result_dict_list:
    for key in result_dict.keys():
      if key in final_result.keys():
        final_result[key] = (final_result[key] + result_dict[key])/2
      else:
        final_result[key] = result_dict[key]
  return final_result
```

## paper
[slide](https://tenxor.sh/jkWW)
[paper]()
