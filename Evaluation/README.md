## Training Curve
To plot the training curve: `./training_plot.py`  

KTH dataset = 25 people* 4 scenarios* 6 actions = 600 video clips. Each video clip includes about 400 frames.  
* During the first try of training, sets allocation is:  
  TRAIN_PEOPLE_ID = [11, 12, 13, 14, 15, 16, 17, 18]  
  DEV_PEOPLE_ID = [19, 20, 21, 23, 24, 25, 1, 4]  
  TEST_PEOPLE_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]  
  The model suffers from overfitting. Abandon it.

* During the second try of training, sets allocation is:  
  TEST_PEOPLE_ID = [5, 10, 15, 20, 25]  
  TRAIN_PEOPLE_ID = the remaining 20 sets  
  > Finally, I choose the model trained for 46 epoches to do the following evaluation.

## Evaluation
To evaluate: `python3 ./eval_cnn_block_frame_flow.py`
1. During this evaluation, mean image is calculated from training dataset. However, results show that if the mean image is calculated from test set, the accuracy increase about 1-2%. This means updating the mean image before thr inference will be helpful.  

1. Accuracy test on the model trained for 46 epochs:  
CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]  
1 snippet = 5 blocks = 5\*15 frames. 1 snippet generates 1 prediction.
    ```
     snippet_predection  video_pre/labels
    [5. 0. 0. 0. 0. 0.]       0     0
    [5. 0. 0. 0. 0. 0.]       0     0
    [5. 0. 0. 0. 0. 0.]       0     0
    [5. 0. 0. 0. 0. 0.]       0     0
    [5. 0. 0. 0. 0. 0.]       0     0
    [5. 0. 0. 0. 0. 0.]       0     0
    [7. 0. 0. 0. 0. 0.]       0     0
    [6. 0. 0. 0. 0. 0.]       0     0
    [5. 0. 0. 0. 0. 0.]       0     0
    [3. 0. 3. 0. 0. 0.]       0     0
    [5. 0. 0. 0. 0. 0.]       0     0
    [3. 0. 0. 0. 0. 0.]       0     0
    [5. 0. 0. 0. 0. 0.]       0     0
    [6. 0. 0. 0. 0. 0.]       0     0
    [6. 0. 0. 0. 0. 0.]       0     0
    [5. 0. 0. 0. 0. 0.]       0     0
    [5. 0. 0. 0. 0. 0.]       0     0
    [6. 0. 0. 0. 0. 0.]       0     0
    [6. 0. 0. 0. 0. 0.]       0     0
    [6. 0. 0. 0. 0. 0.]       0     0
    [0. 5. 0. 0. 0. 0.]       1     1
    Done 20/120 videos
    [2. 3. 0. 0. 0. 0.]       1     1
    [0. 5. 0. 0. 0. 0.]       1     1
    [0. 6. 0. 0. 0. 0.]       1     1
    [0. 3. 0. 0. 0. 0.]       1     1
    [0. 4. 0. 0. 0. 0.]       1     1
    [0. 5. 0. 0. 0. 0.]       1     1
    [0. 4. 0. 0. 0. 0.]       1     1
    [0. 4. 0. 0. 0. 0.]       1     1
    [0. 3. 0. 0. 0. 0.]       1     1
    [0. 4. 0. 0. 0. 0.]       1     1
    [0. 4. 0. 0. 0. 0.]       1     1
    [0. 5. 0. 0. 0. 0.]       1     1
    [0. 5. 0. 0. 0. 0.]       1     1
    [0. 5. 1. 0. 0. 0.]       1     1
    [0. 6. 0. 0. 0. 0.]       1     1
    [1. 4. 0. 0. 0. 0.]       1     1
    [0. 6. 0. 0. 0. 0.]       1     1
    [0. 5. 0. 0. 0. 0.]       1     1
    [0. 6. 0. 0. 0. 0.]       1     1
    [2. 1. 3. 0. 0. 0.]       2     2
    Done 40/120 videos
    [0. 3. 3. 0. 0. 0.]       1     2 ---wrong video predection
    [0. 3. 3. 0. 0. 0.]       1     2 ---wrong video predection
    [1. 0. 4. 0. 0. 0.]       2     2
    [0. 0. 6. 0. 0. 0.]       2     2
    [0. 0. 6. 0. 0. 0.]       2     2
    [0. 0. 6. 0. 0. 0.]       2     2
    [0. 0. 7. 0. 0. 0.]       2     2
    [0. 1. 6. 0. 0. 0.]       2     2
    [0. 1. 8. 0. 0. 0.]       2     2
    [0. 0. 7. 0. 0. 0.]       2     2
    [1. 0. 7. 0. 0. 0.]       2     2
    [0. 0. 7. 0. 0. 0.]       2     2
    [0. 0. 6. 0. 0. 0.]       2     2
    [0. 0. 6. 0. 0. 0.]       2     2
    [0. 0. 6. 0. 0. 0.]       2     2
    [0. 0. 6. 0. 0. 0.]       2     2
    [0. 0. 6. 0. 0. 0.]       2     2
    [0. 0. 6. 0. 0. 0.]       2     2
    [0. 0. 6. 0. 0. 0.]       2     2
    [0. 0. 0. 2. 0. 0.]       3     3
    Done 60/120 videos
    [0. 0. 0. 1. 2. 0.]       4     3 ---wrong video predection
    [0. 0. 0. 2. 0. 0.]       3     3
    [0. 0. 0. 2. 0. 1.]       3     3
    [0. 0. 0. 3. 0. 0.]       3     3
    [0. 0. 0. 4. 0. 2.]       3     3
    [0. 0. 0. 0. 2. 0.]       4     3 ---wrong video predection
    [0. 0. 0. 3. 0. 0.]       3     3
    [0. 0. 0. 2. 0. 1.]       3     3
    [0. 0. 0. 2. 0. 2.]       3     3
    [0. 0. 0. 2. 0. 0.]       3     3
    [0. 0. 0. 3. 0. 1.]       3     3
    [0. 0. 0. 2. 0. 0.]       3     3
    [0. 0. 0. 4. 0. 0.]       3     3
    [0. 0. 0. 0. 2. 0.]       4     3 ---wrong video predection
    [0. 0. 0. 3. 0. 0.]       3     3
    [0. 0. 0. 2. 0. 0.]       3     3
    [0. 0. 0. 3. 0. 1.]       3     3
    [0. 0. 0. 3. 0. 0.]       3     3
    [0. 0. 0. 2. 1. 0.]       3     3
    [0. 0. 0. 0. 1. 0.]       4     4
    Done 80/120 videos
    [0. 0. 0. 0. 2. 0.]       4     4
    [0. 0. 0. 0. 2. 0.]       4     4
    [0. 0. 0. 0. 2. 0.]       4     4
    [0. 0. 0. 0. 1. 0.]       4     4
    [0. 0. 0. 1. 0. 2.]       5     4 ---wrong video predection
    [0. 0. 0. 0. 1. 0.]       4     4
    [0. 0. 0. 0. 2. 0.]       4     4
    [0. 0. 0. 0. 2. 0.]       4     4
    [0. 0. 0. 1. 1. 0.]       3     4 ---wrong video predection
    [0. 0. 0. 0. 2. 0.]       4     4
    [0. 0. 0. 1. 2. 0.]       4     4
    [0. 0. 0. 0. 1. 0.]       4     4
    [0. 0. 0. 1. 2. 0.]       4     4
    [0. 0. 0. 0. 1. 0.]       4     4
    [0. 0. 0. 0. 2. 0.]       4     4
    [0. 0. 0. 0. 2. 0.]       4     4
    [0. 0. 0. 0. 0. 3.]       5     4 ---wrong video predection
    [0. 0. 0. 1. 1. 0.]       3     4 ---wrong video predection
    [0. 0. 0. 0. 2. 0.]       4     4
    [0. 0. 0. 0. 0. 4.]       5     5
    Done 100/120 videos
    [0. 0. 0. 0. 0. 6.]       5     5
    [0. 0. 0. 0. 0. 4.]       5     5
    [0. 0. 0. 0. 0. 5.]       5     5
    [0. 0. 0. 0. 0. 4.]       5     5
    [0. 0. 0. 1. 0. 9.]       5     5
    [0. 0. 0. 1. 0. 3.]       5     5
    [0. 0. 0. 0. 0. 5.]       5     5
    [0. 0. 0. 0. 0. 6.]       5     5
    [0. 0. 0. 0. 0. 7.]       5     5
    [0. 0. 0. 0. 0. 4.]       5     5
    [0. 0. 0. 0. 0. 7.]       5     5
    [0. 0. 0. 0. 0. 5.]       5     5
    [0. 0. 0. 0. 0. 8.]       5     5
    [0. 0. 0. 0. 0. 4.]       5     5
    [0. 0. 0. 0. 0. 5.]       5     5
    [0. 0. 0. 0. 0. 5.]       5     5
    [0. 0. 0. 0. 0. 8.]       5     5
    [0. 0. 0. 0. 0. 5.]       5     5
    [0. 0. 0. 0. 0. 5.]       5     5
    
    Rows: Label, Columns: Prediction
    [[104.   0.   3.   0.   0.   0.]	 ---104/107 0.972
     [  3.  92.   1.   0.   0.   0.]	 ---92/96   0.958
     [  4.   9. 115.   0.   0.   0.]	 ---115/128 0.898
     [  0.   0.   0.  45.   7.   8.]	 ---45/60   0.75
     [  0.   0.   0.   5.  29.   5.]	 ---29/39   0.744
     [  0.   0.   0.   2.   0. 109.]]	---109/111 0.982
    ```
494/541 Accuracy on Snippet: 0.913 
111/120 Accuracy on Videos: 0.925

