# 4.8
- the competition can be work through different types of models.
    - neural networks (deberta-v3)
    - gradient boosting (LGBM) (it might needs thousands of features to keep up with NN)
    - Tdidf
- optimize function can be square error or quadratic weighted kappa
- start from NN then try GB later?
    - like other competitions, start with relatively small model first for higher pace of experimenting.

- unicode inside essays may does not need any preprocessing, the model itself will take care of them.

# 4.9
- Writing up baseline training pipeline
- task for tomorrow
    - look up quadratic weighted kappa
    - finish up baseline pipeline

# 4.10
- begin with simple squared error?
    - try qwk(quardratic weighted kappa) as loss function later


# 4.11
- task for tommorrow fix CUDA bug

# 4.18
- seems like the bug is due to consistency between layers and inputs
    - bug fixed: cross entropy need [0, cls-1], so before calculating loss the label tensor needs to be minus - 1

- ran train with base model without any technique around qwk 0.59

# 4.20
- still can't find the reason why the pipeline can't reproduce result like others
    - found that random_state seed is not stated, but it is not fixed
    - model is not the problem
    - perhaps not data misalighnment
    - maybe something in train loop 
    - found that collator was fotten in valid loop (This might fix the issue)
        - meaning that validation was incorrectly calculated

- tasks for tmr
    - modify train nb to responding to upcoming inference nb (DONE)
    - write inference nb (DONE)

# 4.21
- making registerable inferencing class should be good for easing model organization (DONE)
    - this should be modidify later to be able to handle different kind like GBDT
- haven't decided yet about which to go with, regression or classification
    - reportedlyi classification has shown better result, although regression should be easier when ensembling models 

- things I can do later
    - clean data like special utf-8 symbol can help?
    - multiple drop
    - try larger model
    - different pooling layer
    - treated as regression problems
    - train with GDBT
    - read past competition


# 4.22
- notes from previous competition
    - max len 512, 768, or to max length
    - use different max_len in train and infer 
    - poolings
        - mean
        - concat
        - weightedlayer
        - gem
        - lstm

    - AWP (for later training epoch)
    - differential LR
    - pseudo labels (reportedly not worked in the compettition)(haven't tried yet)
    - max_norm 10
    - deal with special token

- in best sing model discussion, most of the competitor seemed to have gap between CV and LB around 0.03 and some with 0.05 difference

>- **exp001**: baseline
>    - 4 folds CV: 0.8139, LB: 0.769 
>    - fold0 CV: 0.8087, LB: 0.771

>- **exp002**: try with wiped-out unicode full_text
>    - based: exp001
>    - [RESULT] fold0 CV: 0.8096, LB: 0.779 (REVISED)
>    - [POSITIVE]

>- **exp003**: change grad_max_norm from 1000 to 10 
>    - based: exp002
>    - [RESULT] fold0 CV: 0.8118, LB: 0.780 (REVISED)
>    - [POSITIVE]
 
>- **exp004**: extends token max_len from 512 to 768
>    - based: exp003
>    - max_len 512 takes around 22 mins(train loop base model), 768 took about 35 mins
>    - [RESULT] fold0 CV: 0.8265 , LB: 0.794 (REVISED)
>    - [POSITIVE]
 
>- **exp005**: try concat pooling layer (pooling last 4 layers)
>    - based: exp004
>    - [RESULT] fold0 CV: 0.8275, LB: 0.798 (REVISED)
>    - wait for exp004 result, couldn't determine if this pooling layer good or not
>    - [POSITIVE]

- try evaluation with 512 vs 768
    - check if there are any impacts


# 4.23
- rechecking if CV is aligning with LB (with only one fold)
- try exp004 submit again tommorrow (don't forget run in gpu)

>- **exp007**: weighted pooling layer (12 layers)
>   - based: exp004
>   - [RESULT] fold0 CV: 0.8173, LB: 0.787 (REVISED)

>- **exp007**: lstm pooling layer (hidden_dim 768)
>    - based: exp004
>    - [RESULT] fold0 CV: 0.8215, LB: 0.795 (REVISED)

- check out LGBM notebook tmr

# 4.24

- Need to check \n to |?
    - is \n\n to | better?
>    - **exp008**: check the early result in the first epoch base off exp001
>        - based: exp001
>        - comparing with exp002
>        - [RESULT] fold0 CV: 0.8107, LB: 0.777 (REVISED)

>- **exp009**: changed classification to regression
>    - it make sense more because it is ordinal numbers (its order is meaningful)
>    - based: exp002
>    - [RESULT] fold0 cv: 0.8132, LB: 0.777 (REVISED)

>- **exp010**: 
>    - base: 009
>    - max_len 512 to 768, grad_clip 1000 to 10
>    - trying to replicate exp001 to exp004 result but in regression (mse)
>    - [RESULT] fold0 CV: 0.8322 , LB: 0.796 (REVISED)

# 4.25

- GPU quota is not much left may be I should continue doing experiment then upsscale next week
    - haven't decided yet it will be large model or base model

- maybe ensemble CLS and REG at the end?
- try BCE, turn labels to 0 to 5 to range of 0 to 1

>- **exp011**: change MSE to Binary Cross-entropy
>    - based: exp002
>    - as it proved that max_len 768 and grad_clip 10 improved in previous exps, this exp will start with this
>    - scaled labels 0-5 to 0-1 then scaled back went scoring
>    - compare with exp010
>    - [RESULT] fold2 CV: 0.8265, LB: 0.791 (REVISED)

- Current good candicates
    - exp004 for cls
        - fairly good in both cv and lb
    - exp010 for reg(ce)
        - very good on cv side not so much on lb
    - exp011 for reg(bce)
        - mediocre on both side, but it was reportedly good in the past competitions
    - all three exps got the same setting except their loss function

- Most of experiments based on only 1 fold, which means they are argruably reliable
- From observing exp001_4f to 001_f0, 

- forgot to add preprocess function in inference notebook
    - might need resub all past exps
    - resubbed
        - [x] exp011_f0_2
        - [x] exp010_f0_2
        - [x] exp009_f0_2
        - [x] exp008_f0_3
        - [x] exp007_f0_2
        - [x] exp006_f0_2
        - [x] exp005_f0_2
        - [x] exp004_f0_2
        - [x] exp003_f0_2
        - [x] exp002_f0_2
    - [x] update .csv and redraw corr graph

- tasks for tmr read tf-idf articles and study LGBM notebook

# 4.26

- found "<blank>" in the essay(2239), not sure how to deal with it
    - there were "<" included in the essay but it was writer's will to include it
    - 9849
    - NEED FARTHER EXPLORING

- Have feeling that Tf-idf+LGBM might overfit
    - deberta-v3 + (Tf-idf + LGBM) + Mistral (or something from LLM)?

>- **exp012**: use concat pooling layer with regression (mse)
>    - based: exp005, exp010 
>    - [RESULT] fold0 CV: 0.8316, LB: 0.791

- task for tmr, finish LGBM notebook

# 4.27
- calculate and ensemble a few nn model first
    - store them in model_dict
- made a prediction
- add prediction as a feeatures to lgbm
- after then, ensemble metamodel and nn model

- Needs to gather more idea for more experiments

- Write OOF qwk in train notebook DONE

- tasks for tmr read Expanded QWK blog and the notebook

# 4.28

>- **exp013** increase max_len from 768 to 1024
>    - add oof_cv.csv
>    - based: exp010
>    - [RESULT] fold0 CV: 0.8324, LB: 0.793


- maybe I should also change from base model to small for fast iteration of experimenting?

>- **exp014**: change batch size from 16 to 8
>    - based: exp010
>    - [RESULT] fold0 CV: 0.8294, LB: 0.800

- In train lgbm notebook, deberta-v3 features will be using oof.csv
    - but in infer notebook will use predict from the test.csv
    - all these features columns needs to be the same name 

- there is extra data, add to train later

>- **exp015**: try small for the first time
>    - based: exp014
>    - [RESULT] fold0 CV: 0.8219, LB: 0.794
>    - Need to see how 4 fold goes, then I might can assume the pattern around cv and lb relation
>    - After that, I can try stacking LGBM
>    - [RESULT] 4fold oof CV: 0.8284, LB: 0.798
>    - [RESULT] 4fold+stacking CV: 0.8329, LB: 0.773
>        - something wrong with how stacking implemented


- task for tmr calculate CV and check LB, decide what to do next

# 4.29

- need to revise eval notebook as oof_df.csv is get more consistant (aligning with submitted)

- what to do next
    - integrate persuade 2.0 into training
    - train LLM (mistral 7b)

>- **exp016**: add non-overlapped data from persuade2.0
>    - this exp mixed them into train.csv
>    - based: exp015
>    - [RESULT] fold0_1 CV: 0.8616, LB: 0.792 LEAKED
>        - removed 1 duplicated full_text from persaude2.0
>    - [RESULT] fold0_2 CV: 0.8599, LB: 0.800 LEAKED2
>        - removed 2 similar full_text from persuade2.0
>    - [RESULT] fold0_3 CV: 0.8605, LB: 0.791
>        - removed 1 duplicated essay_ids as it can could bug in lgbm in the future
>    - LB score diff may cause by diffences on how data how had been splited
>        - because each time data is removed, fold is also slightly shifted

    
- Assumption, because multiple topics doesn't exist in the original dataset, public LB may doesn't incease because it is also doesn't share the common topics.
- need to through data again

- maybe I should pre-train on extra data for 1 epoch then fully train on default one?

- tasks for tomorrow:
    - check exp015 stacking if it is correctly implemented
        - maybe fixed? not yet
        - check throuh past competition how stacking was done
        - **If this can't be fixed in the mean time, fixed it later**
        - maybe it need more models to be able to perform
        
    - [x] check data leakage in exp016
        - there was 1 exactly full_text with different essay_id
        - get rid of it fixed

- re-calculate a and b score improved 

- tasks for tomorrow:
    - submit third exp016, see if it is fixed the data leaked
        - found duplicated essay_id but it should not be a problem since the content is different
    - check out focal loss and smoothl1

# 5.1

- by just mixed persuade 2.0 to lgbm with pure feats improve LB score from 0.802 (CV: 0.8099) to 0.810 (CV: 0.8492)

>- **exp017**: using only persuade 2.0 data to train
>    - based: exp015, exp016
>    - [RESULT] fold0 CV: 0.8722, LB: 0.766
>    - pretty sure that there are still data leakage 


- found 2 more duplicates full_text
- found a lot similar text
    - a lot of them are using the same exact hook

>- **exp018**: using only persuade 2.0 but removed duplicated
>    - based: exp017
>    - [RESULT] fold0 CV: 0.8743, LB: 0.758
>    - may be I have to remove the text that use the same hook
>    - still, fold distribution is different, yet this dataset may not included in the public lb

 - What should I do with the essay that using the same hoook phrase?
    - what experiment I can do?
        - remove all of them except one and mix them into the default one
            - using clean persuade won't help because it may or may not exist in the public lb

>- **exp019**: mixing cleaned removed the-same-hook essay
>   - based: exp016
>   - [RESULT] fold0 CV: 0.8642, LB: 0.794
>   - [RESULT] 4fold oof CV: , LB: 0.797

- tasks:
    - get colab pro and try torch.compile
    - submit exp019_4f
    - decide what to do next

# 5.2

- due to breaking down ConfusionMatrix and F1 score
    - The overall performance is greatly increase, but if filtered and only focused on the default data predictions
    - Found that it slightly hurt how predictions is made on the default one
    - This explains why CV is so high because it did very good overall but not on LB because it decent the default one
    - the improved parts were not taken into account since it was not existed in public LB

>- **exp020**: pretrained on persuade for 1 epoch then train on default one for 3 epoch
>   - based: exp015, exp016
>   - if this exp goes well, might have to revise some part how to stack to model
>       - one idea is to inference all mixed data to apply them later on stacking model
>   - [RESULT] fold0 CV: 0.8224, LB: 0.789
>   - f1 score is slightly higher when compare with against exp015 (0.66 vs 0.67)

>- **exp021**: try to replicate exp015
>    - based: exp015
>    - the only diff is removed 3 similar text
>    - [RESULT] fold0 CV: 0.8269, LB: 0.793
>    - [RESULT] 4fold CV: 0.8255, LB: 0.798
>    - nothing reaaly change
>    - check the 3 similar text again
>        - checked they should be delete, this may explains the slight deceased of CV

>- **exp022**: pretrained with cleaned persuade2.0 then train for 3 epoch
>    - based: exp021
>    - directly comparison with exp021
>    - [RESULT] 4fold CV: 0.8241, LB: 0.796

- try max norm to from 10 to 1.0
- try learn rate 3e-5 to 1e-5
- try 1024 max_len instead of 768

>- **exp023**: try learning rate from 2e-5 and 3e-5 to 1e-5 and 2e-5
>    - based: exp015
>    - [RESULT] fold0 CV: 0.8242, LB: 0.792


>- **exp024**: try max_norm 10 t 1.0
>    - based: exp023, exp015
>    - [RESULT] fold0 CV: 0.8228, LB: 0.794


- tasks for tmr
    - sub exp023, exp024 DONE
    - write oof cv in train notebook DONE

# 5.3

- removed exp016-19 from csv
 4.29,016_f0_1,0.8616,0.792,0.0696,small,1
> 4.29,016_f0_2,0.8599,0.800,0.0599,small,1
> 4.29,016_f0_3,0.8605,0.791,0.0695,small,1
> 5.1,019_f0,0.8642,0.794,0.0704,small,1
> 5.1,017_f0,0.8722,0.766,0.1062,small,1
> 5.1,018_f0,0.8743,0.758,0.1163,small,1

- mix data and downsample imbalance data

>- **exp025**: downsampling by all balanced to around 800 samples of each class
>   - based: exp015
>   - a bit lost here cause fold distribution is diff
>   - [RESULT] fold0 CV: 0.9308, LB: 0.780

- use 1024 max_len as default?
- try stratified_group_kfold

>- **exp026**: use stratified group kfold with data from persuade and train combined but excluded kaggle-only data
>    - based: none (compare with past exps)
>    - used 1024 max_len
>    - [RESULT] fold0 CV: 0.8733, LB: 0.773

>- **exp027**: use stratified group kfold, all train with prompt(kaggle-only excluded) and scored5-6 from persuade mitigate imbalanced
>    - v2
>    - based: exp026
>    - used 1024 max_len
>    - [RESULT] fold0 CV: 0.772, LB: 0.778

- tasks for tmr
    - submit stack+exp015+exp022
    - submit exp025f0, exp026f0,exp027f0
    - submit

# 5.4

- needs to check \n\n again?

- Needs new baseline as a lot things changed and a lost, it became harder and harder to evaluate each exp
    - assign this task to exp0E
    - train fold0 with removed some similar text
    - 1024 maxlen, 2e-5 and 3e-5 learning rate, mean pooling, small model, cosine scheduler with warmup ratio 0.2
    - max_norm 1.0 <- test later
    >- **exp0E**
    >   - [RESULT] fold0 CV: 0.8296, LB: 0.791

- with groupkfold
    1. create data with prompt_name excluded the one that doesn't have one from both train_df and persuade2.0
    1. predict prompt name that doesn't have one
    1. integrate prompt_name to excluded train_df
    1. use this integrated train_df to train with GroupKFold or StratifiedGroupKFold

    - this idea can apply in lgbm model too?
- after add more models lgbm upward trends can be seen
- early ensemble strategy
    - predict multiple models with variations like diff-pool, diff-seed, dff-architects, optimize them with optuna
    - predict multiple models, treats them as meta-features in lgbm
    - blending both previous result together


>- **exp028**: use train dataset with prompt predicted from pormpt_name prediction
>    - based: exp0E
>    - [RESULT] fold0 CV: 0.7352, LB: 0.743

>- **exp029**: use max_norm 10 instead of 1.0
>    - based: exp028
>    - [RESULT] fold0 CV: 0.7217 LB: 0.729

- exp028 and exp029 is using StratifiedGroupKFold

>- **exp030**: use groupkfold instead (max_norm 1.0)(lr 2e-5 and 3e-5) 
>    - based: ex028
>    - [RESULT] fold0 CV: 0.7755, LB: 0.795

>- **exp031**: same as exp030 but with max_norm 10 (lr 2e-5 and 3e-5)
>    - based: exp030
>    - [RESULT] fold0 CV: 0.7727, LB: 0.789

>- **exp032**: same as exp031 but with (max_norm 10) lr 1e-5 and 2e-5 
>    - based: exp031
>    - [RESULT] fold0 CV: 0.7693, LB: 0.788

>- **exp033**: lr 1e-5 and 2e-5 and max_norm 1.0
>    - based: exp032
>    - [RESULT] fold0 CV: 0.7696, LB: 0.793

>- **exp034**: use groupkfold instead (max_norm 1.0)(lr 2e-5 and 3e-5) 
>    - based: exp030
>    - in exp030 trained 3 epoch, it become overfit in 3rd epoch
>    - changed to 2 epoch
>    - [RESULT] fold0 CV: 0.7377, LB: 0.779
>    - does not help

- StratifiedGroupKFold is not equally distributed because it is trying to perserved percentage of the samples
    - not sure if it gets a good result

- check later if infer notebook check inference cls model
    - In the moement, it can't create raw output
    - It can now, and infer notebook now works with cls
- submit from exp030 first then to exp034, then exp028, exp029

# 5.5

>- **exp035**: use exp030 as base then change warmup ratio to 0
>    - based: exp030
>    - [RESULT] fold0 CV: 0.7674, LB: 0.795

>- **exp036**: use exp030 as base then change warmup ratio to 0.1
>    - based: exp030
>    - [RESULT] fold0 CV: 0.7703, LB: 0.796

>- **exp037**: expanded to 4 epoch
>    - based: exp030
>    - [RESULT] fold0 CV: 0.7842, LB:
>    - it is always overfit after epoch 2, but using 2 epoch also does not work
>        - meaning that when goes under certain LR it will overfitting
>        - set as epoch 4 then break at epoch 2?
>    - [RESULT] 4fold oof CV: 0.7160, LB: 0.773


- use FB3 & FB1 as pb?
    - add  'cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions' features into LGBM
- Should I try longer epochs from 3 to 4?
- stratified prompt_name?

- What if add prompt_name to the start of the full_text?
    - as special token?

>- **exp038**: add prompt_name at the start of full_text goes by prompt_name||full_text
>    - based: exp030
>    - [RESULT] fold0 CV: 0.7637, LB: 0.789
>    - did not help

- tasks for tmr
     Train full 4 fold with exp030?

# 5.6

- Avoid overfitting by adjust LR?
    - try warmup restart

>- **exp039**: try warmup restart  2 cycles
>    - change | to added token
>    - based: exp030
>    - [RESULT] fold0 CV: 0.7798, LB:
>    - did not help with overfitting

>- **exp040**: try 1e-5 on both
>    - based: exp030
>    - [RESULT] fold0 CV: 0.7616, LB:

# 5.7

- GroupKFold always overfitted at the last epoch
    - StratifiedGroupKFold does not

- try submit exp028, exp029
    - decide what to do later

- StratifiedGroupKFold tends not to overfitting at the last epoch
    - because all score labels are more balanced?
    - from exp037_4f, all folds score are very different from each others


>- **exp041**: try grouped LLRD
>    - based: exp028
>    - [RESULT] fold0 CV: 0.7285, LB:

>- **exp042**: added new sample with similar distribution
>    - based: exp028
>    - [RESULT] fold0 CV: 0.7658, LB: 0.765


- need to check what's going on on the 4th fold
    - because GroupKFold does not take imbalance dataset into account, it led to overfitting

- thinking about utilizing persuade2.0
    - if it is carefully selected, it could be useful
    - in exp027, added score 5, 6 help a lot 


- lets say public lb test dataset does not contain any topics in persuade2.0
    - only thing to consider is imbalance
    - but using StratifiedKFold does not have good correlation between CV and LB
    - GroupKFold have good LB but not so much on LB, and it always get overfitting
    - StratifiedGroupKFold bring good correlation between CV and LB but it is extremely hard to get higher score
        - rebalancing MIGHT help.
        - eventhough, it is importance to split using group but group does not necessary relfect on bot CV and LB

# 5.8

> @chaudharypriyanshu
> Kaggle, persuade datapoints = x
> Kaggle samples = y
> We do normal 5 Kfold split on y (call them y1,y2,y3,y4,y5):
> The folds will be :
> fold 1: train = x + [y2,y3,y4,y5] , validate on y1
> fold 2: train = x + [y1,y3,y4,y5] , validate on y2
> fold 3: train = x + [y1,y2,y4,y5] , validate on y3

- spliting
    - on kaggle samples split using StratifiedGropuKFold
    - on NOL persuade split using StratifiedFold on score
        - on each fold still has similar score distribution
    
    - testing this idea
        - **exp043**:
            - [RESULT] fold0 CV: 0.8213, LB: 0.789
            - there were overlapping data this CV can be invalid because there might be some leakage
        - **exp044**: fixed exp043
            - [RESULT] fold0 CV: 0.7947, LB: 0.790
            - [RESULT] fold1 CV: 0.7621, LB: 0.797
            - [RESULT] fold2 CV: 0.7534, LB: 0.785
            - [RESULT] fold3 CV: 0.5923, LB: 0.774
            - [RESULT] oof 4fold CV: 0.7461, LB: 
        - **exp045**: this time use both StratifiedKFold on two dataset
            - [RESULT] fold0 CV: 0.8200, LB: 0.796

- try 7 fold on groupkfold

# 5.9

> i meant you should use full samples of persuade 2.0 present in training data and keep the fold splitting to the ones that are there in kaggle data.
> So the corrected one should be
>
> P=P1+P2+P3+P4
> 4 folds of default train(removing persuade samples) dataset [K1, K2, K3, K4]
>
> trains P + [K2, K3, K4] validates K1
> trains P + [K1, K3, K4] validates K2
> trains P + [K1, K2, K4] validates K3
> trains P + [K1, K2, K3] validates K4
>
> where P is full pursuade corpus.

- **exp046**: setup new CV scheme
    - follow the above method
    - score is from 1st epoch
    - P in is exp is only persuade the overlapping with default training set
    - [RESULT] fold0 CV: 0.7820, LB: 0.768


- by using kaggle-only data as validation set, cv can be calculated more aligned with test dataset