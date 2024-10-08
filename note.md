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


>- **exp024**: try max_norm 10 to 1.0
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
   
 >   - testing this idea
 >       - **exp043**:
 >           - [RESULT] fold0 CV: 0.8213, LB: 0.789
 >           - there were overlapping data this CV can be invalid because there might be some leakage
 >       - **exp044**: fixed exp043
 >           - [RESULT] fold0 CV: 0.7947, LB: 0.790
 >           - [RESULT] fold1 CV: 0.7621, LB: 0.797
 >           - [RESULT] fold2 CV: 0.7534, LB: 0.785
 >           - [RESULT] fold3 CV: 0.5923, LB: 0.774
 >           - [RESULT] oof 4fold CV: 0.7461, LB: 
 >       - **exp045**: this time use both StratifiedKFold on two dataset
 >           - [RESULT] fold0 CV: 0.8200, LB: 0.796

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
> where P is full persuade corpus.

>- **exp046**: setup new CV scheme
>    - follow the above method
>    - score is from 1st epoch
>    - P in this exp is only persuade that overlapping with default training set
>    - [RESULT] fold0 CV: 0.7820, LB: 0.768
>    - from 1st epoch

>- **exp047**: 
>    - based: exp046
>    - still using Kaggle-only data as validation dataset
>    - this exp added extra data from persuade 2.0 that does not appear in default dataset split
>    - also, how training set mix is a bit unusual
>        - training set is split then split persuade dataset was added
>    - [RESULT] fold0 CV: 0.7808, LB: 0.778
>    - from 2nd epoch

>- **exp048**:
>    - based: exp047
>    - same setup this time with base model
>    - [RESULT] fold0 CV: 0.7765, LB: 0.792
>    - from the last epoch

>- **exp049**: change decoder_lr to 2e-5
>    - based: exp047
>    - small model
>    - [RESULT] fold0 CV: 0.7774, LB: 0.775
>    - does not help

>- **exp050**:
>    - try LLRD
>    - stick to this as it show a good correlation, see if cv goes up, will LB up too
>    - this training get worse at 2nd epoch
>    - based: exp047
>    - [RESULT] fold0 CV: 0.7761, LB: 0.793
>    - from the last epoch

- by using kaggle-only data as validation set, cv can be calculated more aligned with test dataset

- *try these after find the right CV scheme*
    - right now I am using only differential learning rate, haven't tried LLRD yet
    - freezing embedding layers or layers close to it can help

# 5.10

>- **exp051**:
>    - using LLRD on exp046
>    - based: epx046
>    - [RESULT] fold0 CV: 0.7805, LB: 0.772
>    - [POS]
>    - from the last epoch

- replace \n\n with "[PARAGRAPH]" speical token instead of using AddToken

>- **exp052**:
>    - based: exp051
>    - replace \n\n with [PARAGRAPH] and add it as special token
>    - [RESULT] fold0 CV: 0.7823, LB: 0.777
>    - [POS]
>    - from 2nd epoch
>    - diff: 0.0053
>    - CV and LB are correlating when compared to based exp

- In the previouse exps, all weight_decay config  were 0.0
    - changed to 0.01 to layer except layer that does not contains in no_decay list

>- **exp053**: weight_decay to 0.01
>    - based: exp052
>    - [RESULT] fold0 CV: 0.787, LB: 0.774
>    - from the 2nd epoch
>    - diff: 0.013

>- **exp054**: freeze top 4 layers
>    - based: exp053
>    - [RESULT] fold0 CV: 0.7805, LB: 0.796
>    - from the last epoch
>    - diff: 0.0155

>- **exp055**: use gem pooling
>    - based: exp053
>    - [RESULT] fold0 CV: 0.7815, LB: 0.770
>    - from 2nd epoch
>    - diff: 0.0115

- gap between CV and LB is getting closer but it is very unstable

- tasks for tmr
    - see exp053, 54, 55 results and decide whether run exp056 or not

# 5.11

>- **exp056**: use both 1-e5 lr on warmup 0.2
>    - goal: try to fix overfit at the last epoch, as previous exp possibly reached local minimum
>    - based: exp052
>    - [RESULT] fold0 CV: 0.7915, LB: 0.779
>    - [POS]

>- **exp057**: use both 2-e5 lr on warmup 0.2
>    - based: exp054
>    - [RESULT] fold0 CV: 0.7843, LB: 0.773

- Since the validatiion dataset is very small, it is very hard for evaluation
    - Should I consider the smaller gap between CV and LB better
    - Or, when LB improves but did not show on CV because CV is relatively unstable

- Does weight_decay hurt the performance because AdamW has already apply it?
    - especially, on regreesion task

- In training stack use readability as features

>- **exp058**: max_norm from 1.0 to 10
>    - based: exp054
>    - [RESULT] fold0 CV: 0.7837, LB:

>- **exp059**: lr 1-e5, warmp up ratio to 0
>    - based: exp052
>    - [RESULT] fold0 CV: 0.7841, LB:

- no warmup does not change to fact that it best score came from the first epoch
- In exp056, 0.7915 is from 1st epoch
    - this score achieve when lr was close to 1-e5 which is peak of the cycle
    - what if I start at 5-e5?

>- **exp060**: lr start from 5-e5 with warmup_ratio of 0.2
>    - based: exp052
>    - [RESULT] fold0 CV: 0.7786, LB:
>    - this one from last epoch but it did not converge?

>- **exp061**: lr 1e-5 with warmup_ratio of 0.2 now wih weight decay 0.01
>    - based: exp056
>    - [RESULT] fold0 CV: 0.7909, LB: 0.779
>    - diff: 0.0119

>- **exp062**: lr 3e-5 no weight decay
>    - based: exp056
>    - [RESULT] fold0 CV: 0.7792, LB:

>- **exp063**: lr 1e-5 linear decay lr scheduler
>    - based: exp056
>    - [RESULT] fold0 CV: 0.7869, LB:


# 5.12

- check oof CV of train stack notebook

>- **exp064**: decrease batch_size to 4 from 8
>    - based: exp056
>    - [RESULT] fold0 CV: 0.7826, LB: 0.783
>    - diff: 0.0004

>- **exp065**: freeze top 4 layers
>    - accidentally turn off clip_grad_norm
>    - need to re run with clip_grad_norm turn on
>    - based: exp056
>    - [RESULT] fold0 CV: 0.7799, LB: 0.778
>    - diff: 0.0019

- CV and LB is now aligning nicely
    - now, the problem is how I can increase the CV

- why weight decay does not work?

>- **exp066**: added more data from non-overlap persuade 2.0 to train only score 1, 5, 6
>    - based: exp064
>    - [RESULT] fold0 CV: 0.7701, LB:

>- **exp067**: try freeze top 4 layers 
>    - based: exp056
>    - [RESULT] fold0 CV: 0.7915, LB:
>    - does not change except more available memory
>        - should be used only on large model to save memory with big batch size

>- **exp068**: use normal training set which is default StratifiedKfold but validating using only kaggle_only data
>    - based: exp056
>    - save hparams as exp056 
>    - [RESULT] fold0 CV: 0.7867, LB: 0.782
>    - diff: 0.0047
>    - if this looks better, I should try mix persuade 2.0

>- **exp069**: added persuade 2.0 and validates on kaggle_only
>    - based: exp068
>    - [RESULT] fold0 CV: 0.7899, LB: 0.777
>    - train time: 2 hours

>- **exp070**: same as exp069 but include not_kaggle_only in validataion dataset that had been removed
>    - based: exp069
>    - [RESULT] fold0 CV: 0.7908, LB: 0.796
>    - train time: 2.5 hours

- tasks for tmr
    - work on readability more
    - figure out about rounding post-processing
        - use optuna to optimize rounding?

# 5.13

- would base or large boost the score maybe on CV but not so much on LB

- exp068 and (exp069 and exp070) have different validation set

>- **exp071**: remove persuade out of the training set but remained not_kaggle_only that belonged in the validation dataset
>    - based: exp068
>    - [RESULT] fold0 CV: 0.7924, LB: 0.789
>    - [POS]
>    - dff: 0.0034

>- **exp072**: change lr scheuduler to constant with warmup
>    - based: exp071
>    - [RESULT] fold0 CV: 0.7953, LB: 0.790
>       - diff: 0.0053
>    - [RESULT] fold0_2 CV: 0.8099, LB: 0.797 (optimized rounding)
>       - diff: 0.0129
>    - [POS]

>- **exp073**: try differentiate LR with cosine warmpup 0.2 with 1e-5 and 2e-5 with 0.01 wd
>    - based: exp071
>    - [RESULT] fold0 CV: 0.7965, LB: 0.784
>        - diff: 0.0125
>    - [RESULT] fold0_2 CV: 0.8044, LB: 0.790 (optimized rounding)
>        - diff: 0.0144

- integrate optunarounder to training and infer notebook later
    - I think each threshold is bit different in every models
    - Does it need to calculate in inferencing?
        - No, do it seprately
        - Calculate it after ensemble with mean of raw output
- So now on, when I trained 4 fold I wiil run optimization after finished the training to get most


# 5.14

- tried nelder-merd method. It improved CV butt not good as optuna
    - ~~sometimes one other is better~~
    - so run both and get better one
        - optuna is usually better even with slightly lower CV

>- **exp074**: same hparams as exp072 but change lr scheduler to polynomial power of 0.05
>    - based: exp072
>    - [RESULT] fold0 CV: 0.7937, LB:

- There are two topics in NOL persuade that has similar distribution as validation set
    - more 5 and 6 score sample
    - Can use?

>- **exp075**: mixed two topics in training dataset ['Cell phones at school','Mandatory extracurricular activities']
>    - based: exp072
>    - [RESULT] fold0 CV: (0.80105), LB: (0.796) (optimized rounding)
>    - did not improve

>- **exp076**: use exp072 as base, this time use base model
>    - based: exp072
>    - [RESULT] fold0 CV: 0.7829 (0.8121), LB: (0.793) (submitted only optimized one)

>- **exp077**: increase lr to 2e-5
>    - based: exp076
>    - [RESULT] fold0 CV: 0.7608, LB:

>- **exp078**: I want to see how warmpup 0.1 goes return to small model for this
>    - linear decay and weight decay 0.01
>    - based: exp071
>    - [RESULT] fold0 CV: 0.7939 (0.8100), LB: (0.795)

>- **exp079**: try groupkfold
>    - based: exp072
>    - [RESULT] fold0 CV: 0.7530, LB:
>    - did not go well?


- tasks for tmr
    - train 4 folds exp078
        - base model after that?

# 5.15

>- **exp080**: same setup with exp078 but 4 folds
>    - based: exp078
>    - [RESULT] 4fold oof CV: 0.7855, LB: 0.795
>    - [RESULT] 4fold oof CV: 0.7971, LB: 0.794 (NM)
>    - [RESULT] 4fold oof CV: 0.7964, LB: 0.803 (OPTUNA)
>       - diff: 0.0066

- Since CV scheme changes best qwk score is always showing up at middle on the training (not last epoch)
    - despite of having more data to train the score is almost always lower
    - Is there are any chances I saved the wrong epoch?

- Maybe split then filtered out is not the good way
    -  So I should filtered kaggle-only data first to k0 to k4 then add addtional data


- Let's organize CV scheme again
    1. FIRST
        - splited train_df_with_prompt with StratifiedKFold
        - training set
            - (!= fold_num) + (== fold_num & kaggle_only == False)
        - validating set
            - (== fold_num)

    2. SECOND
        - splited `train_df_with_prompt[train_dfwith_prompt['kaggle_only'] == True]` as ko
        - splited `train_df_with_prompt[train_dfwith_prompt['kaggle_only'] == False]` as nok
        - training set
            - (ko != fold_num) + (nok)
        - validating set
            - (ko == fold_num)

    2. THIRD
        - splited `train_df_with_prompt[train_dfwith_prompt['kaggle_only'] == True]` as ko
        - splited `train_df_with_prompt[train_dfwith_prompt['kaggle_only'] == False]` as nok
        - training set
            - (ko != fold_num) + (nok != fold_num)
        - validating set
            - (ko == fold_num)

    2. FOUTH
        - splited `train_df_with_prompt[train_dfwith_prompt['kaggle_only'] == True]` as ko
        - splited `train_df_with_prompt[train_dfwith_prompt['kaggle_only'] == False]` as nok
        - training set
            - (ko != fold_num) + (nok == fold_num)
        - validating set
            - (ko == fold_num)

- try to balance distribution
    - add 'Distance learning'  (have more 5 and 6 samples)
    - add 'Seeking multiple opinions' (majority of samples is 4 )

- I think I need to from now on I have to train 4 folds
    - 1 fold tends to overfitted 1 validation set more than others
    - when I found one good score from 1 fold it does not transfer to others

- tasks for tmr
    - train 4 fold of exp081

# 5.16

>- **exp081**: add 'Distance learning' topic (have more 5 and 6 samples)
>    - add before split
>        - validation set changed
>            - by intersected around 300 of datapoint and 800 that were difference
>            - some intersected datapoint does not have score 6
>    - based: exp078
>    - all best epoch from the last one
>    - [RESULT] 4fold oof CV: 0.7858, LB: 0.796
>    - [RESULT] 4fold oof CV: 0.7973, LB: 0.799 (OPTUNA)

- ensembled exp080 and exp081 CV: 0.7996, LB: 0.804

- **I have to be very be carefully about mixing new data**
    - **score might improved but there are more chances that It might start to overfit**

- cls taks can also use optimized thresholds method using raw predict

- ensemble between
    - diff pooling
    - diff task
    - diff kfold (skf, sgkf)
    - diff architects (longformer, roberta, deberta)
    - SVR
    - (LGBM is bit concerning)

- Comparison between raw predictions exp080 and exp081
    - diff between mean and std
        - exp081 is clearly leaning toward high score mean across all folds is around 2.6-7 and a bit more fluctuating by std is around 0.83-4
        - exp089 tends to overall predict lower score mean arcoss all folds is aroudn 2.4-6 but std is quite consistant around 0.78
    - This had shown that raw predict is also share the similar distribution as its training set 
    - If I try to manipulate the distribution, would it become overfit to validation set?

>- **exp082**: try FOUTH CV scheme as it has bell curve
>    - better result is not expect here, it should be quick training
>    - [RESULT] fold0 CV: 0.8101, LB: 0.779 diff: 0.0311
>    - [RESULT] fold0 CV: 0.8261, LB: 0.788 diff: 0.0381 (Optuna)
>    - Why cv is so high despite the size of the training set
>        - this looks similar to normal stratifiedKfold so the diff gap
    
- How can I balance distribution between training set and validations set

- Maybe to only way to use extra Persuade 2.0 is to predict later by trained model without it then add to LGBM later

# 5.17

- Higher score predict seems to share to same shape of histogram
    - this has shown in exp078(fold0) and exp080 (4folds)

>- **exp083**: take some topics out ['Car-free cities', 'Does the electoral college work?']
>    - these topics does not contain in kaggle-only
>    - [RESULT] fold0 CV: 0.7925, LB: 0.800 diff: 0.0075
>    - [RESULT] fold0 CV: 0.8049, LB: 0.802 (Optuna) diff: 0.0029
>    - train time: around 14 mins per epoch
>    - It worked?
>    - Optuna did not adjust much in the middle range but adjust heavily on both end

- Predicted and evaluate using exp083 on these two topics
    - f1: 0.594
    - qwk: 0.7797
    - not neither bad or good, let's see how 4 folds perform

- Predcited and evaluate using exp083 on persuade 2.0
    - f1: 0.494
    - qwk: 0.7756
    - pretty bad 
- On the other hand
    - When used exp080f0 as predictor
        - f1: 0.5197
        - qwk: 0.7916
        - not significantly better but still betterr

- I needs to find a way to handle this
        - There are chances that others topics could appear in private dataset

- Is that mean that these topics are noises?
    - How do they perform on unseen topics then?

- check print loss average and value

- In LGBM notebook, when apply new CV scheme training dataset is accordingly get larger
    - this cause notebook kernel died early in the process

>- **exp084**: exp083 but 4 folds
>    - based: exp083
>    - This exp validation set is (== fold_num & kaggle_only == True)
>    - [RESULT] 4folds oof CV: 0.7860, LB: 0.793 diff: 0.007
>    - [RESULT] 4folds oof CV: 0.7976, LB: 0.800 diff: 0.0024 (OPTUNA)

>- **exp085**: use typical StratifiedKFold no filter or exclude any thing
>    - as in, train: (!= fold_num), valid (== fold_num)
>    - based: exp084
>    - this exp still removed two topics
>    - [RESULT] 4folds oof CV: 0.8199, LB: 0.796 diff: 0.0239
>    - [RESULT] 4folds oof CV: 0.8295, LB: 0.808 diff: 0.0215 (OPTUNA)
>    - but both CV and LB were up
>    - Every folds picked last epoch
>    - But gap between CV and LB is quite large 

- analyze more on exp80, exp84

>- **exp086**: train: (!= fold_num), valid: (== fold_num & kaggle_only == True)
>    - this exp included all 7 prompt_name in default training dataset
>    - [RESULT] 4folds oof CV: 0.7899, LB: 
>    - [RESULT] 4folds oof CV: 0.7972, LB: 0.800 (OPTUNA)

- If exp085 or exp086 does not give a good result, might have to think about next step

# 5.18

- After looking throught competition's discussion, it seems that those removed two topics does not appear in test dataset
    - So going forward removed these two topics should be must

- Sine removing two topics make current remainning topics only 5
    - should I change to 5 folds split?
    - that means StratifiedGroupKFold and GroupKFold are basically the same split
    - kaggele_only == True might make training pick the wrong epoch of training
        - the stratifiedfkfold MIGHT cause topics-leakage, that could be the reasons why CV is higher than LB

>- **exp087**: 5 folds without two topics
>    - GroupKFold -> see if topics leakage really occured
>    - [RESULT] fold0 CV: 0.7524, LB: 0.791
>    - [RESULT] fold0 CV: 0.7840, LB: 0.758 (OPTUNA)

- StratifiedKFold but remove one topic from each training set

>- **exp088**: scale exp085 to base model same hparams
>    - based: exp085
>    - 1 hours and 24 mins
>    - [RESULT] fold0 CV: 0.8289, LB:
>    - [RESULT] fold0 CV: 0.8377, LB: 0.784 (OPTUNA)

>- **exp089**: change lr from 1e-5 to 2e-5
>    - based: exp085
>    - [RESULT] fold0 CV: 0.8367, LB: 
>    - [RESULT] fold0 CV: 0.8417, LB: 0.790 (OPTUNA)

>- **exp90**: try CORN loss
>    - based: exp085
>    - [RESULT] fold0 CV: 0.7548, LB:

# 5.19

- let's organize how different CV schemes have been working so far
    1. using all 7 prompt_names
        1. use kaggle-only data as validation set and add remained validation data to training set
            - used in: *exp080*
                - exposed to all prompt
                - [RESULT] 4fold oof CV: 0.7964, LB: 0.803 diff: 0.0066(OPTUNA)
            - training set
                - (!= fold_num) + (== fold_num & kaggle_only == False)
            - validataing set
                - (== fold_num & kaggle_only == True)

        2. use kaggle-only data as validation set 
            - training set
                - (!= fold_num)
            - validating set
                - (== fold_num & kaggle_only == True)

        3. use typical StratifiedKFold
            - used in: *exp015*
                - [RESULT] 4fold oof CV: 0.8284, LB: 0.798 diff: 0.0304 (np.rint)
            - training set
                - (!= fold_num)
            - validating set
                - (== fold_num)

    2. using only 5 prompt_names (removed ['Car-free cities', 'Does the electoral college work?'])
        1. use kaggle-only data as validation set and add remained validation data to training set
            - used in: *exp084*
                - exposed to all prompt
                - [RESULT] 4folds oof CV: 0.7976, LB: 0.800 diff: 0.0024 (OPTUNA)
            - training set
                - (!= fold_num) + (== fold_num & kaggle_only == False)
            - validataing set
                - (== fold_num & kaggle_only == True)

        2. use kaggle-only data as validation set 
            - training set
                - (!= fold_num)
            - validating set
                - (== fold_num & kaggle_only == True)

        3. use typical StratifiedKFold
            - used in: *exp085*
                - [RESULT] 4folds oof CV: 0.8295, LB: 0.808 diff: 0.0215 (OPTUNA)
            - training set
                - (!= fold_num)
            - validating set
                - (== fold_num)

- So adding removed-not-kaggle-only from validation set later is different from splitting kaggle-only then add whole not-kaggle-only
    - former is containing all 7 prompt_name while latter does not

>- **exp091**: split kaggle_only first then add ol_persaude to each training set
>    - [RESULT] fold0 CV: 0.7968, LB: 0.784 diff: 0.0128
>    - [RESULT] fold0 CV: 0.8167, LB: 0.795 diff: 0.0217

>- **exp092**: based on exp085 change 1e-5 to 3e-5
>    - based: exp085
>    - [RESULT] fold0 CV: 0.8344, LB:
>    - loss is higher than exp085

>- **exp093**: based on exp085 large model lr 2e-5
>    - just to test how large perform
>    - 7 prompt_name is still safer to use
>    - based: exp085
>    - [RESULT] fold0 CV: 0.8295, LB:
>    - [RESULT] fold0 CV: 0.8389, LB: 0.804
>    - from first epoch

- split fold as if it is a multilabel problem?
    - based on score and prompt_name

>- **exp094**: 
>    - small model hparams is the same as exp085 but all 7 prompt_name
>    - test more with 5 prompt_name later
>    - [RESULT] fold0 CV: 0.8245, LB: 0.794 diff: 0.0305
>    - [RESULT] fold1 CV: 0.8208, LB: 0.797 diff: 0.0238
>    - [RESULT] fold2 CV: 0.8232, LB: 0.789 diff: 0.0342
>    - [RESULT] fold3 CV: 0.8196, LB: 0.798 diff: 0.0216

>    - [RESULT] fold0 CV: 0.8333, LB: 0.808 (LB is OPTUNA Threshold)
>    - [RESULT] 4folds oof CV: 0.8220, LB: 0.797
>    - [RESULT] 4folds oof CV: 0.8299, LB: 0.805 diff: 0.0249

# 5.20

- new weight initialization?
- multitask learning? train both prompt_name and score at the same time

>- **exp095**: added prompt_name as target
>    - exp
>    - this MIGHT make score share the weight with prompt
>    - [RESULT] fold0 CV: 0.8221, LB: 0.794
>    - [RESULT] fold0 CV: 0.8294, LB: 0.808

>- **exp096**: based on exp094 turn hparams lr from 1e-5 to 2-e5
>    - based: exp094
>    - [RESULT] fold0 CV: 0.8235, LB:

>- **exp097**: try kaggle-only validation
>    - split to ko and nko
>    - train: (ko != fold_num) + (nko != fold_num), valid: (ko == fold_num)
>    - [RESULT] fold0 CV: 0.7919, LB:
>    - [RESULT] fold0 CV: 0.7971, LB: 0.798

>- **exp098**: try kaggle-only validation
>    - split to ko and nko
>    - train: (ko != fold_num) + (nko), valid: (ko == fold_num)
>    - [RESULT] fold0 CV: 0.7889, LB: 
>    - [RESULT] fold0 CV: 0.7951, LB: 0.790


- tasks for tmr
    - submit exp094 with np.rint
    - decide whether submit all folds from exp094 of not
    - there is something wrong why some essay ground truth is 1 but model scored 4 or 5
        - maybe model learn only from legnth of the full_text of something else
        - need to look through this in EDA

# 5.21

- exp094 each fold using np.rint
    - fold 0:
        - train:
            - fold 1, 2, 3
         - result: 0.794 
    - fold 1:
        - train:
            - fold 0, 2, 3
         - result: 0.797
    - fold 2: 
        - train:
            - fold 0, 1, 3
         - result: 0.789
    - fold 3: 
        - train:
            - fold 0, 1, 2
        - result: 0.798

- seems like pred difference is cause by length of the essay
    - by typical lower score wold have around 1400 to 1500 chars or around from 300 to 400 of token length 
    - On higher side of score length go higher from 2600 and up to 4500 chars or 500 to 800 of token
        - predictions differences are often occured when ground truth score is lower but model decide to predict higher than it should based on length of the essay


- train start point of the discourse element first
    - then inference the train model into the full_text to give full_text more rich information.
    - the point is the give model more information about how certain setence mean to the whole essay
    - not sure if this possible with multilabel because some does not contain some discourse type
        - if the essay does not have some type label it as -1?
    - try with CLAIM first
    - Or I can predict how many unique discourse type has
        - not that is not possible because there is no way to integrate this formation into training set
    - this approach harder to execute than I thought maybe comeback later?


> Bias-Variance Trade-off:
> A higher number of splits (e.g., 5) typically reduces bias but increases variance, while a lower number of splits (e.g., 4) has the opposite effect.
> Bias refers to the error introduced by approximating a real-world problem, while variance refers to the sensitivity of the model to the specific data it was trained on.

 
>- **exp099**: train max_len 1024, valid max_len 768
>    - use shorter valid max_len to mitigate text length dependent
>    - based: exp094
>    - [RESULT] fold0 CV: 0.8182, LB:
>    - [RESULT] fold0 CV: 0.8309, LB: 
>    - shorter length of validation make OOF prediction between 5 and 6 become hard
>        - as this exp mostly predicted 5 instead 6 in y_trues


# 5.22

- Gathered discourse_type and found that higher score tends to have move discoure_type_num
    - thus, higher scores possibly have a clear type of discourse statement
    - To test idea is meaning full, 
        - let's first try with feed sentence by sentence
        - labels the sentence that which class that sentence belongs to
        - I need to seperate train and valid with multilabel score and prompt_name
    - f1_score for fold0 is 0.78, quite mediocre but not unusable
    - Rebuttal and evidence seem to have a significant impact on how essays were score
    - choose rebuttal and evidence to mix into training dataset first
    - finished created training set with new special token
        - this one coming from only 1 fold, can get better result with 4 folds


- task for tmr
    - fork train notebook to train on this new training dataset

# 5.23

- before fork new notebook, I need to find threshold for prediction confident
    - get it from OOF CV

- found some token weirdly placed in persaude dataset
    - unannotated right after s in such as -> s[UANNOTATED]uch as
    - should not be problem becaue unannotated will be ditched anyway?

- inferencing 1 batch at the time have a lot of overhead thus increase unnecessary time
    - need to fix this infer notebook

- need to study more about f1_score, auc_roc_score

>- **exp100**:
>    - added special token to text (added rebutta and evidence)
>    - [RESULT] fold0 CV: 0.8198, LB: 0.790
>    - [RESULT] fold0 CV: 0.8253, LB: 0.799 (OPTUNA)
>    - [RESULT] fold0 CV: 0.8261, LB: 0.799 (NM)
>    - *found bugs: tokens from not_kaggle_only are only from the first one*
>    - So basically this one is failed experiment

- tasks for tmr:
    - train sen_discourse again this time with 4fold with colab
        - or large 1 fold? May 1 fold of large is better?
    - infer on kaggle_only
    - created new train_combined_tokened
    - train tokened_text -> exp101

# 5.24

- predict text dependent and indepedent?
    - because depends on type of essay writting style will be largely different
    - text depedent will quote and mention thing that is specific and context from one source
    - On the other hand, Indepedent will represent more about writer opinion which is more general
    - In default training dataset, all essays is text dependent
        - topics (prompt_name) that does not exist in the first is independent
        - that could explain why hidden test is giving bad score because they (test) are all text dependent?

- 2 topics are different from others
    - ['Car-free cities', 'Does the electoral college work?']
    - they have multiple sources
    - their essay could have a wide range of things to write about
        - which means, they may mention or give evidence from more than 1 source
        - evidence from 2 sources and 3 sources might make difference
    - this explained a bit why 5 prompt_name (without)
    - Should I try to seperate subtype of these essays first?
        - Maybe using text with special tokened could help as it is providing which sentence indicated as Evidence discourse

>- **exp101**: fixed exp100
>    - (added rebuttal and evidence)
>    - [RESULT] fold0 CV: 0.8359, LB: 0.784
>    - [RESULT] fold0 CV: 0.8426, LB: 0.792

- put selected token inside config?
    - no

>- **exp102**: add all tokens except Unannotated
>    - [RESULT] fold0 CV: 0.8382, LB: 0.754

>- **exp103**: add all tokens 
>    - [RESULT] fold0 CV: 0.8420, LB:

- Need to look more into how to split sentence

# 5.25

- discourse type is being replace by a set of sentences
    - merge them with when the current one is the same as previous discourse type

- re-organize steps
    1. create discourse text
        - extract from persaude1.0 and persuade2.0 (part1 of data) -> sen_discourse.csv
        - train model on extracted discourse text 
        - infer on data that does not have discourse type (part2 of data) (mostly kaggle_only) -> dt_pred.csv
            - post-processing: remove discourse type when there is consecutive type
                - this make sentence with type become set of sentence instead
        - concatenate infered discourse text with extracted discourse text (send_discourse + dt_pred) -> train_tokened_text
    2. use newly created tokened text train mode and predict score

>- **exp104**: 
>    - same record as typical training
>    - all token is trained
>    - if token appeared consecutively remove it
>    - [RESULT] fold0 CV: 0.8395, LB: 0.777
>    - [RESULT] fold0 CV: 0.8454, LB: 0.799 (OPTUNA)
>    - [RESULT] fold0 CV: 0.8476, LB: 0.795 (NM)

>- **exp105**:
>    - removed Unannotated, threshold 0.75 on inferred kaggle_only data
>    - cleaned up data more
>    - [RESULT] fold0 CV: 0.8380, LB: 0.784


# 5.26
- found bugs \n\n got removed in not_kaggle_only data

# 5.27

- So idea is working only when is token correctly placed
    - in training tokened text phrase, qwk went up.
        - this proved that the idea is working
    - the problem is most of the training data is from persaude which discourse type is labeled by human
        - the labeled type is considered as correct one, the tokened text in  kaggle_only data is created from prediction
        - It is not accurate as the hand labeled one
        - thus, when the type predictor model encounter new data, it did fairly poor job that led to score predictor to worse result
    - if I can find a way to improved performance of type predictor and decrease inferencing time.
        - this idea will work out

    - to reduce inferencing time, NN model should not being used. rather, LGBM or SVR can help?
        - should try SVR first? As LGBM will consume of features engineering, but SVR only needs embedding from NN model

>- **exp106**:
>    - try this last time, need to move on if this does not works
>    - D005
>    - include only evidence token because it has over 0.8 on f1_score
>    - [RESULT] fold0 CV: 0.8295, LB: 0.798 
>        - diff: 0.0315
>        - RESLUT from exp094, fold0 CV: 0.8245, LB: 0.794 diff: 0.0305
>            - CV+0.0055, LB+0.004
>        - this result, agian, re-proved the idea. I just have to make this right on discourse type predicting part
>            - first, improved model performance
>            - second, decease inferring time

- maybe change all pandas to polars to speed up?
    - changed in predicting and creating discourse type part
    - predict_cv remained to the same
	

>- **exp107**
>    - D005 <- this is only 1 fold
>        - can improved by add more data or increase size of the model
>        - add find threshold for each discourse type
>    - add claim token as precision is high enough (around 0.8)
>    - purpose to test the result
>    - [RESULT] fold0 CV: 0.838, LB: 0.785

- maybe only evidence as it is most accurate?

# 5.28

- DT train recap
    - D001 small model
        - all token included
        - bugged
        - did not used anywhere
    - D002 small model
        - all token included
        - lr 2e-5
    - D003 base model
        - all token included
    - D004 small model
        - did not used anywhere
    - D005 small model
        - excluded unannotated token
    - D006
        - train binary between Evidence vs. Others

- discourse type predictor inferencing time
    - small model takes around 1 hrs and 30 mins (3.19it/s in sample sub)
    - base model takes around (4.29it/s in sample sub)
    - turned out, it took so long before copying between polars and pandas
        - test sub took only 1 hours
        - can still improve this
            - reduce reliance on pandas

- In final submission, `lead`, `claim`, `evidence` can be used as they have the most obvious trends
    - more num of discourse, higher score 
    - others has the similar trend, but they not obvious the gradient are steep, some has slightly drop.

- Large model should be used as discourse type because it will influence others models in second stage so much
    - could extend more options for usable token
    - D009 trained excluded unnanotated
        - 1 epoch takes around 1 hours

- not removing repeated have lower score
    - take 1 hours and 25 mins to finish

- lead and conclude should appear only once in the essay
    - filter out if there is one more, if use
    - do this after remove repeated discouse type
    - also position?
        - no, as almost every essay has one

- `claim` might be difficult to use as train as kaggle_only data produce too much of claim
    - discourse type predictor was not good on this particurly discourse type even in large model

- try add lead and counter claim as the next experiment

- cleaned up train_tokened_text more
    - lead and conclude only appear once in each essay if exist
    - remove consecutive discourse type

>- **exp106.5**
>    - same condition as exp106, but with newly adjusted data
>    - [RESULT] fold0 CV: 0.8287, LB: 0.794
>        - prev [RESULT] fold0 CV: 0.8295, LB: 0.798 
>    - did not help
>    - going forward remove consecutive discourse type should only be done in post-process

# 5.29

- on previous train_tokened_text 
    - consecutive discourse type in dataset was not removed but get higher score (exp106)
    - but when removed consecutive type score get a bit lower (exp106.5)
        - let's see on how LB would do
    - both removed consecutive on inferencing

>- **exp108**:
>    - try max_len 2048 instead of 1024
>    - taking more tokens
>    - [RESULT] fold0 CV: 0.8297, LB: 0.798

>- **exp109**:
>    - add lead (now have evidence and lead)
>    - use D005
>    - [RESULT] fold0 CV: 0.8282, LB: 0.795


>- **exp110**:
>    - add claim (now have evidence and claim)
>    - use D005
>    - [RESULT] fold0 CV: 0.8375, LB: 0.785
>    - 1 hours and 40 mins when submit
>       - score predictor small model usually takes around 7-10mins

>- **exp111**:
>    - add counter claim (now have evidence and counter claim)
>    - use D005
>    - [RESULT] fold0 CV: 0.8288, LB: 0.786

- even 1 fold of base model take close to 10 mins on 3 records
    - then it should be SVR or LGBM
    - lets first try SVR as it is closer to NN
        - embeddings took time so much
        - use them in colab notebook with faster gpu
    - lgb is almost impossible to train because lack of features
        - in the past competition, it was used for stacking

# 5.30

- removed repeated only on claim and position

>- **exp112**
>    - add claim only
>    - [RESULT] fold0 CV: 0.8351, LB: 0.786

>- **exp113**
>    - add claim only
>    - same as exp112 but classification
>    - [RESULT] fold0 CV: 0.8295, LB: 0.786

- recap token strat
    - all of these exps are not using the same D model (D005)
    - tried
        - all except unannotated*
            - exp103 [RESULT] fold0 CV: 0.8382, LB: 0.754
        - evidence + rebuttal* old data
            - exp101 [RESULT] fold0 CV: 0.8359, LB: 0.784
        - evidence + claim
            - exp110 [RESULT] fold0 CV: 0.8375, LB: 0.785
        - evidence + lead
            - exp109 [RESULT] fold0 CV: 0.8282, LB: 0.795
        - evidence + counter
            - exp111 [RESULT] fold0 CV: 0.8288, LB: 0.786
        - evidence + position
            - exp115 [RESULT] fold0 CV: 0.8313, LB: 0.797
        - evidence + conclude
            - exp114 [RESULT] fold0 CV: 0.8272, LB: 0.802
        - evidence + conclude + rebuttal
            - exp118 [RESULT] fold0 CV: 0.8295, LB: 0.801
        - evidence
            - exp108 [RESULT] fold0 CV: 0.8297, LB: 0.798
        - claim 
            - exp112 [RESULT] fold0 CV: 0.8351, LB: 0.786
        - conclude 
            - exp116 [RESULT] fold0 CV: 0.8223, LB: 0.800
        - rebuttal 
            - exp117 [RESULT] fold0 CV: 0.8238, LB: 0.797

- found big bugged
    - a lot of tokens had been incorrectly mapped
    - fixed from exp114

- evidence help model discriminate in lower score better
    - but higher score still hard identify

>- **exp114**:
>    - add conclude (now have evidence and conclude)
>    - [RESULT] fold0 CV: 0.8272, LB: 0.802

>- **expp115**:
>    - add position (now have evidence and position)
>    - [RESULT] fold0 CV: 0.8313, LB: 0.797

# 5.31

>- **expp116**:
>    - conclude only
>    - [RESULT] fold0 CV: 0.8223, LB: 0.800

- Maybe try training MLM to predict discourse type
    - can massively reduce inferencing as it will be feeding the full essay
        - thus, it will take sam amount of time as the score predictor
    - mask all discourse type
    - let mlm predict added special token

>- **exp117**:
>    - rebuttal only
>    - [RESULT] fold0 CV: 0.8238, LB: 0.797

- step of training mlm
    - use masked text for training
    - target: tokenized addtitional special token
    - use cross-entropy

- tasks for tmr
    - train mlm model M001
        - wish it can replace earlier discourse type predictor model

# 6.1

- fix out of memory
    - Can not store out of fold prediction as its prediction contained 10k+ class
    - evaluate only on valid loss
    - It can be use as perplexity metric later
- maybe I have to train mlm max_len 2048 to cover all sentence

- finished inference mlm notebook
    - have not write how ensemble goes yet
    - as raw_output is too big to store within notebook ram

- default training dataset is quite unique from usual essay. They are all text dependent essay
    - Meaning that the range of vocabulary appear in the essay is limited to some point.
    - They will be writing about sources which is different for general essay that giving opinions on topics
    - Examples for general will be from experience or particularly created situation
    - on the other hands, the example from source dependent essay will quoting  from sources more 

- max_len 1024 is not enough, the prediction value will go shorter than number of sentence
    - thus, go with 4096 to be safe as 2048 might not cover in some case
    - how about none on max_len? it wil go to longest
    - maybe 4096 is enough

- would the model do better without [CLS] and [SEP] in every sentence?
    - try later

>- **exp118**:
>    - 3 tokens (evidence and conclude, and rebuttal)
>    - [RESULT] fold0 CV: 0.8295, LB: 0.801

# 6.2

- cleaning up dt_pred steps
    - merge consecutive discourse type
    - make sure LEAD and CONCLUDE remain only one in essay
        - keep the first LEAD
        - keep the last CONCLUDE
        - maybe in MLM predictions is not necessary

>- **exp119**:
>    - 2 tokens (evidence and conclude)
>    - use MLM predictions in Kaggle_only
>    - score is a bit different because data was not sorted
>        - out-of-fold is dfferent from previous exp
>        - already fixed for the future exp
>    - [RESULT] fold0 CV: 0.8300, LB: 0.783

- Need to work on multi-folds ensembler on MLM prediction

- Maybe `[CLS][MASK]...[SEP]` make model confused as someone has said it in the past competition
    - yes, the one with `[CLS][MASK]...[SEP]` data is getting lower accuracy and higher loss

# 6.3

- since MLM predictions is a bit different from sen_discourse predictions
    - need to check how removing consecutive discourse type influence CV and LB

>- **exp120**:
>    - add only evidence
>    - using M005 large model fold0
>    - remove consecutive discourse type on training time and inferencing time
>    - [RESULT] fold0 CV: 0.8305, LB: 0.770

- evidence only even lower together with conclude
    - maybe MLM predictions included need different approach
    - all predicted special tokens have contextual information of each others 
    - unlike sen_discourse preditions which training and predicting seprately

>- **exp121**:
>    - add all tokens except unnanotated
>    - using M005 large model fold0
>    - remove consecutive discourse type on training time and inferencing time
>    - [RESULT] fold0 CV: 0.83847, LB: 0.756

>- **exp122**:
>    - add evidence only
>    - using M005 large model fold0
>    - remove consecutive discourse type only on inferencing time
>    - [RESULT] fold0 CV: 0.8377, LB: 0.787

- maybe split '?' is not good
- Should I add Unnanotated? in mlm?

- M006:
    - will included unannotated
    - small model

- tasks for tmr
    - compare between predicted sen_discourse and predicted mlm

# 6.4

- I need better way to split sentence
    - `nltk.tokenizer.sent_toknize` can do better job than `str.split('.')`
    - lets see if speed would cause any problem
    - remove consecutive discourse type on training time and inferencing time

>- **exp123**:
>    - add evidence only
>    - using M006
>    - remove consecutive discourse type only on inferencing time
>        - [RESULT] fold0 CV: 0.8335, LB: 0.785
>    - not remove anything
>        - [RESULT] fold0 CV: 0.8335, LB: 0.786

- maybe removing consecutive discourse type only on inferencing time is not necessary on mlm?
    - lets try both

- found bug in infer notebook where `pred0` is still being used

- in discourse predictions, I finally found why it was taking so long to run
    - run 1 batchsize and using only `collator` not `DataCollatorWithPadding`
    - from 1 hrs 30 mins to 30 mins

-  should go back to discourse predictions?
    - since MLM is not working at all

- still dont understand why MLM does worse than discourse type predictor
    - with much higher F1 score and seemingly correctly labeled it made 

# 6.5

- maybe user-defined special tokens are harder for model to understand as it did not appear in pretrained before
    - so instead of `[LEAD]` or `[EVIDENCE]` trying `<Lead>` or `<Evidence>` might make more sense

>- **exp124**:
>    - same as exp108
>    - but using D012
>        - similar to D005 but changed batch size from 32 to 16 and using multiple-samples dropout
>    - [RESULT] fold0 CV: 0.8291, LB: 0.801

>- **exp125**:
>    - using D012
>    - changed `[SPTOKEN]` -> `(SpToken)`
>    - [RESULT] fold0 CV: 0.8301, LB: 0.799

- note that loss of D005 is still the lowest at	0.6280
    - while D011 at 0.6344 and D012 at 0.6353
    - diff
        - D005: bs 32
        - D011: bs 16
        - D012: be 16 multi-drop

- been thinking about how final ensemble would look like
    - 1st stage
        - discourse type preditors
            - which size?
    - 2nd stage
        - ensemble deberta trained with tokened_text
        - ensemble deberta-v3 with different pooling strat trained with tokened_text
            - mean pooling
            - GEM pooling
            - lstm pooling
            - attention pooling
        - ensemble deberta-v3 trained with mlm tokened_text?
        - ensemble longformer  trained with tokened_text
    - 3rd stage
        - use ridge or lgb stacking


- in auxiliary training
    - maybe I can use prompt_name as different head
    - from which layers

>- **exp126**:
>    - using D012
>    - changed `[SPTOKEN]` -> `<SpToken>`
>    - [RESULT] fold0 CV: 0.8332, LB: 0.800

- D013
    - same hparms as D012
    - base model

>- **exp127**:
>    - using D013
>    - changed `[SPTOKEN]` -> `<SpToken>`
>    - [RESULT] fold0 CV: 0.8315, LB: 0.798

- a fold of base model took 1 hour (D013) to submit
    - maybe small model is a better choice for speed and performance


- deciding whether go with `[TOKEN]`, `(TOKEN)`, or  `<TOKEN>`
    - <> improved on CV and LB 
    - [] worse on CV and better on LB 
    - () better a bit on CV but worse on LB
    - thus, going forward <> will be used

- tasks for tmr
    - train 1-3 fold of D012
        - create new ko_sen
    - while training, look through why MLM does not work
    - investigate more about gap between CV and LB (start from past exps)

# 6.6

- how can I analyze MLM?
    - maybe MLM need some threshold?
    - threshold by recall?

>- **exp128**:
>    - add evidence only
>    - using D012 4folds
>    - based: exp126
>    - [RESULT] fold0 CV: 0.8324, LB: 0.796
>    - [RESULT] fold0 CV: 0.8392, LB: 0.810 (NM)
>    - [RESULT] fold0 CV: 0.8375, LB: 0.805 (OPTUNA)

>- **exp129**:
>    - add conclude only
>    - using D012 4folds
>    - based: exp126
>    - [RESULT] fold0 CV: 0.8224, LB: 0.797

>- **exp130**:
>    - add rebuttal only
>    - using D012 4folds
>    - based: exp126
>    - [RESULT] fold0 CV: 0.8234, LB: 0.798

>- **exp131**:
>    - add lead only
>    - using D012 4folds
>    - based: exp126
>    - [RESULT] fold0 CV: 0.8222, LB: 0.795

>- **exp132**:
>    - add position only
>    - using d012 4folds
>    - based: exp126
>    - [RESULT] fold0 CV: 0.8228, LB: 0.798

>- **exp133**:
>    - add claim & counterclaim
>    - using d012 4folds
>    - based: exp126
>    - [RESULT] fold0 CV: 0.8360, LB: 0.790


- recap of D012_4folds
    - Lead
        - exp131 [RESULT] fold0 CV: 0.8222, LB: 0.795
    - Position
        - exp132 [RESULT] fold0 CV: 0.8228, LB: 0.798
    - Claim & Counterclaim
        - exp133 [RESULT] fold0 cv: 0.8360, lb: 0.790
    - Evidence
        - exp128 [RESULT] fold0 CV: 0.8324, LB: 0.796
    - Concluding
        - exp129 [RESULT] fold0 CV: 0.8224, LB: 0.797
    - Rebuttal
        - exp130 [RESULT] fold0 CV: 0.8234, LB: 0.798

>- **exp134**
>    - add position, evidence, concluding, rebuttal
>    - using d012 4folds
>    - based: exp126
>    - [RESULT] fold0 CV: 0.8306, LB: 0.800

>- **exp135**
>    - add position, evidence, concluding, rebuttal, lead
>    - using d012 4folds
>    - based: exp126
>    - [RESULT] fold0 CV: 0.8294, LB: 0.798

# 6.8

- so go without `<Lead>`

- For discourse type predictor approach is settled.
    - In exp134, it had shown consistency improvement both in CV and LB
        - its based exp, exp094, [RESULT] fold0 CV: 0.8245, LB: 0.794 diff: 0.0305
        - exp134 was [RESULT] fold0 CV: 0.8306, LB: 0.800, diff: 0.0306
    - 4 folds of D012
    - chose position, evidence, concluding, rebuttal
        - meaning that, excluded lead, claim, counterclaim
    - maybe hparams like betas and lr still needs a bit of adjustment

- There are something I want to try before go full-scale
    - why MLM is not working?
        - maybe single fold is too bias and simply not enough despite its performance
            - testing by train 4 folds and select only position, evidence, concluding, rebuttal
            - maybe `[TOKEN]` and `<Token>` diffrence?
            - if this does not work out, I have to leave this approach
        - maybe the preprocess and postprocess might had been done inconsistently and incorrectly
            - checked: done the same 
            - changed to how dt predictor steps (split by '.')
    - GroupKFold split
        - To make sense out of this approach, I have to train 7 splits based on each topics
        - This will take incredibly amount of time to train as large model takes around 4 hours per fold to train

- M007
    - small model
    - 4 folds


>- **exp136**:
>    - using M007
>    - split the same as dt predictor
>    - add position, evidence, concluding, rebuttal
>    - using `<Token>`
>    - [RESULT] fold0 CV: 0.8326, LB: 0.793

- okay, not working. lets move on

>- **exp137**:
>    - using D012
>    - add position, evidence, concluding, rebuttal (4 defaults)
>    - increase from 3 to 4 epoch
>        - for comparing AWP later
>    - [RESULT] fold0 CV: 0.8341, LB: 0.797
>        - overfitting?

- tried sub with pure feats lgbm
    - CV: 0.807 , LB: 0.802

- have to clean up infer notebook a bit where replace class to `<token>`
    - done

- try awp?

>- **exp138**:
>    - based: exp134
>    - start awp at 2 out of 3
>    - [RESULT] fold0 CV: , LB:
>    - train loss increasing nonstop
>        - pick wrong lr or eps?
>        - Ex. from others adv_lr = 0.0001 adv_eps = 0.01 (not in this exp)

# 6.9

- **exp139**:
    - based: exp134
    - added 'Holistic esssay score [SEP]' at the start of every essay
    - [RESULT] fold0 CV: 0.8285, LB:
        - don't need to submit because even it improved on LB, I will use it anyway

- try SiFT?

>- **exp140**:
>    - based: exp134
>    - try SiFT from last 2 epochs out of 3 epochs
>    - [RESULT] fold0 CV: , LB:
>    - maybe adversial training is not working
>        - SiFT also increased train loss a lot and took up to 3 times longer than usual
>        - This is not practical for larger model

>- **exp141**:
>    - based: exp134
>    - pretrain MLM first 2 epoch
>    - load that model and train like usual
>    - [RESULT] fold0 CV: , LB:
>    - cannot get it to work

- start trying different pooling
    - LSTM
    - Attention
    - GeM
    - Concat last 4
    - WeightedLayer?

>- **exp142**:
>    - based: exp134
>    - 4 defaults
>    - LSTM pooling
>    - [RESULT] fold0 CV: 0.8166, LB: 0.794

>- **exp143**:
>    - based: exp134
>    - 4 defaults
>    - Attention pooling
>    - [RESULT] fold0 CV: 0.8340, LB: 0.793

>- **exp144**:
>    - based: exp134
>    - 4 defaults
>    - GeM pooling
>    - [RESULT] fold0 CV: 0.8350, LB: 0.799

>- **exp145**:
>    - based: exp134
>    - 4 defaults
>    - Concat last 4 layrs pooling
>    - [RESULT] fold0 CV: 0.8338, LB: 0.793

- maybe others pooling needs different set of discourse type

# 6.10

>- **exp146**
>    - based: exp134, exp143
>    - instead of 4 defaults, go with evidence and conclude
>    - Attention pooling
>    - [RESULT] fold0 CV: 0.8301, LB: 0.795

- put question in front of essay?
    - to do that, prompt_name predictor needs to be included
    - predict what the prompt name -> map with prepared questions

>- **exp147**
>    - add assignment at the beginning
>        - `<Assignment>` assignment `[SEP]` `<Essay>` essay
>    - [RESULT] fold0 CV: 0.8296, LB: 
>        - did not submit because even LB is better I won't be using it

>- **exp148**
>    - using groupKfold split 7 base on prompt_name instead
>    - 4 defaults and mean pooling first
>    - [RESULT] fold0 CV: 0.762, LB: 
    
>- **exp149**
>   - same as exp134 but 4 folds
>   - [RESULT] 4folds CV: 0.8321, LB: 0.802
>   - [RESULT] 4folds CV: 0.8383, LB: 0.808 (OPTUNA) upper
>   - [RESULT] 4folds CV: 0.8391, LB: 0.808 (NM)

- comparing exp094 vs exp149
    - exp094
        - [RESULT] 4folds CV: 0.8220, LB: 0.797
         -[RESULT] 4folds CV: 0.8299, LB: 0.805 (OPTUNA)
    - exp149
        - [RESULT] 4folds CV: 0.8321, LB: 0.802
        - [RESULT] 4folds CV: 0.8383, LB: 0.808 (OPTUNA)


- max length of essay is 1536 

- What can add to lgbm features
    - from tokened_text
        - discourse types count
    - from nltk_eng
        - POS tag count
        - simple, past, future tense
    - variety of sentence structures count
    - total type of sentence structures

- how can I identify that quoting, mention, or taking from source text

- Should I try remove all discourse type and train with attention first?

>- **exp150**
>    - train without any discourse type to compare with next exp
>    - Attention pooling
>    - [RESULT] fold0 CV: 0.8202, LB: 0.788


>- **exp151**
>    - train with evidence only
>    - Attention pooling
>    - [RESULT] fold0 CV: 0.8300, LB: 0.792


- task for tmr
    - make train lgbm save oof_df for optimizing
    - check if oof_df and splits in train_lgbm leak or not
    - maybe think about using persuade 2.0 in training both NN and lgbm
        - try 7 GroupKFold with persuade 2.0?
    - check diff between tokened_text and full_text
        - not many
    - submit add_pos_tagged

>- exp149 + lgbm
>    - [RESULT] CV: 0.8336, LB: 0.812
>    - CV+0.0015, LB 0.01
>    - so using stacking in leakage free manner worked!

>- **exp152**
>    - based: exp134
>    - everything is the same except no warmup (following chris baseline)
>    - [RESULT] fold0 CV: 0.8317, LB: 0.800
>    - full-precision sorting in public score is higher than exp143.
>        - go without warmup

# 6.11

>- **exp153**
>    - based: exp152
>    - expand to 4 epoch
>    - [RESULT] fold0 CV: 0.8334, LB: 0.805
>    - but why warmup 0.1 make it worse?

- not submitted features
    - pos_tag (nltk features)
        - CV+0.002 from pure_feats
    - discourse type features
        - CV+0.002 from nn_feats

- I can add non-overlapping persuade in lgbm later
    - split 4 folds with multilabelstratifiedkfold as usual
    - then split 4 folds non-overlapping persuade seperately
        - these raw_preds (soft labels) will be coming from simple mean ensemble of all models

- finished FB3 features notebook
    - don't forget to add to lgbm notebook

- Try SWA or EMA

>- **exp154**
>    - based: exp153
>    - 4 epoch 
>    - apply EMA
>    - [RESULT] fold0 CV: 0.8347, LB: 0.793

- lgbm added POS_tag
    - CV: 0.8091, LB: 0.800
    - CV:+0.002 , LB:-0.002

# 6.12

- revisit AWP

- EMA does not help

>- **exp155**
>    - based: exp153
>        - as ema does not help, this exp will base on exp153
>    - 4 epoch start from 3rd epoch
>    - adv_lr 1e-4, adv_eps 1e-2
>    - [RESULT] fold0 CV: , LB:
>    - same, train loss increasing non-stop

- exp140+nn_feats+dt_count+fb3_feats
    - before selected CV: 0.8385
    - CV: 0.8368, LB: 0.815
    - no select CV: 0.8376, LB: 0.816

>- **exp156**
>    - based: exp153
>    - max_grad_norm from 1.0 to 10
>    - [RESULT] fold0 CV: 0.8319, LB:
>    - overfitting at the last epoch, this result is from 3rd epoch
 
>- **exp157**
>    - based: exp153
>    - change lr from 1e-5 to 2e-5
>    - [RESULT] fold0 CV: 0.8297, LB:

>- **exp158**
>    - based: exp153
>    - change linear to cosine scheduler
>    - [RESULT] fold0 CV: 0.8326, LB:


- This might be optimal hparams
    - linear decay no warmup
    - lr 1-e5
    - 4 epoch
    - weight_decay 0.01
    - batch size 8
    - gradient clip 1.0

- time to scale up?
- spend time cleaning incorrectly placed discourse type? 
    - too much to process

- tasks for tmr
    - train base model
        - start with single fold see if tuning needed
    - write ridge model notebook while waiting
        - wrote one but reaaly bad result
        - train more model adn revisit again

# 6.13

>- **exp159**
>    - try base model
>    - based: exp153
>    - [RESULT] fold0 CV: 0.8363, LB: 0.792
>    - [RESULT] fold1 CV: 0.8391, LB: 0.796
>    - don't know why base model is different from others
>        - It was reportedly a bit off 
>        - maybe lower learning rate? because it get ovefiting

- if adding persuade in stacking worked, would adding fb3 data in stacking worked too?
    - because lgbm relying less on topics
    - try persuade first then add fb3 later
    - Can I use FB3 data?
        - need to check licensing

- recap on lgbm
    - (s) selected_feats, (ns) no_selected_feats
    - pure_feats (s)
        - CV: 0.8073, LB: 0.802

    - pure_feats + pos_tag (s)
        - CV: 0.8091, LB: 0.800

    - pure_feats + fb3_feats (s)
        - (ns) 0.8134
        - CV: 0.8131, LB: 0.804

    - pure_feats + dt_count (s)
        - CV: 0.8180, LB: 0.801
        
    - pure_feats + nn (s)
        - CV: 0.8336, LB: 0.812 

    - pure_feats + nn + dt_count (s)
        - CV: 0.8356, LB: 

    - pure_feats + nn + fb3 (s)
        - CV: 0.8325, LB: 0.809

    - pure_feats + nn + dt_count + fb3 (s)
        - CV: 0.8368, LB: 0.815

    - pure_feats + nn + dt_count + fb3 (ns)
        - CV: 0.8376, LB: 0.816

    

- colab compute unit
    - T4: 1.76/hr 
        - most efficient but took long for large model
        - 4 folds of large model will be around 36 credit
        - 1 fold takes 12 hours (might to train 1 fold/day)
    - L4: 4.82/hr
        - might be a good trade off
        - 4 folds of large model will be around 48 credit
        - 1 fold takes 6 hours (might to train 2 folds/day)
    - A100: 11.77/hr
        - might have to use when time is running out
        - 4 folds of large model will be around 56 credit
        - 1 fold takes 1-2 hours (4 folds/day)


>- **exp160**
>    - large model
>    - [RESULT] fold0 CV: 0.8362, LB: 0.792
>    - time on colab
>        - 6/13 18:17, CU 87.47 
>        - 6/13 19:42, CU 70.91
>        - took around 1hr and 25 minues with 16.56 CU used
>    - maybe try lower learn rate?
>        - 5-e6

>- **exp161**
>    - large model
>    - lower learn rate to 5e-6 from 1e-5 
>    - [RESULT] fold0 CV: 0.843298, LB: 0.795
>    - time on colab
>        - 6/13 20:00, CU 70.91
>        - 6/14 01:08, CV 47.64
>        - L4 ended up use more CU
>        - took 5 hr and 8 minutes with 23.27 CU used

- considering freezing layer 
    - large model has 24 layers
    - base model has 12 layers
    - small model has 6 layers

- tasks for tmr
    - submit pure_feats + nn + fb3 (s)
    - submit exp161

# 6.14

>- **exp162**
>    - large model
>    - lr 5-e6
>    - freezed 12 top layers
>    - [RESULT] fold0 CV: 0.843205, LB:

- if there is submit left submit pure_feats + nn + dt_count
    - in trian-lgbm-ds

>- **exp163**
>    - large model
>    - lr 5-e6
>    - reduced to 3 epoch from 4 epoch
>        - maybe 4 epoch is causing overfitting?
>    - [RESULT] fold0 CV: 0.84266, LB: 0.799

>- **exp164**
>    - large model
>    - lr 5-e6
>    - 3 epoch
>    - redcue batch_size to 4 from 8
>    - [RESULT] fold0 CV: 0.8444, LB: 0.783

>- **exp165**
>    - large model
>    - lr 5-e6
>    - 4 epoch
>    - batch_size 4
>    - [RESULT] fold0 CV: 0.842317, LB: 0.795

- quick recap large model
    - exp166 lr 1e-5, epoch 3, bs 8 -> fold0 CV: 0.8437  , LB: 0.803 
    - exp163 lr 5e-6, epoch 3, bs 8 -> fold0 CV: 0.84266 , LB: 0.799
    - exp165 lr 5e-6, epoch 4, bs 4 -> fold0 CV: 0.842317, LB: 0.795
    - exp161 lr 5e-6, epoch 4, bs 8 -> fold0 CV: 0.843298, LB: 0.795
    - exp160 lr 1e-5, epoch 4, bs 8 -> fold0 CV: 0.8362  , LB: 0.792
    - exp164 lr 5e-6, epoch 3, bs 4 -> fold0 CV: 0.8444  , LB: 0.783

>- **exp166**
>    - moving toward exp163
>    - large model
>    - lr 1e-5
>    - 3 epoch
>    - batch_size 8
>    - LLRD*0.9
>    - [RESULT] fold0 CV: 0.8437, LB: 0.803

# 6.15

- exp166 have almost the same hyperpaarmeters as exp149
    - except warmup being 0.0 not 0.1 in exp149
    - I want to tuning hparams a bit more
    - let's change 1e-5 to 2-e5
    - after that, lets try diff lr 1-e5 on encoder and 2-e5 on decoder
    - lastly LLRD 1e-5 but *0.95 instead of 0.9
    - can't go with 4 epoch, because it will cause overfitting? then do with 3 epoch

>- **exp167**
>    - comparing with exp152
>    - back to small model
>    - LLRD 2-e5
>    - 3 epoch
>    - [RESULT] fold0 CV: 0.8347, LB: 0.796

>- **exp168**
>    - Diff LR Encoder: 1-e5, Decoder: 2-e5
>    - [RESULT] fold0 CV: 0.8331, LB:

>- **exp169**
>    - LLRD 3-e5
>    - [RESULT] fold0 CV: 0.8319, LB:

- Maybe I should newly create training dataset from D012
    - No need to use nko_sen, all data will from infer_discourse
    - So data is more consistent in term of preprocessing, training and inferencing

- might have to go to square one which discourse type to choose

>- **exp170**
>    - use new fully predicted from D012
>    - start with evidence
>    - replicated hparams from exp152
>    - [RESULT] fold0 CV: 0.8224, LB: 0.797

>- **exp171**
>    - use new fully predicted from D012
>    - 4 defaults dt
>    - replicating exp152
>    - [RESULT] fold0 CV: 0.8241, LB: 0.801

>- **exp172**
>    - use new fully predicted from D012
>    - 4 defaults dt
>    - LLRD from 1e-5 to 2e-5
>    - [RESULT] fold0 CV: 0.8243, LB: 0.795

>- **exp173**
>    - 4 defaults dt
>    - dataset created entirely from M007
>    - based: exp171
>    - [RESULT] fold0 CV: 0.8248, LB: 

- maybe dataset fully created by D012 or M007 working better than the original one
    - stick to orignal where half of them from persuade 1.0/2.0

>- **exp174**
>    - original tokened_text dataset
>    - based: exp152
>    - trying Huberloss instead of MSE
>    - [RESULT] fold0 CV: 0.8318, LB: 0.802

- put warmup 0.1 back on?

# 6.16

- so using huberloss is good?
    - note that loss must be *2 when comparing to mse

>- **exp175**
>    - use new fully predicted from D012
>    - comparing with exp171 
>    - 4 epoch
>    - so 4 epoch not really good for the strategy
>    - [RESULT] fold0 CV: 0.8242, LB: 0.799

>- **exp176**
>    - use huber with warmup 0.1
>    - 3 epoch
>    - [RESULT] fold0 CV: 0.8308, LB: 0.801
>    - without warmup is CV0.001 and LB0.001 higher 

- should look into max_len more, currently using 2048

>- **exp177**
>    - huber loss
>    - layerwise_lr_decay 0.95
>    - [RESULT] fold0 CV: 0.8332, LB: 0.804
>    - CV and LB increased almost at the same amount
>        - from exp174 fold0 CV: 0.8318, LB: 0.802

>- **exp178**
>    - huber loss
>    - layerwise_lr_decay 0.975
>    - [RESULT] fold0 CV: 0.8321, LB:

>- **exp179**
>    - huber loss
>    - layerwise_lr_decay 0.95
>    - LLRD 2-e5
>    - [RESULT] fold0 CV: 0.8329, LB:

>- **exp180**
>    - huber loss
>    - layerwise_lr_decay 0.95
>    - LLRD 1.5-e5
>    - [RESULT] fold0 CV: 0.8355, LB: 0.800

>- **exp181**
>    - large model 4 folds
>    - based: 177
>    - start: CU 77.72 at 6/16, 21:49
>    - end:   CU 27.70 at 6/17, 02:05
>    - used:  CU 50.02 took 4 hrs 16 mins
>    - [RESULT] 4 folds oof CV: 0.8436, LB: 0.802
>    - [RESULT] fold0 CV: 0.8363, LB: 0.798
>    - [RESULT] fold1 CV: 0.8465, LB: 0.798
>    - [RESULT] fold2 CV: 0.8466, LB: 0.796
>    - [RESULT] fold3 CV: 0.8448, LB: 0.794
>    - submit D012+exp181 took around just over 1 hour

# 6.17

>- **exp182**
>    - huberloss
>    - 4 epoch
>    - small model
>    - LLRD 1e-5 decay *0.95
>    - [RESULT] fold0 CV: 0.8326, LB: 
>        - from the 3rd epoch out of 4 epoch
>    - maybe decay *0.95 is pushing too hard

>- **exp183**
>    - huberloss
>    - hparams are same as 166 except the loss
>    - start: CU 127.61 at 6/17, 15:09
>    - end:   CU  77.58 at 6/17, 19:23
>    - used:  CU  50.03 took 4 hrs 15 mins
>    - [RESULT] 4 folds oof CV: 0.8452, LB: 0.806


- wanted to try longformer-base but not really worth it 
    - if train, will go with lonformer-large

- **exp184**
    - replacing exp149
    - following exp153 setup
        - 4 epoch
        - no warmup
        - lr 1-e5
        - max_len 2048
    - [RESULT] 4 folds oof CV: 0.8352, LB: 

- the last submission time limit
    - currently have
        - 4 folds of 1 small DT model (This took around 10 mins)
        - 4 folds of 1 small model (This took around 25 mins)
            - current exp149 replacing with exp184 if it is better
        - 4 folds of 2 large models (1 model took aroudn 1 hr)
        - 4 folds of 1 lgbm model (takes around 7-10 mins)
    - if include everything excluding large model will take approximately
        -  2 hrs and 45 mins
    - so maybe last submission can fit around 5-7 large models
    - check real time tmr

- What models to train next?
    - deberta-large
    - deberta-v3-base
    - longformer-large
    - distilled-roberta-uncased?

# 6.18

- exp183_4f submit time 
    - at 09:16
    - finished 10:27
    - took around 1 hr and 11 mins

- exp149+exp181+exp184 submit time
    - at 09:16
    - finished 11:46
    - took around 2 hr and 30 mins

- exp184 when ensemble with all others exp it gave lower CV
    - it can't replace 149

- what exp that have not submitted yet?
    - lgbm pure_feats with persuade
    - pure_feats + nn + dt_count (s)

- see exp149+exp181+exp184 result then decide what to do next
    - longformer-large and deberta-large can not train on kaggle
        - can but with loss accumulation
    - Roberta-large could be able to fit in P100 ram

- exp149+exp181+exp184
    - CV: 0.848144
    - LB: 0.808

>- **exp185**
>    - small model
>    - classification with class weight
>    - [RESULT] fold0 CV: 0.8281, LB: 
>    - did not pick the last epoch (this reuslt is from 2nd epoch)

>- **exp186**
>    - deberta-large
>    - start: CU 77.58 at 6/18, 13:32
>    - end:   CU 22.40 at 6/18, 18:15
>    - used:  CU 55.18 took 4 hrs and 43 mins
>    - [RESULT] 4 folds oof CV: 0.8483, LB: 0.797

- maybe I should train 1 large with plain data, just for safety

- fb3+dt_count+149+181+183+186
    - (s)  CV: 0.8537, LB: 0.821
    - (ns) CV: 0.8541, LB: 0.819
    - This result CV is incorrectly calculated because it was wrongly splitted
    - submission time:
        - start: 17:05
        - end: 20:33
        - took 3 hr and 28 mins
- pure_feats+149+181+183+186
    - CV: 0.8549, LB: 0.817

# 6/19

- Recap left tasks (what I can do in the last 2 weeks)
    - train more models
        - longformer-large (colab)
        - deberta-v3-large using mse and GeMPooling (colab)?

        - distilbert-base-uncased (kaggle)
            - train 2 folds in one run
        - deberta-v3-base (kaggle) (if have time left)
            - train 2 folds in one run

    - inferencing soft-label on NOL data from Persuade to include in lgbm model
        - have to inference from all trained model
        - this will be almost last step to do after trained all models

- fb3_feats is potentially leak
    - the model was trained on feedback1 which is the same data
        - some were not exclude
    - it is indirect leakage because it does not directly target the y
        - leakage from knowledge distilled pseudo labelling 

- found bugs that mskf was split differently on NN and lgbm because of order of dummies columns 
- fixed leakage (bugs)
    - (s)  CV: 0.85487, LB: 0.822
    

- tasks for tmr
    - train longformer-large
    - looking more for features


# 6.20

- pure_feats+persuade with LB 0.810 was 15 folds
    - while 4 folds is only 0.805
    - increase folds should help in efficiency track

>- **exp187**
>    - large model
>        - gempooling
>        - mse
>    - start: CU 113.36 at 6/20, 13:41
>    - end:   CU  62.35 at 6/20, 17:58
>    - used:  CU  51.01 took 4 hrs and 17 mins
>    - [RESULT] 4 folds oof CV: 0.8452, LB: 0.799

- don't know why longformer-large give nan loss
    - skip for now
    - set autocast to bfloat16 should help

- bigbird-large
    - loss does not decrease

- lgbm+persuade 15 folds
    - CV: 0.8519, LB: 0.807
    - only around 4 minutes

- what model to train next
    - longformer-large
    - deberta-v3-large (cls)
    - deberta-v3-base

- fb3+dt_count+149+181+183+186+187
    - CV: 0.8557, LB: 0.819
    - this took around 4 and 20 mins
        - can fit 3-4 large models more

- task for tmr
    - train longformer, this time wiht dtype=bfloat16
    - decide whether to replace 181 with proper one

# 6.21

- **exp188**
	- longformer-large
	- 24-layer, 1024-hidden, 16-heads, ~435M parameters
	- start: CU 162.35 at 6/21, 15:07
	- end: CU 82.43 at 6/21, 21:56
	- used: CU 79.92 took 6 hrs 49 mins
	- [RESULT] 4 folds oof CV: 0.8427, LB: 0.798

- maybe I don't need mean pooling mse?
	- huberloss is almost the same except it will penelize heavily when loss goes high (like mae)
	- in general, it will act like mse

- when kaggle gpu reset
	- train deberta-v3-large (cls) (this will use around 16-17 of gpu hours)
    	- use 1024 max len for speed (maximize effiency)

- potential improve over lgbm
	- try different hyperameters on lgbm
		- train tonight
	- try universal sentence encoder
	- add soft label to fb3 train data
		- add to efficiency track lgbm

- task for tmr
	- train deberta-v3-large (cls)

# 6.22

> - **exp189**
>   - bigbird-roberta-large
>   - this time wiht bfloat16
>     - attention_type = 'original_full' is not working
>     - try sparse_block
>     - still no working, training loss constantly going up
>       - maybe because mask format is `<>` while others is `[]`
>   - [RESULT] 4 folds oof CV: , LB:

> - **exp190**
>   - try classification model
>   - multi-dropout on
>   - xsmall model
>   - [RESULT] fold0 CV: 0.8136, LB: 0.782

> - **exp191**
>   - try 'funnel-transformer/small'
>   - diff lr e: 1-e5, d: 2-e5
>   - [RESULT] fold0 CV: 0.83381, LB: 0.798

> - **exp192**
>   - try 'funnel-transformer/small'
>   - diff lr e: 1-e5, d: 1-e5
>   - [RESULT] fold0 CV: 0.8341, LB: 0.800
>   - [RESULT] fold2 CV: 0.8373, LB:

- task for tmr
    - train funnel-transformer
        - done
    - convert .bin to .plk for eff track
        - done
        - not submitted yet
    - check data diff in merge_data notebook for eff
        - checked

> - **exp193**
>   - funnel-transformer/large
>   - 26 layers: 3 blocks of 8 layers then 2 layers decoder, 1024-hidden, 12-heads, 386M parameters
>   - start: CU 70.89 6/23 at 12:50
>   - end: CU 41.30 6/23 at 15:23
>   - used: CU 29.59 took 2 hrs and 32 mins
>   - [RESULT] 4folds CV: 0.8361, LB: 0.796

# 6.23

- try classication

> - **exp194**
>   - diff lr e: 1e-5, d: 2e-5
>   - max_len 1024
>   - multidropout
>   - [RESULT] fold0 CV: 0.8287, LB: 0.793

> - **exp195**
>   - diff lr e: 1e-5, d: 2e-5
>   - max_len 2048
>   - multidrpopout
>   - concat pooling
>   - [RESULT] fold0 CV: 0.82885, LB: 0.796

- funnel-transformer/large did not increase CV nor LB
- what's left

  - cls model (deberta-v3-large concat)
    - kaggle 2 folds per run
  - deberta-v3-base
    - next week?
  - if time left try out oof from persuade1.0

- task for tmr
  - submit exp195 see result
    - decide whether to train large model following the setup
  - submit eff
  - train longformer-base
    - this will use up all colab credit

# 6.24

- mixing different pooling layer seems interesting

> - **exp196**
>   - deberta-v3-large
>   - classification
>   - concat last 4 layers
>   - [RESULT] 4 folds CV: 0.8416, LB: 0.805

- when training lgbm included NOL persuade

  - validation should not include NOL persuade?

    - since it can train on cpu, take a bit of time should not be a problem

  - so it would look like this

    - train set
      - default[1, 2, 3] + NOL_persuade[1, 2, 3]
    - valid set
      - default[0]

  - doing this to prevent from over optimistic CV and leakage

- if train base model on kaggle it would take around 2 hours per fold

  - 8 hrs for the model
  - it'd better use another pooling layer
    - attention?
    - was thinking about LSTM but it gave not so good result on CV

- be careful with order of features in lgbm

- task for tmr
  - try including NOL persuade in lgbm
  - looking into optim notebook for next model
    - maybe deberta-base? or longformer-base?
  - submit new eff
    - done LB: 0.805
      - stick to the last one

# 6.25

- maybe there is bug where selected_features.yaml is loaded, the order of features is also swapped
  - importance features go up to the first of the row
  - when using non-selective features
    - the order will follow generating order
  - when using seletive features
    - input order will follow selected_features.yaml
    - model will take what ever will take what ever it get because it is numpy
    - to make ensure the safety, predictor needs validate_features=True on

> - **exp197**
>   - deberta-base
>   - max_len 1024
>   - attention pooling
>   - [RESULT] 4 folds CV: 0.8363, LB: 0.804

- task for tmr
  - combine exp197 and submit
  - explore more about ensemble
  - note for last submit
    - 1 for eff
    - 1 for highest CV
    - 1 for highest LB

# 6.26

- when add cls model

  - the CV is increase
  - but LB is slightly decrease

- recap stacking (all of these model included fb3 + dt)
  - 181 + 186 + 187 + 188
    - (s) CV: 0.85574, LB: 0.823
    - (ns) CV: 0.85725, LB: 0.823 (higher pos) (currently highest LB)

  - 181 + 186 + 187 + 188 (clip)
    - (s) CV: 0.85622, LB:
    - (ns) CV: 0.85772, LB: 0.821
      - forgot to clip in infer notebook?
      - resub tmr?

  - 181 + 186 + 187 + 188 (new_vec)
    - (s) CV: 0.85610, LB:
    - (ns) CV: 0.85526, LB:

  - when included exp196 (cls) model
    - 181 + 186 + 187 + 188 + 196
      - (s) CV: 0.85866, LB: 0.819
      - (ns) CV: 0.85552, LB: 0.816
    - 181 + 186 + 187 + 188 + 196 (up_vocab)
      - (s) CV: 0.85912, LB: 0.820
      - (ns) CV: 0.85764, LB:

- try stacking without fb3 or without dt ot without any of those

  - some reported that fb3 increase CV but decrease LB
  - (s) 18_000
  - dt + 181 + 186 + 187 + 188
    - (s) CV: 0.85718, LB: 0.821
    - (ns) CV: 0.85536, LB:

  - fb3 + 181 + 186 + 187 + 188
    - (s) CV: 0.85593, LB: 0.821
    - (ns) CV: 0.85436, LB:

  - pure + 181 + 186 + 187 + 188
    - (s) CV: 0.85654, LB: 0.818
    - (ns) CV: 0.85537, LB:

- today submit

  - train both ns and s, then choose one with higher
    - as its always get better result

  1. fb3 + dt + 181 + 186 + 187 + 188 (ns) (clip) - submitted
  2. dt + 181 + 186 + 187 + 188 (s) - submitted
  3. fb3 + 181 + 186 + 187 + 188 (ns or s)? - submitted
  4. pure_feats + 181 + 186 + 187 + 188 - submitted
  5. fb3 + dt + 181 + 186 + 187 + 188 (new_vec) - submitted

- if clipped model get better result then re-train lgbm with clipped method

  - maybe have to do it tmr

- might have to fix count vector and tf-idf later, but stick to this for a while just for comparison

  - or remove them altogether
  - there were report that almot half these features have zero features_important
  - or even input word an char vector

- task for tmr
  - resub oof clip model
    - done - nothing changed

# 6.27

- need to test how effective count_vec and tfidf_vec

- fb3 + dt + 181 + 186 + 187 + 188 (16_500)
  - CV: 0.85816, LB: 0.820

- fb3 + dt + 181 + 186 + 187 + 188 + 197 (no_vec)
  - CV: 0.85911, LB: 0.823

- without count_vec and tfidf_vec
  - fb3 + dt + 181 + 186 + 187 + 188
    - (ns) CV: 0.85823, LB: 0.820

  - fb3 + dt + 181 + 186 + 187 + 188 + 197
    - (ns) CV: 0.85911, LB: 0.823

- last 2 submits of today

  - fb3 + dt + 181 + 186 + 187 + 188 + 197 (tfidf (3, 6) only)
    - CV: 0.85736, LB: 0.823
  - fb3 + dt + 181 + 186 + 187 + 188 + 197 (count (2, 3) only)
    - CV: 0.85741, LB: 0.824 (close to 0.825)

- fb3 + dt + 181 + 186 + 187 + 188 + 197 (new_count (1, 1) only)
  - CV: 0.85951, LB:

# 6.28

- 4 days left
- deadline is 3 july 2024, 06.59
  - so last selected should be done in 2 july midnight

- what to sub today?

  1. fb3 + dt + 181 + 186 + 187 + 188 + 197 (new_count (1, 1) only)
     - CV: 0.85951, LB: 0.822

  2. fb3 + dt + 181 + 186 + 187 + 188 + 197 (count (1, 3) only)
     - CV: 0.85862, LB: 0.821

  3. fb3 + dt + 181 + 186 + 187 + 188 + 197 (new_tfidf (1, 5) max=100)
     - CV: 0.85867, LB: 0.823

  4. fb3 + dt + 181 + 186 + 187 + 188 + 197 (tfidf (3, 6) max=300)
     - CV: 0.85828, LB: 0.822

  5. fb3 + dt + 181 + 186 + 187 + 188 + 197 (count (2, 3)) (up_v)
     - CV: 0.85838, LB:

- Not sure why incorrected one is better than corrected one?

  - minimize features number using max_features help a lot
    - new_count (1, 1)
      - CV: 0.85951, LB: 0.822
    - new_tfidf (1, 5) max=100
      - CV: 0.85867, LB: 0.823

- just realized that exp197 did not submit yet
  - submit if slot left, or maybe after competition end

- today I just submit all type of vec
  - all of theem generated and selected by the best CV score
    - potential ovefit can not ignore
  - let's see how they perform
  - up vocab is not used yet

- try train 5 epoch of small model
  - concat 4 lasy layers
  - train on gpu reset

- task for tmr
  - decided what to do next
  - explore more

# 6.29

- have not tried submit + 197 + 196

- today submit

  1. fb3 + dt + 181 + 186 + 187 + 188 + 197 (optim)
     - [1.4990845648767515, 2.5000698104807086, 3.499945414943945, 4.4996127434483135, 5.509749609049589]
       - from 0.8591147 to 0.8591752 (+ 0.00006)
       - LB: 0.823 (lower than original)
  2. fb3 + dt + 181 + 186 + 187 + 188 + 197 + 196
     - it give better on f1 score when added 196(cls model)
     - CV: 0.85777, LB: 0.818
  3. exp197_4f
  4. fb3 + dt + 181 + 186 + 187 + 188 + 197 (ns) both vec
     - CV: , LB: 0.822

- task for tmr
  - explore nol data lgbm
  - check on exp94 for non-tokened-text possiblity

# 6.30

- it is pretty clear that using vectorizers is not very good idea due to
  - limited range of vocabulary

- **exp199**
  - deberta-v3-large
  - 2 epoch
  - [RESULT] 4 folds CV: 0.84483, LB:

- **exp200**
  - deberta-v3-base
  - full_text as target_col
  - [RESULT] 4 folds CV: 0.824406, LB:

- **exp201**
  - longformer-base
  - [RESULT] 4 folds CV: 0.82978, LB:

- to maximize dt_count potential
  - dt_count per paragraph
  - mean dt_count per paragraph
  - sum dt_count per parapgraph
  - not helping very much

- wait for nol_oof
  - nol_oof not work

- ran out of ideas
- submit some combinations of models

  1. fb3+dt+181+186+187+188+197+198 (no_vec)
    - CV: 0.85844, LB: 0.819
  2. fb3+dt+181+186+187+188+197+199 (no_vec)
    - CV: 0.85761, LB: 0.820
  3. fb3+dt+181+186+187+188+197+201 (no_vec)
    - CV: 0.85719, LB: 0.822
  4. fb3+dt+181+186+187+188+197+183 (no_vec)
    - CV: 0.85759, LB: 0.820
  5. fb3+dt+181+186+187+188+197+193 (no_vec)
    - CV: 0.85835, LB: 0.821

# 7.1

- last 2 days

>- **exp202**
>    - roberta-large
>    - [RESULT] 4 folds CV: 0.8325, LB:

>- **exp203**
>    - roberta-base
>    - [RESULT] 4 folds CV: 0.8162, LB:

>- **exp204**
>    - distilbert-base-uncased
>    - [RESULT] 4 folds CV: 0.7958, LB:

# 7.2

- last day

- try ensembling didfferent seed of lgbm (just the model)

# 7.3
- competition ended
- public 21st/ private 15th
- gold medal

CV table in markdown format
| model_num | model_name       | note                      | fold_0  | fold_1  | fold_2  | fold_3  | oof     | public |
|-----------|------------------|---------------------------|---------|---------|---------|---------|---------|--------|
| exp181    | deberta-v3-large | huber_loss, lr_decay 0.95 | 0.83637 | 0.84654 | 0.84660 | 0.84487 | 0.84362 | 0.802  |
| exp186    | deberta-large    |                           | 0.84634 | 0.84623 | 0.85516 | 0.84567 | 0.84839 | 0.797  |
| exp187    | deberta-v3-large | gem pooling               | 0.84229 | 0.84475 | 0.85143 | 0.84234 | 0.84523 | 0.799  |
| exp188    | longformer-large |                           | 0.83744 | 0.83544 | 0.84183 | 0.82966 | 0.83610 | 0.798  |
| exp197    | deberta-base     | attention pooling         | 0.83544 | 0.83634 | 0.84462 | 0.82870 | 0.83633 | 0.804  |


# 7.4

- Tried CV strategy of the top winners
    - pretrain on not-kaggle-only data
    - finetune with kaggle-only data

- **exp205**
    - small model
    - 2 stage training
    - higher lr on finetune
    - [RESULT] fold0
        - CV: 0.79480
        - LB: 0.78122
        - PB: 0.79880

- **exp206**
    - base model
    - 2 stage training
    - higher lr on finetune
        - [RESULT] fold0
            - CV: 0.79876
            - LB: 0.77714
            - PB: 0.79596

    - lower lr on finetune
        - [RESULT] fold0
            - CV: 0.79959
            - LB: 0.78367
            - PB: 0.80570
    

- re-calculate the CV of the current rank (15th) 
    - using kaggle-only: 0.81440
    - using not-kaggle-only: 0.871564