# Methodology
## What worked

- start with small model for fast iteration

- write almost complete pipeline with scalability in mind
    - make modular so it it easy to make change
    - easy to just copy code to colab with minimal additional code

- name experiment with expXXX annd special experiments with tag
    - using based: in record is good when trying to backtrack the experiment

- taking note, everything in mind

- note for tomorrow task to remind what to do first before next moving to next thing

- read discussion and do experiment by myself 

- gather information and idea from the similar past compeitions.

- spend a lot of time doing EDA
    - but need to focus more on goal, "What am I looking for?" kind of question

- get outside of comfortzone to learn new things

- ask question without fear of being judged


## What did not worked

- experiments records is a bit messy
    - mediocre at best
    - but this is probably better than using spreadsheet or some table
        - because it is easy to add extra detail because of its flexibity

- unhealthy
    - pushing too hard on sleep schedule
    - not so much exercise
    - drinking too less of water


# What I could have done better

- take interval time off, reduce stress

- pacing
    - good overall
        - but a bit too rushing and hustle in the last week
        - should give it a bit more time at the last part

- going forward without stable CV
    - this part is hard to fix as it might take a lot of time

- pipeline still a bit too manual like need change of exp name everytime
    - sometime I forget to change

- should not have give up easily on some experiment
    - pretrain MLM
    - Ordinal Regression Loss
    - fixing: write-up what to expect when experiment is done


# What others done and worked

## 1st place solution

    This made me believe there were actual differences in the criteria that the raters used.
    (~13k Persuade essays ("old") and 4.5k unseen ("new") essays in the competition data.)
    Since the new data was clearly using slightly different scoring criteria, I set up my training as a pre-train -> fine-tune, two-staged process. Pre-train on the old data (+ maybe aux data) and fine-tune on the new. This gave a boost of at least 0.015 on public LB and made CV-LB-correlation much better.
    Within each stage I used stratified 5-fold split using prompt_id+score as labels.
    
    risk of using threshold
    1. (over-) fit the targets
    2. correct errors caused by impalanced score labels 
        - it would lean toward more to 3 because sample number is greater than 1
    3. Optimize quadratic weighted cohen's kappa (QWK).
        - moving predicions a bit could increase score but it will learn toward minority class 

- [Link](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)

- differentiate data from each others
    - old data -> essays came from Persaude
    - new data -> essays that only appear in kaggle
    - because they have different distribution and pottentially using **different criteria for rating**
    - different criteria assumption in this appoarch is very important because it help author distinguishing and treating data differently
    - for better correlation and better score as it has similar distrition to new data 

- train processs
    1. pre-tain with old data
    2. finetune with new data

- using StratifiedKFold 5 folds based on prompt_name + score as labels 
    - when splits only?

- Learn rate
    - diffrentiate LR
    - using half of the LR when finetune
    - pretrain
        - encoder 1e-5
        - decoder 2e-5
    - finetune
        - encoder 5-e6
        - decoder 1e-5

- Pseudo Labelling
    - predict new score for old data using ensemble of new data fine-tuned models
    - use soft label
    - persumably, this help realigning/adjusting criteria of the old data to new data

- Threshold Optimizing
    - risky to use because it significantly increases bias in predictions due to the imbalance sample class
        - There were more samples for 3 and 4, thus the optimizing method would learn toward the majority class
    - author used 3 differnt seeds to calculate threshold
    - for method, `scikit.optimize` with `Powell` on targe `1 - qwk` (minizing)

- Ensembling
    - simple average was safe to use rather than pushing it

## 2nd place solution
    1. Two-Stage Training. Training on Kaggle-Persuade first, load the weights and training on Kaggle-Only with same loss function.
    2. 10 epochs of MLM on all the training set.
    3. CV: 4 folds SKF. Validate on both Kaggle-Persuade and Kaggle-Only data.

- [Link](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516790)

- 2 stage training
    - pretrain on kaggle-persuade-overlapped data
    - fine-tune with kaggle-only data

- Pretrain MLM
    - 10 epoch of MLM

- CV strategy
    - StratifiedKFold 4 folds
    - validating on both kaggle-persaude-overlapped and kaggle-only

- Using Ordinal Regression Loss

- Optimizing threshold on the full training dataset rather than kaggle-only
    - author suspect that test dataset distribution is somewhat similar to


## 3rd place solution
    CV: 0.836
    Public: 0.832
    Private: 0.839

    - Pre-training with MLM
    - Two-stage training:
        First stage: Train with Kaggle data and validate with Kaggle-only data.
        Second stage: Load the weights from the first stage model, train on the Kaggle-only data, and validate on the non-Kaggle-only data.
        　(Validation used data other than the Kaggle-only data to avoid overfitting to the Kaggle-only data)
    - Ensemble using the following combinations of backbone, head, and max_length:
        Backbone: DeBERTa-v3-base / DeBERTa-v3-large
        Head: Mean&Pool / Attention / LSTM
        Max_length: 1024 / 1536



- [Link](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516790)

- a lot of things similar to 2nd place solution except validation on non-kaggle-only data
    - to prevent overfitting

## 4th place solution

- [Link](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516639)

- author did some experiments and developed hypothesis that two data sources are different
    - conclusion, mix them would make model not properly fit

- author addd data source tags to distinguish orgin of the essay
    - `[A] {full_text}` for kaggle-only data
    - `[B] {full_text}` for not-kaggle-only data

- modeling
    - add classification head for data source
    - validating based on kaggle-only data 
    
### Dynamic Micro Batch Collation
        Since text lengths are variable, to reduce the impact of padding, I split a batch into micro batches with a limit of total number of tokens. This makes training faster and memory usage stable.
        The gradient accumulation training loop looks like this:
```python
        for batch in dataloader:
            optimizer.zero_grad()
            total_batch_size = sum(micro_batch["batch_size"] for micro_batch in batch)
            for micro_batch in batch:
                micro_batch = to_gpu(micro_batch)
                loss = model(micro_batch)
                scale = micro_batch["batch_size"] / total_batch_size
                (loss * scale).backward()
        optimizer.step()
```

## Others
- thresholds for each prompt_name
> - I tuned the thresholds directly on public LB. Here is a systematic way to do it.
>    - First set all 5 prompts to [1.5, 2.5, 3.5, 4.5, 5.5]
>    - On first day make 5 submissions. In sub 1 change 1st prompt to [1.6, 2.5, 3.5, 4.5, 5.5]. In sub 2 change 2nd prompt to [1.6, 2.5, 3.5, 4.5, 5.5].
>    - On second day, all subs that improved LB move the threshold more in same direction. All subs that got worse on LB move the thresholds the other way like [1.4, 2.5, 3.5, 4.5, 5.5].
>    - On third day make 5 submissions. In sub 1 change 1st prompt to [1.5, 2.5, 3.5, 4.5, 5.4] etc etc
>    - On fourth day, all subs that improved LB more the threshold more in same direction. All subs that got worse on LB more the thresholds the other way.

> Then on days 5-6 we focus on [1.5, XXX, 3.5, 4.5, 5.5]. And on days 7-8 we focus on [1.5, 2.5, 3.5, XXX, 5.5]. By the end of the week we have LB 0.827 on public LB 🎉

# What I should do for my next competitions

- put priority more no health
    - good sleep come with good decision making
    - less bugs as focus increase

- colab is a bit too expensive
    - maybe change cloud
        - lambdalab
        - vast.ai

- exploring more, learn more about similarity between LB and CV

