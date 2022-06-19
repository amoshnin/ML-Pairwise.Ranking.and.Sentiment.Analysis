import pandas as pd
from copy import deepcopy
from joblib import load, dump
from statistics import mean
from collections import Counter
import sys

sys.path.append('./utils')
from utils import review_feature
import argparse
import time

rf = review_feature()

start = time.time()
parser = argparse.ArgumentParser(description='Ranking Reviews')
parser.add_argument('--spell_threshold', action='store', type=int, default=0.9, help='Spell checking threshold')
parser.add_argument('--model_path', action='store', type=str, default='randomforest.joblib', help='Model Path')
parser.add_argument('--file_name', action='store', type=str, default='data/test.csv',
                    help='File to Rank (Product, Review)')
parser.add_argument('--testing', action='store', type=str, default='False', help='Get Ranking Score Test or Not (True/False)')
args = parser.parse_args()

classifier = load(args.model_path)
df = pd.read_csv(args.file_name)
print(df.head(5))


bad_reviews = set()
language_error = set()
gibberish = set()
swear = set()
company_tag = set()
for indx in df.index:
    review = df.at[indx, 'answer_option']
    ## Language Detection
    try:
        b = rf.language_detection(review)
        if b == 'hi' or b == 'mr':
            language_error.add(indx)
    except:
        language_error.add(indx)

    ## Gibberish Detection
    if rf.gibberish_detection(review, prefix_path='utils'):
        gibberish.add(indx)

    ## Swear Words Check
    if rf.english_swear_check(review) or rf.hindi_swear_check(review):
        swear.add(indx)

    ## Identify reviews on Competitive Brands
    if rf.competitive_brand_tag(review):
        company_tag.add(indx)

print("Number of Bad Reviews for Language Error: {} \n Number of Bad Reviews for Gibberish: {} \n Number of Bad Reviews for Swear: {} \n Number of Bad Reviews for Competitive Brand: {}".format(len(language_error), len(gibberish), len(swear), len(company_tag)))
bad_reviews = list(bad_reviews.union(swear, company_tag, gibberish, language_error))
print("DELETED REVIEWS: \n", df[df.index.isin(bad_reviews)])

df = df[~df.index.isin(bad_reviews)].reset_index(drop=True)

# ## Spell Correct
# for indx in df.index:
#     review = df.at[indx, 'answer_option']
#     try:
#         df.at[indx, 'answer_option'] = rf.spell_correct(review, args.spell_threshold)
#     except:
#         bad_reviews.append(indx)
# df = df[~df.index.isin(bad_reviews)].reset_index(drop=True)
# print("Stage Spell Correction Complete")

df = df.sort_values(by=['product'], ignore_index=True)

df['review_len'] = df['answer_option'].apply(lambda x: len(x.split()))
df['Rn'] = 0.0
df['Rp'] = 0.0
df['Rs'] = 0.0
df['Rc'] = 0.0
df['Rd'] = 0.0
df['Rsc'] = 0.0

product_list = df['product'].unique()
for product in product_list:
    data = df[df['product'] == product]
    unique_bag = set()
    for review in data['answer_option']:
        review = review.lower()
        words = review.split()
        unique_bag = unique_bag.union(set(words))

    for indx in data.index:
        review = data.at[indx, 'answer_option']
        df.at[indx, 'Rp'] = rf.polarity_sentiment(review)
        df.at[indx, 'Rs'] = rf.subjectivity_sentiment(review)
        df.at[indx, 'Rd'] = rf.service_tag(review)
        df.at[indx, 'Rsc'] = rf.slang_emoji_polarity_compoundscore(review)
        df.at[indx, 'Rc'] = float(len(set(review.split()))) / float(len(unique_bag))

    df.loc[df['product'] == product, 'Rn'] = rf.noun_score(data['answer_option'].values).values

product_list = df['product'].unique()
df['win'] = 0
df['lose'] = 0
df['review_score'] = 0.0
df.reset_index(inplace=True, drop=True)


def score_giver(C, D):
    E = pd.merge(C, D, how='outer', on='j')
    E.drop(columns=['j'], inplace=True)
    q = classifier.predict(E.values)
    return Counter(q)


for i, product in enumerate(product_list):
    data = df[df['product'] == product]
    for indx in data.index:
        review = df.loc[indx, ['review_len', 'Rn', 'Rp', 'Rs', 'Rc', 'Rd', 'Rsc']]
        review['j'] = 'jn'
        C = pd.DataFrame([review])
        D = data[data.index != indx].loc[:, ['review_len', 'Rn', 'Rp', 'Rs', 'Rc', 'Rd', 'Rsc']]
        D['j'] = 'jn'
        E = pd.merge(C, D, how='outer', on='j')
        score = score_giver(C, D)
        df.at[indx, 'win'] = 0 if score.get(1) is None else score.get(1)
        df.at[indx, 'lose'] = 0 if score.get(0) is None else score.get(0)
        df.at[indx, 'review_score'] = float(0 if score.get(1) is None else score.get(1)) / len(data) * 1.0
    print("Iteration: {} Reviews of Product: {} Ranked".format(i + 1, product))

df = df.sort_values(by=['product', 'review_score'], ascending=False)

if args.testing == 'True':
    data_split = pd.crosstab(df['product'], df['label'])
    r_accuracy = []
    for product in product_list:
        x = data_split[data_split.index == product][1][0]
        number_of_1_in_x = Counter(df[df['product'] == product].iloc[:x, ]['label']).get(1)
        rank_accuracy = float(number_of_1_in_x * 1.0 / x * 1.0)
        print("Product: {} | Rank Accuracy: {}".format(product, rank_accuracy))
        r_accuracy.append(rank_accuracy)
    print("TEST DATA: Mean Rank Accuracy: {}".format(mean(r_accuracy)))

print(df[['product', 'answer_option', 'review_score']])
df[['product', 'answer_option', 'review_score']].to_csv('data/test_ranked_output.csv', index=False)
print('RANKING COMPLETE TIME TAKEN: {}'.format(time.time() - start))
