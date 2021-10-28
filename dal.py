import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
import random

cred = credentials.Certificate("artifacts/serviceAccountKey.json")

firebase_admin.initialize_app(cred)

db = firestore.client()
fashions = db.collection('fashion_designers')
designs = db.collection('fashions_data')


def get_designers_df():
    designers_dict = list(map(lambda x: x.to_dict(), list(fashions.stream())))
    df = pd.DataFrame(designers_dict)
    df['tag'] = df['tag'].apply(lambda x: x.strip())
    sorted_df = df.sort_values(by=['id'], ascending=True)
    print(sorted_df)
    return sorted_df


def get_base_design(gender, event, dress_type, article_type, skin_color):
    smilar_designs = []
    if gender != '':
        docs = designs. \
            where("gender", "==", gender)
    if event != '':
        docs = docs. \
            where("event", "==", event)
    if dress_type != '':
        docs = docs. \
            where("dress_type", "==", dress_type)
    if article_type != '':
        docs = docs. \
            where("article_type", "==", article_type)
    if skin_color != -1:
        docs = docs. \
            where("skin_color", "==", int(skin_color))
    docs = docs.stream()

    for doc in docs:
        #print('{} => {} '.format(doc.id, doc.to_dict()['image']))
        smilar_designs.append(doc.to_dict()['image'])
    print(smilar_designs)
    if len(smilar_designs) > 0:
        return int(random.choice(smilar_designs).replace(".jpg", ""))
    else:
        return -1


def get_gender_by_id(id):
    docs = designs.where("id", "==", id).stream()
    gender = ''
    for doc in docs:
        gender = doc.to_dict()['gender']
    return gender



""""
df = pd.read_csv("C:/Users/user/Documents/Projects/Fashion Recommender System/Datasets/designers.csv", nrows=5000, error_bad_lines=False)
for index, row in df.iterrows():
    print(row.tag)
    doc_ref = db.collection('fashion_designers').document('designer_'+str(index))
    doc_ref.set({
        'id': row.id,
        'tag': row.tag,
        'popularity': row.popularity,
        'location': row.location,
        'lat': row.lat,
        'lng': row.lng,
        'designer_name': row.designer_name,
    })
    print('done')
"""

"""
df = pd.read_csv("C:/Users/user/Documents/Projects/Fashion Recommender System/Datasets/designs.csv", nrows=5000, error_bad_lines=False)
for index, row in df.iterrows():
    print(index)
    doc_ref = db.collection('fashions_data').document('fshn_'+str(index))
    doc_ref.set({
        'id': row.id,
        'gender': row.gender,
        'event': row.event,
        'dress_type': row.dress_type,
        'article_type': row.article_type,
        'skin_color': row.skin_color,
        'image': str(row.image) + '.jpg'
    })
    print('done')
"""
