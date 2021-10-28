from sklearn.metrics.pairwise import pairwise_distances
import joblib
from firebase_admin import storage
import datetime
import artifacts.firebase_config as fb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import dal
from geopy.distance import geodesic
import cv2
import base64
import pywt
import numpy as np


# designs


def get_similar_designs(gender, event, dress_type, article_type, img_url):
    skin_col = -1
    idx_rec_modified = []
    if img_url != '':
        skin_col = detect_color(img_url)[0]
    print(skin_col)
    idx = dal.get_base_design(gender, event, dress_type, article_type, skin_col)
    if idx != -1:
        print('idx', idx)
        df_embs = joblib.load('artifacts/cnnembmodel.pkl')
        cosine_sim = 1 - pairwise_distances(df_embs, metric='cosine')
        sim_scores = list(enumerate(cosine_sim[idx - 1]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:7]
        idx_rec = [i[0] for i in sim_scores]
        idx_sim = [i[1] for i in sim_scores]
        print('id_rec', idx_rec)
        for idrec in idx_rec:
            if dal.get_gender_by_id(idrec) == gender:
                idx_rec_modified.append(idrec)
        print('idx_rec_modified', idx_rec_modified)
        urls = get_similar_design_urls(idx_rec_modified)
        return urls, idx_sim
    else:
        return idx


def get_similar_design_urls(idx_rec):
    urlList = []
    for id in idx_rec:
        urlList.append(get_image_url(str(id + 1) + ".jpg"))
        print(get_image_url(str(id + 1) + ".jpg"))
    print(idx_rec)
    return urlList


def get_image_url(img_name):
    img_path = "images_dir/" + img_name
    bucket = storage.bucket(app=fb.fb_init())
    blob = bucket.blob(img_path)
    # blob.make_public()
    """
    print(
        "Blob {} is publicly accessible at {}".format(
            blob.name, blob.public_url
        )
    )
    """
    # print(blob.generate_signed_url(datetime.timedelta(days=2), method='GET'))
    return blob.generate_signed_url(datetime.timedelta(days=2), method='GET')


# designers

df = dal.get_designers_df()


# df = pd.read_csv("artifacts/designers.csv")
# current_loc = (5.946,80.534)
def get_data_by_id(id):
    return (df[df.id == id]["designer_name"].values[0], df[df.id == id]["location"].values[0],
            df[df.id == id]["popularity"].values[0])


def get_id_by_tag(tag):
    try:
        return df[df.tag == tag]["id"].values[0]
    except:
        print('Except')


def get_latlng_by_id(id):
    return id, (df[df.id == id]["lat"].values[0], df[df.id == id]["lng"].values[0])


def get_similar_designers(gender, event, dress_type, lat, lng):
    current_loc = (lat, lng)
    tag = gender + ' ' + event + ' ' + dress_type
    tag = tag.strip()
    designers_list = []
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["tag"])
    cosine_sim = cosine_similarity(count_matrix)
    designer_index = get_id_by_tag(tag)
    print('designer_index : ', designer_index)
    if designer_index is None:
        return -1
    else:
        similar_designers = list(enumerate(cosine_sim[designer_index]))
        sorted_similar_designers = sorted(similar_designers, key=lambda x: x[1], reverse=True)

        distance_list = []
        for element in sorted_similar_designers[:10]:
            distance_list.append((get_latlng_by_id(element[0])[0], geodesic(current_loc, get_latlng_by_id(element[0])[1])))
        sorted_distance_list = sorted(distance_list, key=lambda x: x[1])
        print(sorted_distance_list)
        for des in sorted_distance_list:
            designers_list.append(get_data_by_id(des[0]))
        print(designers_list)
        return designers_list


def detect_color(img_url):
    try:
        skin_color_model = joblib.load('artifacts/skin_color_dector_model.pkl')
        data_uri = img_url
        img = data_uri_to_cv2_img(data_uri)
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        X = np.array(combined_img).reshape(1, 4096).astype(float)
        return skin_color_model.predict(X)
    except:
        return -1


def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255;
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0;
    imArray_H = pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H = np.uint8(imArray_H)
    return imArray_H
