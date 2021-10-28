import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("artifacts/serviceAccountKey.json")
app = firebase_admin.initialize_app(cred, {
    'storageBucket': 'fashion-recommendation-cade6.appspot.com',
}, name='storage')


def fb_init():
    return app
