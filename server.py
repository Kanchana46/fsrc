from flask import Flask, request, jsonify
import recommender
import artifacts.firebase_config

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return 'fs'


@app.route('/v1/getSimilarDesigns', methods=['GET', 'POST'])
def get_similar_designs():
    try:
        content = request.json
        gender = content["gender"]
        event = content["event"]
        dress_type = content["dress_type"]
        article_type = content["article_type"]
        img_url = content["img_url"]
        res = recommender.get_similar_designs(gender, event, dress_type, article_type, img_url)
        if res != -1:
            response = jsonify(
                payload=str(res),
                status=200,
                mimetype='application/json'
            )
        else:
            response = jsonify(
                payload='No matches',
                status=500,
                mimetype='application/json'
            )
        return response
    except:
        return jsonify(
            payload='Server Error',
            status=500,
            mimetype='application/json'
        )


@app.route('/v1/getSimilarDesigners', methods=['GET', 'POST'])
def get_similar_designers():
    try:
        content = request.json
        gender = content["gender"]
        event = content["event"]
        dress_type = content["dress_type"]
        lat = content["lat"]
        lng = content["lng"]
        res = recommender.get_similar_designers(gender, event, dress_type, lat, lng)
        if res != -1:
            response = jsonify(
                payload=str(res),
                status=200,
                mimetype='application/json'
            )
        else:
            response = jsonify(
                payload='No matches',
                status=500,
                mimetype='application/json'
            )
        return response
    except:
        return jsonify(
            payload='Server Error',
            status=500,
            mimetype='application/json'
        )


if __name__ == "__main__":
    app.run(debug=True)
