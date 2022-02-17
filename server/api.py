from flask import Flask, request, jsonify, make_response, abort
from load_classifier import CategoryCLF

api = Flask(__name__)

@api.route("/v1/categorize", methods = ["POST"])
def api_categorize():
    """
    the API will receive a POST request containing a JSON with the labels title and tags,
    with the data provided it'll serve a category that best fit the data
    """
    model = CategoryCLF()
    all_predicted = list()

    if not request.is_json:
        r = {"message": "Input is not a JSON file"}

        res = make_response(jsonify(r), 400)

        return res

    req = request.get_json()

    for product in req.get("products"):
        if not (product.get("tags") and product.get("title")):
            r = {"message": "your input should have `title` and `tags` keys for each product"}

            res = make_response(jsonify(r), 400)

            return res

        predicted = model.predict_cat(
            product.get("tags"),
            product.get("title")
        )

        all_predicted.append(predicted)

    r = {"categories": all_predicted}

    res = make_response(jsonify(r), 200)

    return res

if __name__ == "__main__":
    api.run()
