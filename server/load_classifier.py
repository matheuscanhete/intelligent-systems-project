from scipy.sparse import hstack
import pickle
import os

class CategoryCLF:
    """
    this object was meant to facilitate the readability of the code
    in a way that in the API code only one function is called
    """
    def __init__(self):
        self.model_path = os.environ["MODEL_PATH"]
        self.count_vec_path = os.environ["COUNT_VEC_PATH"]
        self.label_enc_path = os.environ["LABEL_ENC_PATH"]

    def predict_cat(self, tags: str, title: str) -> str:
        """
        method to predict the best category for the given data
        """
        model, count_vec, label_enc = self.load_model()

        x_tags = count_vec.transform([tags])
        x_title = count_vec.transform([title])

        x_array = hstack((x_tags, x_title))

        predicted = model.predict(x_array)

        predicted = label_enc.inverse_transform(predicted)[0]

        return predicted

    def load_model(self):
        """
        method to load the model trained at '../training/trainer.ipynb'
        and the count vectorizer used to tokenize the new provided
        data
        """
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        with open(self.count_vec_path, "rb") as f:
            count_vec = pickle.load(f)

        with open(self.label_enc_path, "rb") as f:
            label_enc = pickle.load(f)

        return model, count_vec, label_enc

