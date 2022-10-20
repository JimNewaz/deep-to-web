from flask import Flask, render_template, request
import pickle
import model as m


# model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)



@app.route('/', methods = ["GET","POST"])
def model():
    if request.method == "POST":
        sentence = request.form['a']
        pred = m.predict_text(sentence)
        print(pred)
        score = pred

    return render_template('index.html', final=score)





if __name__ == "__main__":
    app.run(debug=True)

