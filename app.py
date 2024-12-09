from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from werkzeug.utils import secure_filename
from mlxtend.frequent_patterns import apriori, association_rules
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded!"
        file = request.files['file']
        if file.filename == '':
            return "No selected file!"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                data = pd.read_csv(file_path)
                results = run_apriori(data)
                return render_template('results.html', results=results)
            except Exception as e:
                return f"Error processing file: {e}"
    return redirect(url_for('home'))

def run_apriori(data):
    """
    Run the Apriori algorithm and return results.
    """
    frequent_itemsets = apriori(data, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    rules['support'] = rules['support'].round(2)
    rules['confidence'] = rules['confidence'].round(2)
    rules['lift'] = rules['lift'].round(2)

    best_lift = rules.loc[rules['lift'].idxmax()].to_dict()
    best_confidence = rules.loc[rules['confidence'].idxmax()].to_dict()
    best_support = rules.loc[rules['support'].idxmax()].to_dict()

    top_support = rules.nlargest(10, 'support').to_dict(orient='records')
    top_confidence = rules.nlargest(10, 'confidence').to_dict(orient='records')
    top_lift = rules.nlargest(10, 'lift').to_dict(orient='records')

    return {
        "rules": rules.to_dict(orient='records'),
        "best_lift": best_lift,
        "best_confidence": best_confidence,
        "best_support": best_support,
        "top_support": top_support,
        "top_confidence": top_confidence,
        "top_lift": top_lift,
    }

if __name__ == '__main__':
    app.run(debug=True)
