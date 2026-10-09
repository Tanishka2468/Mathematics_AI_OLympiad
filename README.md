# Math Problem Topic Classifier: KDD Project Dashboard

An interactive dashboard for the KDD project that predicts the topic of a math problem (Algebra, Geometry, Number Theory, and four more) from its text.

**What it includes**
- Predict the topic of a new problem with all four algorithms (Logistic Regression, Linear SVM, Naive Bayes, Random Forest), then an analysis of the most accurate one: confidence, agreement between models, how reliable that topic is, and the words that drove the result
- Batch prediction for many problems at once, with agreement between algorithms and CSV download
- The full KDD report: cleaning, selection, transformation, four algorithms, interpretation, clustering
- Model comparison, per-topic scores, confusion matrix, dataset overview, top words per topic

The dashboard is one file (`index.html`). All four trained models are embedded in it (about 7.5 MB) and run in the browser, so no server is needed. The page takes a few seconds to load the first time.

## Publish on GitHub Pages
1. Create a new repository on GitHub and upload `index.html` and this `README.md`.
2. Open **Settings > Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**, select the `main` branch and the `/ (root)` folder, then save.
4. After a minute your site is live at `https://<your-username>.github.io/<repository-name>/`.

## Results
Logistic Regression 75.2% accuracy, Linear SVM 74.9%, Random Forest 74.5%, Naive Bayes 71.7% (20% held-out test set, 5-fold CV alongside).

## Dataset
ML_PROJECT.csv (MATH + GSM8K). Only the labelled MATH rows (about 12,500 problems) are used for training.
