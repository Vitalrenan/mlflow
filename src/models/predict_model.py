import mlflow
logged_model = 'runs:/25f25af2231d4f5a858d4c23eaa7ae2c/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
df = pd.read_csv('data/processed/casas.csv')
df.drop(columns=['preco'],inplace=True)
pred = loaded_model.predict(pd.DataFrame(df))

df['pred'] = pred
df.to_csv('precos2.csv')