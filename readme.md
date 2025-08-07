This is the spectrum framework to evaluate nonlinear systems modeling methods and model adaptation.

Related article>>
Spectrum: A holistic evaluation framework for nonlinear system modeling methods.

DOI: 10.1109/MOCAST65744.2025.11083951

# Installation requirement:

Language: Python 3.11

# Use

Put your methods in the folder "user/".

Write functions to be called in this form: 

  ------------

  def train(data_sample_train):
    return [params_list, x_fit]

  def test(params, U_test, y, ahead_step):
    return yhat

  ------------

Start with the file "framework_start.py" for method training and test.






