# Copyright DarkStarQuantumLab, Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from src.QSVM import *

if __name__ == "__main__":
    # train the QSVM object
    N = 70
    validation_pts = 10
    QSVM_object = QSVM(file_path='data/Covid_Sym2.csv', train_record_number=N, validation_pts=validation_pts, solver_type="simulated_annealing", verbose=False)
    alpha, b = QSVM_object.train_SVM()
    QSVM_object.calculateKPI()

    # make prediction
    age = 20
    fever = True
    throat_pain = False
    taste_smell = False

    x_test = [age, fever, throat_pain, taste_smell]
    c = QSVM_object.predict(x_test)
    prediction = "Negative" if c < 0.50 else "Positive"
    print("Prediction is ", prediction)