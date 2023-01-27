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

import numpy as np
import math
import time
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from dimod.binary_quadratic_model import BinaryQuadraticModel
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave.samplers import SimulatedAnnealingSampler
from typing import List, Dict


class QSVM:
	"""A class for the Quantum Support Vector Machine (QSVM)."""
	def __init__(self, file_path:str, train_record_number:int, 
				validation_pts:int, solver_type:str, B:int = 2, K:int = 2, 
				C:int = 3, gamma: int = 20, xi:int = 0.001, 
				verbose:bool = True):
		self.train_record_number = train_record_number
		self.validation_pts = validation_pts
		self.solver_type = solver_type
		self.B = B
		self.K = K
		self.C = C
		self.verbose = verbose

		self.xi = xi
		self.gamma = gamma

		self.init_data(file_path)

	def init_data(self, file_path:str):
		"""
		Prepares dataset for QSVM model training.

		Args:
			file_path:(str) path to a  file with information about a patient's 
			health conditions (age, fever, taste, smell, etc.) 
		Returns:
			None.
        """
		print("[1] Initiating dataset...")
		self.dataframe = np.loadtxt(file_path, delimiter=',' , skiprows=1)
		for i in range(self.train_record_number+self.validation_pts):
			if(self.dataframe[i][-1] == 0):
				self.dataframe[i][-1] = -1

		self.data = self.dataframe[:self.train_record_number + self.validation_pts, :4]
		self.train_data = self.dataframe[:self.train_record_number + self.validation_pts, -1]
		
		x_min, x_max = 1000, 0
		y_min, y_max = 1000, 0
		w_min, w_max = 1000, 0
		z_min, z_max = 1000, 0
		# rescalling data
		for i in range(self.train_record_number + self.validation_pts):
			x_min = min(self.data[i][0], x_min)
			x_max = max(self.data[i][0], x_max)
			y_min = min(self.data[i][1], y_min)
			y_max = max(self.data[i][1], y_max)
			w_min = min(self.data[i][2], w_min)
			w_max = max(self.data[i][2], w_max)
			z_min = min(self.data[i][3], z_min)
			z_max = max(self.data[i][3], z_max)

		for i in range(self.train_record_number + self.validation_pts):
			self.data[i][0] = (self.data[i][0] - x_min) / (x_max - x_min)
			self.data[i][1] = (self.data[i][1] - y_min) / (y_max - y_min)
			self.data[i][2] = (self.data[i][2] - w_min) / (w_max - w_min)
			self.data[i][3] = (self.data[i][3] - z_min) / (z_max - z_min)

	def kernel(self, x:np.array, y:np.array, gamma:int):
		"""
        Constracts a kernel (RBF).

		Args:
			x: (np.array) data points.
			y: (np.array) data labels.
			gamma: (int) gamma parameter.
		Returns:
			kernel: constacted kernel
        """
		y = np.arange(4).reshape(1, 4)
		if gamma == -1:
			kernel = np.dot(x, y)
		elif gamma >= 0:
			kernel = np.exp( -gamma * (np.linalg.norm(x-y, ord=None)))
		
		return kernel

	def delta(self, i, j):
		return 1 if i == j else 0

	def train_SVM(self) -> list:
		"""
			Construct a QUBO model that is optimized on a quantum annealier device
			or as a simulated annealing algorithm. The QSVM model is trained through 
			a QUBO optimization process.

			Args:
				None

			Returns:
				alpha
				b
		"""
		print("[0] Training SVM...")
		Q_tilde = np.zeros((self.K * self.train_record_number, self.K * 
							self.train_record_number))
		
		for n in range(self.train_record_number):
			for m in range(self.train_record_number):
				for k in range(self.K):
					for j in range(self.K):
						Q_tilde[(self.K * n + k, self.K * m + j)] = (0.5 * (self.B**(k+j)) *
								self.train_data[n]*self.train_data[m] * 
								(self.kernel(self.data[n], self.data[m], self.gamma) + self.xi) - 
								(self.delta(n, m) * self.delta(k, j) * (self.B**k)))
		
		# construct QUBO matrix
		Q = np.zeros((self.K*self.train_record_number, self.K * self.train_record_number))

		for j in range(self.K * self.train_record_number):
			Q[(j, j)] = Q_tilde[(j, j)]
			
			for i in range(self.K*self.train_record_number):
				if i < j:
					Q[(i, j)] = Q_tilde[(i, j)] + Q_tilde[(j, i)]

		size_of_q = Q.shape[0]
		qubo = 	{
					(i, j): Q[i, j] 
					for i, j in product(range(size_of_q), range(size_of_q))
				}
		print("[1] QUBO matrix formed...")
		now = time.time()
		

		#TODO: add an option to pass user-defined hardware parameters such as chain_strength, anneal_schedule, etc.
		# for 3 cities the chose of chain_strength is 15. Were there any other
		# tests in variation in chain strenght depending on the number of cities? 

		if self.solver_type == "quantum":
			sampler = DWaveSampler()
			print("[2] Quantum Sampler initiating...")
			sampler = EmbeddingComposite(sampler)
			print("[3] Solving QUBO...")
			response = sampler.sample_qubo(Q, chain_strength=15, num_reads=1000)
			print("[4] QUBO Solved")
		elif self.solver_type == "simulated_annealing":
			print("[2] Simulated Annealing Sampler initiating...")
			sampler = SimulatedAnnealingSampler()
			print("[3] Solving QUBO...")
			response = sampler.sample_qubo(Q)
			print("[4] QUBO Solved")
		else:
			raise ValueError("Unsupported Solver type. Select from quantum or simulated_annealing.")

		if self.verbose:
				print("Total time to solve: ", time.time() - now)

		a = response.first.sample
		if self.verbose:
			print("Sample Set:\n",a)
			print("Sample Set Energy:\n", response.first.energy)

		self.alpha = {}
		for n in range(self.train_record_number):
			self.alpha[n] = sum([(self.B**k) * a[self.K * n + k] for k in range(self.K)])

		self.b = (
					sum([self.alpha[n] * (self.C - self.alpha[n]) * 
					(self.train_data[n] - ( sum([self.alpha[m] * self.train_data[m] * 
    										self.kernel(self.data[m], self.data[n], self.gamma) 
    										for m in range(self.train_record_number)])
    									  )
					) for n in range(self.train_record_number)]) / sum(
							[self.alpha[n] * (self.C - self.alpha[n]) 
							for n in range(self.train_record_number)])
    			)

		if self.verbose:
			print("alpha: \n",self.alpha)
			print("b:\n",self.b)

		return self.alpha, self.b

	def predict(self, x_test: list, external: bool = True) -> int:
		"""
		Make prediction.

		Args:
			x_test: a list of x test points.
			esternal: (bool) default is True. 
		Returns:
			f: (int)
        """
		N = len(self.alpha)
		f = (sum([self.alpha[n] * self.train_data[n]
			* self.kernel(self.data[n], x_test, self.gamma)
                for n in range(self.train_record_number)]) + self.b)
		
		if external:
			print("prediction = Negative") if f < 0.50 else print("prediction = Positive")
		return f

	def plotQSVM(self):
		"""
        Plot and save trained QSVM model.

		Args:
			None.
		Returns:
			None.
        """
		print("[0] Plotting QSVM")
		plt.figure()
		cm = plt.cm.RdBu
		xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 80),
							np.linspace(0.0, 1.0, 80))
		Z = []
		for row in range(len(xx)):
			Z_row = []
			for col in range(len(xx[row])):
				target = np.array([xx[row][col], yy[row][col]])
				Z_row.append(self.predict(target, external=False))
			
			Z.append(Z_row)

		cnt = plt.contourf(xx, yy, Z, levels=np.arange(-1, 1.1, 0.1), 
			cmap=cm, alpha=0.8, extend="both")
		plt.contour(xx, yy, Z, levels=[0.0], colors=("black",), 
			linestyles=("--",), linewidths=(1.5,))
		plt.colorbar(cnt, ticks=[-1,0,1])

		red_sv = []; blue_sv = []; red_pts = []; blue_pts = []

		for i in range(self.train_record_number):
			if self.alpha[i]:
				(blue_sv.append(self.data[i, :2]) 
					if self.train_data[i] == 1 else red_sv.append(self.data[i, :2]))
			else:
				(blue_pts.append(self.data[i, :2]) 
					if self.train_data[i] == 1 else red_pts.append(self.data[i, :2]))

		plt.scatter([el[0] for el in blue_sv],[el[1] for el in blue_sv],
			color='b', marker='^', edgecolors='k', label="Type 1 SV")
		
		plt.scatter([el[0] for el in red_sv],[el[1] for el in red_sv],
			color='r', marker='^', edgecolors='k', label="Type -1 SV")
		
		plt.scatter([el[0] for el in blue_pts],[el[1] for el in blue_pts],
			color='b', marker='o', edgecolors='k', label="Type 1 Train")
		
		plt.scatter([el[0] for el in red_pts],[el[1] for el in red_pts], 
			color='r', marker='o', edgecolors='k', label="Type -1 Train")
		
		plt.legend(loc='lower right', fontsize='x-small')
		plt.plot()
		plt.savefig("../results/QSVM_plot.png")
		print("[1] Figure saved at current directory as QSVM_plot.png")
		
	def calculateKPI(self):
		"""
        Calculates the ML evaluation metrics: precision, recall, f_score,
		accuracy.

		Args:
			None.
		Returns:
			None.
        """
		print("[0] Calculating KPI")
		tp , fp, tn, fn = 0, 0, 0, 0
		for i in range(self.train_record_number,
						self.train_record_number + self.validation_pts):
			cls = self.predict(self.data[i], external=False)
			y_i = self.train_data[i]
			if y_i == 1:
				if cls > 0:
					tp += 1
				else: 
					fp += 1
			else:
				if cls < 0:
					tn += 1 
				else:
				 fn += 1
		
		precision = tp / (tp +  fp)
		recall = tp / (tp + fn)
		f_score = tp / (tp + 1 / 2 * (fp + fn))
		accuracy = (tp + tn) / (tp + tn + fp + fn)

		print("f1_score = {} accuracy = {} precision = {} recall = {}"
				.format(f_score, accuracy,precision,recall))

	def printconfusionmatrix(self):
		"""
        Plot and save a confusion matrix of the QSVM model.

		Args:
			None.
		Returns:
			None.
        """
		print("[0] Confusion Matrix")
		y_test=[]
		for i in range(0, self.train_record_number + self.validation_pts):
			cls = self.predict(self.data[i], external=False)
			y_test.append(1.) if cls > 0 else y_test.append(-1)

		if self.verbose:
			print("y_test: \n",y_test)
			print("train dataset: \n", self.train_data)
		
		label = [1., -1]
		cm = confusion_matrix(y_test, self.train_data,labels=label)
		ax= plt.subplot()
		if self.verbose:
			print("Confusion Matrix: \n", cm)
		
		ax= plt.subplot()
		sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

		ax.set_xlabel('Predicted labels')
		ax.set_ylabel('True labels')
		ax.set_title('Confusion Matrix')
		ax.xaxis.set_ticklabels(['positive', 'negative'])
		ax.yaxis.set_ticklabels(['True', 'False'])
		plt.savefig("../results/Confusion_Matrix.png")
		print("[1] Confusion matrix plot saved as Confusion_Matrix.png")

	def _pref_metrics(self, threshold):
		tp, fp = 0, 0
		for i in range(self.train_record_number, 
			self.train_record_number + self.validation_pts):
			cls = self.predict(self.data[i], external=False)
			if self.train_data[i] >= threshold:
				if cls > 0:
					tp += 1
				else:
					fp += 1
			elif self.train_data[i] < threshold:
				if cls > 0:
					tp += 1
				else:
					fp += 1
		tpf = tp / (tp + fp)
		fpf = fp / (tp + fp)

		return tpf, fpf

	def AUROC_result(self):
		"""
        Calculates the area under the ROC curve

		Args:
			None.
		Returns:
			None.
        """
		TPF=[]
		FPF=[]
		THRESHOLD=[]
		i=0
		#increemental step size for threshold
		dx_step=0.0002
		while(i<=1):
			threshold = i
			tpf,fpf = self._pref_metrics(threshold)
			TPF.append(tpf)
			FPF.append(fpf)
			THRESHOLD.append(threshold)
			i += dx_step
		plt.plot(THRESHOLD,THRESHOLD,'--')
		plt.plot(FPF,TPF,marker='o')
		plt.xlabel("False Positive fraction (FPF)--->")
		plt.ylabel("True Positive fraction (TPF)--->")
		plt.title("ROC Curve")
		plt.show()
		area = np.trapz(TPF, dx=dx_step)
		print("AUC:Area under the ROC curve is", area)