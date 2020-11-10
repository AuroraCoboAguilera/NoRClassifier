import torch
from torch import nn

# Inference Network
class InferenceNet(nn.Module):
	def __init__(self, x_dim, z_dim, w_dim, K, hidden_dim, dropout, layers=3):
		super(InferenceNet, self).__init__()

		# For sampling in forward
		self.z_dim = z_dim
		self.w_dim = w_dim

		# self.activation = nn.ReLU()
		self.activation = nn.Tanh()

		# q(z|x)
		if layers == 3:
			# 3 layers for mean and for var
			self.qz_x_mean_layers = nn.ModuleList([
				nn.Linear(x_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, z_dim)
			])
			self.qz_x_var_layers = nn.ModuleList([
				nn.Linear(x_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, z_dim)
			])
		elif layers == 6:
			# 6 layers for mean and for var
			self.qz_x_mean_layers = nn.ModuleList([
				nn.Linear(x_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, z_dim)
			])
			self.qz_x_var_layers = nn.ModuleList([
				nn.Linear(x_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, z_dim)
			])

		# q(w|x)
		if layers == 3:
			# 3 layers for mean and for var
			self.qw_x_mean_layers = nn.ModuleList([
				nn.Linear(x_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, w_dim)
			])
			self.qw_x_var_layers = nn.ModuleList([
				nn.Linear(x_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, w_dim)
			])
		elif layers == 6:
			# 6 layers for mean and for var
			self.qw_x_mean_layers = nn.ModuleList([
				nn.Linear(x_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, w_dim)
			])
			self.qw_x_var_layers = nn.ModuleList([
				nn.Linear(x_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, w_dim)
			])

		# p(y|w, z)
		if layers == 3:
			# 3 layers
			self.py_wz_layers = nn.ModuleList([
				nn.Linear(w_dim + z_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, K)
			])
		elif layers == 6:
			# 6 layers
			self.py_wz_layers = nn.ModuleList([
				nn.Linear(w_dim + z_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, K)
			])



	# q(z|x)
	def qz_x(self, x):
		z_x_mean = x.clone()
		z_x_var = x.clone()
		for layer in self.qz_x_mean_layers:
			z_x_mean = layer(z_x_mean)
		for layer in self.qz_x_var_layers:
			z_x_var = layer(z_x_var)
		return z_x_mean, nn.functional.softplus(z_x_var)

	# q(w|x)
	def qw_x(self, x):
		w_x_mean = x.clone()
		w_x_var = x.clone()
		for layer in self.qw_x_mean_layers:
			w_x_mean = layer(w_x_mean)
		for layer in self.qw_x_var_layers:
			w_x_var = layer(w_x_var)
		return w_x_mean, nn.functional.softplus(w_x_var)		#TODO: torch.add(var, 0.1) ?

	# p(y|w, z)
	def py_wz(self, w, z):
		y_wz_logit = torch.cat((w, z), dim=1)
		for layer in self.py_wz_layers:
			y_wz_logit = layer(y_wz_logit)
		return y_wz_logit, nn.functional.softmax(y_wz_logit, dim=1)


	def forward(self, x):
		# Pass the input tensor through each of our operations

		# q(z|x)
		self.z_x_mean, self.z_x_var = self.qz_x(x)
		self.z_x_logvar = torch.log(self.z_x_var)

		# Sampling from q(z|x)
		std = torch.sqrt(self.z_x_var)		# torch.add(var, 1e-10)
		eps = torch.randn_like(std)
		self.z_x_sample = self.z_x_mean + eps * std

		# q(w|x)
		self.w_x_mean, self.w_x_var = self.qw_x(x)
		self.w_x_logvar = torch.log(self.w_x_var)

		# Sampling from q(w|x)
		std = torch.sqrt(self.w_x_var)  # torch.add(var, 1e-10)
		eps = torch.randn_like(std)
		self.w_x_sample = self.w_x_mean + eps * std

		# p(y|w, z)
		self.y_wz_logit, self.y_wz_prob = self.py_wz(self.w_x_sample, self.z_x_sample)
		self.y_wz_logprob = torch.log(torch.add(self.y_wz_prob, 1e-10))
