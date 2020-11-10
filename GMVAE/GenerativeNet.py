import torch
from torch import nn


# Generative Network
class GenerativeNet(nn.Module):
	def __init__(self, x_dim, z_dim, w_dim, K, hidden_dim, dropout, use_cuda, sigma, layers=3):
		super(GenerativeNet, self).__init__()

		self.K = K
		# For sampling in forward
		self.w_dim = w_dim
		self.z_dim = z_dim

		# self.activation = nn.ReLU()
		self.activation = nn.Tanh()

		self.use_cuda = use_cuda

		self.sigma = sigma


		# p(z|w, y)
		if layers == 3:
			# 2 layers for the list of mean and of var
			self.qz_wy_mean_layers = nn.ModuleList([
				nn.Linear(w_dim, hidden_dim),
				self.activation,
			])
			self.qz_wy_var_layers = nn.ModuleList([
				nn.Linear(w_dim, hidden_dim),
				self.activation,
			])
		elif layers == 6:
			# 6 layers for the list of mean and of var
			self.qz_wy_mean_layers = nn.ModuleList([
				nn.Linear(w_dim, hidden_dim),
				self.activation,
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
			])
			self.qz_wy_var_layers = nn.ModuleList([
				nn.Linear(w_dim, hidden_dim),
				self.activation,
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
			])
		self.mean_linear_K = list()
		self.var_linear_K = list()
		for i in range(self.K):
			if self.use_cuda:
				self.mean_linear_K.append(nn.Linear(hidden_dim, z_dim).cuda())
				self.var_linear_K.append(nn.Linear(hidden_dim, z_dim).cuda())
			else:
				self.mean_linear_K.append(nn.Linear(hidden_dim, z_dim))
				self.var_linear_K.append(nn.Linear(hidden_dim, z_dim))

		# p(x|z)
		if layers == 3:
		# 3 layers for mean; var is a value sigma
			self.px_z_mean_layers = nn.ModuleList([
				nn.Linear(z_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, hidden_dim),
				self.activation,
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim, x_dim)
			])
		elif layers == 6:
			# 6 layers for mean; var is a value sigma
			self.px_z_mean_layers = nn.ModuleList([
				nn.Linear(z_dim, hidden_dim),
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
				nn.Linear(hidden_dim, x_dim)
			])

	# p(z|w, y)
	def pz_wy(self, w):
		z_wy_mean = w.clone()
		z_wy_var = w.clone()
		for layer in self.qz_wy_mean_layers:
			z_wy_mean = layer(z_wy_mean)
		for layer in self.qz_wy_var_layers:
			z_wy_var = layer(z_wy_var)

		mean_list = list()
		var_list = list()
		for i in range(self.K):
			mean = self.mean_linear_K[i](z_wy_mean)
			mean_list.append(mean)

			var = nn.functional.softplus(self.var_linear_K[i](z_wy_var))  # TODO: torch.add(var, 0.1)
			var_list.append(var)

		return torch.stack(mean_list), torch.stack(var_list)

	# p(x|z)
	def px_z(self, z):
		x_z_mean = z.clone()
		for layer in self.px_z_mean_layers:
			x_z_mean = layer(x_z_mean)
		return x_z_mean


	def forward(self, z):

		# Sampling from p(w)
		self.w_sample = torch.randn(z.size()[0], self.w_dim)
		if self.use_cuda:
			self.w_sample = self.w_sample.cuda()

		# p(z|w, y)
		self.z_wy_mean_stack, self.z_wy_var_stack = self.pz_wy(self.w_sample)
		self.z_wy_logvar_stack = torch.log(self.z_wy_var_stack)

		# Sampling from p(z|w, y)
		std = torch.sqrt(self.z_wy_var_stack)  # torch.add(var, 1e-10)
		eps = torch.randn_like(std)
		self.z_wy_sample_stack = self.z_wy_mean_stack + eps * std

		# p(x|z)
		self.x_z_mean_list = list()
		for i in range(self.K):
			self.x_z_mean_list.append(self.px_z(self.z_wy_sample_stack[i, :, :]))
		self.x_z_mean_stack = torch.stack(self.x_z_mean_list)

		# Sampling from p(x|z)
		eps = torch.randn_like(self.x_z_mean_stack)
		self.x_z_sample_stack = self.x_z_mean_stack + eps * torch.sqrt(torch.tensor(self.sigma))

		# Reconstruction
		# p(x|z)
		self.x_z_mean = self.px_z(z)

		# Sampling from p(x|z)
		eps = torch.randn_like(self.x_z_mean)
		self.x_z_sample = self.x_z_mean + eps * torch.sqrt(torch.tensor(self.sigma))
