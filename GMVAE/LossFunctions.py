'''

Loss functions of the ELBO

'''

import torch


def reconstruction_loss(sigma, x_batch, x_recons_mean):

	loss = 0.5 / sigma * torch.sum(torch.pow((x_recons_mean - x_batch), 2), dim=1)

	return -torch.mean(loss)


def cond_prior_loss(z_x, z_x_mean, z_x_var, z_x_logvar, K, z_wy_mean_stack, z_wy_var_stack, z_wy_logvar_stack, py_wz):

	logq = -0.5 * torch.sum(z_x_logvar, dim=1) - 0.5 * torch.sum(torch.div(torch.pow((z_x - z_x_mean), 2), z_x_var), dim=1)

	z_wy = z_x.repeat(K, 1, 1)																		# [K, batch_size, z_dim]
	log_det_sigma = torch.transpose(torch.sum(z_wy_logvar_stack, dim=2), 1, 0)						# [batch_size, K]
	aux = torch.sum(torch.div(torch.pow((z_wy - z_wy_mean_stack), 2), z_wy_var_stack), dim=2) 		# [K, batch_size]
	aux = torch.transpose(aux, 1, 0)																# [batch_size, K]
	logp = -0.5 * torch.sum(torch.mul(py_wz, log_det_sigma), dim=1) -0.5 * torch.sum(torch.mul(py_wz, aux), dim=1)

	loss = logq - logp

	return torch.mean(loss)


def w_prior_loss(w_x_mean, w_x_var, w_x_logvar):

	loss = 0.5 * torch.sum(w_x_var + torch.pow(w_x_mean, 2) - 1 - w_x_logvar, dim=1)

	return torch.mean(loss)


def y_prior_loss(K, log_py_wz):

	loss = -torch.log(torch.tensor(K).float()) - 1/K * torch.sum(log_py_wz, dim=1)

	return torch.mean(loss)


def log_lik(sigma, x_batch, x_recons_mean):

	loss = -0.5 / sigma * torch.sum(torch.pow((x_recons_mean - x_batch), 2), dim=1)

	return torch.mean(loss)