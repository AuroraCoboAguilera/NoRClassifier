'''

Gaussian Mixture Variational Autoencoder

'''

import os
import sys
import numpy as np
from tensorboardX import SummaryWriter

from torch.autograd import Variable
from LossFunctions import *
from InferenceNet import *
from GenerativeNet import *


class GMVAE:
    def __init__(self, args, seq2seq=False):

        # Read the parameters
        self.dataset_name = args.dataset_name

        self.cuda = args.cuda

        self.epochs = args.epochs
        self.l_rate = args.l_rate
        self.dropout = args.dropout
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay

        self.model_name = args.model_name

        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.sigma = args.sigma
        self.z_dim = args.z_dim
        self.w_dim = args.w_dim
        self.K = args.K

        self.remove = args.remove
        self.verbose = args.verbose
        self.board_dir = args.board_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.restore = args.restore
        self.summary = args.summary
        self.step_restore = args.step_restore
        self.checkpoint_step = args.checkpoint_step
        self.layers = args.layers

        # Create the generative network
        self.generative = GenerativeNet(self.input_dim, self.z_dim, self.w_dim, self.K, self.hidden_dim, self.dropout,
                                        self.cuda, self.sigma, self.layers)
        # Create the inference network
        self.inference = InferenceNet(self.input_dim, self.z_dim, self.w_dim, self.K, self.hidden_dim, self.dropout,
                                      self.layers)

        # Create the optimizers
        self.optimizer_inf = torch.optim.Adam(self.inference.parameters(), lr=self.l_rate, betas=(0.9, 0.999),
                                              weight_decay=self.weight_decay)
        self.optimizer_dec = torch.optim.Adam(self.generative.parameters(), lr=self.l_rate, betas=(0.9, 0.999),
                                              weight_decay=self.weight_decay)

        self.global_step = 0

        if not seq2seq:
            self.save_config(args)

    def train_epoch(self, train_loader):
        '''
			Train one epoch in the model

			Args:
				data_loader: (DataLoader) corresponding loader containing the training data

			Returns:
				average of all contributions of the loss function
		'''

        # Training mode (enable dropout)
        self.generative.train()
        self.inference.train()

        # Initialize variables
        total_loss = 0.
        log_lik = 0.
        cond_loss = 0.
        w_loss = 0.
        y_loss = 0.
        num_batches = 0.

        # iterate over the dataset
        for data in train_loader:
            if hasattr(train_loader.dataset, 'targets'):
                data = data[0].squeeze()
            # Pass data to cuda if GPUs are used
            if self.cuda == 1:
                x = Variable(data.type(torch.FloatTensor)).cuda()
            else:
                x = Variable(data.type(torch.FloatTensor))

            # Zero gradients, since optimizers will accumulate gradients for every backward.
            self.optimizer_inf.zero_grad()
            self.optimizer_dec.zero_grad()

            # Forward call and compute losses
            self.ELBO(x)

            # accumulate values
            total_loss += self.ELBO_loss.item()
            log_lik += self.log_lik.item()
            cond_loss += self.cond_prior_l.item()
            w_loss += self.w_prior_l.item()
            y_loss += self.y_prior_l.item()

            # Backward and optimize
            self.ELBO_loss.backward()

            self.optimizer_inf.step()
            self.optimizer_dec.step()

            # Grad norm clipping
            # torch.nn.utils.clip_grad_norm_(self.generative.parameters(), 5)
            # torch.nn.utils.clip_grad_norm_(self.inference.parameters(), 5)

            num_batches += 1.

            # Free memory
            del x

        # average per batch
        total_loss /= num_batches
        log_lik /= num_batches
        cond_loss /= num_batches
        w_loss /= num_batches
        y_loss /= num_batches

        return total_loss, log_lik, cond_loss, w_loss, y_loss

    def test_epoch(self, test_loader):
        '''
			Test the model with validation data

			Args:
				test_loader: (DataLoader) corresponding loader containing the test/validation data

			Return:
				accuracy for the given test data

		'''

        # Evaluation mode (disenable dropout)
        self.inference.eval()
        self.generative.eval()

        # Initialize variables
        total_loss = 0.
        log_lik = 0.
        cond_loss = 0.
        w_loss = 0.
        y_loss = 0.
        num_batches = 0.

        with torch.no_grad():
            for data in test_loader:
                if hasattr(test_loader.dataset, 'targets'):
                    data = data[0].squeeze()
                # Pass data to cuda if GPUs are used
                if self.cuda == 1:
                    x = Variable(data.type(torch.FloatTensor)).cuda()
                else:
                    x = Variable(data.type(torch.FloatTensor))

                # forward call and compute losses
                self.ELBO(x)

                # accumulate values
                total_loss += self.ELBO_loss.item()
                log_lik += self.log_lik.item()
                cond_loss += self.cond_prior_l.item()
                w_loss += self.w_prior_l.item()
                y_loss += self.y_prior_l.item()

                num_batches += 1.

                del x,

        # average per batch
        total_loss /= num_batches
        log_lik /= num_batches
        cond_loss /= num_batches
        w_loss /= num_batches
        y_loss /= num_batches

        return total_loss, log_lik, cond_loss, w_loss, y_loss

    def train(self, train_loader, val_loader, bunch_n=0):
        '''
			Train the model

			Args:
				train_loader: (DataLoader) corresponding loader containing the training data
				val_loader: (DataLoader) corresponding loader containing the validation data
				bunch_n: Index of the corresponding bunch of data to train

			Returns:
				output: (dict) contains the history of train/val loss
		'''

        # Sum quantity of parameters
        total_params_inference = sum(p.numel() for p in self.inference.parameters() if p.requires_grad)
        total_params_generative = sum(p.numel() for p in self.generative.parameters() if p.requires_grad)
        print('Trainable params: {}'.format(total_params_inference + total_params_generative))

        # Restore model if necessary
        if self.restore and bunch_n == 0:
            self.restore_model()

        # Move models to GPU (need time for initial run)
        if self.cuda:
            self.inference = self.inference.cuda()
            self.generative = self.generative.cuda()

        # Create object to write in the tensorboard if necessary
        if self.summary:
            writer = SummaryWriter(self.board_dir)

        # for name, param in self.generative.named_parameters():
        # 	print('name: ', name)
        # 	print(type(param))
        # 	print('param.shape: ', param.shape)
        # 	print('param.requires_grad: ', param.requires_grad)
        # 	print('=====')
        # for name, param in self.inference.named_parameters():
        # 	print('name: ', name)
        # 	print(type(param))
        # 	print('param.shape: ', param.shape)
        # 	print('param.requires_grad: ', param.requires_grad)
        # 	print('=====')

        # Bucle of epochs
        for epoch in range(1, self.epochs + 1):

            # Training epoch
            train_total_loss, train_log_lik, train_cond_loss, train_w_loss, train_y_loss = self.train_epoch(
                train_loader)

            self.global_step += 1

            if np.isnan(train_total_loss):
                print('ERROR: Encountered NaN, stopping training. Please check the learning_rate settings.')
                sys.exit()

            # Validation epoch
            val_total_loss, val_log_lik, val_cond_loss, val_w_loss, val_y_loss = self.test_epoch(val_loader)

            # Print losses in train and validation
            if self.verbose == 1:
                print("(Epoch %d / %d)" % (epoch, self.epochs))
                print("Train - Log-lik: %.5lf;  CP: %.5lf;  KL_w: %.5lf;  KL_y: %.5lf;" % \
                      (train_log_lik, train_cond_loss, train_w_loss, train_y_loss))
                print("Valid - Log-lik: %.5lf;  CP: %.5lf;  KL_w: %.5lf;  KL_y: %.5lf;" % \
                      (val_log_lik, val_cond_loss, val_w_loss, val_y_loss))
                print(
                    "Total Loss=Train: %.5lf; Val: %.5lf" % \
                    (train_total_loss, val_total_loss))
            else:
                print('(Epoch %d / %d) Train_Loss: %.3lf; Val_Loss: %.3lf' % \
                      (epoch, self.epochs, train_total_loss, val_total_loss))

            # Save checkpoint
            if (epoch) % self.checkpoint_step == 0:
                # Don't remove file for the first one saved
                if epoch == self.checkpoint_step and bunch_n == 0:
                    self.save_checkpoint(self.global_step, remove=0)
                else:
                    self.save_checkpoint(self.global_step, remove=self.remove)

            # Save losses in tensorboard
            if self.summary:
                self.write_to_tensorboard(writer, self.global_step, train_total_loss, train_log_lik, train_cond_loss,
                                          train_w_loss, train_y_loss, val_total_loss, val_log_lik, val_cond_loss,
                                          val_w_loss, val_y_loss)

    def save_checkpoint(self, global_step, remove=0):
        checkpoint = {
            'global_step': global_step,
            'inference_state_dict': self.inference.state_dict(),
            'generative_state_dict': self.generative.state_dict(),
            'optimizer_inf_state_dict': self.optimizer_inf.state_dict(),
            'optimizer_dec_state_dict': self.optimizer_dec.state_dict(),
        }

        checkpoint_path = '%s/step_%d.pt' % (self.checkpoint_dir, global_step)
        torch.save(checkpoint, checkpoint_path)

        # Remove previous checkpoint because of memory usage
        if remove:
            os.remove('%s/step_%d.pt' % (self.checkpoint_dir, global_step - self.checkpoint_step))

        print('=' * 100)
        print('Saving checkpoint to "{}".'.format(checkpoint_path))
        print('=' * 100 + '\n')

    def load_checkpoint(self, global_step, add_path=None):
        checkpoint_path = '%s/step_%d.pt' % (self.checkpoint_dir, global_step)
        if add_path != None:
            checkpoint_path = add_path + '/' + checkpoint_path
        if not (os.path.isfile(checkpoint_path)):
            print('ERROR: checkpoint not found. Revise the step to be loaded or change the restore flag to 0: %s' % (
                checkpoint_path))
            sys.exit()
        return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    def restore_model(self, add_path=None):
        print('Loading model from checkpoint in step %d...' % (self.step_restore))
        checkpoint = self.load_checkpoint(self.step_restore, add_path)
        self.inference.load_state_dict(checkpoint['inference_state_dict'])
        self.generative.load_state_dict(checkpoint['generative_state_dict'])
        self.optimizer_inf.load_state_dict(checkpoint['optimizer_inf_state_dict'])
        self.optimizer_dec.load_state_dict(checkpoint['optimizer_dec_state_dict'])
        self.global_step = checkpoint['global_step']

        if self.cuda:
            self.inference = self.inference.cuda()
            self.generative = self.generative.cuda()
            for state in self.optimizer_inf.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            for state in self.optimizer_dec.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    def write_to_tensorboard(self, writer, global_step, train_total_loss, train_log_lik, train_cond_loss, train_w_loss,
                             train_y_loss,
                             val_total_loss, val_log_lik, val_cond_loss, val_w_loss, val_y_loss):
        # scalars
        writer.add_scalar('train_total_loss', train_total_loss, global_step)
        writer.add_scalar('train_log_lik', train_log_lik, global_step)
        writer.add_scalar('train_cond_loss', train_cond_loss, global_step)
        writer.add_scalar('train_w_loss', train_w_loss, global_step)
        writer.add_scalar('train_y_loss', train_y_loss, global_step)
        writer.add_scalar('val_total_loss', val_total_loss, global_step)
        writer.add_scalar('val_log_lik', val_log_lik, global_step)
        writer.add_scalar('val_cond_loss', val_cond_loss, global_step)
        writer.add_scalar('val_w_loss', val_w_loss, global_step)
        writer.add_scalar('val_y_loss', val_y_loss, global_step)

    # for name, f in self.network.named_parameters():  # model is the NN model, f is one set of parameters of the model
    # 	# Create a dynamic name for the histogram summary
    # 	# Use current parameter shape to identify the variable
    # 	hist_name = 'hist' + name
    #
    # 	# Save the entire list of gradients of parameters f
    # 	writer.add_histogram(hist_name, f, global_step)
    #
    # 	# Save the norm of list of gradients of parameters f
    # 	# scalar_name = 'scalar' + name
    # 	# writer.add_scalar(scalar_name, torch.norm(f.grad.data).item(), global_step)

    def save_config(self, config):
        checkpoint = {
            'config': config,
        }

        checkpoint_path = '%s/config.pt' % (self.checkpoint_dir)
        torch.save(checkpoint, checkpoint_path)

        print('=' * 100)
        print('Saving config to "{}".'.format(checkpoint_path))
        print('=' * 100 + '\n')

    def ELBO(self, x):

        self.inference.forward(x)
        self.generative.forward(self.inference.z_x_sample)

        # ELBO regularizer
        self.log_lik = log_lik(self.sigma, x, self.generative.x_z_mean)

        self.cond_prior_l = cond_prior_loss(self.inference.z_x_sample, self.inference.z_x_mean, self.inference.z_x_var,
                                            self.inference.z_x_logvar, self.K, self.generative.z_wy_mean_stack,
                                            self.generative.z_wy_var_stack, self.generative.z_wy_logvar_stack,
                                            self.inference.y_wz_prob)
        self.w_prior_l = w_prior_loss(self.inference.w_x_mean, self.inference.w_x_var, self.inference.w_x_logvar)
        self.y_prior_l = y_prior_loss(self.K, self.inference.y_wz_logprob)

        self.ELBO_loss = -(self.log_lik - self.cond_prior_l - self.w_prior_l - self.y_prior_l)

    def reconstruct_data(self, data_loader):
        """Reconstruct Data

		Args:
			data_loader: (DataLoader) loader containing the data

		Returns:
			original: (array) array containing the original data
			reconstructed: (array) array containing the reconstructed data
		"""

        self.inference.eval()
        self.generative.eval()

        # obtain values
        it = iter(data_loader)
        test_batch_data = it.next()
        original = test_batch_data.data.numpy()

        # Pass data to cuda if GPUs are used
        if self.cuda == 1:
            test_batch_x = Variable(test_batch_data.type(torch.FloatTensor)).cuda()
        else:
            test_batch_x = Variable(test_batch_data.type(torch.FloatTensor))

        # Obtain reconstructed data
        self.inference.forward(test_batch_x)
        self.generative.forward(self.inference.z_x_sample)

        reconstructed = self.generative.x_z_sample.data.cpu().numpy()

        return original, reconstructed

    def random_generation(self, data_loader, number_batches=10):
        """Random generation for each category

		Args:
			number_batches: (int) number of elements to generate
		Returns:
			generated data according to number_batches
		"""

        self.inference.eval()
        self.generative.eval()

        # obtain values
        it = iter(data_loader)

        x_true_list = list()
        x_generated_list = list()
        for i in range(number_batches):
            batch_data = it.next()
            x_true_list.append(batch_data)

            # Pass data to cuda if GPUs are used
            if self.cuda == 1:
                z = Variable(torch.zeros((self.batch_size, self.z_dim))).cuda()
            else:
                z = Variable(torch.zeros((self.batch_size, self.z_dim)))

            # Generate a batch of samples
            self.generative.forward(z)

            generated = self.generative.x_z_sample_stack.cpu().detach().numpy()

            x_generated_list.append(generated)

        return np.stack(x_true_list), np.stack(x_generated_list)

    ###############################################################################################
    # METHODS TO USE WITH THE SEQ2SEQ
    ###############################################################################################

    def batch_generation(self):
        """Batch generation. We add the sampling from p(y)

			Returns:
			generated data
		"""

        self.inference.eval()
        self.generative.eval()

        # Pass data to cuda if GPUs are used
        if self.cuda == 1:
            z = Variable(torch.zeros((self.batch_size, self.z_dim))).cuda()
        else:
            z = Variable(torch.zeros((self.batch_size, self.z_dim)))

        # Generate a batch of samples
        self.generative.forward(z)

        generated = self.generative.x_z_sample_stack.cpu().detach().numpy()

        # Sampling from p(y), an uniform distribution to choose the class
        py = torch.distributions.Categorical(torch.tensor(1 / self.K * np.ones(self.K)))
        index = py.sample().item()

        return generated[index]

    def batch_reconstruction(self, x):
        """Reconstruct Data

		Args:
			x: (tensor: batch_size * input_dim) data to be reconstructed
		Returns:
			reconstructed: (array) array containing the reconstructed data
		"""
        self.inference.eval()
        self.generative.eval()

        # Pass data to cuda if GPUs are used
        if self.cuda == 1:
            test_batch_x = Variable(x.type(torch.FloatTensor)).cuda()
        else:
            test_batch_x = Variable(x.type(torch.FloatTensor))

        # Obtain reconstructed data
        self.inference.forward(test_batch_x)
        self.generative.forward(self.inference.z_x_sample)

        reconstructed = self.generative.x_z_sample.data.cpu().numpy()
        return reconstructed
