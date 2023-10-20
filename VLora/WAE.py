def gData(data):
    tensor=data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if use_cuda:
        tensor=tensor.cuda()
    return tensor
def gVar(data):
    return gData(data)


def train_D(self, context, context_lens, utt_lens, floors, response, res_lens):
    self.context_encoder.eval()
    self.discriminator.train()

    self.optimizer_D.zero_grad()

    batch_size = context.size(0)

    c = self.context_encoder(context, context_lens, utt_lens, floors)
    x, _ = self.utt_encoder(response[:, 1:], res_lens - 1)
    post_z = self.sample_code_post(x, c)
    errD_post = torch.mean(self.discriminator(torch.cat((post_z.detach(), c.detach()), 1)))
    errD_post.backward(one)

    prior_z = self.sample_code_prior(c)
    errD_prior = torch.mean(self.discriminator(torch.cat((prior_z.detach(), c.detach()), 1)))
    errD_prior.backward(minus_one)

    alpha = gData(torch.rand(batch_size, 1))
    alpha = alpha.expand(prior_z.size())
    interpolates = alpha * prior_z.data + ((1 - alpha) * post_z.data)
    interpolates = Variable(interpolates, requires_grad=True)
    d_input = torch.cat((interpolates, c.detach()), 1)
    disc_interpolates = torch.mean(self.discriminator(d_input))
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=gData(torch.ones(disc_interpolates.size())),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.contiguous().view(gradients.size(0), -1).norm(2,
                                                                                 dim=1) - 1) ** 2).mean() * self.lambda_gp
    gradient_penalty.backward()

    self.optimizer_D.step()
    costD = -(errD_prior - errD_post) + gradient_penalty
    return [('train_loss_D', costD.item())]
