# we next create our weight matrix and then do update rule to retrieve a half pattern
import torch.nn.functional as F

def continuous_update_rule(X,z,beta):
  return X.T @ F.softmax(beta * X @ z,dim=0)

def halve_continuous_img(img):
  H,W = img.reshape(200,200).shape
  i = deepcopy(img.reshape(200,200))
  i[H//2:H,:] = 0
  return i

def retrieve_store_continuous(imgs,N, beta=18,num_plot = 5):
  X = imgs[0:N,:]
  X = X.reshape(N,40000)
  print(X.shape)

  for j in range(num_plot):
    z = halve_continuous_img(X[j,:])
    z = z.reshape(40000)
    out = continuous_update_rule(X,z,beta)
    # plot
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    imgs = [X[j,:], z, out]
    titles = ["Original","Masked","Reconstruction"]
    for i, ax in enumerate(axs.flatten()):
      plt.sca(ax)
      plt.imshow(imgs[i].reshape(200,200))
      plt.title(titles[i])
    plt.show()


retrieve_store_continuous(imgs,5)
retrieve_store_continuous(imgs, 20,beta=0.25,num_plot = 10)

betas = [0.001,0.2,0.5,1,2,4,8]
X = imgs[0:10,:].reshape(10,40000)
z = halve_continuous_img(X[0,:])
z = z.reshape(40000)
for beta in betas:
  print("Beta: ", beta)
  out = continuous_update_rule(X,z,beta)
  out = out.reshape(200,200)
  plt.imshow(out)
  plt.show()