
import torch
class Diffusion:
    def __init__(self,start_schedule=0.001,end_schedule=0.2, timesteps=300):
        self.start_schedule=start_schedule
        self.end_schedule=end_schedule
        self.timesteps=timesteps

        #Create linear scheduler
        #Beta=[0.01 0.02 0.03 0.04 0.05]
        #alfa=[0.99 0.98 0.97 0.96 0.95]
        #alfa_cumprod=[0.99 (0.99*0.98) (0.99*0.98*0.97) (0.99*0.98*0.97*0.96)]
        self.betas=torch.linspace(start_schedule, end_schedule, timesteps) 
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)


        def forward(self, x_0, t, device):
            """
            Function to introduce noise to the image
            Loss_function=(ground_truth_noise-predicted_noise)
            predicted_noise=((\sqrt{alpha_cum_{t}})*image)+\sqrt{1-alpha_cum_{t}}*ground_truth_noise}
            where t is the time step that we want to find the noise
            It can be 4. The result will be as we pass the image 4 times through the filter
            This function finds a noise image in a given time step


            In theory the unet will learn the noise that we are introducing.
            We pass to the unet the number of time steps
            We can pass the label if we have many classes. Thus, it can be conditional unet
            """
            noise = torch.randn_like(x_0)  #create random noise with the same shape of x_0
            sqrt_alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod.sqrt(), t, x_0.shape)  #find the element in the list of alphas_cumprod and applies the squareroot
            sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x_0.shape) #find the element in the list of alphas_cumprod and applies the squareroot and substract from 1
            mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)  #find the mean 
            variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)  # find the std
            
            return mean + variance, noise.to(device)  #return the image with noise
    
    @torch.no_grad()
    def backward(self, x, t, model, **kwargs):

        """
        This function finds x_{t-1} from x_{t}. It does the reverse process
        x_{t-1}=\frac{1}{\sqrt{alpha_{t}}} *   ( x_{t}-  (\frac{Beta_{t}}{\sqrt{1-alpha_cum_{t}}})*model(x_{t}) ) + beta*normal_noise

        """
        
        betas_t = self.get_index_from_list(self.betas, t, x.shape)  #find beta at the position that we want
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)
        mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, **kwargs) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = betas_t  #this is the last term 

        if t == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            variance = torch.sqrt(posterior_variance_t) * noise 
            return mean + variance

    @staticmethod
    def get_index_from_list(values, t, x_shape):
        batch_size = t.shape[0]
        """
        pick the values from vals
        according to the indices stored in `t`
        """
        result = values.gather(-1, t.cpu())
        """
        if 
        x_shape = (5, 3, 64, 64)
            -> len(x_shape) = 4
            -> len(x_shape) - 1 = 3
            
        and thus we reshape `out` to dims
        (batch_size, 1, 1, 1)
        
        """
        return result.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

        