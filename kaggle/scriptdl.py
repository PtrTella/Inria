from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()  
kernels = api.kernels_list(user='ptrtella')
print(f"Found {len(kernels)} kernels in the 'ptrtella/rawclip' dataset.")
print(f"Listing kernels: {[kernel.ref for kernel in kernels]}")
api.kernels_output('ptrtella/rawclip', path='./kaggle_output', force=True) 