"""Entry point for MikoshiLang Jupyter kernel."""

if __name__ == "__main__":
    from ipykernel.kernelapp import IPKernelApp
    from .kernel import MikoshiLangKernel
    
    IPKernelApp.launch_instance(kernel_class=MikoshiLangKernel)
