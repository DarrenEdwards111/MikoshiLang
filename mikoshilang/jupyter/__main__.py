"""Entry point for MikoshiLang Jupyter kernel."""

if __name__ == "__main__":
    # Fix for Anaconda/Windows ProactorEventLoop conflicts
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass  # Not required on Linux/macOS with SelectorEventLoop
    
    from ipykernel.kernelapp import IPKernelApp
    from .kernel import MikoshiLangKernel
    
    IPKernelApp.launch_instance(kernel_class=MikoshiLangKernel)
