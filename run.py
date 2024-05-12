from lightning.pytorch.cli import LightningCLI

from xlstm import xLSTM
from xlstm.stories import TinyStoriesLightning

def cli_main():
    '''
    Main function for the training script.
    '''
    
    # That's all it takes for LightningCLI to work!
    # No need to call .fit() or .test() or anything like that.
    cli = LightningCLI(
        xLSTM,
        TinyStoriesLightning,
    )

if __name__ == '__main__':    
    cli_main()